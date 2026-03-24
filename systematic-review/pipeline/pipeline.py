"""
Main pipeline orchestrator.
Runs records through all stages with full state persistence.
"""
import json
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from datetime import datetime

from .models import (
    RawRecord, DedupRecord, ScreenedRecord, PipelineRecord,
    DecisionLabel, PipelineStage, ExtractedRecord
)
from .importer import load_all_from_directory, save_records, load_records
from .deduplicator import deduplicate, filter_unique, save_deduped, load_deduped
from agents.screener import screen_title_abstract, screen_fulltext, screen_second_pass, screen_fulltext_round2
from agents.extractor import extract_and_assess
from agents.phase1_extractor import run_phase1_for_record
from config.settings import settings

logger = logging.getLogger(__name__)

# State file paths
STATE_FILE = "data/pipeline_state.jsonl"


class PipelineRunner:
    """
    Manages the full systematic review pipeline.
    State is persisted after each stage.
    Records are separated into four output buckets.
    """

    def __init__(self):
        self.records: Dict[str, PipelineRecord] = {}
        # Stores results of the last run of each stage for UI display
        self.stage_log: Dict[str, Any] = {}
        # Set to the name of the stage currently running ("" when idle)
        self.running_stage: str = ""
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [settings.raw_dir, settings.deduped_dir, settings.screened_dir,
                  settings.extracted_dir, settings.output_dir, settings.pdf_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────
    # STAGE 1: IMPORT
    # ──────────────────────────────────────────────────

    def run_import(self) -> Dict:
        """Import all files from raw data directory.
        Replaces any previously imported records — safe to re-run.
        """
        self.running_stage = "import"
        logger.info("=== STAGE: IMPORT ===")
        raw_records, file_stats = load_all_from_directory(settings.raw_dir)

        # Clear old import-stage records so re-running import is idempotent.
        # Records that have already progressed beyond IMPORT (screened / extracted)
        # are preserved so work isn't lost if someone accidentally re-runs import.
        self.records = {
            rid: pr for rid, pr in self.records.items()
            if pr.pipeline_stage not in (PipelineStage.IMPORT, PipelineStage.DEDUP)
        }

        for r in raw_records:
            pr = PipelineRecord(record_id=r.record_id, raw=r)
            pr.pipeline_stage = PipelineStage.IMPORT
            pr.final_decision = DecisionLabel.UNCERTAIN
            self.records[r.record_id] = pr

        self._save_state()
        stats = {
            "imported": len(raw_records),
            "per_file": file_stats,
            "completed_at": datetime.utcnow().isoformat(),
        }
        self.stage_log["import"] = stats
        self.running_stage = ""
        logger.info(f"Import complete: {stats}")
        return stats

    # ──────────────────────────────────────────────────
    # STAGE 2: DEDUPLICATION
    # ──────────────────────────────────────────────────

    def run_deduplication(self) -> Dict:
        """Deduplicate imported records."""
        self.running_stage = "dedup"
        logger.info("=== STAGE: DEDUPLICATION ===")
        raw_list = [pr.raw for pr in self.records.values() if pr.raw is not None]
        deduped_list, stats = deduplicate(raw_list)

        for drec in deduped_list:
            pr = self.records.get(drec.record_id)
            if pr:
                pr.dedup = drec
                if drec.is_duplicate:
                    pr.update_stage(PipelineStage.DEDUP, DecisionLabel.EXCLUDE)
                else:
                    pr.pipeline_stage = PipelineStage.DEDUP

        # Save deduped records
        save_deduped(deduped_list, os.path.join(settings.deduped_dir, "deduped.jsonl"))
        self._save_state()
        self.stage_log["dedup"] = {**stats, "completed_at": datetime.utcnow().isoformat()}
        self.running_stage = ""
        logger.info(f"Dedup complete: {stats}")
        return stats

    # ──────────────────────────────────────────────────
    # STAGE 3: TITLE/ABSTRACT SCREENING
    # ──────────────────────────────────────────────────

    def run_title_screening(self, limit: Optional[int] = None) -> Dict:
        """Run two-agent title/abstract screening on unique records."""
        self.running_stage = "title_screening"
        logger.info("=== STAGE: TITLE/ABSTRACT SCREENING ===")

        candidates = [
            pr for pr in self.records.values()
            if pr.dedup is not None and not pr.dedup.is_duplicate
            and (
                # Normal path: records not yet screened
                pr.pipeline_stage in (PipelineStage.IMPORT, PipelineStage.DEDUP)
                # Re-screen path: records that errored mid-run (stage advanced to
                # TITLE_SCREENING but no result was stored).  Records where agents
                # ran successfully but disagreed (UNCERTAIN + result exists) are
                # left alone for human review.
                or (
                    pr.pipeline_stage == PipelineStage.TITLE_SCREENING
                    and pr.final_decision == DecisionLabel.UNCERTAIN
                    and not (pr.screened and pr.screened.title_screening)
                )
            )
        ]

        if limit:
            candidates = candidates[:limit]

        counts = {
            DecisionLabel.INCLUDE: 0,
            DecisionLabel.EXCLUDE: 0,
            DecisionLabel.UNCERTAIN: 0,
        }
        lock = threading.Lock()
        completed = [0]

        def _screen_one(pr: PipelineRecord):
            logger.info(f"Screening [{pr.record_id[:8]}]: {pr.dedup.title[:60]}...")
            result = screen_title_abstract(pr.dedup)
            return pr, result

        with ThreadPoolExecutor(max_workers=settings.screening_workers) as executor:
            futures = {executor.submit(_screen_one, pr): pr for pr in candidates}
            for future in as_completed(futures):
                try:
                    pr, result = future.result()
                    if pr.screened is None:
                        pr.screened = ScreenedRecord(**pr.dedup.model_dump())
                    pr.screened.title_screening = result
                    pr.screened.current_decision = result.final_decision
                    pr.update_stage(PipelineStage.TITLE_SCREENING, result.final_decision)
                    with lock:
                        counts[result.final_decision] = counts.get(result.final_decision, 0) + 1
                        completed[0] += 1
                        n = completed[0]
                except Exception as e:
                    pr = futures[future]
                    logger.error(f"Screening error for {pr.record_id}: {e}")
                    pr.update_stage(PipelineStage.TITLE_SCREENING, DecisionLabel.UNCERTAIN)
                    with lock:
                        counts[DecisionLabel.UNCERTAIN] += 1
                        completed[0] += 1
                        n = completed[0]

                if n % 50 == 0:
                    with lock:
                        self._save_state()
                    logger.info(f"Checkpoint saved at {n}/{len(candidates)} records")

        self._save_state()
        self._export_screening_lists()
        log = {str(k): v for k, v in counts.items()}
        log["completed_at"] = datetime.utcnow().isoformat()
        self.stage_log["title_screening"] = log
        self.running_stage = ""
        logger.info(f"Title screening complete: {counts}")
        return counts

    # ──────────────────────────────────────────────────
    # STAGE 3b: SECOND-PASS SCREENING
    # ──────────────────────────────────────────────────

    def run_second_pass_screening(self, limit: Optional[int] = None) -> Dict:
        """Run strict two-agent second-pass screening on title-screened INCLUDE records.

        Applies tighter exclusion criteria (lecture notes, non-QDA LLM use,
        writing assistance, chatbot UX, education-only, user perception studies,
        conceptual/theoretical papers, quantitative-only, human-only QDA).

        Auto-excludes when both agents agree (≥0.60 conf). Auto-includes when
        both agents agree clearly (≥confidence_threshold). Everything else →
        Needs Human Verification (sent to human review queue).
        """
        self.running_stage = "second_pass_screening"
        logger.info("=== STAGE: SECOND-PASS SCREENING ===")

        candidates = [
            pr for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
            and pr.screened is not None
            and pr.screened.second_pass_screening is None  # not yet second-pass screened
        ]

        if limit:
            candidates = candidates[:limit]

        logger.info(f"Second-pass screening {len(candidates)} included records")

        counts = {
            DecisionLabel.INCLUDE: 0,
            DecisionLabel.EXCLUDE: 0,
            DecisionLabel.UNCERTAIN: 0,
        }
        lock = threading.Lock()
        completed = [0]

        def _screen_one(pr: PipelineRecord):
            # Extract first-pass rationale to give context to second-pass agents
            a1_rat = a2_rat = ""
            if pr.screened and pr.screened.title_screening:
                ts = pr.screened.title_screening
                if ts.agent1:
                    a1_rat = ts.agent1.rationale or ""
                if ts.agent2:
                    a2_rat = ts.agent2.rationale or ""
            result = screen_second_pass(pr.dedup, a1_rat, a2_rat)
            return pr, result

        with ThreadPoolExecutor(max_workers=settings.screening_workers) as executor:
            futures = {executor.submit(_screen_one, pr): pr for pr in candidates}
            for future in as_completed(futures):
                try:
                    pr, result = future.result()
                    pr.screened.second_pass_screening = result
                    pr.screened.current_decision = result.final_decision
                    pr.update_stage(PipelineStage.TITLE_SCREENING, result.final_decision)
                    with lock:
                        counts[result.final_decision] = counts.get(result.final_decision, 0) + 1
                        completed[0] += 1
                        n = completed[0]
                except Exception as e:
                    pr = futures[future]
                    logger.error(f"Second-pass error for {pr.record_id}: {e}")
                    # On error, keep as INCLUDE (do not lose records silently)
                    with lock:
                        counts[DecisionLabel.INCLUDE] += 1
                        completed[0] += 1
                        n = completed[0]

                if n % 50 == 0:
                    with lock:
                        self._save_state()
                    logger.info(f"Second-pass checkpoint at {n}/{len(candidates)}")

        self._save_state()
        log = {str(k): v for k, v in counts.items()}
        log["completed_at"] = datetime.utcnow().isoformat()
        self.stage_log["second_pass_screening"] = log
        self.running_stage = ""
        logger.info(f"Second-pass screening complete: {counts}")
        return counts

    # ──────────────────────────────────────────────────
    # STAGE 3c: FULL-TEXT AUTO-DOWNLOAD
    # ──────────────────────────────────────────────────

    def run_fulltext_download(self, limit: Optional[int] = None) -> Dict:
        """Auto-download full texts for the 302 included records.

        Tries Unpaywall → Semantic Scholar → OpenAlex → Europe PMC for each
        record (by DOI). Records where a PDF is obtained are kept as INCLUDE
        with fulltext_available=True. Records that cannot be auto-fetched are
        set to FULL_TEXT_NEEDED so the researcher can upload them manually.

        A PRISMA snapshot is saved automatically at the start of this stage
        to record the post-human-verification counts.
        """
        from agents.fulltext_downloader import try_download_fulltext

        self.running_stage = "fulltext_download"
        logger.info("=== STAGE: FULL-TEXT AUTO-DOWNLOAD ===")

        # Save PRISMA snapshot BEFORE download (captures post-human-verification state)
        self.save_prisma_snapshot("post_human_verification_pre_fulltext_download")

        candidates = [
            pr for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
            and pr.screened is not None
            and not pr.screened.fulltext_available
        ]

        if limit:
            candidates = candidates[:limit]

        logger.info(f"Attempting auto-download for {len(candidates)} records")

        counts = {
            "auto_downloaded": 0,
            "manual_needed": 0,
            "already_available": 0,
            "total_candidates": len(candidates),
            "processed": 0,
        }
        # Publish initial progress so UI can show candidate count immediately
        self.stage_log["fulltext_download"] = {**counts, "in_progress": True}

        download_log: list = []
        lock = threading.Lock()
        pdf_dir = Path(settings.pdf_dir)

        def _download_one(pr: PipelineRecord):
            doi = pr.dedup.doi if pr.dedup else ""
            title = pr.dedup.title if pr.dedup else ""
            url = pr.dedup.url if pr.dedup else ""
            success, source = try_download_fulltext(doi, title, pr.record_id, pdf_dir, record_url=url)
            return pr, success, source, doi, title

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_download_one, pr): pr for pr in candidates}
            for future in as_completed(futures):
                try:
                    pr, success, source, doi, title = future.result()
                    with lock:
                        counts["processed"] += 1
                        # Update live progress in stage_log for UI polling
                        self.stage_log["fulltext_download"] = {**counts, "in_progress": True}
                        if success:
                            # Resolve actual saved file (may be .pdf or .txt)
                            txt_path = pdf_dir / f"{pr.record_id}.txt"
                            saved = (txt_path if txt_path.exists()
                                     else pdf_dir / f"{pr.record_id}.pdf")
                            pr.screened.pdf_path = str(saved)
                            pr.screened.fulltext_available = True
                            pr.updated_at = datetime.utcnow()
                            counts["auto_downloaded"] += 1
                            download_log.append({
                                "record_id": pr.record_id,
                                "title": title[:100],
                                "doi": doi,
                                "status": "downloaded",
                                "source": source,
                            })
                        else:
                            pr.final_decision = DecisionLabel.FULL_TEXT_NEEDED
                            pr.updated_at = datetime.utcnow()
                            counts["manual_needed"] += 1
                            download_log.append({
                                "record_id": pr.record_id,
                                "title": title[:100],
                                "doi": doi,
                                "status": "manual_needed",
                                "source": None,
                            })
                except Exception as e:
                    pr = futures[future]
                    logger.error(f"Download error for {pr.record_id}: {e}")
                    with lock:
                        pr.final_decision = DecisionLabel.FULL_TEXT_NEEDED
                        counts["manual_needed"] += 1

        self._save_state()
        self._export_fulltext_needed_list()
        self._export_download_log(download_log)

        counts.pop("in_progress", None)
        log = {**counts, "completed_at": datetime.utcnow().isoformat()}
        self.stage_log["fulltext_download"] = log
        self.running_stage = ""
        logger.info(f"Full-text download complete: {counts}")
        return counts

    def retry_fulltext_download(self, limit: Optional[int] = None) -> Dict:
        """Retry downloading for all FULL_TEXT_NEEDED records using extended sources
        (arXiv, CORE, direct URL added on top of original sources).
        Records that succeed are moved back to INCLUDE + fulltext_available=True.
        """
        from agents.fulltext_downloader import try_download_fulltext

        self.running_stage = "fulltext_retry"
        logger.info("=== STAGE: FULL-TEXT RETRY DOWNLOAD ===")

        candidates = [
            pr for pr in self.records.values()
            if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
        ]
        if limit:
            candidates = candidates[:limit]

        logger.info(f"Retrying download for {len(candidates)} FULL_TEXT_NEEDED records")

        counts = {
            "auto_downloaded": 0,
            "still_needed": 0,
            "total_candidates": len(candidates),
            "processed": 0,
        }
        self.stage_log["fulltext_retry"] = {**counts, "in_progress": True}

        lock = threading.Lock()
        pdf_dir = Path(settings.pdf_dir)

        def _download_one(pr: PipelineRecord):
            doi = pr.dedup.doi if pr.dedup else ""
            title = pr.dedup.title if pr.dedup else ""
            url = pr.dedup.url if pr.dedup else ""
            success, source = try_download_fulltext(doi, title, pr.record_id, pdf_dir, record_url=url)
            return pr, success, source

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_download_one, pr): pr for pr in candidates}
            for future in as_completed(futures):
                try:
                    pr, success, source = future.result()
                    with lock:
                        counts["processed"] += 1
                        self.stage_log["fulltext_retry"] = {**counts, "in_progress": True}
                        if success:
                            txt_path = pdf_dir / f"{pr.record_id}.txt"
                            saved = (txt_path if txt_path.exists()
                                     else pdf_dir / f"{pr.record_id}.pdf")
                            if pr.screened:
                                pr.screened.pdf_path = str(saved)
                                pr.screened.fulltext_available = True
                            pr.final_decision = DecisionLabel.INCLUDE
                            pr.updated_at = datetime.utcnow()
                            counts["auto_downloaded"] += 1
                        else:
                            counts["still_needed"] += 1
                except Exception as e:
                    logger.error(f"Retry download error: {e}")
                    with lock:
                        counts["still_needed"] += 1

        self._save_state()
        counts.pop("in_progress", None)
        log = {**counts, "completed_at": datetime.utcnow().isoformat()}
        self.stage_log["fulltext_retry"] = log
        self.running_stage = ""
        logger.info(f"Retry download complete: {counts}")
        return counts

    def _export_download_log(self, log_rows: list):
        """Write download attempt log to output directory."""
        import json as _json
        out_path = Path(settings.output_dir) / "fulltext_download_log.json"
        with open(out_path, "w") as f:
            _json.dump(log_rows, f, indent=2)

    # ──────────────────────────────────────────────────
    # PRISMA SNAPSHOTS (for plot generation)
    # ──────────────────────────────────────────────────

    def save_prisma_snapshot(self, label: str = "") -> Dict:
        """Save current PRISMA counts as a timestamped snapshot.

        Snapshots accumulate in data/output/prisma_snapshots.jsonl so the
        complete PRISMA flow can be reproduced and plotted at any time.
        """
        import json as _json
        counts = self.get_prisma_counts()
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label or f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "counts": counts,
            "total_records": len(self.records),
            "bucket_totals": {
                "included": sum(1 for pr in self.records.values()
                                if pr.final_decision == DecisionLabel.INCLUDE),
                "excluded": sum(1 for pr in self.records.values()
                                if pr.final_decision == DecisionLabel.EXCLUDE),
                "uncertain": sum(1 for pr in self.records.values()
                                 if pr.final_decision == DecisionLabel.UNCERTAIN),
                "full_text_needed": sum(1 for pr in self.records.values()
                                        if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED),
            },
        }
        out_path = Path(settings.output_dir) / "prisma_snapshots.jsonl"
        with open(out_path, "a") as f:
            f.write(_json.dumps(snapshot) + "\n")
        logger.info(f"PRISMA snapshot saved: {label} → {counts}")
        return snapshot

    def get_prisma_snapshots(self) -> list:
        """Return all saved PRISMA snapshots."""
        import json as _json
        out_path = Path(settings.output_dir) / "prisma_snapshots.jsonl"
        if not out_path.exists():
            return []
        snapshots = []
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        snapshots.append(_json.loads(line))
                    except Exception:
                        pass
        return snapshots

    # ──────────────────────────────────────────────────
    # STAGE 4: FULL-TEXT SCREENING
    # ──────────────────────────────────────────────────

    def run_fulltext_screening(self, limit: Optional[int] = None) -> Dict:
        """Run full-text screening on records that passed title screening."""
        self.running_stage = "fulltext_screening"
        logger.info("=== STAGE: FULL-TEXT SCREENING ===")

        candidates = [
            pr for pr in self.records.values()
            if pr.screened is not None
            and (
                # First pass: records that passed title screening, not yet fulltext-screened
                (pr.pipeline_stage == PipelineStage.TITLE_SCREENING
                 and pr.final_decision == DecisionLabel.INCLUDE)
                or
                # Uncertain at title screening: given benefit of the doubt, need fulltext review
                (pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
                 and pr.final_decision == DecisionLabel.UNCERTAIN
                 and (pr.screened.fulltext_screening is None
                      or not pr.screened.fulltext_screening.human_verified))
                or
                # Retry pass: records that previously had no PDF (manual upload case)
                (pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
                 and pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED)
            )
        ]

        if limit:
            candidates = candidates[:limit]

        counts = {
            DecisionLabel.INCLUDE: 0,
            DecisionLabel.EXCLUDE: 0,
            DecisionLabel.UNCERTAIN: 0,
            DecisionLabel.FULL_TEXT_NEEDED: 0,
        }

        # Separate records with/without PDFs before launching threads
        pdf_ready = []
        for pr in candidates:
            pdf_path = self._find_pdf(pr)
            if not pdf_path:
                pr.screened.fulltext_available = False
                pr.update_stage(PipelineStage.FULLTEXT_SCREENING, DecisionLabel.FULL_TEXT_NEEDED)
                counts[DecisionLabel.FULL_TEXT_NEEDED] += 1
            else:
                pr.screened.pdf_path = pdf_path
                pr.screened.fulltext_available = True
                pdf_ready.append(pr)

        lock = threading.Lock()
        completed = [0]

        def _fulltext_one(pr: PipelineRecord):
            fulltext = self._extract_pdf_text(pr.screened.pdf_path)
            result = screen_fulltext(pr.dedup, fulltext)
            return pr, result

        with ThreadPoolExecutor(max_workers=settings.screening_workers) as executor:
            futures = {executor.submit(_fulltext_one, pr): pr for pr in pdf_ready}
            for future in as_completed(futures):
                try:
                    pr, result = future.result()
                    pr.screened.fulltext_screening = result
                    pr.screened.current_decision = result.final_decision
                    pr.update_stage(PipelineStage.FULLTEXT_SCREENING, result.final_decision)
                    with lock:
                        counts[result.final_decision] = counts.get(result.final_decision, 0) + 1
                        completed[0] += 1
                        n = completed[0]
                except Exception as e:
                    pr = futures[future]
                    logger.error(f"Full-text screening error for {pr.record_id}: {e}")
                    pr.update_stage(PipelineStage.FULLTEXT_SCREENING, DecisionLabel.UNCERTAIN)
                    with lock:
                        counts[DecisionLabel.UNCERTAIN] += 1
                        completed[0] += 1
                        n = completed[0]

                if n % 50 == 0:
                    with lock:
                        self._save_state()
                    logger.info(f"Checkpoint saved at {n}/{len(pdf_ready)} records")

        self._save_state()
        self._export_fulltext_needed_list()
        log = {str(k): v for k, v in counts.items()}
        log["completed_at"] = datetime.utcnow().isoformat()
        self.stage_log["fulltext_screening"] = log
        self.running_stage = ""
        logger.info(f"Full-text screening complete: {counts}")
        return counts

    # ──────────────────────────────────────────────────
    # STAGE 4b: SECOND ROUND FULL-TEXT SCREENING
    # Applied to the 115 studies included after round-1 full-text screening.
    # Uses refined criteria requiring evaluation of LLM-generated outputs.
    # ──────────────────────────────────────────────────

    def run_second_fulltext_screening(self, limit: Optional[int] = None) -> Dict:
        """Run second-round full-text screening on studies included after round-1.

        Targets records where:
          - final_decision == INCLUDE
          - pipeline_stage == FULLTEXT_SCREENING  (confirmed included after round-1 + human review)
          - second_fulltext_screening is None (not yet screened in round-2)

        Also retries records left as UNCERTAIN at SECOND_FULLTEXT_SCREENING stage
        that have not been human-verified.
        """
        self.running_stage = "second_fulltext_screening"
        logger.info("=== STAGE: SECOND ROUND FULL-TEXT SCREENING ===")

        candidates = [
            pr for pr in self.records.values()
            if pr.screened is not None
            and pr.screened.fulltext_available
            and (
                # Round-1 included — not yet round-2 screened
                (pr.final_decision == DecisionLabel.INCLUDE
                 and pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
                 and pr.screened.second_fulltext_screening is None)
                or
                # Previously sent to human at round-2 and not yet verified
                (pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
                 and pr.final_decision == DecisionLabel.UNCERTAIN
                 and pr.screened.second_fulltext_screening is not None
                 and not pr.screened.second_fulltext_screening.human_verified)
            )
        ]

        if limit:
            candidates = candidates[:limit]

        logger.info(f"Round-2 screening candidates: {len(candidates)}")

        counts = {
            DecisionLabel.INCLUDE: 0,
            DecisionLabel.EXCLUDE: 0,
            DecisionLabel.UNCERTAIN: 0,
        }

        lock = threading.Lock()
        completed = [0]

        def _round2_one(pr: PipelineRecord):
            fulltext = self._extract_pdf_text(pr.screened.pdf_path)
            # Pass round-1 rationale as context for agents
            round1_rationale = ""
            if pr.screened.fulltext_screening:
                r1 = pr.screened.fulltext_screening
                parts = []
                if r1.agent1:
                    parts.append(f"Agent 1 ({r1.agent1.model_used}): {r1.agent1.rationale}")
                if r1.agent2:
                    parts.append(f"Agent 2 ({r1.agent2.model_used}): {r1.agent2.rationale}")
                if r1.human_verified and r1.human_rationale:
                    parts.append(f"Human reviewer: {r1.human_rationale}")
                round1_rationale = "\n".join(parts)
            result = screen_fulltext_round2(pr.dedup, fulltext, round1_rationale)
            return pr, result

        with ThreadPoolExecutor(max_workers=settings.screening_workers) as executor:
            futures = {executor.submit(_round2_one, pr): pr for pr in candidates}
            for future in as_completed(futures):
                try:
                    pr, result = future.result()
                    pr.screened.second_fulltext_screening = result
                    pr.screened.current_decision = result.final_decision
                    pr.update_stage(PipelineStage.SECOND_FULLTEXT_SCREENING, result.final_decision)
                    with lock:
                        counts[result.final_decision] = counts.get(result.final_decision, 0) + 1
                        completed[0] += 1
                        n = completed[0]
                except Exception as e:
                    pr = futures[future]
                    logger.error(f"Round-2 screening error for {pr.record_id}: {e}")
                    pr.update_stage(PipelineStage.SECOND_FULLTEXT_SCREENING, DecisionLabel.UNCERTAIN)
                    with lock:
                        counts[DecisionLabel.UNCERTAIN] += 1
                        completed[0] += 1
                        n = completed[0]

                if n % 20 == 0:
                    with lock:
                        self._save_state()
                    logger.info(f"Checkpoint saved at {n}/{len(candidates)} records")

        self._save_state()
        log = {str(k): v for k, v in counts.items()}
        log["completed_at"] = datetime.utcnow().isoformat()
        self.stage_log["second_fulltext_screening"] = log
        self.running_stage = ""
        logger.info(f"Round-2 full-text screening complete: {counts}")
        return counts

    def reset_second_fulltext_screening(self, force: bool = False) -> Dict:
        """Reset round-2 full-text screening results, reverting records to INCLUDE at FULLTEXT_SCREENING stage."""
        reset_count = 0
        for pr in self.records.values():
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING or (
                force and pr.screened and pr.screened.second_fulltext_screening is not None
            ):
                if pr.screened:
                    pr.screened.second_fulltext_screening = None
                # Revert to round-1 included state
                pr.final_decision = DecisionLabel.INCLUDE
                pr.pipeline_stage = PipelineStage.FULLTEXT_SCREENING
                pr.updated_at = datetime.utcnow()
                reset_count += 1
        self._save_state()
        logger.info(f"Reset {reset_count} records from round-2 full-text screening")
        return {"reset": reset_count}

    def get_round2_exclusion_summary(self) -> Dict:
        """Return a breakdown of round-2 exclusion reasons for PRISMA reporting."""
        from collections import Counter
        excluded = [
            pr for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
            and pr.screened is not None
            and pr.screened.second_fulltext_screening is not None
        ]

        code_counter: Counter = Counter()
        for pr in excluded:
            r2 = pr.screened.second_fulltext_screening
            # Prefer agent1 exclusion code; fall back to agent2
            code = None
            if r2.agent1 and r2.agent1.exclusion_code:
                code = r2.agent1.exclusion_code
            elif r2.agent2 and r2.agent2.exclusion_code:
                code = r2.agent2.exclusion_code
            code_counter[code or "unknown"] += 1

        from config.settings import ROUND2_EXCLUSION_CODES
        summary = {
            "total_excluded": len(excluded),
            "by_code": dict(code_counter),
            "by_code_with_labels": {
                code: {
                    "count": cnt,
                    "label": ROUND2_EXCLUSION_CODES.get(code, code),
                }
                for code, cnt in code_counter.items()
            },
        }
        return summary

    # ──────────────────────────────────────────────────
    # STAGE 5: DATA EXTRACTION
    # ──────────────────────────────────────────────────

    def run_extraction(self, limit: Optional[int] = None, record_id: Optional[str] = None) -> Dict:
        """Run data extraction on confirmed included studies."""
        logger.info("=== STAGE: DATA EXTRACTION ===")

        # Accept records at any pipeline stage as long as they are INCLUDE + have fulltext.
        # This covers manually uploaded PDFs (still at TITLE_SCREENING) and papers that
        # bypassed fulltext screening.
        included = [
            pr for pr in self.records.values()
            if pr.final_decision == DecisionLabel.INCLUDE
            and pr.screened is not None
            and pr.screened.fulltext_available
        ]

        # If a specific record_id is requested, filter to just that one
        if record_id:
            included = [pr for pr in included if pr.record_id == record_id]
            if not included:
                # Also allow re-extracting already-extracted records
                included = [
                    pr for pr in self.records.values()
                    if pr.record_id == record_id
                    and pr.final_decision == DecisionLabel.INCLUDE
                    and pr.screened is not None
                    and pr.screened.fulltext_available
                ]

        if limit and not record_id:
            included = included[:limit]

        study_counter = self._get_next_study_id()
        counts = {"extracted": 0, "uncertain": 0, "errors": 0}

        # Resume: skip records already extracted in a previous run
        included = [
            pr for pr in included
            if pr.pipeline_stage != PipelineStage.EXTRACTION or pr.extracted is None
        ]
        if not included and not record_id:
            logger.info("All records already extracted — nothing to do.")
            self.running_stage = ""
            return counts

        CHECKPOINT_EVERY = 5  # save state + CSVs after every N papers
        processed = 0

        for pr in included:
            study_id = f"SR-{datetime.utcnow().year}-{study_counter:03d}"
            study_counter += 1
            pr.study_id = study_id

            try:
                pdf_path = pr.screened.pdf_path
                fulltext = self._extract_pdf_text(pdf_path) if pdf_path else ""

                extracted = extract_and_assess(
                    record_id=pr.record_id,
                    study_id=study_id,
                    title=pr.dedup.title,
                    fulltext=fulltext,
                )
                pr.extracted = extracted

                if extracted.decision == DecisionLabel.UNCERTAIN:
                    pr.update_stage(PipelineStage.EXTRACTION, DecisionLabel.UNCERTAIN)
                    counts["uncertain"] += 1
                else:
                    pr.update_stage(PipelineStage.EXTRACTION, DecisionLabel.INCLUDE)
                    counts["extracted"] += 1

            except Exception as e:
                logger.error(f"Extraction error for {pr.record_id}: {e}")
                pr.update_stage(PipelineStage.EXTRACTION, DecisionLabel.UNCERTAIN)
                counts["errors"] += 1

            processed += 1
            if processed % CHECKPOINT_EVERY == 0:
                self._save_state()
                self._save_checkpoint_csvs()
                logger.info(f"Checkpoint at {processed} papers: {counts}")

        self._save_state()
        self._save_checkpoint_csvs()
        self.stage_log["extraction"] = {**counts, "completed_at": datetime.utcnow().isoformat()}
        self.running_stage = ""
        logger.info(f"Extraction complete: {counts}")
        return counts

    # ──────────────────────────────────────────────────
    # PHASE 1: OPEN DATA EXTRACTION
    # Step 1 – GPT-5 extracts from Methods + Results
    # Step 2 – Verification model checks evidence alignment
    # Outputs saved to data/extracted/phase1/
    # ──────────────────────────────────────────────────

    def run_phase1_extraction(self, limit: Optional[int] = None) -> Dict:
        """
        Run Phase 1 open extraction on all confirmed included studies that
        have full text available.

        Each paper produces two saved outputs:
          • gpt5_extractions.jsonl  — GPT-5 open extraction (step 1)
          • verifications.jsonl     — verification model output (step 2)
          • phase1_complete.jsonl   — combined record

        Records already present in phase1_complete.jsonl are skipped so the
        method is safe to re-run after interruption.
        """
        from pathlib import Path as _Path
        import json as _json

        self.running_stage = "phase1_extraction"
        logger.info("=== PHASE 1: DATA EXTRACTION ===")

        included = [
            pr for pr in self.records.values()
            if pr.final_decision == DecisionLabel.INCLUDE
            and pr.screened is not None
            and pr.screened.fulltext_available
        ]

        if limit:
            included = included[:limit]

        # Build set of already-processed paper IDs to allow safe resume
        complete_path = _Path(settings.phase1_output_dir) / "phase1_complete.jsonl"
        already_done: set = set()
        if complete_path.exists():
            with open(complete_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rec = _json.loads(line)
                            pid = rec.get("record_id") or rec.get("paper_id")
                            if pid:
                                already_done.add(pid)
                        except Exception:
                            pass

        counts = {"extracted": 0, "skipped_already_done": 0, "errors": 0}

        for pr in included:
            paper_id = pr.study_id or pr.record_id
            if pr.record_id in already_done or paper_id in already_done:
                counts["skipped_already_done"] += 1
                continue

            try:
                pdf_path = pr.screened.pdf_path
                fulltext = self._extract_pdf_text(pdf_path) if pdf_path else ""

                run_phase1_for_record(
                    record_id=pr.record_id,
                    study_id=pr.study_id or "",
                    title=pr.dedup.title if pr.dedup else "",
                    fulltext=fulltext,
                )
                counts["extracted"] += 1

            except Exception as e:
                logger.error(f"[Phase1] Error for {pr.record_id}: {e}")
                counts["errors"] += 1

        self.stage_log["phase1_extraction"] = {
            **counts,
            "total_eligible": len(included),
            "output_dir": str(settings.phase1_output_dir),
            "completed_at": datetime.utcnow().isoformat(),
        }
        self.running_stage = ""
        logger.info(f"Phase 1 extraction complete: {counts}")
        return counts

    # ──────────────────────────────────────────────────
    # HUMAN VERIFICATION
    # ──────────────────────────────────────────────────

    def apply_human_decision(
        self,
        record_id: str,
        decision: DecisionLabel,
        rationale: str,
        reviewer: str = "human",
        corrections: Optional[dict] = None,
    ) -> bool:
        """Apply a human reviewer's decision to a record."""
        pr = self.records.get(record_id)
        if not pr:
            return False

        pr.final_decision = decision
        pr.updated_at = datetime.utcnow()

        # Update screening result if at screening stage
        if pr.screened:
            active_screening = (
                pr.screened.second_fulltext_screening
                if pr.screened.second_fulltext_screening
                else (
                    pr.screened.fulltext_screening
                    if pr.screened.fulltext_screening
                    else pr.screened.title_screening
                )
            )
            if active_screening:
                active_screening.human_verified = True
                active_screening.human_decision = decision
                active_screening.human_rationale = rationale
                active_screening.human_reviewer = reviewer
                active_screening.human_timestamp = datetime.utcnow()

        # Update extraction if at extraction stage
        if pr.extracted and corrections:
            pr.extracted.human_verified = True
            pr.extracted.human_corrections = corrections
            pr.extracted.human_reviewer = reviewer
            # Apply corrections to final extraction
            if pr.extracted.extraction_final:
                for field, value in corrections.items():
                    if hasattr(pr.extracted.extraction_final, field):
                        setattr(pr.extracted.extraction_final, field, value)

        self._save_state()
        return True

    def mark_included_for_review(self) -> Dict:
        """Mark all title-screened INCLUDE records as UNCERTAIN for second-pass human review.

        This sends the 1038 AI-included records back to the Human Verification
        queue so a human can confirm or override each decision before full-text
        screening begins.
        """
        count = 0
        for pr in self.records.values():
            if (pr.pipeline_stage == PipelineStage.TITLE_SCREENING
                    and pr.final_decision == DecisionLabel.INCLUDE):
                pr.final_decision = DecisionLabel.UNCERTAIN
                pr.updated_at = datetime.utcnow()
                # Tag the screening result so the UI can display context
                if pr.screened and pr.screened.title_screening:
                    pr.screened.title_screening.human_verified = False
                count += 1
        if count:
            self._save_state()
        logger.info(f"mark_included_for_review: {count} records queued for human review")
        return {"updated": count}

    # ──────────────────────────────────────────────────
    # OUTPUT / REPORTING
    # ──────────────────────────────────────────────────

    def get_records_by_decision(self) -> Dict[str, List[PipelineRecord]]:
        """Return records grouped by final decision."""
        groups: Dict[str, List[PipelineRecord]] = {
            DecisionLabel.INCLUDE: [],
            DecisionLabel.EXCLUDE: [],
            DecisionLabel.UNCERTAIN: [],
            DecisionLabel.FULL_TEXT_NEEDED: [],
        }
        for pr in self.records.values():
            bucket = groups.get(pr.final_decision, groups[DecisionLabel.UNCERTAIN])
            bucket.append(pr)
        return groups

    def get_prisma_counts(self) -> Dict:
        """Return PRISMA 2020 flow counts."""
        total = len(self.records)
        dups = sum(1 for pr in self.records.values()
                   if pr.dedup and pr.dedup.is_duplicate)
        after_dedup = total - dups

        # Records waiting for title screening (dedup done, not yet screened)
        awaiting_screening = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage in (PipelineStage.IMPORT, PipelineStage.DEDUP)
            and not (pr.dedup and pr.dedup.is_duplicate)
        )

        title_screen_excluded = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
        )
        # Title-included = all records that passed title/abstract screening
        # (INCLUDE + FULL_TEXT_NEEDED + those that advanced to fulltext/extraction)
        title_screen_included = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision in (DecisionLabel.INCLUDE, DecisionLabel.FULL_TEXT_NEEDED)
        ) + sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage in (
                PipelineStage.FULLTEXT_SCREENING,
                PipelineStage.SECOND_FULLTEXT_SCREENING,
                PipelineStage.EXTRACTION,
            )
        )
        # Of those, how many have full text already retrieved
        # Count all records with a PDF at any fulltext-screening stage
        fulltext_retrieved = sum(
            1 for pr in self.records.values()
            if pr.screened is not None
            and pr.screened.fulltext_available
            and pr.pipeline_stage in (
                PipelineStage.FULLTEXT_SCREENING,
                PipelineStage.SECOND_FULLTEXT_SCREENING,
                PipelineStage.EXTRACTION,
            )
        ) + sum(
            # Still at TITLE_SCREENING with PDF (fulltext screening not yet run)
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
            and pr.screened is not None
            and pr.screened.fulltext_available
        )
        # How many still need manual upload
        fulltext_needed = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
        )
        # Round-1 fulltext screening breakdown
        fulltext_r1_excluded = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
        )
        fulltext_r1_uncertain = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.UNCERTAIN
        )
        # "passed round 1" = moved to SECOND_FULLTEXT_SCREENING, or still INCLUDE at FULLTEXT_SCREENING
        fulltext_r1_passed = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
        ) + sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
        )
        # Round-2 fulltext screening breakdown
        fulltext_r2_excluded = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
        )
        fulltext_r2_included = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
        )
        fulltext_r2_uncertain = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.UNCERTAIN
        )
        # backward-compat alias: r1 excluded (what old code called fulltext_excluded)
        fulltext_excluded = fulltext_r1_excluded
        final_included = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.INCLUDE
            and pr.pipeline_stage in (
                PipelineStage.FULLTEXT_SCREENING,
                PipelineStage.SECOND_FULLTEXT_SCREENING,
                PipelineStage.EXTRACTION,
            )
        )
        # "Needs human verification" — broken down by stage so the UI can show
        # separate labels for title, fulltext, and extraction reviews.
        needs_human_title = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.UNCERTAIN
            and pr.pipeline_stage == PipelineStage.TITLE_SCREENING
        )
        needs_human_fulltext = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.UNCERTAIN
            and pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
        )
        needs_human_round2 = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.UNCERTAIN
            and pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
        )
        needs_human_extraction = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.UNCERTAIN
            and pr.pipeline_stage == PipelineStage.EXTRACTION
        )
        needs_human = needs_human_title + needs_human_fulltext + needs_human_round2 + needs_human_extraction
        final_extracted = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.EXTRACTION
            and pr.final_decision == DecisionLabel.INCLUDE
        )

        # Verification checks (logged for debugging)
        # fulltext_retrieved + fulltext_needed should equal title_screen_included
        # fulltext_r1_excluded + fulltext_r1_passed + fulltext_r1_uncertain + fulltext_needed
        #   should equal title_screen_included
        # fulltext_r2_excluded + fulltext_r2_included + fulltext_r2_uncertain
        #   should equal fulltext_r1_passed (when round-2 has run)

        return {
            "identified": total,
            "duplicates_removed": dups,
            "after_dedup": after_dedup,
            "awaiting_title_screening": awaiting_screening,
            "title_abstract_excluded": title_screen_excluded,
            "title_abstract_included": title_screen_included,
            # Full-text retrieval
            "fulltext_retrieved": fulltext_retrieved,
            "full_text_needed": fulltext_needed,
            # Round-1 full-text screening
            "full_text_excluded": fulltext_r1_excluded,          # backward compat
            "fulltext_r1_excluded": fulltext_r1_excluded,
            "fulltext_r1_uncertain": fulltext_r1_uncertain,
            "fulltext_r1_passed": fulltext_r1_passed,
            # Round-2 full-text screening
            "fulltext_r2_excluded": fulltext_r2_excluded,
            "fulltext_r2_included": fulltext_r2_included,
            "fulltext_r2_uncertain": fulltext_r2_uncertain,
            # Final
            "final_included": final_included,
            # Extraction
            "final_extracted": final_extracted,
            # Needs-human breakdown
            "needs_human_verification": needs_human,
            "needs_human_title_screening": needs_human_title,
            "needs_human_fulltext_screening": needs_human_fulltext,
            "needs_human_round2_screening": needs_human_round2,
            "needs_human_extraction": needs_human_extraction,
        }

    def get_full_prisma_report(self) -> Dict:
        """Return a complete PRISMA 2020 report for publication.

        Includes:
        - Per-database record counts (source_db breakdown)
        - All PRISMA flow counts (both screening rounds)
        - Round-2 exclusion reasons grouped by R2-EC code with counts
        - Formatted ASCII flowchart string ready to print / paste
        """
        from collections import Counter
        from config.settings import ROUND2_EXCLUSION_CODES

        p = self.get_prisma_counts()

        # ── Source-database breakdown ─────────────────────────────────
        all_db: Counter = Counter()
        unique_db: Counter = Counter()
        for pr in self.records.values():
            d = pr.raw or pr.dedup
            db = (d.source_db if d else None) or "Unknown"
            all_db[db] += 1
            if not (pr.dedup and pr.dedup.is_duplicate):
                unique_db[db] += 1

        # ── Round-2 exclusion reason breakdown ───────────────────────
        r2_excl = [
            pr for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.SECOND_FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
            and pr.screened is not None
            and pr.screened.second_fulltext_screening is not None
        ]
        ec_counter: Counter = Counter()
        for pr in r2_excl:
            r2 = pr.screened.second_fulltext_screening
            # Human-overridden exclusion: prefer human note, fall back to agent code
            code = None
            if r2.human_verified and r2.human_decision == DecisionLabel.EXCLUDE:
                # Try agent codes first (agents identified the reason)
                code = (r2.agent1.exclusion_code if r2.agent1 else None) or \
                       (r2.agent2.exclusion_code if r2.agent2 else None) or \
                       "human_decision"
            else:
                code = (r2.agent1.exclusion_code if r2.agent1 else None) or \
                       (r2.agent2.exclusion_code if r2.agent2 else None) or \
                       "unknown"
            ec_counter[code] += 1

        # User-friendly label map
        friendly = {
            "R2-EC1": "Non-empirical / ineligible publication type",
            "R2-EC2": "No real-world qualitative data (synthetic/simulated only)",
            "R2-EC3": "LLM not used for qualitative analysis (no coding/thematic/content analysis)",
            "R2-EC4": "Low-level NLP task (sentiment analysis, classification, text mining)",
            "R2-EC5": "Non-analytic or auxiliary LLM use only (writing, chatbot, interview only)",
            "R2-EC6": "Focus on LLM evaluation / perception / interaction, not analysis",
            "R2-EC7": "Coding-only, no theme development or interpretive synthesis",
            "R2-EC8": "No evaluation of LLM-generated outputs",
            "R2-EC9": "Methodological unclearity (LLM role unclear)",
            "R2-EC10": "Platform/tool-only study (no analytic performance evaluated)",
            "R2-EC11": "Unclear eligibility",
            "human_decision": "Human decision (exclusion reason noted in rationale)",
            "unknown": "Exclusion code not recorded",
        }

        exclusion_reasons = [
            {
                "code": code,
                "count": cnt,
                "label": friendly.get(code, code),
                "full_description": ROUND2_EXCLUSION_CODES.get(code, ""),
            }
            for code, cnt in sorted(ec_counter.items(), key=lambda x: -x[1])
        ]

        # ── Round-1 exclusion breakdown (title/abstract stage codes) ─
        r1_excl = [
            pr for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
            and pr.screened is not None
            and pr.screened.fulltext_screening is not None
        ]
        r1_ec_counter: Counter = Counter()
        for pr in r1_excl:
            r1 = pr.screened.fulltext_screening
            code = None
            if r1.human_verified and r1.human_decision == DecisionLabel.EXCLUDE:
                code = (r1.agent1.exclusion_code if r1.agent1 else None) or \
                       (r1.agent2.exclusion_code if r1.agent2 else None) or \
                       "human_decision"
            else:
                code = (r1.agent1.exclusion_code if r1.agent1 else None) or \
                       (r1.agent2.exclusion_code if r1.agent2 else None) or \
                       "unknown"
            r1_ec_counter[code] += 1

        r1_friendly = {
            "EC-A": "Non-empirical or non-primary research",
            "EC-B": "LLM not used for qualitative analysis",
            "EC-C": "Code-only / low-level processing without synthesis",
            "EC-D": "Non-qualitative or NLP-only tasks",
            "EC-E": "Writing, assistance, or content generation only",
            "EC-F": "Focus on user interaction/perception/evaluation of LLMs",
            "EC-G": "Non-analytic or irrelevant LLM use cases",
            "EC-H": "Misaligned research focus",
            "human_decision": "Human decision",
            "unknown": "Exclusion code not recorded",
        }
        r1_exclusion_reasons = [
            {
                "code": code,
                "count": cnt,
                "label": r1_friendly.get(code, code),
            }
            for code, cnt in sorted(r1_ec_counter.items(), key=lambda x: -x[1])
        ]

        # ── Formatted ASCII flowchart ─────────────────────────────────
        flowchart = self._format_prisma_flowchart(p, all_db, exclusion_reasons)

        return {
            "prisma_counts": p,
            "source_databases_all": dict(all_db),
            "source_databases_unique": dict(unique_db),
            "round1_total_excluded": len(r1_excl),
            "round1_exclusion_by_reason": r1_exclusion_reasons,
            "round2_total_excluded": len(r2_excl),
            "round2_exclusion_by_reason": exclusion_reasons,
            "prisma_flowchart_text": flowchart,
        }

    def _format_prisma_flowchart(self, p: Dict, db_counts: Dict, r2_reasons: list) -> str:
        """Return a formatted ASCII PRISMA 2020 flowchart."""
        # Database row
        dbs = [(k, v) for k, v in sorted(db_counts.items(), key=lambda x: -x[1])]
        db_cells = "  |  ".join(f"{k} (n={v})" for k, v in dbs)

        # Round-2 exclusion list (sorted by count desc)
        r2_excl_lines = "\n".join(
            f"   • {r['label']} (n={r['count']})"
            for r in r2_reasons if r['count'] > 0
        ) or "   (none)"

        ta = p.get("title_abstract_included", "?")
        r1_passed = p.get("fulltext_r1_passed", "?")
        final = p.get("final_included", "?")
        r1_excl_n = p.get("fulltext_r1_excluded", "?")
        r2_excl_n = p.get("fulltext_r2_excluded", "?")
        r2_unc_n = p.get("fulltext_r2_uncertain", 0)
        ft_retr = p.get("fulltext_retrieved", "?")
        ft_need = p.get("full_text_needed", "?")

        lines = [
            "══════════════════════════════════════════════════════════",
            "  PRISMA 2020 Flow Diagram",
            "══════════════════════════════════════════════════════════",
            "",
            f"  DATABASES SEARCHED",
            f"  {db_cells}",
            f"  Total records identified: {p.get('identified', '?')}",
            "",
            f"  ↓  {p.get('duplicates_removed','?')} duplicates removed",
            "",
            f"  After deduplication: {p.get('after_dedup','?')} unique records",
            "",
            f"  ↓  {p.get('title_abstract_excluded','?')} excluded at title/abstract screening",
            "",
            f"  Full-text assessed for eligibility: {ta}",
            f"  │  Retrieved: {ft_retr}",
            f"  │  Not obtained (manual upload needed): {ft_need}",
            "",
            "  ─────────────────────────────────────────────────────────",
            "  ROUND-1 FULL-TEXT SCREENING",
            "  ─────────────────────────────────────────────────────────",
            f"  ↓  {r1_excl_n} excluded at round-1 full-text screening",
            "",
            f"  Passed round-1 (sent to round-2): {r1_passed}",
            "",
            "  ─────────────────────────────────────────────────────────",
            "  ROUND-2 FULL-TEXT SCREENING (refined criteria)",
            "  ─────────────────────────────────────────────────────────",
            f"  ↓  {r2_excl_n} excluded at round-2 full-text screening:",
            r2_excl_lines,
        ]
        if r2_unc_n:
            lines.append(f"  ↓  {r2_unc_n} resolved via human verification")
        lines += [
            "",
            "  ══════════════════════════════════════════════════════",
            f"  Studies included in final synthesis: {final}",
            "  ══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)

    def export_evidence_table(self) -> List[Dict]:
        """Export final evidence table for included studies."""
        rows = []
        for pr in self.records.values():
            if (pr.final_decision == DecisionLabel.INCLUDE
                    and pr.extracted
                    and pr.extracted.extraction_final):
                ef = pr.extracted.extraction_final
                rows.append({
                    "study_id": pr.study_id,
                    "title": ef.title or pr.dedup.title if pr.dedup else "",
                    "authors": ef.authors,
                    "year": ef.year,
                    "journal_venue": ef.journal_venue,
                    "domain": ef.domain,
                    "model_name": ef.model_name,
                    "analytic_task": ", ".join(ef.analytic_task or []),
                    "workflow_structure": ef.workflow_structure,
                    "qualitative_approach": ef.qualitative_approach,
                    "human_comparison": ef.human_comparison,
                    "qa_score": pr.extracted.qa_score.total_score if pr.extracted.qa_score else None,
                    "key_findings": ef.key_findings,
                    "limitations_reported": ef.limitations_reported,
                })
        return rows

    def reset_screening(self) -> Dict:
        """Roll back all non-human-verified title screening results to DEDUP stage.
        Use this to re-run title screening from scratch with updated settings.
        Human-verified decisions are preserved.
        """
        rolled_back = 0
        kept = 0
        for pr in self.records.values():
            if pr.pipeline_stage != PipelineStage.TITLE_SCREENING:
                continue
            human_verified = (
                pr.screened
                and pr.screened.title_screening
                and pr.screened.title_screening.human_verified
            )
            if human_verified:
                kept += 1
                continue
            # Roll back to DEDUP stage, clear screening result
            pr.pipeline_stage = PipelineStage.DEDUP
            pr.final_decision = DecisionLabel.UNCERTAIN
            if pr.screened:
                pr.screened.title_screening = None
                pr.screened.current_decision = None
            rolled_back += 1
        self._save_state()
        stats = {"rolled_back": rolled_back, "human_verified_kept": kept}
        self.stage_log["reset_screening"] = {**stats, "completed_at": datetime.utcnow().isoformat()}
        logger.info(f"Reset screening: {stats}")
        return stats

    def reset_failed_screenings(self) -> Dict:
        """Roll back ONLY error-path records (stage=TITLE_SCREENING, UNCERTAIN, no result stored).
        Safe to call while valid INCLUDE/EXCLUDE results are preserved.
        """
        rolled_back = 0
        for pr in self.records.values():
            if pr.pipeline_stage != PipelineStage.TITLE_SCREENING:
                continue
            if pr.final_decision != DecisionLabel.UNCERTAIN:
                continue
            # Only reset records where no screening result was stored (i.e. crashed before saving)
            if pr.screened and pr.screened.title_screening:
                continue
            pr.pipeline_stage = PipelineStage.DEDUP
            if pr.screened:
                pr.screened.current_decision = None
            rolled_back += 1
        self._save_state()
        stats = {"rolled_back": rolled_back}
        self.stage_log["reset_failed_screenings"] = {**stats, "completed_at": datetime.utcnow().isoformat()}
        logger.info(f"Reset failed screenings: {stats}")
        return stats

    def restore_bulk_excluded(self, rationale_marker: str = "manual bulk decision") -> Dict:
        """Reverse a bulk human-exclusion by restoring records to UNCERTAIN.

        Targets records at FULLTEXT_SCREENING + EXCLUDE that have no fulltext_screening
        result — meaning they were excluded before any full-text review happened.
        This is the reliable fingerprint of the accidental bulk exclusion.

        Also clears any human_verified flags on title_screening where the rationale
        matches (belt-and-suspenders for cases where the flag was stored).
        """
        restored = 0
        for pr in self.records.values():
            if pr.pipeline_stage != PipelineStage.FULLTEXT_SCREENING:
                continue
            if pr.final_decision != DecisionLabel.EXCLUDE:
                continue
            # Only restore records that were never actually fulltext-screened
            if pr.screened and pr.screened.fulltext_screening:
                continue
            # Clear any human_verified flag on title_screening if present
            if pr.screened and pr.screened.title_screening:
                ts = pr.screened.title_screening
                ts.human_verified = False
                ts.human_decision = None
                ts.human_rationale = None
                ts.human_reviewer = None
                ts.human_timestamp = None
            pr.final_decision = DecisionLabel.UNCERTAIN
            pr.updated_at = datetime.utcnow()
            restored += 1
        self._save_state()
        stats = {"restored": restored}
        self.stage_log["restore_bulk_excluded"] = {**stats, "completed_at": datetime.utcnow().isoformat()}
        logger.info(f"Restore bulk excluded: {stats}")
        return stats

    def reset_fulltext_screening(self, force: bool = False) -> Dict:
        """Reset all fulltext screening so it can be re-run on all 302 records.

        For each record at FULLTEXT_SCREENING stage:
        - Clears the fulltext_screening result
        - Restores stage/decision based on original title_screening outcome:
            title INCLUDE   → TITLE_SCREENING + INCLUDE   (picked up by fulltext_screening)
            title UNCERTAIN → FULLTEXT_SCREENING + UNCERTAIN (picked up by fulltext_screening)
            no title result → FULLTEXT_SCREENING + UNCERTAIN
        Human-verified fulltext decisions are preserved unless force=True.
        Use force=True to also reset bulk-excluded records.
        """
        reset = 0
        skipped_human = 0
        for pr in self.records.values():
            if pr.pipeline_stage not in (
                PipelineStage.FULLTEXT_SCREENING, PipelineStage.EXTRACTION
            ):
                continue
            # Preserve human-verified fulltext decisions unless force=True
            if (not force
                    and pr.screened
                    and pr.screened.fulltext_screening
                    and pr.screened.fulltext_screening.human_verified):
                skipped_human += 1
                continue
            # Clear fulltext result
            if pr.screened:
                pr.screened.fulltext_screening = None
            # Restore stage/decision from title screening
            title_decision = (
                pr.screened.title_screening.final_decision
                if pr.screened and pr.screened.title_screening
                else None
            )
            if title_decision == DecisionLabel.INCLUDE:
                pr.pipeline_stage = PipelineStage.TITLE_SCREENING
                pr.final_decision = DecisionLabel.INCLUDE
            else:
                # Uncertain at title screening → keep at FULLTEXT_SCREENING for AI review
                pr.pipeline_stage = PipelineStage.FULLTEXT_SCREENING
                pr.final_decision = DecisionLabel.UNCERTAIN
            pr.updated_at = datetime.utcnow()
            reset += 1
        self._save_state()
        stats = {"reset": reset, "human_verified_kept": skipped_human}
        self.stage_log["reset_fulltext_screening"] = {
            **stats, "completed_at": datetime.utcnow().isoformat()
        }
        logger.info(f"Reset fulltext screening: {stats}")
        return stats

    # ──────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────

    def _find_pdf(self, pr: PipelineRecord) -> Optional[str]:
        """Look for full-text file (.pdf or .txt) by record_id or DOI."""
        pdf_dir = Path(settings.pdf_dir)
        if not pdf_dir.exists():
            return None

        # Try record_id with both extensions
        for ext in (".pdf", ".txt"):
            candidate = pdf_dir / f"{pr.record_id}{ext}"
            if candidate.exists():
                return str(candidate)

        # Try DOI (sanitized) with both extensions
        if pr.dedup and pr.dedup.doi:
            doi_safe = pr.dedup.doi.replace("/", "_").replace(":", "_")
            for ext in (".pdf", ".txt"):
                candidate = pdf_dir / f"{doi_safe}{ext}"
                if candidate.exists():
                    return str(candidate)

        return None

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from a .pdf or .txt full-text file."""
        path = Path(file_path)

        # Plain-text files (HTML-extracted by downloader)
        if path.suffix == ".txt":
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Could not read txt {file_path}: {e}")
                return ""

        # PDF files
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text
            return extract_text(file_path)
        except ImportError:
            pass

        logger.warning(f"No PDF extraction library available for {file_path}")
        return ""

    def _get_next_study_id(self) -> int:
        """Get next available study counter."""
        existing = [
            pr.study_id for pr in self.records.values()
            if pr.study_id
        ]
        if not existing:
            return 1
        nums = []
        for sid in existing:
            try:
                nums.append(int(sid.split("-")[-1]))
            except (ValueError, IndexError):
                pass
        return max(nums) + 1 if nums else 1

    def _export_screening_lists(self):
        """Write current decision buckets to JSON files."""
        groups = self.get_records_by_decision()
        for decision, records in groups.items():
            safe_name = decision.replace(" ", "_").lower()
            out_path = os.path.join(settings.screened_dir, f"{safe_name}.jsonl")
            with open(out_path, "w") as f:
                for pr in records:
                    f.write(pr.model_dump_json() + "\n")

    def _export_fulltext_needed_list(self):
        """Export list of papers needing full text."""
        needed = [
            pr for pr in self.records.values()
            if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
        ]
        out_path = os.path.join(settings.output_dir, "full_text_needed.json")
        rows = []
        for pr in needed:
            rows.append({
                "record_id": pr.record_id,
                "title": pr.dedup.title if pr.dedup else "",
                "authors": pr.dedup.authors if pr.dedup else "",
                "year": pr.dedup.year if pr.dedup else None,
                "doi": pr.dedup.doi if pr.dedup else "",
                "journal_venue": pr.dedup.journal_venue if pr.dedup else "",
                "url": pr.dedup.url if pr.dedup else "",
                "source_db": pr.dedup.source_db if pr.dedup else "",
            })
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Full text needed: {len(needed)} papers → {out_path}")

    def _save_state(self):
        """Persist all records to state file, and sync to Drive if configured."""
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            for pr in self.records.values():
                f.write(pr.model_dump_json() + "\n")
        drive_path = os.environ.get("DRIVE_STATE_FILE")
        if drive_path:
            try:
                import shutil
                os.makedirs(os.path.dirname(drive_path), exist_ok=True)
                shutil.copy2(STATE_FILE, drive_path)
            except Exception as e:
                logger.warning(f"Drive sync failed: {e}")

    def _save_checkpoint_csvs(self) -> None:
        """Write the four extraction CSV files and sync to Drive.

        Files written to <APP_DIR>/output/:
          extraction_agent1.csv       — Agent 1 (GPT-5) raw fields
          extraction_agent2.csv       — Agent 2 (GPT-5.4-mini) raw fields
          extraction_merged.csv       — Merged/final result + agree flag
          extraction_disagreements.csv — Field-level diff for every hard/soft disagreement
        """
        import csv as _csv
        output_dir = os.path.join(os.path.dirname(STATE_FILE), "..", "output")
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        def _flat(val):
            if isinstance(val, list):
                return " || ".join(str(v) for v in val)
            return val if val is not None else ""

        rows_a1, rows_a2, rows_merged, rows_diff = [], [], [], []
        HARD_FIELDS = {
            "model_name", "workflow_structure", "analytic_task", "human_comparison",
            "fine_tuned", "rag_used", "formal_methodology", "qualitative_approach",
        }

        for pr in self.records.values():
            if pr.pipeline_stage != PipelineStage.EXTRACTION or pr.extracted is None:
                continue
            ext = pr.extracted
            title = (pr.dedup.title if pr.dedup else None) or (pr.raw.title if pr.raw else "") or ""
            base = {
                "record_id": pr.record_id,
                "study_id":  pr.study_id or ext.study_id or "",
                "title":     title,
            }

            if ext.extraction_agent1:
                row = {**base}
                for k, v in ext.extraction_agent1.model_dump().items():
                    row[k] = _flat(v)
                rows_a1.append(row)

            if ext.extraction_agent2:
                row = {**base}
                for k, v in ext.extraction_agent2.model_dump().items():
                    row[k] = _flat(v)
                rows_a2.append(row)

            if ext.extraction_final:
                row = {**base}
                for k, v in ext.extraction_final.model_dump().items():
                    row[k] = _flat(v)
                row["disagreement_fields"] = " || ".join(ext.disagreement_fields)
                row["agents_agree"]        = ext.agents_agree_extraction
                row["human_verified"]      = ext.human_verified
                rows_merged.append(row)

            for field in ext.disagreement_fields:
                a1_val = getattr(ext.extraction_agent1, field, None) if ext.extraction_agent1 else None
                a2_val = getattr(ext.extraction_agent2, field, None) if ext.extraction_agent2 else None
                mg_val = getattr(ext.extraction_final,  field, None) if ext.extraction_final  else None
                rows_diff.append({
                    "record_id":     pr.record_id,
                    "title":         title[:80],
                    "field":         field,
                    "agent1":        _flat(a1_val),
                    "agent2":        _flat(a2_val),
                    "merged":        _flat(mg_val),
                    "is_hard_field": field in HARD_FIELDS,
                })

        def _write_csv(path, rows):
            if not rows:
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

        paths = {
            "agent1":        os.path.join(output_dir, "extraction_agent1.csv"),
            "agent2":        os.path.join(output_dir, "extraction_agent2.csv"),
            "merged":        os.path.join(output_dir, "extraction_merged.csv"),
            "disagreements": os.path.join(output_dir, "extraction_disagreements.csv"),
        }
        _write_csv(paths["agent1"],        rows_a1)
        _write_csv(paths["agent2"],        rows_a2)
        _write_csv(paths["merged"],        rows_merged)
        _write_csv(paths["disagreements"], rows_diff)
        logger.info(
            f"Checkpoint CSVs saved: {len(rows_merged)} merged, "
            f"{len(rows_diff)} disagreements → {output_dir}"
        )

        # Sync all four files to Drive if configured
        drive_state = os.environ.get("DRIVE_STATE_FILE")
        if drive_state:
            drive_dir = os.path.dirname(drive_state)
            try:
                import shutil
                for csv_path in paths.values():
                    if os.path.exists(csv_path):
                        shutil.copy2(csv_path, os.path.join(drive_dir, os.path.basename(csv_path)))
            except Exception as e:
                logger.warning(f"Drive CSV sync failed: {e}")

    def load_state(self):
        """Load records from state file."""
        if not os.path.exists(STATE_FILE):
            return
        self.records = {}
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pr = PipelineRecord.model_validate_json(line)
                    self.records[pr.record_id] = pr
        logger.info(f"Loaded {len(self.records)} records from state")
