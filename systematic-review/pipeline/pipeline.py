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
from agents.screener import screen_title_abstract, screen_fulltext, screen_second_pass
from agents.extractor import extract_and_assess
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
        }
        download_log: list = []
        lock = threading.Lock()
        pdf_dir = Path(settings.pdf_dir)

        def _download_one(pr: PipelineRecord):
            doi = pr.dedup.doi if pr.dedup else ""
            title = pr.dedup.title if pr.dedup else ""
            success, source = try_download_fulltext(doi, title, pr.record_id, pdf_dir)
            return pr, success, source, doi, title

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_download_one, pr): pr for pr in candidates}
            for future in as_completed(futures):
                try:
                    pr, success, source, doi, title = future.result()
                    with lock:
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

        log = {**counts, "completed_at": datetime.utcnow().isoformat()}
        self.stage_log["fulltext_download"] = log
        self.running_stage = ""
        logger.info(f"Full-text download complete: {counts}")
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
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
            and pr.screened is not None
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
    # STAGE 5: DATA EXTRACTION
    # ──────────────────────────────────────────────────

    def run_extraction(self, limit: Optional[int] = None) -> Dict:
        """Run data extraction on confirmed included studies."""
        logger.info("=== STAGE: DATA EXTRACTION ===")

        included = [
            pr for pr in self.records.values()
            if pr.final_decision == DecisionLabel.INCLUDE
            and pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
            and pr.screened is not None
            and pr.screened.fulltext_available
        ]

        if limit:
            included = included[:limit]

        study_counter = self._get_next_study_id()
        counts = {"extracted": 0, "uncertain": 0, "errors": 0}

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

        self._save_state()
        self.stage_log["extraction"] = {**counts, "completed_at": datetime.utcnow().isoformat()}
        self.running_stage = ""
        logger.info(f"Extraction complete: {counts}")
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
                pr.screened.fulltext_screening
                if pr.screened.fulltext_screening
                else pr.screened.title_screening
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
        title_screen_included = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
            and pr.final_decision == DecisionLabel.INCLUDE
        )
        fulltext_needed = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
        )
        fulltext_excluded = sum(
            1 for pr in self.records.values()
            if pr.pipeline_stage == PipelineStage.FULLTEXT_SCREENING
            and pr.final_decision == DecisionLabel.EXCLUDE
        )
        final_included = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.INCLUDE
            and pr.pipeline_stage in (PipelineStage.FULLTEXT_SCREENING, PipelineStage.EXTRACTION)
        )
        # "Needs human verification" = records that have BEEN processed by agents
        # but agents disagreed; does NOT include pre-screening records (those are
        # "awaiting_title_screening").
        needs_human = sum(
            1 for pr in self.records.values()
            if pr.final_decision == DecisionLabel.UNCERTAIN
            and pr.pipeline_stage in (
                PipelineStage.TITLE_SCREENING,
                PipelineStage.FULLTEXT_SCREENING,
                PipelineStage.EXTRACTION,
            )
        )

        return {
            "identified": total,
            "duplicates_removed": dups,
            "after_dedup": after_dedup,
            "awaiting_title_screening": awaiting_screening,
            "title_abstract_excluded": title_screen_excluded,
            "title_abstract_included": title_screen_included,
            "full_text_needed": fulltext_needed,
            "full_text_excluded": fulltext_excluded,
            "final_included": final_included,
            "needs_human_verification": needs_human,
        }

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
