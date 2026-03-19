"""
Main pipeline orchestrator.
Runs records through all stages with full state persistence.
"""
import json
import os
import logging
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from datetime import datetime

from .models import (
    RawRecord, DedupRecord, ScreenedRecord, PipelineRecord,
    DecisionLabel, PipelineStage, ExtractedRecord
)
from .importer import load_all_from_directory, save_records, load_records
from .deduplicator import deduplicate, filter_unique, save_deduped, load_deduped
from agents.screener import screen_title_abstract, screen_fulltext
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
        logger.info(f"Import complete: {stats}")
        return stats

    # ──────────────────────────────────────────────────
    # STAGE 2: DEDUPLICATION
    # ──────────────────────────────────────────────────

    def run_deduplication(self) -> Dict:
        """Deduplicate imported records."""
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
        logger.info(f"Dedup complete: {stats}")
        return stats

    # ──────────────────────────────────────────────────
    # STAGE 3: TITLE/ABSTRACT SCREENING
    # ──────────────────────────────────────────────────

    def run_title_screening(self, limit: Optional[int] = None) -> Dict:
        """Run two-agent title/abstract screening on unique records."""
        logger.info("=== STAGE: TITLE/ABSTRACT SCREENING ===")

        candidates = [
            pr for pr in self.records.values()
            if pr.dedup is not None and not pr.dedup.is_duplicate
            and pr.pipeline_stage in (PipelineStage.IMPORT, PipelineStage.DEDUP)
        ]

        if limit:
            candidates = candidates[:limit]

        counts = {
            DecisionLabel.INCLUDE: 0,
            DecisionLabel.EXCLUDE: 0,
            DecisionLabel.UNCERTAIN: 0,
        }

        for pr in candidates:
            logger.info(f"Screening [{pr.record_id[:8]}]: {pr.dedup.title[:60]}...")
            try:
                result = screen_title_abstract(pr.dedup)

                if pr.screened is None:
                    pr.screened = ScreenedRecord(**pr.dedup.model_dump())
                pr.screened.title_screening = result
                pr.screened.current_decision = result.final_decision
                pr.update_stage(PipelineStage.TITLE_SCREENING, result.final_decision)
                counts[result.final_decision] = counts.get(result.final_decision, 0) + 1

            except Exception as e:
                logger.error(f"Screening error for {pr.record_id}: {e}")
                pr.update_stage(PipelineStage.TITLE_SCREENING, DecisionLabel.UNCERTAIN)
                counts[DecisionLabel.UNCERTAIN] += 1

        self._save_state()
        self._export_screening_lists()
        log = {str(k): v for k, v in counts.items()}
        log["completed_at"] = datetime.utcnow().isoformat()
        self.stage_log["title_screening"] = log
        logger.info(f"Title screening complete: {counts}")
        return counts

    # ──────────────────────────────────────────────────
    # STAGE 4: FULL-TEXT SCREENING
    # ──────────────────────────────────────────────────

    def run_fulltext_screening(self, limit: Optional[int] = None) -> Dict:
        """Run full-text screening on records that passed title screening."""
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

        for pr in candidates:
            # Check PDF availability
            pdf_path = self._find_pdf(pr)
            if not pdf_path:
                pr.screened.fulltext_available = False
                pr.update_stage(PipelineStage.FULLTEXT_SCREENING, DecisionLabel.FULL_TEXT_NEEDED)
                counts[DecisionLabel.FULL_TEXT_NEEDED] += 1
                continue

            pr.screened.pdf_path = pdf_path
            pr.screened.fulltext_available = True

            try:
                fulltext = self._extract_pdf_text(pdf_path)
                result = screen_fulltext(pr.dedup, fulltext)
                pr.screened.fulltext_screening = result
                pr.screened.current_decision = result.final_decision
                pr.update_stage(PipelineStage.FULLTEXT_SCREENING, result.final_decision)
                counts[result.final_decision] = counts.get(result.final_decision, 0) + 1

            except Exception as e:
                logger.error(f"Full-text screening error for {pr.record_id}: {e}")
                pr.update_stage(PipelineStage.FULLTEXT_SCREENING, DecisionLabel.UNCERTAIN)
                counts[DecisionLabel.UNCERTAIN] += 1

        self._save_state()
        self._export_fulltext_needed_list()
        log = {str(k): v for k, v in counts.items()}
        log["completed_at"] = datetime.utcnow().isoformat()
        self.stage_log["fulltext_screening"] = log
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

    # ──────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────

    def _find_pdf(self, pr: PipelineRecord) -> Optional[str]:
        """Look for PDF by DOI, title, or study_id in pdf_dir."""
        pdf_dir = Path(settings.pdf_dir)
        if not pdf_dir.exists():
            return None

        # Try by record_id
        candidate = pdf_dir / f"{pr.record_id}.pdf"
        if candidate.exists():
            return str(candidate)

        # Try by DOI (sanitized)
        if pr.dedup and pr.dedup.doi:
            doi_safe = pr.dedup.doi.replace("/", "_").replace(":", "_")
            candidate = pdf_dir / f"{doi_safe}.pdf"
            if candidate.exists():
                return str(candidate)

        return None

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfminer or PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text
            return extract_text(pdf_path)
        except ImportError:
            pass

        logger.warning(f"No PDF extraction library available for {pdf_path}")
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
        """Persist all records to state file."""
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            for pr in self.records.values():
                f.write(pr.model_dump_json() + "\n")

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
