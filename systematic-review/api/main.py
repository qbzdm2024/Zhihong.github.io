"""
FastAPI backend for the Systematic Review Automation System.
Provides REST endpoints for the UI and pipeline management.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings, ModelConfig
from pipeline.models import DecisionLabel, PipelineStage, PipelineRecord
from pipeline.pipeline import PipelineRunner
from agents.screener import get_agent_errors
from agents.openai_client import get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Systematic Review Automation API",
    description="Human-in-the-loop systematic review pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/app", StaticFiles(directory=str(ui_path), html=True), name="ui")

# Global pipeline runner (loaded on startup)
runner = PipelineRunner()


@app.on_event("startup")
async def startup():
    runner.load_state()
    logger.info(f"Pipeline loaded: {len(runner.records)} records")


# ─────────────────────────────────────────────────────
# PIPELINE CONTROL
# ─────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    stage: str  # "import", "dedup", "title_screening", "fulltext_screening", "extraction"
    limit: Optional[int] = None
    force: Optional[bool] = False
    record_id: Optional[str] = None  # target a specific record (used for pilot extraction)


@app.post("/api/pipeline/run")
async def run_pipeline_stage(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Trigger a pipeline stage.
    import and dedup run synchronously (fast, no AI calls) and return
    their results directly. Screening and extraction run as background tasks.
    """
    if request.stage == "import":
        stats = runner.run_import()
        return {"status": "completed", "stage": "import", "stats": stats}

    if request.stage == "dedup":
        stats = runner.run_deduplication()
        return {"status": "completed", "stage": "dedup", "stats": stats}

    stage_map = {
        "title_screening": lambda: runner.run_title_screening(request.limit),
        "second_pass_screening": lambda: runner.run_second_pass_screening(request.limit),
        "fulltext_download": lambda: runner.run_fulltext_download(request.limit),
        "fulltext_screening": lambda: runner.run_fulltext_screening(request.limit),
        "second_fulltext_screening": lambda: runner.run_second_fulltext_screening(request.limit),
        "extraction": lambda: runner.run_extraction(request.limit, request.record_id),
    }

    if request.stage == "reset_screening":
        stats = runner.reset_screening()
        return {"status": "completed", "stage": "reset_screening", "stats": stats}

    if request.stage == "reset_failed_screenings":
        stats = runner.reset_failed_screenings()
        return {"status": "completed", "stage": "reset_failed_screenings", "stats": stats}

    if request.stage == "restore_bulk_excluded":
        stats = runner.restore_bulk_excluded()
        return {"status": "completed", "stage": "restore_bulk_excluded", "stats": stats}

    if request.stage == "reset_fulltext_screening":
        stats = runner.reset_fulltext_screening(force=request.force)
        return {"status": "completed", "stage": "reset_fulltext_screening", "stats": stats}

    if request.stage == "reset_second_fulltext_screening":
        stats = runner.reset_second_fulltext_screening(force=request.force)
        return {"status": "completed", "stage": "reset_second_fulltext_screening", "stats": stats}

    if request.stage == "mark_included_for_review":
        stats = runner.mark_included_for_review()
        return {"status": "completed", "stage": "mark_included_for_review", "stats": stats}

    if request.stage not in stage_map:
        raise HTTPException(400, f"Unknown stage: {request.stage}")

    background_tasks.add_task(stage_map[request.stage])
    return {"status": "started", "stage": request.stage}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status and PRISMA counts."""
    from agents.screener import get_agent_errors
    counts = runner.get_prisma_counts()
    groups = runner.get_records_by_decision()

    # Determine which stage is currently "active" (furthest along)
    stage_order = ["import", "dedup", "title_screening", "fulltext_screening", "extraction"]
    active_stage = None
    for s in reversed(stage_order):
        if s in runner.stage_log:
            active_stage = s
            break

    # Detect probable API key / auth failure: many UNCERTAIN + 0 included/excluded
    warnings = []
    if (counts.get("needs_human_verification", 0) > 10
            and counts.get("title_abstract_included", 0) == 0
            and counts.get("title_abstract_excluded", 0) == 0):
        recent_errors = get_agent_errors(n=3)
        if recent_errors:
            warnings.append(
                "⚠ All records show 'Needs Human Verification' with 0 included/excluded — "
                "likely an API key error. Check /api/debug/agent-errors. "
                "Fix the key, restart the server, then call reset_failed_screenings."
            )

    return {
        "prisma_counts": counts,
        "bucket_counts": {k: len(v) for k, v in groups.items()},
        "total_records": len(runner.records),
        "stage_log": runner.stage_log,
        "active_stage": active_stage,
        "running_stage": runner.running_stage,
        "warnings": warnings,
        "last_updated": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────
# RECORD QUERIES
# ─────────────────────────────────────────────────────

@app.get("/api/records")
async def list_records(
    decision: Optional[str] = None,
    stage: Optional[str] = None,
    human_needed: Optional[bool] = None,
    page: int = 1,
    page_size: int = 50,
):
    """List records with optional filtering."""
    records = list(runner.records.values())

    if decision:
        records = [r for r in records if r.final_decision == decision]
    if stage:
        records = [r for r in records if r.pipeline_stage == stage]
    if human_needed is True:
        records = [r for r in records if r.final_decision == DecisionLabel.UNCERTAIN]

    # Sort by updated_at descending
    records.sort(key=lambda r: r.updated_at, reverse=True)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_records = records[start:end]

    return {
        "total": len(records),
        "page": page,
        "page_size": page_size,
        "records": [_serialize_record_summary(r) for r in page_records],
    }


@app.get("/api/records/{record_id}")
async def get_record(record_id: str):
    """Get full details of a single record."""
    pr = runner.records.get(record_id)
    if not pr:
        raise HTTPException(404, f"Record {record_id} not found")
    return json.loads(pr.model_dump_json())


@app.get("/api/records/uncertain/list")
async def list_uncertain_records():
    """Get all records needing human verification."""
    uncertain = [
        _serialize_record_summary(pr)
        for pr in runner.records.values()
        if pr.final_decision == DecisionLabel.UNCERTAIN
    ]
    return {"count": len(uncertain), "records": uncertain}


@app.get("/api/records/second-pass/list")
async def list_second_pass_records(page: int = 1, page_size: int = 50, reviewed: Optional[str] = None):
    """Return included title-screened records enriched with abstract + agent rationale
    for second-pass human review. reviewed=no filters to unreviewed only."""
    records = [
        pr for pr in runner.records.values()
        if pr.pipeline_stage == PipelineStage.TITLE_SCREENING
        and pr.final_decision == DecisionLabel.INCLUDE
    ]

    if reviewed == "no":
        records = [
            pr for pr in records
            if not (pr.screened and pr.screened.title_screening
                    and pr.screened.title_screening.human_verified)
        ]

    records.sort(key=lambda r: r.updated_at, reverse=True)
    total = len(records)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": [_serialize_second_pass(pr) for pr in records[start:end]],
    }


@app.get("/api/records/fulltext-needed/list")
async def list_fulltext_needed():
    """Get list of papers where full text is needed."""
    needed = [
        pr for pr in runner.records.values()
        if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
    ]
    return {
        "count": len(needed),
        "records": [
            {
                "record_id": pr.record_id,
                "title": pr.dedup.title if pr.dedup else "",
                "authors": pr.dedup.authors if pr.dedup else "",
                "year": pr.dedup.year if pr.dedup else None,
                "doi": pr.dedup.doi if pr.dedup else "",
                "journal_venue": pr.dedup.journal_venue if pr.dedup else "",
                "url": pr.dedup.url if pr.dedup else "",
                "source_db": pr.dedup.source_db if pr.dedup else "",
            }
            for pr in needed
        ]
    }


# ─────────────────────────────────────────────────────
# HUMAN VERIFICATION
# ─────────────────────────────────────────────────────

class HumanDecisionRequest(BaseModel):
    decision: str  # "Included" | "Excluded" | "Needs Human Verification"
    rationale: str
    reviewer: str = "human"
    corrections: Optional[Dict[str, Any]] = None


@app.post("/api/records/{record_id}/verify")
async def apply_human_decision(record_id: str, request: HumanDecisionRequest):
    """Apply human reviewer's decision to a record."""
    try:
        decision = DecisionLabel(request.decision)
    except ValueError:
        raise HTTPException(400, f"Invalid decision: {request.decision}")

    success = runner.apply_human_decision(
        record_id=record_id,
        decision=decision,
        rationale=request.rationale,
        reviewer=request.reviewer,
        corrections=request.corrections,
    )

    if not success:
        raise HTTPException(404, f"Record {record_id} not found")

    return {"status": "updated", "record_id": record_id, "decision": request.decision}


@app.patch("/api/records/{record_id}/extraction")
async def update_extraction(record_id: str, updates: Dict[str, Any]):
    """Update specific extraction fields for a record."""
    pr = runner.records.get(record_id)
    if not pr:
        raise HTTPException(404, f"Record {record_id} not found")

    if not pr.extracted or not pr.extracted.extraction_final:
        raise HTTPException(400, "No extraction data to update")

    for field, value in updates.items():
        if hasattr(pr.extracted.extraction_final, field):
            setattr(pr.extracted.extraction_final, field, value)
        else:
            raise HTTPException(400, f"Unknown extraction field: {field}")

    pr.extracted.human_verified = True
    pr.extracted.human_corrections.update(updates)
    pr.updated_at = datetime.utcnow()
    runner._save_state()

    return {"status": "updated", "record_id": record_id, "updated_fields": list(updates.keys())}


# ─────────────────────────────────────────────────────
# PDF UPLOAD
# ─────────────────────────────────────────────────────

@app.post("/api/pdfs/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    record_id: str = Form(...),
):
    """Upload a PDF for a record that needs full text."""
    pr = runner.records.get(record_id)
    if not pr:
        raise HTTPException(404, f"Record {record_id} not found")

    allowed = (".pdf", ".txt")
    if not any(file.filename.endswith(ext) for ext in allowed):
        raise HTTPException(400, "Only .pdf or .txt files are accepted")

    suffix = Path(file.filename).suffix.lower()
    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / f"{record_id}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update record state
    if pr.screened:
        pr.screened.pdf_path = str(save_path)
        pr.screened.fulltext_available = True

    # Move from Full Text Needed back to title-screened included (for fulltext screening)
    if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED:
        pr.final_decision = DecisionLabel.INCLUDE  # ready for fulltext screening

    pr.updated_at = datetime.utcnow()
    runner._save_state()

    return {
        "status": "uploaded",
        "record_id": record_id,
        "file_path": str(save_path),
        "message": f"File uploaded ({suffix}). Run full-text screening to process this record.",
    }


@app.post("/api/pdfs/upload-batch")
async def upload_pdfs_batch(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs. Filename should be record_id.pdf or doi_safe.pdf."""
    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for file in files:
        if not any(file.filename.endswith(ext) for ext in (".pdf", ".txt")):
            results.append({"filename": file.filename, "status": "skipped (not .pdf or .txt)"})
            continue

        save_path = pdf_dir / file.filename
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Auto-mark matching record as fulltext_available
        stem = Path(file.filename).stem  # record_id part
        pr = runner.records.get(stem)
        if pr and pr.screened:
            pr.screened.pdf_path = str(save_path)
            pr.screened.fulltext_available = True
            if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED:
                pr.final_decision = DecisionLabel.INCLUDE
            pr.updated_at = datetime.utcnow()

        results.append({"filename": file.filename, "status": "uploaded", "path": str(save_path)})

    runner._save_state()
    return {"uploaded": len([r for r in results if r["status"] == "uploaded"]), "results": results}


@app.post("/api/pdfs/register-from-disk")
async def register_pdfs_from_disk():
    """Scan data/pdfs/ and register every .pdf/.txt file found there.

    Matching order for each file stem:
      1. Exact record_id match
      2. Sanitised-DOI match  (10.1016_j.foo… → original DOI)
      3. Title similarity match (≥ 0.65 Jaccard on word tokens)

    Records in FULL_TEXT_NEEDED are promoted back to INCLUDE automatically.
    """
    import unicodedata, re as _re

    def _normalise(s: str) -> set:
        """Lower-case, strip punctuation, return word-token set."""
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return set(_re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())

    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # Pre-build title token sets for all included records (cheap, done once)
    title_index: list = []
    for rec in runner.records.values():
        d = rec.dedup or rec.raw
        if d and d.title:
            title_index.append((rec, _normalise(d.title)))

    pdf_dir = Path(settings.pdf_dir)
    if not pdf_dir.exists():
        return {"registered": 0, "unmatched": 0, "results": []}

    results = []
    for f in sorted(pdf_dir.iterdir()):
        if f.suffix.lower() not in (".pdf", ".txt"):
            continue

        stem = f.stem
        pr = runner.records.get(stem)

        # 2. DOI reverse-lookup
        if pr is None:
            doi_candidate = stem.replace("_", "/", 1)
            for rec in runner.records.values():
                d = rec.dedup or rec.raw
                if d and d.doi:
                    doi_safe = d.doi.replace("/", "_").replace(":", "_")
                    if doi_safe == stem:
                        pr = rec
                        break

        # 3. Title similarity fallback
        if pr is None:
            stem_tokens = _normalise(stem.replace("_", " ").replace("-", " "))
            best_score, best_rec = 0.0, None
            for rec, title_tokens in title_index:
                score = _jaccard(stem_tokens, title_tokens)
                if score > best_score:
                    best_score, best_rec = score, rec
            if best_score >= 0.65:
                pr = best_rec

        if pr and pr.screened:
            # Prefer PDF over TXT: never overwrite a .pdf path with a .txt path
            existing = pr.screened.pdf_path or ""
            is_downgrade = (
                existing.lower().endswith(".pdf")
                and f.suffix.lower() == ".txt"
            )
            if not is_downgrade:
                pr.screened.pdf_path = str(f)
                pr.screened.fulltext_available = True
                if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED:
                    pr.final_decision = DecisionLabel.INCLUDE
                pr.updated_at = datetime.utcnow()
            results.append({
                "file": f.name,
                "status": "registered",
                "record_id": pr.record_id,
            })
        else:
            results.append({"file": f.name, "status": "unmatched"})

    runner._save_state()
    registered = sum(1 for r in results if r["status"] == "registered")
    unmatched  = sum(1 for r in results if r["status"] == "unmatched")
    return {"registered": registered, "unmatched": unmatched, "results": results}


# ─────────────────────────────────────────────────────
# FULL-TEXT AUTO-DOWNLOAD
# ─────────────────────────────────────────────────────

class DownloadEmailRequest(BaseModel):
    email: Optional[str] = None  # contact email for Unpaywall/OpenAlex polite pool
    limit: Optional[int] = None


@app.post("/api/fulltext/download")
async def trigger_fulltext_download(
    request: DownloadEmailRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger automatic full-text download for all INCLUDE records lacking a PDF.

    Tries Unpaywall → Semantic Scholar → OpenAlex → Europe PMC for each record.
    Records that cannot be auto-fetched are set to FULL_TEXT_NEEDED.
    A PRISMA snapshot is saved before download begins.
    """
    if runner.running_stage:
        raise HTTPException(409, f"Pipeline stage already running: {runner.running_stage}")

    # Override contact email if provided
    if request.email:
        import agents.fulltext_downloader as dl_mod
        dl_mod.CONTACT_EMAIL = request.email

    def _run():
        runner.run_fulltext_download(limit=request.limit)

    background_tasks.add_task(_run)
    return {"status": "started", "stage": "fulltext_download"}


@app.get("/api/fulltext/download-status")
async def get_download_status():
    """Return the result of the last fulltext download run."""
    log = runner.stage_log.get("fulltext_download", {})
    manual_needed_list = [
        {
            "record_id": pr.record_id,
            "title": (pr.dedup or pr.raw).title if (pr.dedup or pr.raw) else "",
            "authors": pr.dedup.authors if pr.dedup else "",
            "year": pr.dedup.year if pr.dedup else None,
            "doi": pr.dedup.doi if pr.dedup else "",
            "journal": pr.dedup.journal_venue if pr.dedup else "",
            "url": pr.dedup.url if pr.dedup else "",
            "source_db": pr.dedup.source_db if pr.dedup else "",
        }
        for pr in runner.records.values()
        if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
    ]
    return {
        "running": runner.running_stage in ("fulltext_download", "fulltext_retry"),
        "last_run": log,
        "manual_needed_count": len(manual_needed_list),
        "manual_needed": manual_needed_list,
    }


@app.post("/api/fulltext/retry")
async def retry_fulltext_download(background_tasks: BackgroundTasks):
    """Retry downloading full texts for all FULL_TEXT_NEEDED records
    using extended sources (arXiv, CORE, direct URL)."""
    if runner.running_stage:
        raise HTTPException(409, f"Stage '{runner.running_stage}' already running")

    def _run():
        runner.retry_fulltext_download()

    background_tasks.add_task(_run)
    return {"status": "started", "stage": "fulltext_retry"}


@app.get("/api/fulltext/retry-status")
async def get_retry_status():
    """Return progress of the retry download run."""
    log = runner.stage_log.get("fulltext_retry", {})
    still_needed = sum(
        1 for pr in runner.records.values()
        if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
    )
    return {
        "running": runner.running_stage == "fulltext_retry",
        "last_run": log,
        "still_needed_count": still_needed,
    }


@app.get("/api/fulltext/manual-list/csv")
async def export_manual_fulltext_csv():
    """Download a CSV of papers needing manual full-text retrieval."""
    import csv, io
    records = [
        pr for pr in runner.records.values()
        if pr.final_decision == DecisionLabel.FULL_TEXT_NEEDED
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "record_id", "title", "authors", "year", "journal_venue",
        "doi", "url", "source_db",
        "upload_filename_hint",
    ])
    for pr in records:
        d = pr.dedup or pr.raw
        if not d:
            continue
        writer.writerow([
            pr.record_id,
            d.title or "",
            d.authors or "",
            d.year or "",
            d.journal_venue or "",
            d.doi or "",
            d.url or "",
            d.source_db or "",
            f"{pr.record_id}.pdf",  # hint: save file with this name for auto-match
        ])
    output.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=manual_fulltext_needed.csv"},
    )


# ─────────────────────────────────────────────────────
# PRISMA SNAPSHOTS
# ─────────────────────────────────────────────────────

class PrismaSnapshotRequest(BaseModel):
    label: str = ""


@app.post("/api/prisma/snapshot")
async def save_prisma_snapshot(request: PrismaSnapshotRequest):
    """Save a labelled PRISMA flow snapshot to the audit trail."""
    snapshot = runner.save_prisma_snapshot(request.label)
    return {"status": "saved", "snapshot": snapshot}


@app.get("/api/prisma/snapshots")
async def get_prisma_snapshots():
    """Return all saved PRISMA flow snapshots."""
    return {"snapshots": runner.get_prisma_snapshots()}


@app.get("/api/prisma/report")
async def get_full_prisma_report():
    """Return complete PRISMA 2020 report with source DB breakdown,
    round-1 and round-2 exclusion reason groupings, and formatted flowchart text."""
    return runner.get_full_prisma_report()


@app.get("/api/prisma/flowchart")
async def get_prisma_flowchart():
    """Return the formatted ASCII PRISMA flowchart as plain text."""
    report = runner.get_full_prisma_report()
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(report["prisma_flowchart_text"])


# ─────────────────────────────────────────────────────
# ROUND-2 FULL-TEXT SCREENING
# ─────────────────────────────────────────────────────

@app.get("/api/round2-screening/status")
async def get_round2_screening_status():
    """Return counts and status of second-round full-text screening."""
    from pipeline.models import PipelineStage as PS
    r2_records = [
        pr for pr in runner.records.values()
        if pr.pipeline_stage == PS.SECOND_FULLTEXT_SCREENING
    ]
    included = [pr for pr in r2_records if pr.final_decision == DecisionLabel.INCLUDE]
    excluded = [pr for pr in r2_records if pr.final_decision == DecisionLabel.EXCLUDE]
    uncertain = [pr for pr in r2_records if pr.final_decision == DecisionLabel.UNCERTAIN]
    pending = [
        pr for pr in runner.records.values()
        if pr.final_decision == DecisionLabel.INCLUDE
        and pr.pipeline_stage == PS.FULLTEXT_SCREENING
        and pr.screened is not None
        and pr.screened.second_fulltext_screening is None
    ]
    return {
        "total_screened_round2": len(r2_records),
        "included": len(included),
        "excluded": len(excluded),
        "needs_human_verification": len(uncertain),
        "pending_round2_screening": len(pending),
        "running": runner.running_stage == "second_fulltext_screening",
        "last_run": runner.stage_log.get("second_fulltext_screening", {}),
    }


@app.get("/api/round2-screening/exclusion-summary")
async def get_round2_exclusion_summary():
    """Return PRISMA-ready breakdown of round-2 exclusion reasons."""
    summary = runner.get_round2_exclusion_summary()
    return summary


@app.get("/api/round2-screening/excluded-records")
async def list_round2_excluded_records(page: int = 1, page_size: int = 50):
    """List all records excluded in round-2 with their exclusion codes and rationale."""
    from pipeline.models import PipelineStage as PS
    excluded = [
        pr for pr in runner.records.values()
        if pr.pipeline_stage == PS.SECOND_FULLTEXT_SCREENING
        and pr.final_decision == DecisionLabel.EXCLUDE
        and pr.screened is not None
        and pr.screened.second_fulltext_screening is not None
    ]
    excluded.sort(key=lambda r: r.updated_at, reverse=True)
    start = (page - 1) * page_size
    page_records = excluded[start: start + page_size]

    def _fmt(pr: PipelineRecord):
        d = pr.dedup or pr.raw
        r2 = pr.screened.second_fulltext_screening
        ec_code = None
        ec_rationale = None
        if r2.agent1 and r2.agent1.exclusion_code:
            ec_code = r2.agent1.exclusion_code
            ec_rationale = r2.agent1.rationale
        elif r2.agent2 and r2.agent2.exclusion_code:
            ec_code = r2.agent2.exclusion_code
            ec_rationale = r2.agent2.rationale
        return {
            "record_id": pr.record_id,
            "title": d.title if d else "",
            "authors": d.authors if d else "",
            "year": d.year if d else None,
            "journal_venue": d.journal_venue if d else "",
            "doi": d.doi if d else "",
            "exclusion_code": ec_code,
            "agent1_exclusion_code": r2.agent1.exclusion_code if r2.agent1 else None,
            "agent2_exclusion_code": r2.agent2.exclusion_code if r2.agent2 else None,
            "agent1_rationale": r2.agent1.rationale if r2.agent1 else None,
            "agent2_rationale": r2.agent2.rationale if r2.agent2 else None,
            "consensus_confidence": r2.consensus_confidence,
            "human_verified": r2.human_verified,
            "human_rationale": r2.human_rationale,
        }

    return {
        "total": len(excluded),
        "page": page,
        "page_size": page_size,
        "records": [_fmt(pr) for pr in page_records],
    }


@app.get("/api/round2-screening/uncertain-records")
async def list_round2_uncertain_records():
    """List records needing human verification after round-2 screening."""
    from pipeline.models import PipelineStage as PS
    uncertain = [
        pr for pr in runner.records.values()
        if pr.pipeline_stage == PS.SECOND_FULLTEXT_SCREENING
        and pr.final_decision == DecisionLabel.UNCERTAIN
        and pr.screened is not None
        and pr.screened.second_fulltext_screening is not None
        and not pr.screened.second_fulltext_screening.human_verified
    ]

    def _fmt(pr: PipelineRecord):
        d = pr.dedup or pr.raw
        r2 = pr.screened.second_fulltext_screening
        return {
            "record_id": pr.record_id,
            "title": d.title if d else "",
            "authors": d.authors if d else "",
            "year": d.year if d else None,
            "journal_venue": d.journal_venue if d else "",
            "doi": d.doi if d else "",
            "agent1_decision": r2.agent1.decision if r2.agent1 else None,
            "agent1_exclusion_code": r2.agent1.exclusion_code if r2.agent1 else None,
            "agent1_rationale": r2.agent1.rationale if r2.agent1 else None,
            "agent2_decision": r2.agent2.decision if r2.agent2 else None,
            "agent2_exclusion_code": r2.agent2.exclusion_code if r2.agent2 else None,
            "agent2_rationale": r2.agent2.rationale if r2.agent2 else None,
            "consensus_confidence": r2.consensus_confidence,
        }

    return {"total": len(uncertain), "records": [_fmt(pr) for pr in uncertain]}


# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────

@app.get("/api/config/setup-status")
async def get_setup_status():
    """Check whether the system is configured and ready to run."""
    has_key = bool(settings.openai_api_key and settings.openai_api_key != "sk-your-key-here")
    env_exists = Path(".env").exists()
    return {
        "ready": has_key,
        "has_api_key": has_key,
        "env_file_exists": env_exists,
        "total_records": len(runner.records),
        "data_dir_exists": Path(settings.data_dir).exists(),
        "raw_dir_exists": Path(settings.raw_dir).exists(),
    }


class ApiKeyRequest(BaseModel):
    api_key: str
    base_url: Optional[str] = None


@app.post("/api/config/api-key")
async def save_api_key(request: ApiKeyRequest):
    """Save OpenAI API key to .env file and update runtime settings."""
    key = request.api_key.strip()
    if not key.startswith("sk-"):
        raise HTTPException(400, "API key must start with 'sk-'")

    settings.openai_api_key = key
    if request.base_url:
        settings.openai_base_url = request.base_url

    # Write to .env
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()

    # Replace or append OPENAI_API_KEY
    found = False
    new_lines = []
    for line in lines:
        if line.startswith("OPENAI_API_KEY="):
            new_lines.append(f"OPENAI_API_KEY={key}\n")
            found = True
        elif line.startswith("OPENAI_BASE_URL=") and request.base_url:
            new_lines.append(f"OPENAI_BASE_URL={request.base_url}\n")
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"OPENAI_API_KEY={key}\n")
        if request.base_url:
            new_lines.append(f"OPENAI_BASE_URL={request.base_url}\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)

    # Reinitialize the OpenAI client with new key
    from agents import openai_client as _oc
    _oc._client = None  # force re-creation next call

    return {"status": "saved", "message": "API key saved to .env and applied to runtime."}


@app.post("/api/config/test-api-key")
async def test_api_key():
    """Test that the configured API key works by making a minimal API call."""
    if not settings.openai_api_key or settings.openai_api_key == "sk-your-key-here":
        raise HTTPException(400, "No API key configured")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url)
        # Minimal call: list models
        models = client.models.list()
        model_ids = [m.id for m in list(models)[:5]]
        return {"status": "ok", "message": "API key is valid.", "sample_models": model_ids}
    except Exception as e:
        raise HTTPException(400, f"API key test failed: {str(e)}")


@app.get("/api/config/models")
async def get_model_config():
    """Get current model configuration."""
    return {
        "model_title_screening": settings.model_title_screening,
        "model_fulltext_screening": settings.model_fulltext_screening,
        "model_extraction": settings.model_extraction,
        "model_qa_assessment": settings.model_qa_assessment,
        "model_synthesis": settings.model_synthesis,
        "model_agent2_screening": settings.model_agent2_screening,
        "model_agent2_extraction": settings.model_agent2_extraction,
        "confidence_threshold": settings.confidence_threshold,
        "agreement_required": settings.agreement_required,
        "has_api_key": bool(settings.openai_api_key and settings.openai_api_key != "sk-your-key-here"),
    }


class ModelConfigUpdate(BaseModel):
    model_title_screening: Optional[str] = None
    model_fulltext_screening: Optional[str] = None
    model_extraction: Optional[str] = None
    model_qa_assessment: Optional[str] = None
    model_synthesis: Optional[str] = None
    model_agent2_screening: Optional[str] = None
    model_agent2_extraction: Optional[str] = None
    confidence_threshold: Optional[float] = None


@app.patch("/api/config/models")
async def update_model_config(updates: ModelConfigUpdate):
    """Update model configuration (runtime, not persisted to env)."""
    changed = {}
    for field, value in updates.model_dump(exclude_none=True).items():
        setattr(settings, field, value)
        changed[field] = value

    # Persist to .env file
    _update_env_file(changed)

    return {"status": "updated", "changed": changed}


# ─────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────

@app.get("/api/export/evidence-table")
async def export_evidence_table():
    """Export evidence table as JSON."""
    table = runner.export_evidence_table()
    return {"count": len(table), "rows": table}


@app.get("/api/export/evidence-table/csv")
async def export_evidence_table_csv():
    """Export evidence table as CSV file."""
    import csv
    import io

    table = runner.export_evidence_table()
    if not table:
        raise HTTPException(404, "No included studies to export")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=table[0].keys())
    writer.writeheader()
    writer.writerows(table)

    from fastapi.responses import Response
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evidence_table.csv"},
    )


@app.get("/api/export/prisma")
async def export_prisma_counts():
    """Export PRISMA 2020 flow diagram counts."""
    return runner.get_prisma_counts()


@app.get("/api/export/all-records")
async def export_all_records():
    """Export complete pipeline state as JSON."""
    out_path = Path(settings.output_dir) / "full_export.json"
    all_data = {
        "exported_at": datetime.utcnow().isoformat(),
        "prisma_counts": runner.get_prisma_counts(),
        "records": [json.loads(pr.model_dump_json()) for pr in runner.records.values()],
    }
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    return FileResponse(str(out_path), filename="systematic_review_export.json")


# ─────────────────────────────────────────────────────
# DEBUG ENDPOINTS
# ─────────────────────────────────────────────────────

@app.get("/api/debug/sample-screenings")
async def debug_sample_screenings(n: int = 10):
    """Return n records that have screening results, showing what agent1/agent2 decided.
    Use this to diagnose why records are all UNCERTAIN.
    """
    from pipeline.models import PipelineStage, DecisionLabel
    samples = []
    for pr in list(runner.records.values()):
        if pr.screened and pr.screened.title_screening:
            ts = pr.screened.title_screening
            a1 = ts.agent1
            a2 = ts.agent2
            samples.append({
                "record_id": pr.record_id[:8],
                "title": (pr.dedup.title if pr.dedup else "")[:80],
                "final_decision": str(pr.final_decision),
                "agent1_decision": str(a1.decision) if a1 else None,
                "agent1_confidence": a1.confidence if a1 else None,
                "agent1_flags": a1.flagged_criteria if a1 else [],
                "agent2_decision": str(a2.decision) if a2 else None,
                "agent2_confidence": a2.confidence if a2 else None,
                "agent2_flags": a2.flagged_criteria if a2 else [],
                "agents_agree": ts.agents_agree,
                "consensus_confidence": ts.consensus_confidence,
            })
        if len(samples) >= n:
            break

    # Tally what agents decided
    from collections import Counter
    a1_decisions = Counter(s["agent1_decision"] for s in samples if s["agent1_decision"])
    a2_decisions = Counter(s["agent2_decision"] for s in samples if s["agent2_decision"])
    agreements = sum(1 for s in samples if s["agents_agree"])

    return {
        "sampled": len(samples),
        "agent1_decision_counts": dict(a1_decisions),
        "agent2_decision_counts": dict(a2_decisions),
        "agents_agreed_count": agreements,
        "records": samples,
    }


@app.get("/api/debug/agent-errors")
async def debug_agent_errors():
    """Return recent agent screening errors (API failures, parse errors, etc.)."""
    errors = get_agent_errors(n=50)
    return {
        "error_count": len(errors),
        "errors": errors,
        "hint": "If you see AuthenticationError or InvalidAPIKey, check OPENAI_API_KEY in settings.",
    }


@app.get("/api/debug/test-api-key")
async def debug_test_api_key():
    """Send a minimal test request to verify the OpenAI API key works."""
    try:
        client = get_client()
        raw, usage = client.chat_json(
            system_prompt="You are a test assistant. Reply with valid JSON only.",
            user_prompt='Reply with: {"status": "ok", "message": "API key works"}',
            model=settings.model_title_screening,
        )
        return {
            "status": "ok",
            "model": settings.model_title_screening,
            "response": raw,
            "usage": usage,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "model": settings.model_title_screening,
            "hint": (
                "AuthenticationError → check OPENAI_API_KEY. "
                "RateLimitError → upgrade plan or wait. "
                "NotFoundError → model name wrong in settings."
            ),
        }


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────

def _serialize_second_pass(pr: PipelineRecord) -> Dict:
    """Enriched record summary for second-pass human review (includes abstract + both screening rounds)."""
    d = pr.dedup or pr.raw
    abstract = d.abstract if d else ""

    def _extract_screening(sr):
        if not sr:
            return {"decision": "", "rationale": "", "confidence": 0.0, "ec": "",
                    "a1_decision": "", "a1_rationale": "", "a1_conf": 0.0, "a1_ec": "",
                    "a2_decision": "", "a2_rationale": "", "a2_conf": 0.0, "a2_ec": ""}
        return {
            "decision": sr.final_decision or "",
            "rationale": "",
            "confidence": sr.consensus_confidence or 0.0,
            "ec": "",
            "a1_decision": sr.agent1.decision if sr.agent1 else "",
            "a1_rationale": sr.agent1.rationale if sr.agent1 else "",
            "a1_conf": sr.agent1.confidence if sr.agent1 else 0.0,
            "a1_ec": sr.agent1.exclusion_code if sr.agent1 else "",
            "a2_decision": sr.agent2.decision if sr.agent2 else "",
            "a2_rationale": sr.agent2.rationale if sr.agent2 else "",
            "a2_conf": sr.agent2.confidence if sr.agent2 else 0.0,
            "a2_ec": sr.agent2.exclusion_code if sr.agent2 else "",
        }

    ts = _extract_screening(pr.screened.title_screening if pr.screened else None)
    sp = _extract_screening(pr.screened.second_pass_screening if pr.screened else None)
    human_verified = (pr.screened.title_screening.human_verified
                      if pr.screened and pr.screened.title_screening else False)

    return {
        "record_id": pr.record_id,
        "title": d.title if d else "",
        "authors": d.authors if d else "",
        "year": d.year if d else None,
        "journal": d.journal_venue if d else "",
        "doi": d.doi if d else "",
        "source_db": d.source_db if d else "",
        "abstract": abstract,
        # First-pass agent rationale
        "agent1_decision": ts["a1_decision"],
        "agent1_rationale": ts["a1_rationale"],
        "agent1_confidence": ts["a1_conf"],
        "agent1_ec": ts["a1_ec"],
        "agent2_decision": ts["a2_decision"],
        "agent2_rationale": ts["a2_rationale"],
        "agent2_confidence": ts["a2_conf"],
        "agent2_ec": ts["a2_ec"],
        # Second-pass agent rationale (populated after second_pass_screening stage)
        "sp_agent1_decision": sp["a1_decision"],
        "sp_agent1_rationale": sp["a1_rationale"],
        "sp_agent1_confidence": sp["a1_conf"],
        "sp_agent1_ec": sp["a1_ec"],
        "sp_agent2_decision": sp["a2_decision"],
        "sp_agent2_rationale": sp["a2_rationale"],
        "sp_agent2_confidence": sp["a2_conf"],
        "sp_agent2_ec": sp["a2_ec"],
        "sp_done": pr.screened.second_pass_screening is not None if pr.screened else False,
        "human_verified": human_verified,
        "final_decision": pr.final_decision,
    }


def _serialize_record_summary(pr: PipelineRecord) -> Dict:
    """Lightweight record summary for list views."""
    title = ""
    if pr.dedup:
        title = pr.dedup.title
    elif pr.raw:
        title = pr.raw.title

    agents_agree = None
    if pr.screened:
        active = pr.screened.fulltext_screening or pr.screened.title_screening
        if active:
            agents_agree = active.agents_agree

    uncertain_extraction_fields = []
    if pr.extracted and pr.extracted.extraction_final:
        uncertain_extraction_fields = pr.extracted.extraction_final.uncertain_fields or []

    return {
        "record_id": pr.record_id,
        "study_id": pr.study_id,
        "title": title,
        "authors": pr.dedup.authors if pr.dedup else "",
        "year": pr.dedup.year if pr.dedup else None,
        "source_db": pr.dedup.source_db if pr.dedup else (pr.raw.source_db if pr.raw else ""),
        "final_decision": pr.final_decision,
        "pipeline_stage": pr.pipeline_stage,
        "agents_agree": agents_agree,
        "human_verified": (pr.extracted.human_verified if pr.extracted else False),
        "uncertain_extraction_fields": uncertain_extraction_fields,
        "updated_at": pr.updated_at.isoformat(),
    }


def _update_env_file(changes: Dict[str, Any]):
    """Update .env file with new values."""
    env_path = Path(".env")
    lines = []

    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()

    # Update existing keys or append
    updated_keys = set()
    new_lines = []
    for line in lines:
        key = line.split("=")[0].strip().upper()
        env_key = {
            "model_title_screening": "MODEL_TITLE_SCREENING",
            "model_fulltext_screening": "MODEL_FULLTEXT_SCREENING",
            "model_extraction": "MODEL_EXTRACTION",
            "model_qa_assessment": "MODEL_QA_ASSESSMENT",
            "model_synthesis": "MODEL_SYNTHESIS",
            "model_agent2_screening": "MODEL_AGENT2_SCREENING",
            "model_agent2_extraction": "MODEL_AGENT2_EXTRACTION",
            "confidence_threshold": "CONFIDENCE_THRESHOLD",
        }
        matched = next((k for k, v in env_key.items() if v == key), None)
        if matched and matched in changes:
            new_lines.append(f"{key}={changes[matched]}\n")
            updated_keys.add(matched)
        else:
            new_lines.append(line)

    # Append new keys
    env_key_map = {
        "model_title_screening": "MODEL_TITLE_SCREENING",
        "model_fulltext_screening": "MODEL_FULLTEXT_SCREENING",
        "model_extraction": "MODEL_EXTRACTION",
        "confidence_threshold": "CONFIDENCE_THRESHOLD",
    }
    for k, v in changes.items():
        if k not in updated_keys:
            env_key = k.upper()
            new_lines.append(f"{env_key}={v}\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
