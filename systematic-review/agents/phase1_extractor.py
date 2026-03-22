"""
Phase 1 Data Extraction Module.

Two-step pipeline:
  Step 1 — GPT-5 open extraction from Methods + Results sections.
  Step 2 — GPT-4o-mini (verification model) checks evidence alignment.

Outputs are saved separately so each step can be inspected / re-run:
  data/extracted/phase1/gpt5_extractions.jsonl   — Step 1 raw outputs
  data/extracted/phase1/verifications.jsonl       — Step 2 verification outputs
  data/extracted/phase1/phase1_complete.jsonl     — Combined record per paper
"""
import json
import logging
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from .openai_client import get_client
from .prompts import PHASE1_EXTRACTION_SYSTEM, PHASE1_EXTRACTION_USER
from .prompts import PHASE1_VERIFICATION_SYSTEM, PHASE1_VERIFICATION_USER
from config.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SECTION EXTRACTION UTILITIES
# ─────────────────────────────────────────────

# Ordered list of heading patterns that mark the start of Methods / Results.
# Each tuple: (canonical_name, list_of_regex_patterns)
_SECTION_PATTERNS = {
    "methods": [
        r"(?:^|\n)\s*(?:2\.?\s+)?(?:materials?\s+and\s+)?methods?\b",
        r"(?:^|\n)\s*methodology\b",
        r"(?:^|\n)\s*study\s+design\b",
        r"(?:^|\n)\s*research\s+design\b",
        r"(?:^|\n)\s*data\s+collection\s+(?:and\s+)?(?:analysis|procedure)",
        r"(?:^|\n)\s*analytic\s+approach\b",
    ],
    "results": [
        r"(?:^|\n)\s*(?:3\.?\s+)?results?\b",
        r"(?:^|\n)\s*findings?\b",
        r"(?:^|\n)\s*outcomes?\b",
    ],
}

# Sections that typically follow Results — used to detect section end.
_TRAILING_SECTIONS = [
    r"(?:^|\n)\s*(?:4\.?\s+)?discussion\b",
    r"(?:^|\n)\s*conclusion[s]?\b",
    r"(?:^|\n)\s*limitation[s]?\b",
    r"(?:^|\n)\s*references?\b",
    r"(?:^|\n)\s*bibliography\b",
    r"(?:^|\n)\s*acknowledgements?\b",
    r"(?:^|\n)\s*appendix\b",
    r"(?:^|\n)\s*declaration[s]?\b",
]

# Maximum characters to send per section (keeps API cost reasonable).
_MAX_SECTION_CHARS = 8_000


def _find_section(text: str, patterns: list[str]) -> Optional[int]:
    """Return the character offset of the first pattern match, or None."""
    earliest: Optional[int] = None
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            if earliest is None or m.start() < earliest:
                earliest = m.start()
    return earliest


def extract_sections(fulltext: str) -> Tuple[str, str]:
    """
    Parse Methods and Results sections from a full-text string.

    Returns (methods_text, results_text).
    Falls back to balanced halves of the full text if headings are not found.
    Each section is capped at _MAX_SECTION_CHARS.
    """
    if not fulltext:
        return "", ""

    methods_start = _find_section(fulltext, _SECTION_PATTERNS["methods"])
    results_start = _find_section(fulltext, _SECTION_PATTERNS["results"])

    # ── Methods ──────────────────────────────────────────────────────────
    if methods_start is not None:
        # End of methods = start of results (if after) or first trailing section
        candidates = []
        if results_start is not None and results_start > methods_start:
            candidates.append(results_start)
        for pat in _TRAILING_SECTIONS:
            m = re.search(pat, fulltext[methods_start:], re.IGNORECASE | re.MULTILINE)
            if m:
                candidates.append(methods_start + m.start())
        methods_end = min(candidates) if candidates else methods_start + _MAX_SECTION_CHARS
        methods_text = fulltext[methods_start:methods_end].strip()
    else:
        # Fallback: first half of text
        mid = len(fulltext) // 2
        methods_text = fulltext[:mid].strip()

    # ── Results ──────────────────────────────────────────────────────────
    if results_start is not None:
        # End of results = first trailing section after results_start
        candidates = []
        for pat in _TRAILING_SECTIONS:
            m = re.search(pat, fulltext[results_start:], re.IGNORECASE | re.MULTILINE)
            if m:
                candidates.append(results_start + m.start())
        results_end = min(candidates) if candidates else results_start + _MAX_SECTION_CHARS
        results_text = fulltext[results_start:results_end].strip()
    else:
        # Fallback: second half of text
        mid = len(fulltext) // 2
        results_text = fulltext[mid:].strip()

    # Truncate to safe API size
    methods_text = methods_text[:_MAX_SECTION_CHARS]
    results_text = results_text[:_MAX_SECTION_CHARS]

    return methods_text, results_text


# ─────────────────────────────────────────────
# STEP 1: GPT-5 PRIMARY EXTRACTION
# ─────────────────────────────────────────────

def _run_gpt5_extraction(
    paper_id: str,
    methods_text: str,
    results_text: str,
) -> Dict[str, Any]:
    """
    Run GPT-5 open extraction on Methods + Results sections.
    Returns the parsed JSON dict (or an error envelope on failure).
    """
    client = get_client()
    user_prompt = PHASE1_EXTRACTION_USER.format(
        paper_id=paper_id,
        methods_text=methods_text or "not available",
        results_text=results_text or "not available",
    )

    try:
        raw, usage = client.chat_json(
            system_prompt=PHASE1_EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            model=settings.model_phase1_extraction,
            max_tokens=4000,
        )
        raw["paper_id"] = paper_id  # ensure paper_id is set
        raw["_meta"] = {
            "model": settings.model_phase1_extraction,
            "step": "gpt5_extraction",
            "timestamp": datetime.utcnow().isoformat(),
            "usage": usage,
        }
        logger.info(f"[Phase1-Extract] {paper_id}: extraction complete")
        return raw
    except Exception as e:
        logger.error(f"[Phase1-Extract] {paper_id}: extraction failed — {e}")
        return {
            "paper_id": paper_id,
            "_error": str(e),
            "_meta": {
                "model": settings.model_phase1_extraction,
                "step": "gpt5_extraction",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }


# ─────────────────────────────────────────────
# STEP 2: VERIFICATION (gpt-4o-mini)
# ─────────────────────────────────────────────

def _run_verification(
    paper_id: str,
    gpt5_output: Dict[str, Any],
    methods_text: str,
    results_text: str,
) -> Dict[str, Any]:
    """
    Run verification model against GPT-5 extraction + original sections.
    Returns the parsed JSON dict (or an error envelope on failure).
    """
    client = get_client()

    # Serialize GPT-5 output neatly for the verification prompt
    gpt5_str = json.dumps(
        {k: v for k, v in gpt5_output.items() if not k.startswith("_")},
        indent=2,
        ensure_ascii=False,
    )

    user_prompt = PHASE1_VERIFICATION_USER.format(
        paper_id=paper_id,
        gpt5_output=gpt5_str,
        methods_text=methods_text or "not available",
        results_text=results_text or "not available",
    )

    try:
        raw, usage = client.chat_json(
            system_prompt=PHASE1_VERIFICATION_SYSTEM,
            user_prompt=user_prompt,
            model=settings.model_phase1_verification,
            max_tokens=2000,
        )
        raw["paper_id"] = paper_id
        raw["_meta"] = {
            "model": settings.model_phase1_verification,
            "step": "verification",
            "timestamp": datetime.utcnow().isoformat(),
            "usage": usage,
        }
        logger.info(f"[Phase1-Verify] {paper_id}: verification complete")
        return raw
    except Exception as e:
        logger.error(f"[Phase1-Verify] {paper_id}: verification failed — {e}")
        return {
            "paper_id": paper_id,
            "_error": str(e),
            "_meta": {
                "model": settings.model_phase1_verification,
                "step": "verification",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def _ensure_phase1_dir() -> Path:
    out = Path(settings.phase1_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_phase1_for_record(
    record_id: str,
    study_id: str,
    title: str,
    fulltext: str,
) -> Dict[str, Any]:
    """
    Execute full Phase 1 extraction pipeline for a single included study.

    Steps:
      1. Parse Methods and Results sections from fulltext.
      2. GPT-5 open extraction.
      3. Verification model checks evidence alignment.
      4. Append each step's output to its dedicated JSONL file.
      5. Return combined Phase 1 record (dict).

    The combined record is also appended to phase1_complete.jsonl.
    """
    out_dir = _ensure_phase1_dir()
    gpt5_path = out_dir / "gpt5_extractions.jsonl"
    verify_path = out_dir / "verifications.jsonl"
    complete_path = out_dir / "phase1_complete.jsonl"

    paper_id = study_id or record_id

    # ── 1. Extract sections ────────────────────────────────────────────
    methods_text, results_text = extract_sections(fulltext)
    sections_found = {
        "methods_chars": len(methods_text),
        "results_chars": len(results_text),
    }
    logger.info(
        f"[Phase1] {paper_id}: methods={sections_found['methods_chars']} chars, "
        f"results={sections_found['results_chars']} chars"
    )

    # ── 2. GPT-5 extraction ────────────────────────────────────────────
    gpt5_result = _run_gpt5_extraction(paper_id, methods_text, results_text)
    _append_jsonl(gpt5_path, gpt5_result)

    # ── 3. Verification ────────────────────────────────────────────────
    verification_result = _run_verification(
        paper_id, gpt5_result, methods_text, results_text
    )
    _append_jsonl(verify_path, verification_result)

    # ── 4. Combine ─────────────────────────────────────────────────────
    combined = {
        "record_id": record_id,
        "study_id": study_id,
        "paper_id": paper_id,
        "title": title,
        "phase": "phase1",
        "sections_extracted": sections_found,
        "gpt5_extraction": {k: v for k, v in gpt5_result.items() if not k.startswith("_meta")},
        "verification": {k: v for k, v in verification_result.items() if not k.startswith("_meta")},
        "extraction_model": settings.model_phase1_extraction,
        "verification_model": settings.model_phase1_verification,
        "completed_at": datetime.utcnow().isoformat(),
        "has_extraction_error": "_error" in gpt5_result,
        "has_verification_error": "_error" in verification_result,
    }
    _append_jsonl(complete_path, combined)

    return combined
