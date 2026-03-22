"""
Phase 1 Data Extraction Module.

Two-step pipeline:
  Step 1 — GPT-5 open extraction from Methods + Results sections.
  Step 2 — gpt-4.1-mini verification model checks evidence alignment.

Outputs are saved separately so each step can be inspected / re-run:
  data/extracted/phase1/gpt5_extractions.jsonl   — Step 1 raw outputs
  data/extracted/phase1/verifications.jsonl       — Step 2 verification outputs
  data/extracted/phase1/phase1_complete.jsonl     — Combined record per paper
"""
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .openai_client import get_client
from .prompts import PHASE1_EXTRACTION_SYSTEM, PHASE1_EXTRACTION_USER
from .prompts import PHASE1_VERIFICATION_SYSTEM, PHASE1_VERIFICATION_USER
from config.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SECTION CATALOG
# Each entry: canonical_name → list of bare keyword patterns (no anchors yet).
# Anchors and number-prefix handling are added in _build_heading_re().
# ─────────────────────────────────────────────

_SECTION_CATALOG: Dict[str, List[str]] = {
    "introduction": [
        r"introduction",
        r"background",
        r"overview",
        r"motivation",
    ],
    "related_work": [
        r"related\s+work",
        r"literature\s+review",
        r"prior\s+work",
        r"related\s+literature",
    ],
    "methods": [
        r"methods?",
        r"methodology",
        r"materials?\s+and\s+methods?",
        r"study\s+design",
        r"research\s+design",
        r"experimental\s+(?:setup|design|procedure[s]?)",
        r"data\s+collection(?:\s+and\s+(?:analysis|procedure))?",
        r"analytic(?:al)?\s+(?:approach|framework|strategy|method)",
        r"research\s+method(?:ology)?",
        r"procedure[s]?",
        r"participants?\s+and\s+(?:procedure|method)",
        r"data\s+analysis(?:\s+approach)?",
        r"coding\s+(?:procedure|process|approach)",
        # CS / NLP / computational paper section names
        r"(?:our\s+)?approach",
        r"(?:our\s+)?(?:proposed\s+)?(?:system|model|framework|pipeline|method)",
        r"dataset(?:\s+and\s+(?:method|approach|setup))?",
        r"data(?:\s+and\s+(?:method|setup|collection))?",
        r"corpus(?:\s+and\s+(?:method|setup))?",
        r"experimental\s+(?:evaluation|conditions?)",
        r"(?:task|study)\s+(?:design|setup|description)",
        r"(?:open|axial|inductive|deductive)\s+coding(?:\s+method)?",
        r"qualitative\s+(?:method|procedure|coding|approach)",
        r"(?:data|text)\s+(?:coding|annotation)\s+(?:process|procedure|approach)",
    ],
    "results": [
        r"results?",
        r"findings?",
        r"results?\s+and\s+discussion",
        # NOTE: bare "analysis" removed — too generic, matches mid-paragraph in PDFs
        r"analysis\s+and\s+(?:results?|findings?|discussion)",
        r"outcomes?",
        r"empirical\s+(?:results?|findings?|analysis)",
        r"main\s+(?:results?|findings?)",
        r"themes?\s+and\s+(?:categories|findings?|results?)",
        r"qualitative\s+(?:results?|findings?|analysis)",
        r"evaluation(?:\s+results?)?",
        r"experiments?\s+and\s+(?:results?|findings?|evaluation)",
        r"performance(?:\s+results?)?",
    ],
    "discussion": [
        r"discussion",
        r"discussion\s+and\s+(?:conclusion[s]?|implication[s]?)",
        r"implication[s]?",
        r"interpretation[s]?",
    ],
    "conclusion": [
        r"conclusion[s]?",
        r"concluding\s+remarks?",
        r"summary\s+and\s+conclusion[s]?",
        r"final\s+remarks?",
        r"summary",
    ],
    "limitations": [
        r"limitation[s]?",
        r"limitation[s]?\s+and\s+future\s+(?:work|direction[s]?|research)",
        r"future\s+(?:work|direction[s]?|research)",
    ],
    "references": [
        r"references?",
        r"bibliography",
        r"works?\s+cited",
    ],
    "acknowledgements": [
        r"acknowledgements?",
        r"acknowledgments?",
        r"funding",
    ],
    "appendix": [
        r"appendix",
        r"appendices",
        r"supplementary(?:\s+material[s]?)?",
        r"supplemental(?:\s+material[s]?)?",
    ],
    "declarations": [
        r"declarations?",
        r"(?:competing\s+)?interests?",
        r"conflicts?\s+of\s+interest[s]?",
        r"data\s+availability(?:\s+statement)?",
        r"ethics(?:\s+statement)?",
        r"author\s+contributions?",
    ],
}

# Maximum characters sent per section to the LLM (controls API cost).
_MAX_SECTION_CHARS = 8_000

# Keywords whose density signals methodology / results content (for fallback).
_METHODS_KEYWORDS = [
    "interview", "participant", "coding", "prompt", "llm", "gpt", "claude",
    "sample", "recruited", "collected", "thematic", "inductive", "deductive",
    "framework", "procedure", "analysis", "codebook", "researcher",
]
_RESULTS_KEYWORDS = [
    "theme", "finding", "result", "category", "code", "pattern",
    "identified", "emerged", "revealed", "showed", "reported", "table",
    "figure", "percent", "majority", "participants described",
]


# ─────────────────────────────────────────────
# HEADING DETECTION
# ─────────────────────────────────────────────

def _build_heading_re(patterns: List[str]) -> re.Pattern:
    """
    Build a compiled regex that matches a heading line for the given patterns.

    A valid heading line:
      - Optionally starts with a section number like "2.", "2.1", "II."
      - Is followed (optionally via whitespace) by one of the keyword patterns
      - The entire line content (after stripping number prefix) is ≤ 80 chars
        (prevents matching long sentences that happen to start with a keyword)
      - Preceded by a line boundary (start of string or newline)
    """
    number_prefix = r"(?:\d+(?:\.\d+)*\.?\s+|[IVXivx]+\.\s+)?"
    alts = "|".join(f"(?:{p})" for p in patterns)
    # Full pattern: line start, optional number, keyword(s), end of short line
    pat = (
        r"(?:^|\n)"                        # line boundary
        r"[ \t]*"                          # optional leading whitespace
        + number_prefix +                  # optional "2." / "2.1." / "II."
        r"(?:" + alts + r")"               # the keyword alternatives
        r"[ \t]*(?::|\.)?[ \t]*$"          # optional trailing colon/period
    )
    return re.compile(pat, re.IGNORECASE | re.MULTILINE)


# Pre-compile one regex per catalog entry
_COMPILED: Dict[str, re.Pattern] = {
    name: _build_heading_re(patterns)
    for name, patterns in _SECTION_CATALOG.items()
}


def _find_all_headings(text: str) -> List[Tuple[int, str]]:
    """
    Scan the full text for all recognised section headings.
    Returns a sorted list of (char_offset, canonical_section_name) tuples.
    Only the first match of each canonical name is kept (deduplication).
    """
    found: Dict[str, int] = {}  # name -> offset

    for name, pat in _COMPILED.items():
        m = pat.search(text)
        if m:
            # Skip past the leading newline so the offset points to the heading text
            offset = m.start() + (1 if text[m.start()] == "\n" else 0)
            found[name] = offset

    # Return as (offset, name) sorted by offset
    return sorted(((pos, name) for name, pos in found.items()), key=lambda x: x[0])


# ─────────────────────────────────────────────
# KEYWORD-DENSITY FALLBACK
# ─────────────────────────────────────────────

def _snap_to_paragraph(text: str, offset: int) -> int:
    """
    Snap an offset backwards to the start of its containing paragraph.
    A paragraph boundary is two or more newlines (or start of text).
    """
    if offset <= 0:
        return 0
    # Search backwards for a blank line (\n\n or \n\r\n)
    boundary = text.rfind("\n\n", 0, offset)
    if boundary == -1:
        # Try single newline as last resort
        boundary = text.rfind("\n", 0, offset)
    return boundary + 1 if boundary != -1 else 0


def _density_fallback(text: str, keywords: List[str], window: int = 4000) -> str:
    """
    Slide a window over the text and return the window with the highest
    keyword hit count.  Used when section headings cannot be found.

    The window start is snapped to the nearest paragraph boundary so the
    returned text never begins mid-word or mid-sentence.
    """
    if not text:
        return ""
    lower = text.lower()
    step = window // 2
    best_score = -1
    best_start = 0

    for start in range(0, max(1, len(lower) - window + 1), step):
        chunk = lower[start: start + window]
        score = sum(chunk.count(kw) for kw in keywords)
        if score > best_score:
            best_score = score
            best_start = start

    # Snap to paragraph boundary
    best_start = _snap_to_paragraph(text, best_start)
    return text[best_start: best_start + window].strip()


# ─────────────────────────────────────────────
# PUBLIC: extract_sections
# ─────────────────────────────────────────────

def extract_sections(fulltext: str) -> Tuple[str, str]:
    """
    Extract Methods and Results sections from a full-text string.

    Strategy:
      1. Find ALL recognised section headings and sort them by position.
      2. For each target section (methods, results), the content runs from
         its heading to the start of the next heading — no fixed length.
      3. If a section heading is completely absent, use keyword-density
         sliding-window search on the full text as a fallback.
      4. Each section is capped at _MAX_SECTION_CHARS before being returned.

    Returns (methods_text, results_text).
    """
    if not fulltext:
        return "", ""

    # ── 1. Find all heading positions ─────────────────────────────────
    headings = _find_all_headings(fulltext)   # [(offset, name), ...]
    heading_positions = [pos for pos, _ in headings]
    heading_names = {name: pos for pos, name in headings}

    def _slice_section(start_pos: int) -> str:
        """Return text from start_pos to the next heading (or end of doc)."""
        next_starts = [p for p in heading_positions if p > start_pos]
        end_pos = min(next_starts) if next_starts else len(fulltext)
        return fulltext[start_pos:end_pos].strip()

    # ── 2. Methods ────────────────────────────────────────────────────
    methods_text = ""
    if "methods" in heading_names:
        methods_text = _slice_section(heading_names["methods"])
        logger.debug(f"[SectionParser] Methods found via heading at offset {heading_names['methods']}")
    else:
        methods_text = _density_fallback(fulltext, _METHODS_KEYWORDS)
        logger.debug("[SectionParser] Methods heading not found — using keyword-density fallback")

    # ── 3. Results ────────────────────────────────────────────────────
    results_text = ""
    if "results" in heading_names:
        results_text = _slice_section(heading_names["results"])
        logger.debug(f"[SectionParser] Results found via heading at offset {heading_names['results']}")
    else:
        results_text = _density_fallback(fulltext, _RESULTS_KEYWORDS)
        logger.debug("[SectionParser] Results heading not found — using keyword-density fallback")

    # ── 4. Truncate ───────────────────────────────────────────────────
    methods_text = methods_text[:_MAX_SECTION_CHARS]
    results_text = results_text[:_MAX_SECTION_CHARS]

    return methods_text, results_text


def section_extraction_report(fulltext: str) -> Dict[str, Any]:
    """
    Diagnostic helper: return which headings were found and at what offsets.
    Useful for inspecting / debugging section detection on a specific paper.
    """
    headings = _find_all_headings(fulltext)
    return {
        "headings_found": [{"section": name, "offset": pos} for pos, name in headings],
        "methods_via_heading": any(n == "methods" for _, n in headings),
        "results_via_heading": any(n == "results" for _, n in headings),
        "total_chars": len(fulltext),
    }


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
        raw["paper_id"] = paper_id
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
# STEP 2: VERIFICATION (gpt-4.1-mini)
# ─────────────────────────────────────────────

def _run_verification(
    paper_id: str,
    gpt5_output: Dict[str, Any],
    methods_text: str,
    results_text: str,
) -> Dict[str, Any]:
    """
    Run the verification model against GPT-5 extraction + original sections.
    Returns the parsed JSON dict (or an error envelope on failure).
    """
    client = get_client()

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
      2. GPT-5 open extraction (Step 1).
      3. Verification model checks evidence alignment (Step 2).
      4. Append each step's output to its dedicated JSONL file.
      5. Return combined Phase 1 record.
    """
    out_dir = _ensure_phase1_dir()
    gpt5_path = out_dir / "gpt5_extractions.jsonl"
    verify_path = out_dir / "verifications.jsonl"
    complete_path = out_dir / "phase1_complete.jsonl"

    paper_id = study_id or record_id

    # ── 1. Section extraction ──────────────────────────────────────────
    methods_text, results_text = extract_sections(fulltext)
    report = section_extraction_report(fulltext)
    sections_meta = {
        "methods_chars": len(methods_text),
        "results_chars": len(results_text),
        "methods_via_heading": report["methods_via_heading"],
        "results_via_heading": report["results_via_heading"],
        "headings_found": [h["section"] for h in report["headings_found"]],
    }
    logger.info(
        f"[Phase1] {paper_id}: methods={sections_meta['methods_chars']} chars "
        f"(heading={sections_meta['methods_via_heading']}), "
        f"results={sections_meta['results_chars']} chars "
        f"(heading={sections_meta['results_via_heading']})"
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
        "sections_extracted": sections_meta,
        "gpt5_extraction": {k: v for k, v in gpt5_result.items() if k != "_meta"},
        "verification": {k: v for k, v in verification_result.items() if k != "_meta"},
        "extraction_model": settings.model_phase1_extraction,
        "verification_model": settings.model_phase1_verification,
        "completed_at": datetime.utcnow().isoformat(),
        "has_extraction_error": "_error" in gpt5_result,
        "has_verification_error": "_error" in verification_result,
    }
    _append_jsonl(complete_path, combined)

    return combined
