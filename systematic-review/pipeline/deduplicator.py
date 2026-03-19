"""
Deduplication module.
Strategy (in order of priority):
1. Exact DOI match
2. Fuzzy title + year match (Levenshtein ratio > 0.92)
3. Normalized title exact match
"""
import re
import json
from typing import List, Tuple, Dict
from difflib import SequenceMatcher

from .models import RawRecord, DedupRecord


def normalize_doi(doi: str) -> str:
    """Normalize DOI to bare form: strip URL prefixes, lowercase, strip whitespace."""
    doi = (doi or "").strip().lower()
    # Strip common prefixes
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip()


def normalize_title(title: str) -> str:
    """Normalize title for comparison: lowercase, strip punctuation, collapse whitespace."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_similarity(t1: str, t2: str) -> float:
    """Compute similarity ratio between two normalized titles."""
    return SequenceMatcher(None, t1, t2).ratio()


def deduplicate(records: List[RawRecord],
                fuzzy_threshold: float = 0.92) -> Tuple[List[DedupRecord], Dict]:
    """
    Deduplicate a list of RawRecords.

    Returns:
        - List of DedupRecord (with is_duplicate and duplicate_of set)
        - Stats dict
    """
    deduped: List[DedupRecord] = []
    seen_dois: Dict[str, str] = {}       # doi -> record_id
    seen_titles: Dict[str, str] = {}     # normalized_title -> record_id

    stats = {
        "total_input": len(records),
        "duplicates_by_doi": 0,
        "duplicates_by_exact_title": 0,
        "duplicates_by_fuzzy_title": 0,
        "unique_records": 0,
    }

    for record in records:
        drec = DedupRecord(**record.model_dump())
        is_dup = False
        dup_of = None
        method = None
        confidence = None

        # --- Strategy 1: DOI match ---
        doi = normalize_doi(record.doi or "")
        if doi and doi not in ("n/a", "none"):
            if doi in seen_dois:
                is_dup = True
                dup_of = seen_dois[doi]
                method = "doi"
                confidence = 1.0
                stats["duplicates_by_doi"] += 1

        # --- Strategy 2: Normalized exact title + same year ---
        if not is_dup:
            norm_title = normalize_title(record.title)
            if norm_title in seen_titles:
                canonical_id = seen_titles[norm_title]
                # Optionally check year agreement too
                is_dup = True
                dup_of = canonical_id
                method = "title_exact"
                confidence = 1.0
                stats["duplicates_by_exact_title"] += 1

        # --- Strategy 3: Fuzzy title match ---
        if not is_dup and record.title:
            norm_title = normalize_title(record.title)
            for seen_norm, seen_id in seen_titles.items():
                sim = title_similarity(norm_title, seen_norm)
                if sim >= fuzzy_threshold:
                    is_dup = True
                    dup_of = seen_id
                    method = "title_fuzzy"
                    confidence = sim
                    stats["duplicates_by_fuzzy_title"] += 1
                    break

        if is_dup:
            drec.is_duplicate = True
            drec.duplicate_of = dup_of
            drec.dedup_method = method
            drec.dedup_confidence = confidence
        else:
            # Register as canonical
            if doi and doi not in ("n/a", "none"):
                seen_dois[doi] = record.record_id
            norm_title = normalize_title(record.title)
            seen_titles[norm_title] = record.record_id
            stats["unique_records"] += 1

        deduped.append(drec)

    return deduped, stats


def filter_unique(records: List[DedupRecord]) -> List[DedupRecord]:
    """Return only non-duplicate records."""
    return [r for r in records if not r.is_duplicate]


def save_deduped(records: List[DedupRecord], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def load_deduped(filepath: str) -> List[DedupRecord]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(DedupRecord.model_validate_json(line))
    return records
