"""
Basic pipeline tests — no OpenAI calls required.
Tests import, deduplication, and data model validation.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.models import RawRecord, DedupRecord, DecisionLabel, PipelineRecord, PipelineStage
from pipeline.deduplicator import deduplicate, normalize_title, title_similarity
from pipeline.importer import import_json
import json
import tempfile


# ─────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────

def test_raw_record_creation():
    r = RawRecord(
        source_db="PubMed",
        title="Testing LLMs in Qualitative Research",
        year=2024,
        abstract="An empirical study...",
    )
    assert r.record_id is not None
    assert r.title == "Testing LLMs in Qualitative Research"


def test_decision_labels():
    assert DecisionLabel.INCLUDE == "Included"
    assert DecisionLabel.EXCLUDE == "Excluded"
    assert DecisionLabel.UNCERTAIN == "Needs Human Verification"
    assert DecisionLabel.FULL_TEXT_NEEDED == "Full Text Needed"


def test_pipeline_record_update():
    pr = PipelineRecord()
    pr.update_stage(PipelineStage.TITLE_SCREENING, DecisionLabel.INCLUDE)
    assert pr.pipeline_stage == PipelineStage.TITLE_SCREENING
    assert pr.final_decision == DecisionLabel.INCLUDE


# ─────────────────────────────────────────────
# DEDUPLICATION TESTS
# ─────────────────────────────────────────────

def make_record(title, doi=None, year=2024):
    return RawRecord(source_db="Test", title=title, doi=doi, year=year)


def test_dedup_by_doi():
    records = [
        make_record("Paper A", doi="10.1000/abc"),
        make_record("Paper A (reprint)", doi="10.1000/abc"),  # same DOI
        make_record("Paper B", doi="10.1000/xyz"),
    ]
    deduped, stats = deduplicate(records)
    dups = [r for r in deduped if r.is_duplicate]
    unique = [r for r in deduped if not r.is_duplicate]
    assert len(dups) == 1
    assert len(unique) == 2
    assert dups[0].dedup_method == "doi"
    assert stats["duplicates_by_doi"] == 1


def test_dedup_by_exact_title():
    records = [
        make_record("Using GPT-4 for Thematic Analysis"),
        make_record("Using GPT-4 for Thematic Analysis"),  # exact same
    ]
    deduped, stats = deduplicate(records)
    dups = [r for r in deduped if r.is_duplicate]
    assert len(dups) == 1


def test_dedup_by_fuzzy_title():
    records = [
        make_record("Using GPT-4 for Qualitative Analysis: A Study"),
        make_record("Using GPT-4 for Qualitative Analysis: A Study."),  # minor diff
    ]
    deduped, stats = deduplicate(records)
    dups = [r for r in deduped if r.is_duplicate]
    assert len(dups) == 1


def test_dedup_no_false_positives():
    records = [
        make_record("LLMs in Healthcare Qualitative Research"),
        make_record("ChatGPT for Thematic Analysis in Education"),
        make_record("Grounded Theory with AI Assistance"),
    ]
    deduped, stats = deduplicate(records)
    dups = [r for r in deduped if r.is_duplicate]
    assert len(dups) == 0
    assert stats["unique_records"] == 3


def test_normalize_title():
    t1 = normalize_title("Using GPT-4 for Qualitative Analysis!")
    t2 = normalize_title("using gpt 4 for qualitative analysis")
    # Both should be very similar after normalization
    sim = title_similarity(t1, t2)
    assert sim > 0.9


# ─────────────────────────────────────────────
# IMPORT TESTS
# ─────────────────────────────────────────────

def test_import_json():
    data = [
        {
            "title": "LLM-Assisted Coding in Qualitative Research",
            "authors": "Smith, J; Jones, K",
            "year": 2024,
            "journal_venue": "Qualitative Methods Journal",
            "doi": "10.9999/test1",
            "abstract": "This study examines...",
            "source_db": "TestDB",
        }
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name

    records = import_json(path, source_db="TestDB")
    assert len(records) == 1
    assert records[0].title == "LLM-Assisted Coding in Qualitative Research"
    assert records[0].year == 2024
    os.unlink(path)


def test_import_json_missing_title():
    """Records without titles should be skipped."""
    data = [
        {"title": "", "year": 2024},
        {"title": "Valid Paper", "year": 2024},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name

    records = import_json(path)
    assert len(records) == 1
    assert records[0].title == "Valid Paper"
    os.unlink(path)


# ─────────────────────────────────────────────
# SERIALIZATION TESTS
# ─────────────────────────────────────────────

def test_record_json_roundtrip():
    r = RawRecord(
        source_db="PubMed",
        title="Test Record",
        year=2024,
        abstract="Some abstract text.",
        doi="10.1000/test",
    )
    json_str = r.model_dump_json()
    r2 = RawRecord.model_validate_json(json_str)
    assert r.title == r2.title
    assert r.record_id == r2.record_id
    assert r.year == r2.year
