# Systematic Review Automation System
### LLMs in Qualitative Data Analysis (2023–2026)

A semi-automated, human-in-the-loop systematic review pipeline built with OpenAI API, FastAPI, and a browser-based UI.

---

## Architecture Overview

```
systematic-review/
├── config/
│   └── settings.py          ← All model assignments, thresholds, paths
├── pipeline/
│   ├── models.py             ← Pydantic data models (all stages)
│   ├── importer.py           ← Import RIS/CSV/BibTeX/JSON/PubMed XML
│   ├── deduplicator.py       ← DOI + fuzzy title deduplication
│   └── pipeline.py           ← Orchestrator (runs all stages, saves state)
├── agents/
│   ├── prompts.py            ← All LLM prompts (versioned)
│   ├── openai_client.py      ← OpenAI wrapper with retry + cost tracking
│   ├── screener.py           ← Multi-agent screening (title + full-text)
│   └── extractor.py          ← Multi-agent extraction + QA assessment
├── api/
│   └── main.py               ← FastAPI REST backend
├── ui/
│   ├── index.html            ← Browser UI
│   ├── styles.css            ← Dark theme styles
│   └── app.js                ← Frontend logic
├── data/
│   ├── raw/                  ← ← Place search export files here
│   ├── deduped/
│   ├── screened/
│   ├── extracted/
│   ├── output/
│   └── pdfs/                 ← ← Place/upload PDF files here
├── docs/
│   └── refined_protocol.md   ← Full refined PRISMA protocol v2.0
├── tests/
│   └── test_pipeline.py      ← Unit tests (no API key required)
├── .env.example              ← Environment variable template
├── requirements.txt
├── run.py                    ← Quick-start script
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
cd systematic-review
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add your search results
Place your exported search files in `data/raw/`:
- `.ris` — from Scopus, PubMed, etc.
- `.csv` — from Scopus, Web of Science
- `.bib` — BibTeX
- `.json` — custom JSON array
- `.xml` — PubMed XML export

### 4. Start the system
```bash
python run.py
```
→ API: http://localhost:8000/docs
→ UI:  http://localhost:8000/app

---

## Pipeline Stages

### Stage 1: Import
```bash
python run.py --stage import
```
Reads all files from `data/raw/`. Auto-detects format by extension.

### Stage 2: Deduplication
```bash
python run.py --stage dedup
```
Removes duplicates using (in priority order):
1. Exact DOI match
2. Normalized exact title match
3. Fuzzy title match (Levenshtein ratio ≥ 0.92)

### Stage 3: Title/Abstract Screening
```bash
python run.py --stage title_screening [--limit 50]
```
- **Agent 1** screens using `MODEL_TITLE_SCREENING` (default: `gpt-4o-mini`)
- **Agent 2** screens independently using `MODEL_AGENT2_SCREENING` (default: `gpt-4o`)
- If both agree with confidence ≥ threshold → decision is final
- If they disagree or either is uncertain → **Needs Human Verification**

### Stage 4: Full-Text Screening
```bash
python run.py --stage fulltext_screening [--limit 20]
```
Requires PDFs in `data/pdfs/` named `{record_id}.pdf` or `{doi_safe}.pdf`.
Papers without PDFs → **Full Text Needed** list.

### Stage 5: Data Extraction
```bash
python run.py --stage extraction [--limit 10]
```
- Two agents extract independently
- Field-by-field comparison
- Hard fields (model_name, workflow_structure, etc.) → **Needs Human Verification** on disagreement
- QA assessment scored 0–10

---

## Output Categories

Every record is classified into exactly one bucket:

| Category | Meaning |
|----------|---------|
| **Included** | Both agents agree; meets all criteria |
| **Excluded** | Both agents agree; fails criteria |
| **Needs Human Verification** | Agents disagree, uncertain, or borderline |
| **Full Text Needed** | PDF unavailable for full-text screening |

---

## Human-in-the-Loop Workflow

### Via UI
1. Navigate to **Human Verification** panel
2. Review agent decisions side-by-side
3. Read abstract and rationale
4. Choose: Include / Exclude / Keep Uncertain
5. Enter rationale (required)
6. For extraction disagreements: edit fields directly in the UI

### Via API
```bash
curl -X POST http://localhost:8000/api/records/{record_id}/verify \
  -H "Content-Type: application/json" \
  -d '{"decision": "Included", "rationale": "Both criteria IC1-IC5 met", "reviewer": "ZH"}'
```

---

## PDF Management

### Upload via UI
1. Go to **Full Text Needed** panel
2. Select record → upload PDF

### Batch upload
Name PDFs as `{record_id}.pdf` and place in `data/pdfs/`, or use the batch upload endpoint:
```bash
curl -X POST http://localhost:8000/api/pdfs/upload-batch \
  -F "files=@paper1.pdf" -F "files=@paper2.pdf"
```

---

## Model Configuration

All models are configurable per task. Edit `.env` or use the **Model Config** UI panel:

| Task | Env Variable | Default |
|------|-------------|---------|
| Title screening (Agent 1) | `MODEL_TITLE_SCREENING` | `gpt-4o-mini` |
| Full-text screening (Agent 1) | `MODEL_FULLTEXT_SCREENING` | `gpt-4o` |
| Data extraction (Agent 1) | `MODEL_EXTRACTION` | `gpt-4o` |
| QA assessment | `MODEL_QA_ASSESSMENT` | `gpt-4o` |
| Screening (Agent 2) | `MODEL_AGENT2_SCREENING` | `gpt-4o` |
| Extraction (Agent 2) | `MODEL_AGENT2_EXTRACTION` | `gpt-4o-mini` |

**Cost tip:** Using `gpt-4o-mini` for Agent 1 title screening and `gpt-4o` for Agent 2 balances cost and independence.

---

## Traceability and Audit Trail

Every decision is recorded with:
- Agent ID and model name
- Confidence score (0–1)
- Rationale text
- Criterion codes (IC1–IC5, EC1–EC9)
- Timestamp
- Human reviewer name and rationale (if verified)

Full pipeline state is persisted to `data/pipeline_state.jsonl` after every stage.

Export the complete audit trail:
```bash
curl http://localhost:8000/api/export/all-records -o audit_export.json
```

---

## Evidence Table Export

```bash
# CSV
curl http://localhost:8000/api/export/evidence-table/csv -o evidence_table.csv

# JSON
curl http://localhost:8000/api/export/evidence-table -o evidence_table.json

# PRISMA counts
curl http://localhost:8000/api/export/prisma -o prisma_counts.json
```

---

## Running Tests

```bash
pytest tests/ -v
```
Tests cover: import, deduplication, data model validation, serialization.
No OpenAI API key required for tests.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two independent agents per stage | Catches both false positives and false negatives |
| Conservative uncertainty: uncertain → human | Never hide doubt; traceability over efficiency |
| Field-level disagreement tracking in extraction | Granular; avoids silent errors on critical fields |
| Separate Full Text Needed bucket | Makes missing PDF workflow explicit |
| All prompts in `agents/prompts.py` | Versioned, auditable, reproducible |
| Pydantic models throughout | Type safety; easy serialization/validation |
| State persisted as JSONL | Resumable; no database required |

---

## Protocol Reference

See [`docs/refined_protocol.md`](docs/refined_protocol.md) for the complete PRISMA-style protocol including:
- Refined research questions with rationale
- Operational inclusion/exclusion rules
- Borderline case decision tree
- Full data extraction form
- Quality assessment checklist (QA1–QA10)
