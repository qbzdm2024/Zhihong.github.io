# System Design: Systematic Review Automation

## Agent Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                                  │
└──────────────────────────────────────────────────────────────────────────┘

 [Search Databases] → Export files (RIS/CSV/BibTeX/XML/JSON)
         │
         ▼
 ┌─────────────────┐
 │  1. IMPORT      │  Loads all records → RawRecord objects
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  2. DEDUP       │  DOI → exact title → fuzzy title
 └────────┬────────┘
          │
          ├──── Duplicate → [EXCLUDED: Duplicate]
          │
          ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  3. TITLE/ABSTRACT SCREENING                                  │
 │                                                               │
 │   Agent 1 (model A) ──┐                                       │
 │                        ├──► Compare ──► Agree, high conf?     │
 │   Agent 2 (model B) ──┘                    │                  │
 │                                    Yes     │     No           │
 │                                     ▼      │      ▼           │
 │                              [Finalize]    │  [Human Queue]   │
 └──────────────────────────────────────────────────────────────┘
          │                               │
    Included                         Uncertain
          │                               │
          ▼                               ▼
 ┌──────────────────┐          ┌──────────────────────┐
 │ PDF available?   │          │ HUMAN VERIFICATION    │
 │                  │          │ UI: Review + Decide   │
 └────────┬─────────┘          └──────────────────────┘
          │
     No   │   Yes
          │
    ▼     ▼
[Full Text   Full Text
 Needed]     Available
               │
               ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  4. FULL-TEXT SCREENING                                        │
 │                                                               │
 │   Agent 1 (model A) ──┐                                       │
 │                        ├──► Compare ──► Agree, high conf?     │
 │   Agent 2 (model B) ──┘                    │                  │
 │                                    Yes     │     No           │
 │                                     ▼      │      ▼           │
 │                              [Finalize]    │  [Human Queue]   │
 └──────────────────────────────────────────────────────────────┘
          │
    Included
          │
          ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  5. DATA EXTRACTION                                           │
 │                                                               │
 │   Agent 1 (model A) ──┐                                       │
 │                        ├──► Compare fields ──► Hard field     │
 │   Agent 2 (model B) ──┘    disagree?          disagree?       │
 │                                    │               │          │
 │                                   No              Yes         │
 │                                    ▼               ▼          │
 │                             [Merged auto]   [Human Queue]     │
 │                                                               │
 │   Parallel: QA Assessment (2 agents, averaged)               │
 └──────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌──────────────────┐
 │  6. EVIDENCE     │  Evidence table, PRISMA counts, exports
 │  TABLE OUTPUT    │
 └──────────────────┘
```

---

## Human Verification Checkpoints

| Checkpoint | Trigger | Action Required |
|-----------|---------|----------------|
| T/A agents disagree | Different decisions | Include/Exclude/Keep uncertain |
| T/A agent confidence < threshold | Single agent < 0.80 | Review abstract + agent rationale |
| T/A agent flags "Needs Human Verification" | Any criterion ambiguous | Review and decide |
| Full-text unavailable | No PDF found | Upload PDF |
| Full-text agents disagree | Different decisions | Review full text + decide |
| Extraction hard-field disagree | model_name, workflow_structure, etc. | Edit extraction fields |
| Extraction agent error | API failure | Manual extraction entry |
| QA agents score differs by > 2 | Quality assessment | Review QA items |

---

## Data Flow and State Management

```
State file: data/pipeline_state.jsonl
  ├── One JSON line per PipelineRecord
  ├── Updated after every stage
  ├── Append-safe (rebuilt on each save)
  └── Loadable by runner.load_state()

Output buckets (data/screened/):
  ├── included.jsonl
  ├── excluded.jsonl
  ├── needs_human_verification.jsonl
  └── full_text_needed.jsonl

Output (data/output/):
  ├── full_text_needed.json    ← for PDF procurement
  ├── evidence_table.csv       ← final synthesis table
  └── full_export.json         ← complete audit trail
```

---

## Multi-Agent Comparison Logic

```
Agent 1 decision + Agent 2 decision
         │
         ├── Either is "Needs Human Verification"?
         │       └── YES → Send to human immediately
         │
         ├── Same decision AND conf_diff < 0.15 AND min_conf ≥ threshold?
         │       └── YES → Auto-approve (full agreement)
         │
         ├── Same decision BUT conf_diff ≥ 0.15?
         │       └── Comparison model evaluates → recommendation
         │
         └── Different decisions?
                 └── Comparison model evaluates → almost always send to human
```

---

## Extraction Field Comparison

```
For each extracted field:
  ├── Both null → keep null
  ├── One null, one not → take non-null; flag if HARD_FIELD
  ├── Both list → if sets differ and HARD_FIELD → flag + take union
  ├── Both bool → if differ → null + flag
  ├── Both numeric → if diff > 0.5 and HARD_FIELD → null + flag
  └── Both string → if differ and HARD_FIELD → keep agent1 + flag

HARD_FIELDS (always flag on disagreement):
  model_name, workflow_structure, analytic_task, human_comparison,
  fine_tuned, rag_used, formal_methodology, qualitative_approach

SOFT_FIELDS (minor disagreement tolerated):
  study_aim, key_findings, strengths_reported, limitations_reported,
  human_oversight, codebook_development, extraction_notes
```

---

## API Design

```
POST   /api/pipeline/run               Run a pipeline stage
GET    /api/pipeline/status            PRISMA counts + status

GET    /api/records                    List records (filter by decision/stage)
GET    /api/records/{id}               Full record details
GET    /api/records/uncertain/list     Human verification queue
GET    /api/records/fulltext-needed/list  Papers needing PDFs

POST   /api/records/{id}/verify        Submit human decision
PATCH  /api/records/{id}/extraction    Edit extraction fields

POST   /api/pdfs/upload               Upload single PDF
POST   /api/pdfs/upload-batch         Batch PDF upload

GET    /api/config/models              Get model config
PATCH  /api/config/models             Update model config

GET    /api/export/evidence-table     Evidence table JSON
GET    /api/export/evidence-table/csv Evidence table CSV
GET    /api/export/prisma              PRISMA flow counts
GET    /api/export/all-records         Full audit export
```

---

## UI Panel Map

```
Dashboard         → PRISMA flow diagram, recent records, stats
Human Verification → Queue of uncertain records, review modal
Full Text Needed  → Papers needing PDFs, upload interface
Included          → Final evidence table
Excluded          → Excluded records with reasons
Run Pipeline      → Stage controls, real-time log
Model Config      → Per-task model assignment, API key
Export            → Download evidence table, PRISMA counts
```

---

## Cost Estimation

Approximate OpenAI API costs per 1000 records:

| Stage | Model | Est. tokens/record | Est. cost/1000 |
|-------|-------|--------------------|----------------|
| T/A Screening (Agent 1) | gpt-4o-mini | ~800 | ~$0.12 |
| T/A Screening (Agent 2) | gpt-4o | ~800 | ~$2.00 |
| Full-text Screening (×2) | gpt-4o | ~5000 | ~$25.00 |
| Extraction (×2) | gpt-4o | ~8000 | ~$40.00 |
| QA Assessment (×2) | gpt-4o | ~4000 | ~$20.00 |

*Assumes ~30% pass-through rate from T/A to full-text.*
*Actual costs depend on abstract/full-text lengths.*
