# Refined PRISMA-Style Systematic Review Protocol

**Title:** A Systematic Review of Large Language Models in Qualitative Data Analysis (2023–2026)
**Version:** 2.0 (Refined)
**Date:** 2026-03-18
**Status:** Active

---

## 1. Background and Rationale

Large language models (LLMs) have rapidly emerged as powerful tools for processing and interpreting text data. Their potential applications in qualitative research — such as coding, thematic analysis, and theory development — are expanding. However, there is limited synthesis of how LLMs are being used in qualitative data analysis (QDA), including their methodological roles, workflows, and evaluation approaches.

This systematic review synthesizes current empirical evidence on the use of LLMs in QDA and identifies methodological patterns, strengths, and limitations.

**Refinement note:** The original rationale is sound. Added emphasis on "empirical evidence" as the key filter, distinguishing this review from speculative or opinion pieces.

---

## 2. Research Questions (Refined)

| # | Original | Refined | Rationale |
|---|----------|---------|-----------|
| RQ1 | How are LLMs used in QDA? | **How are LLMs operationally integrated into qualitative data analysis workflows, and at what stages of analysis?** | "How are used" is vague. Specifying "operationally integrated" and "stages" guides extraction. |
| RQ2 | What qualitative methodologies and workflows are employed? | **What qualitative methodological frameworks (e.g., thematic analysis, grounded theory) are combined with LLM-assisted analysis, and how are workflows structured?** | Clarifies that we want named frameworks, not just generic mentions. |
| RQ3 | What evaluation and validation approaches are used? | **How is the quality, validity, and reliability of LLM-assisted qualitative analysis evaluated in these studies?** | "Evaluation approaches" is too broad; revised to focus on quality/validity/reliability. |
| RQ4 | What are the reported strengths, limitations, and challenges? | **What are the reported methodological strengths, limitations, ethical concerns, and practical challenges of using LLMs in QDA?** | Added "ethical concerns" as a distinct dimension given growing literature on AI ethics in research. |

---

## 3. Eligibility Criteria (Refined with Operational Rules)

### 3.1 Inclusion Criteria

All five criteria must be met:

#### IC1 — LLM Tool Usage
**Original:** "The study uses a large language model (e.g., GPT, ChatGPT, Claude, Llama, Gemini)."
**Refined:** The study explicitly names a specific LLM (transformer-based, generative language model with ≥1B parameters) and describes its technical role in the analysis.
**Operational rule:** Terms like "AI tool," "NLP tool," or "machine learning" alone are insufficient. The model must be identifiable. If only "AI" is mentioned with no model name, flag as **Needs Human Verification**.

#### IC2 — Qualitative Analysis Function
**Original:** "The LLM is used to perform or support qualitative data analysis."
**Refined:** The LLM must perform or meaningfully support at least one of the following analytic tasks on primary textual data:
- Inductive or deductive coding
- Codebook development or application
- Thematic analysis (identification, development, review, or naming of themes)
- Content analysis (manifest or latent)
- Grounded theory coding (open, axial, or selective)
- Framework analysis
- Narrative or discourse analysis
- Interpretive phenomenological analysis (IPA)

**Operational rule — Borderline cases:**
- Sentiment analysis → **Exclude** unless framed within a named qualitative methodology
- Topic modeling → **Exclude** unless combined with qualitative interpretation of themes
- Classification into pre-defined categories → **Exclude** (not qualitative analysis)
- Summarization used as a step within a qualitative workflow → **Include** if the LLM then generates codes/themes from summaries; **Exclude** if summarization is the final analytic output
- If unclear, flag as **Needs Human Verification**

#### IC3 — Empirical Application
**Original:** "The study applies the method to real data and reports results."
**Refined:** The study must: (a) apply the LLM to a real dataset (not synthetic or hypothetical), (b) report analytic outputs (codes, themes, categories, findings), and (c) describe the analysis process sufficiently to be reproducible.
**Operational rule:** Proof-of-concept studies with real data qualify. Studies where real data is mentioned but no results are reported → **Needs Human Verification**.

#### IC4 — Publication Date
Published between January 1, 2023 and December 31, 2026 (inclusive).
**Operational rule:** Use the official publication date (not preprint submission date). If only a preprint exists, use preprint date.

#### IC5 — Language
Published in English.
**Operational rule:** If abstract is in English but full text is not, flag as **Needs Human Verification** (may be translatable).

---

### 3.2 Exclusion Criteria

Applied after inclusion criteria:

| Code | Criterion | Operational Rule |
|------|-----------|-----------------|
| EC1 | LLM used only for non-analytic tasks | Transcription cleanup, translation, literature review assistance, writing support, grammar correction, formatting. If LLM *only* does these → Exclude. If these are preliminary steps before QDA → check IC2. |
| EC2 | No empirical application | Framework proposals, theoretical papers, system designs without results, position papers. |
| EC3 | No qualitative results reported | Studies that describe using LLMs for QDA but report only quantitative metrics (e.g., F1, accuracy) without any qualitative interpretation or coding output → Exclude. |
| EC4 | Restricted publication types | Review articles, editorials, commentaries, letters, conference abstracts without full paper, registered protocols without results. |
| EC5 | Superficial extraction only | Studies where LLM performs only keyword extraction, named entity recognition, or metadata tagging without qualitative interpretation → Exclude. |
| EC6 | Non-primary qualitative data | Studies applying LLMs only to structured data (tables, numbers) with no textual/narrative data component → Exclude. |

**Hierarchy rule:** Apply EC codes in order. The first matching exclusion criterion terminates evaluation.

---

## 4. Information Sources

| Database | Coverage | Notes |
|----------|----------|-------|
| PubMed | Biomedical, health sciences | Add MeSH terms: "Natural Language Processing", "Artificial Intelligence" |
| Scopus | Multidisciplinary | Broad coverage; filter by date and language |
| PsycINFO | Psychology, behavioral sciences | Key for qualitative methodology literature |
| Web of Science | Multidisciplinary | Use Core Collection |
| ACM Digital Library | Computer science, HCI | Strong for human-centered AI papers |
| IEEE Xplore | Engineering, CS | Strong for technical LLM papers |
| ERIC | Education | **Added** — important for educational qualitative research |
| arXiv/SSRN | Preprints | **Added** — rapidly evolving field; manual screening required |

**Refinement note:** ERIC added for education domain coverage. arXiv/SSRN added as supplementary sources given the fast-moving nature of LLM research.

---

## 5. Search Strategy (Refined)

### 5.1 Core Search String

```
(
  "LLM" OR "LLMs" OR "large language model" OR "large language models"
  OR "generative AI" OR "generative artificial intelligence"
  OR "GPT-4" OR "GPT-3" OR "GPT-3.5" OR "ChatGPT" OR "GPT"
  OR "Claude" OR "Claude 3" OR "Llama" OR "Llama 2" OR "Llama 3"
  OR "Gemini" OR "Bard" OR "Mistral" OR "Falcon" OR "PaLM"
  OR "foundation model" OR "transformer model"
)
AND
(
  "qualitative analysis" OR "qualitative research" OR "qualitative data analysis"
  OR "qualitative coding" OR "thematic analysis" OR "content analysis"
  OR "grounded theory" OR "framework analysis" OR "narrative analysis"
  OR "discourse analysis" OR "interpretive phenomenological"
  OR "inductive coding" OR "deductive coding" OR "open coding" OR "axial coding"
  OR "codebook" OR "code development" OR "theme development" OR "theme extraction"
  OR "qualitative synthesis" OR "constant comparative"
)
```

**Refinement notes:**
- Added specific model versions (GPT-4, Llama 2/3, Claude 3) to capture version-specific studies
- Added "foundation model" and "transformer model" as broader fallbacks
- Added "inductive coding," "deductive coding," "open coding," "axial coding" for grounded theory specificity
- Added "constant comparative" (grounded theory method)
- Removed generic "themes" and "coding" as standalone terms (too broad; causes false positives)

### 5.2 Database-Specific Adaptations

Each database search string will be adapted for field tags (title, abstract, keywords) and Boolean operator syntax. All adaptations will be documented in Appendix A.

### 5.3 Grey Literature
- Google Scholar: first 200 results (sorted by relevance)
- Reference list screening of all included studies
- Citation tracking (forward) for key studies

---

## 6. Study Selection Process

### Stage 1: Title/Abstract Screening

**Decision rules:**
- **Include** → Clearly meets IC1–IC5 and no EC applies
- **Exclude** → Clearly fails at least one IC or clearly meets EC
- **Uncertain** → Any ambiguity → flag as **Needs Human Verification**

**Liberal inclusion principle at Stage 1:** When in doubt, retain for full-text screening.

### Stage 2: Full-Text Screening

Apply full eligibility criteria. Same three-category classification.

### Disagreement Resolution
- Agent 1 and Agent 2 screen independently
- If both agree → decision is final
- If they disagree OR either flags uncertainty → **Needs Human Verification**
- Human adjudicator reviews and records rationale

### PRISMA Flow Tracking
Records tracked at each stage: identified → deduplicated → title/abstract screened → full-text screened → included.

---

## 7. Data Extraction Form (Refined)

### 7.1 Study Characteristics
| Field | Type | Notes |
|-------|------|-------|
| study_id | Auto-assigned | SR-YYYY-NNN |
| title | Text | Verbatim |
| authors | Text | Last name, First initial |
| year | Integer | Publication year |
| journal_venue | Text | Full name |
| doi | Text | If available |
| country | Text | First author's country |
| discipline | Controlled vocab | Healthcare / Education / HCI / Social Science / CS / Other |
| study_aim | Text | 1-2 sentence summary |

### 7.2 Data and Context
| Field | Type | Notes |
|-------|------|-------|
| data_type | Controlled vocab | Interviews / Focus groups / Documents / Social media / Survey responses / Mixed / Other |
| sample_size | Integer or range | Number of participants or documents |
| corpus_size | Text | E.g., "50 interview transcripts, ~200,000 words" |
| domain | Controlled vocab | Healthcare / Education / HCI / Psychology / Sociology / Other |
| data_language | Text | Language of source data |

### 7.3 LLM Characteristics
| Field | Type | Notes |
|-------|------|-------|
| model_name | Text | Exact name and version |
| model_type | Controlled vocab | Proprietary / Open-source / Not specified |
| model_provider | Text | OpenAI / Anthropic / Meta / Google / Other |
| prompting_strategy | Controlled vocab | Zero-shot / Few-shot / Chain-of-thought / System prompt / Other |
| prompt_provided | Boolean | Is prompt text reported in the paper? |
| fine_tuned | Boolean | Was the model fine-tuned? |
| rag_used | Boolean | Was RAG used? |
| temperature | Float or null | If reported |

### 7.4 Qualitative Analysis Use
| Field | Type | Notes |
|-------|------|-------|
| analytic_task | Multi-select | Coding / Theme development / Content analysis / Grounded theory / Framework analysis / Narrative analysis / Discourse analysis / Other |
| analysis_stage | Controlled vocab | Full analysis / Preliminary coding / Code verification / Theme review / Mixed |
| workflow_structure | Controlled vocab | Human-led (AI assists) / Human-in-the-loop / AI-led (human verifies) / Fully automated |
| pipeline_type | Controlled vocab | Single-step / Multi-step / Multi-agent |
| human_oversight | Text | Description of human role |

### 7.5 Methodological Framework
| Field | Type | Notes |
|-------|------|-------|
| qualitative_approach | Text | Named approach or "not specified" |
| formal_methodology | Boolean | Is a named framework explicitly followed? |
| codebook_development | Text | How was codebook created? |
| epistemological_stance | Text | If stated |

### 7.6 Evaluation and Validation
| Field | Type | Notes |
|-------|------|-------|
| human_comparison | Boolean | Compared to human coder? |
| agreement_method | Text | Cohen's kappa / % agreement / Other |
| agreement_score | Text | Value if reported |
| quantitative_metrics | Text | F1, accuracy, etc. if reported |
| qualitative_validation | Boolean | Member checking, peer debriefing, etc. |
| audit_trail | Boolean | Is analysis process documented? |
| reflexivity | Boolean | Is researcher positionality discussed? |

### 7.7 Outcomes and Assessment
| Field | Type | Notes |
|-------|------|-------|
| key_findings | Text | Main qualitative findings |
| strengths_reported | Text | Verbatim or paraphrased |
| limitations_reported | Text | Verbatim or paraphrased |
| ethical_considerations | Text | If discussed |
| reproducibility_score | 1–4 | 1=None, 2=Partial, 3=Most, 4=Full |

### 7.8 Quality Assessment (Custom Checklist)

Scored 0–1 per item, total /10:

| Item | Description |
|------|-------------|
| QA1 | LLM clearly identified (name + version) |
| QA2 | Prompts or prompt strategy described |
| QA3 | Analysis process described step-by-step |
| QA4 | Human role clearly defined |
| QA5 | Validation or quality check performed |
| QA6 | Results reported with sufficient detail |
| QA7 | Limitations acknowledged |
| QA8 | Data description adequate |
| QA9 | Reproducibility materials available (e.g., code, prompts) |
| QA10 | Ethical considerations addressed |

---

## 8. Data Synthesis Plan

### 8.1 Quantitative Descriptive
- Frequency tables: LLM type, domain, analytic task, workflow structure
- Year-over-year trends
- Quality score distribution

### 8.2 Qualitative Synthesis
- Narrative synthesis grouped by: (a) analytic task, (b) workflow structure, (c) evaluation approach
- Cross-cutting themes identified iteratively
- Evidence tables per RQ

### 8.3 Reporting
- PRISMA 2020 flow diagram
- Study characteristics table
- Evidence synthesis tables per RQ
- Quality assessment summary

---

## 9. Uncertainty and Ambiguity Handling

| Scenario | Action |
|----------|--------|
| Model name unclear (e.g., "AI assistant") | Flag IC1 → Needs Human Verification |
| QDA use ambiguous (e.g., "assisted analysis") | Flag IC2 → Needs Human Verification |
| Study has results but they seem quantitative only | Flag IC3/EC3 → Needs Human Verification |
| Full text unavailable | Flag → Full Text Needed |
| Extraction field missing from paper | Record as "Not reported" — do not infer |
| Agent disagreement on screening | → Needs Human Verification |
| Confidence < threshold on extraction | → Needs Human Verification |

**Core rule: Never infer, fabricate, or guess. All uncertain items surface to the human reviewer with full rationale.**

---

*End of refined protocol. Version 2.0.*
