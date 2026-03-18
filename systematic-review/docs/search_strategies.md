# Executed Search Strategies
**Review:** LLMs in Qualitative Data Analysis (2023–2026)
**Date searched:** 2026-03-18
**Searcher:** ZH

---

## PubMed

**Interface:** PubMed (pubmed.ncbi.nlm.nih.gov)
**Export format:** NBIB (Citation manager) → `pubmed.nbib`

```
(
  (
    "large language model"[Title/Abstract]
    OR "large language models"[Title/Abstract]
    OR ChatGPT[Title/Abstract]
    OR GPT-4[Title/Abstract]
    OR "GPT 4"[Title/Abstract]
    OR Claude[Title/Abstract]
    OR Llama[Title/Abstract]
    OR Gemini[Title/Abstract]
  )
  AND
  (
    "qualitative analysis"[Title/Abstract]
    OR "qualitative data analysis"[Title/Abstract]
    OR "thematic analysis"[Title/Abstract]
    OR "content analysis"[Title/Abstract]
    OR "grounded theory"[Title/Abstract]
    OR coding[Title/Abstract]
    OR codebook[Title/Abstract]
    OR "theme development"[Title/Abstract]
    OR "automated coding"[Title/Abstract]
  )
)
AND ("2023/01/01"[Date - Publication] : "2026/12/31"[Date - Publication])
AND (english[Language])
```

**Notes:** Title/Abstract field restriction applied. Date filter: 2023-01-01 to 2026-12-31. Language: English.

---

## Scopus

**Interface:** Scopus (scopus.com)
**Export format:** RIS → `scopus.ris`

```
TITLE-ABS-KEY (
  ( "large language model*" OR LLM* OR ChatGPT OR GPT-4 OR Claude OR Llama OR Gemini )
  AND
  (
    ( "qualitative" W/3 ( analysis OR coding OR "data analysis" ) )
    OR "thematic analysis"
    OR "content analysis"
    OR "grounded theory"
    OR "framework analysis"
    OR "narrative analysis"
    OR "discourse analysis"
    OR "codebook"
  )
)
AND PUBYEAR > 2022
AND PUBYEAR < 2027
AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) )
AND ( LIMIT-TO ( LANGUAGE , "English" ) )
```

**Notes:** TITLE-ABS-KEY field restriction. Document types: article (ar) and conference paper (cp). Proximity operator W/3 used for qualitative terms. Year: 2023–2026.

---

## Web of Science

**Interface:** Web of Science Core Collection (webofscience.com)
**Export format:** RIS (Other File Formats, Full Record) → `wos.ris`

```
TS=(
  (
    ("large language model*" OR ChatGPT OR GPT-4 OR "GPT 4" OR Claude OR Llama OR Gemini)
    NEAR/10
    (coding OR "qualitative analysis" OR "thematic analysis" OR "content analysis" OR "grounded theory")
  )
  OR
  (
    ("qualitative" NEAR/2 (analysis OR coding))
    AND
    (ChatGPT OR GPT-4 OR Claude OR Llama OR Gemini)
  )
)
AND PY=(2023-2026)
AND LA=(English)
AND DT=(Article OR Proceedings Paper)
```

**Notes:** Topic (TS) field covers title, abstract, author keywords, and KeyWords Plus. NEAR/10 and NEAR/2 proximity operators used. Document types: Article and Proceedings Paper. Year: 2023–2026.

---

## Notes on Search Strategy Decisions

| Decision | Rationale |
|----------|-----------|
| Title/Abstract restriction (PubMed, Scopus) | Reduces noise from papers that merely cite LLMs; ensures LLM use in QDA is a central focus |
| Topic field in WoS | Covers KeyWords Plus in addition to title/abstract — slightly broader to compensate |
| Proximity operators (W/3, NEAR/10) | Captures "qualitative analysis using LLMs" type phrases while avoiding false positives |
| "Claude" as search term | May retrieve unrelated results; screeners instructed to check context |
| coding[Title/Abstract] (PubMed) | Broad term; expect false positives — resolved at screening stage |
| Document type restriction | Excludes reviews, editorials, and letters per protocol EC4 |
| LLM* truncation (Scopus) | Captures LLM, LLMs, LLM-based, LLM-assisted |

---

## Files Generated

| File | Source | Format | Location |
|------|--------|--------|----------|
| `pubmed.nbib` | PubMed | NBIB | `data/raw/pubmed.nbib` |
| `scopus.ris` | Scopus | RIS | `data/raw/scopus.ris` |
| `wos.ris` | Web of Science | RIS | `data/raw/wos.ris` |
