"""
Omaha System Mapping Pipeline — AWS SageMaker Version
======================================================
Maps clinical conversation turns to Omaha System classifications:
  • Signs/Symptoms  (OS_SS_1/2/3)
  • Interventions   (OS_I_1/2/3)

Architecture:
  - RAG: Sentence-Transformers embeddings + cosine-similarity retrieval
  - LLM: local HuggingFace model (data never leaves AWS) OR
         Azure OpenAI with HIPAA BAA (optional)
  - Evaluation: row-level fuzzy matching, micro P/R/F1

PRIVACY NOTE:
  Patient data stays entirely within AWS.
  - Local models: loaded onto the SageMaker GPU instance; no external calls.
  - Azure OpenAI: only use if your organisation has a signed HIPAA BAA with
    Microsoft. Standard OpenAI API (api.openai.com) is NOT covered by HIPAA
    without an enterprise agreement.
  - HuggingFace Inference API: NEVER use with real patient data.

Input  (S3): Omaha_system list with definition.xlsx
             Completed_annotation.xlsx  (23 conversation sheets)
Output (S3): output/<model>_<timestamp>_results.xlsx
             output/<model>_<timestamp>_summary.json

Recommended SageMaker instances for local models:
  Mistral-7B / Llama3-8B (4-bit):  ml.g5.xlarge   (1× A10G 24 GB, ~$1.41/hr)
  Llama3-70B (4-bit, ~35 GB VRAM): ml.g5.12xlarge  (4× A10G 96 GB, ~$5.67/hr)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, re, json, time, logging, warnings
from datetime import datetime
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from tqdm import tqdm
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local model imports (only used when provider == "local")
# These are pre-installed on SageMaker GPU images; no extra pip install needed.
import torch

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Lazy global for local model (loaded once, reused across all calls) ─────────
_local_pipeline = None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# Edit these settings before running.
# ══════════════════════════════════════════════════════════════════════════════

# ── S3 paths ──────────────────────────────────────────────────────────────────
S3_BUCKET        = "rspeech"
OMAHA_KEY        = "Fake example_Sandy/omaha-mapping/Omaha_system list with definition.xlsx"
ANNOTATION_KEY   = "Fake example_Sandy/omaha-mapping/Completed_annotation.xlsx"
OUTPUT_PREFIX    = "Fake example_Sandy/omaha-mapping/output/"

# ── Embedding model (runs on CPU; swap for a stronger model if GPU available) ─
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l"       # higher quality
# EMBEDDING_MODEL = "thenlper/gte-large"                       # strong clinical

# ── LLM configuration ─────────────────────────────────────────────────────────
# Set ACTIVE_LLM to one of the keys below.
#
# PRIVACY GUIDE:
#   provider = "local"       → model runs on SageMaker GPU; no data leaves AWS ✓
#   provider = "azure_openai"→ only if your org has a HIPAA BAA with Microsoft ✓
#   provider = "openai"      → only if your org has an OpenAI Enterprise BAA   ✓
#   provider = "huggingface" → DO NOT use with real patient data               ✗
#
LLM_CONFIGS = {
    # ── Local models (data stays on SageMaker) ────────────────────────────────
    # Requires GPU instance. Model is downloaded from HuggingFace on first run
    # and cached in /tmp or EFS. No patient data is ever sent externally.
    "llama3-8b-local": {
        "provider":    "local",
        "model":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # Recommended instance: ml.g5.xlarge (1× A10G 24 GB)
        # 4-bit quantisation fits comfortably; fast inference (~2 s/call)
        "load_in_4bit": True,
        "kwargs":      {"max_new_tokens": 500, "temperature": 0.05, "do_sample": True},
    },
    "mistral-7b-local": {
        "provider":    "local",
        "model":       "mistralai/Mistral-7B-Instruct-v0.3",
        # Recommended instance: ml.g5.xlarge  (same as above)
        "load_in_4bit": True,
        "kwargs":      {"max_new_tokens": 500, "temperature": 0.05, "do_sample": True},
    },
    "llama3-70b-local": {
        "provider":    "local",
        "model":       "meta-llama/Meta-Llama-3.3-70B-Instruct",
        # Recommended instance: ml.g5.12xlarge (4× A10G 96 GB)
        # 4-bit quantisation needs ~35 GB VRAM; slower (~15 s/call)
        "load_in_4bit": True,
        "kwargs":      {"max_new_tokens": 500, "temperature": 0.05, "do_sample": True},
    },
    # ── Azure OpenAI (HIPAA BAA required) ────────────────────────────────────
    # Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars.
    # Your deployment name goes in "model" field.
    "azure-gpt4o-mini": {
        "provider":    "azure_openai",
        "model":       "gpt-4o-mini",          # Azure deployment name
        "kwargs":      {"temperature": 0.05, "max_tokens": 500},
    },
    # ── Standard OpenAI (only if you have an Enterprise HIPAA BAA) ────────────
    "gpt-4o-mini": {
        "provider": "openai",
        "model":    "gpt-4o-mini",
        "kwargs":   {"temperature": 0.05, "max_tokens": 500},
    },
}

# ← Set this to the model you want to run
ACTIVE_LLM = "llama3-8b-local"

# To run multiple models sequentially and compare results, set:
# COMPARE_ALL_MODELS = True
# MODELS_TO_COMPARE  = ["llama3-8b-local", "mistral-7b-local"]
COMPARE_ALL_MODELS  = False
MODELS_TO_COMPARE   = ["llama3-8b-local", "mistral-7b-local"]

# ── Pipeline settings ─────────────────────────────────────────────────────────
CONTEXT_WINDOW  = 2    # rows before/after current turn to include as context
TOP_K_RETRIEVAL = 15   # number of Omaha options sent to the LLM
FUZZY_THRESHOLD = 80   # minimum fuzz.ratio for a match (0–100)

# ── API / model credentials ────────────────────────────────────────────────────
# For local models: set HF_TOKEN so the model can be downloaded from HuggingFace.
# The token is only used for the initial download; it is not sent with patient data.
HF_TOKEN              = os.environ.get("HF_TOKEN", "")

# For Azure OpenAI (if using "azure_openai" provider):
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")   # e.g. https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY  = os.environ.get("AZURE_OPENAI_API_KEY",  "")
AZURE_API_VERSION     = "2024-08-01-preview"

# For standard OpenAI (enterprise BAA only):
OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — S3 HELPERS
# ══════════════════════════════════════════════════════════════════════════════

s3 = boto3.client("s3")

def s3_read_excel(bucket: str, key: str) -> dict[str, pd.DataFrame]:
    """Read all sheets from an S3-hosted Excel file."""
    log.info(f"Reading s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    buf = BytesIO(obj["Body"].read())
    return pd.read_excel(buf, sheet_name=None, engine="openpyxl")

def s3_write_excel(df_dict: dict[str, pd.DataFrame], bucket: str, key: str):
    """Write a dict of DataFrames to S3 as a multi-sheet Excel file."""
    log.info(f"Writing s3://{bucket}/{key}")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())

def s3_write_json(data: dict, bucket: str, key: str):
    log.info(f"Writing s3://{bucket}/{key}")
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2).encode())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD & INDEX OMAHA SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def load_omaha_system(bucket: str, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        ss_df   — sign/symptom sheet with columns:
                    Domain, Domain definition, Problem, Problem definition,
                    SIGNS/SYMPTOMS OF ACTUAL, Definition/example of signs/symptoms
        int_df  — intervention sheet with columns:
                    Domain (= category), Domain definition, Intervention Target
    """
    sheets = s3_read_excel(bucket, key)
    sheet_names = list(sheets.keys())
    log.info(f"Omaha sheets found: {sheet_names}")

    ss_df  = sheets[sheet_names[0]].copy()
    int_df = sheets[sheet_names[1]].copy()

    # Standardise column names (strip whitespace)
    ss_df.columns  = [c.strip() for c in ss_df.columns]
    int_df.columns = [c.strip() for c in int_df.columns]

    # Drop fully-empty rows
    ss_df  = ss_df.dropna(how="all").reset_index(drop=True)
    int_df = int_df.dropna(how="all").reset_index(drop=True)

    log.info(f"Sign/symptom rows: {len(ss_df)} | Intervention rows: {len(int_df)}")
    return ss_df, int_df


def build_ss_documents(ss_df: pd.DataFrame) -> list[dict]:
    """
    Each document represents one (Problem, Sign/Symptom) pair.
    The text field is used for embedding.
    """
    docs = []
    prob_col = next(c for c in ss_df.columns if "Problem" in c and "definition" not in c.lower() and "Classification" not in c)
    ss_col   = next(c for c in ss_df.columns if "SIGNS" in c.upper() or "SYMPTOM" in c.upper())
    def_col  = next((c for c in ss_df.columns if "Definition" in c or "definition" in c), None)

    for _, row in ss_df.iterrows():
        domain  = str(row.get("Domain", "")).strip()
        problem = str(row.get(prob_col, "")).strip()
        ss      = str(row.get(ss_col, "")).strip()
        defn    = str(row.get(def_col, "")).strip() if def_col else ""

        if problem in ("nan", "") or ss in ("nan", ""):
            continue

        text = f"Domain: {domain}\nProblem: {problem}\nSigns/Symptoms: {ss}"
        if defn and defn != "nan":
            text += f"\nDefinition: {defn[:200]}"   # truncate long definitions

        docs.append({
            "text":    text,
            "domain":  domain,
            "problem": problem,
            "ss":      ss,
            # Canonical key used for evaluation: Problem_SignSymptom
            "label":   f"{problem}_{ss}",
        })
    log.info(f"SS documents built: {len(docs)}")
    return docs


def build_intervention_documents(int_df: pd.DataFrame) -> list[dict]:
    """
    Each document represents one (Category, Target) pair.
    """
    docs = []
    # Column names: Domain (= category), Domain definition, Intervention Target
    cat_col    = "Domain"
    def_col    = next((c for c in int_df.columns if "definition" in c.lower()), None)
    target_col = next(c for c in int_df.columns if "Target" in c or "target" in c)

    for _, row in int_df.iterrows():
        category = str(row.get(cat_col, "")).strip()
        target   = str(row.get(target_col, "")).strip()
        defn     = str(row.get(def_col, "")).strip() if def_col else ""

        if category in ("nan", "") or target in ("nan", ""):
            continue

        text = f"Intervention Category: {category}\nTarget: {target}"
        if defn and defn != "nan":
            text += f"\nCategory Definition: {defn[:150]}"

        docs.append({
            "text":     text,
            "category": category,
            "target":   target,
            # Canonical key: Category_Target
            "label":    f"{category}_{target}",
        })
    log.info(f"Intervention documents built: {len(docs)}")
    return docs


def build_index(docs: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Encode all documents and return embedding matrix (n_docs × dim)."""
    texts = [d["text"] for d in docs]
    log.info(f"Encoding {len(texts)} documents …")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True)
    return embeddings.astype("float32")


def retrieve(query: str, docs: list[dict], embeddings: np.ndarray,
             model: SentenceTransformer, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Return top-k most relevant documents for the query."""
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    sims  = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [docs[i] for i in top_idx]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LLM CLIENTS
# ══════════════════════════════════════════════════════════════════════════════

def _load_local_model(config: dict):
    """
    Load a HuggingFace model onto the local GPU (once; cached in _local_pipeline).
    Uses 4-bit quantisation via bitsandbytes to fit large models on smaller GPUs.

    First call takes 2–5 minutes (model download + load).
    Subsequent calls are instant (model stays in GPU memory).
    """
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

    model_id    = config["model"]
    load_in_4bit = config.get("load_in_4bit", True)

    log.info(f"Loading local model: {model_id}  (4-bit={load_in_4bit})")
    log.info("This takes 2–5 min on first run. Model is cached after that.")

    # Login to HuggingFace for gated models (Llama3 requires accepting licence)
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",          # spreads across all available GPUs
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        token=HF_TOKEN,
    )

    _local_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    log.info(f"Model loaded. Device map: {model.hf_device_map}")
    return _local_pipeline


def _call_local(prompt: str, config: dict) -> str:
    """Run inference locally on the SageMaker GPU. No data leaves AWS."""
    pipe    = _load_local_model(config)
    kwargs  = config.get("kwargs", {"max_new_tokens": 500, "temperature": 0.05})
    outputs = pipe(prompt, return_full_text=False, **kwargs)
    return outputs[0]["generated_text"].strip()


def _call_azure_openai(prompt: str, config: dict) -> str:
    """Azure OpenAI — use only if HIPAA BAA is in place with Microsoft."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    resp = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        **config.get("kwargs", {}),
    )
    return resp.choices[0].message.content.strip()


def _call_openai(prompt: str, config: dict) -> str:
    """Standard OpenAI — only if enterprise HIPAA BAA is in place."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp   = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        **config.get("kwargs", {}),
    )
    return resp.choices[0].message.content.strip()


def call_llm(prompt: str, llm_name: str = ACTIVE_LLM,
             max_retries: int = 3) -> str:
    """
    Unified LLM call with exponential-backoff retry.

    Local models don't need retries (no network); retries only help for API calls.
    """
    config   = LLM_CONFIGS[llm_name]
    provider = config["provider"]

    for attempt in range(max_retries):
        try:
            if provider == "local":
                # Local: no retry needed — if it fails it's a code/memory error
                return _call_local(prompt, config)
            elif provider == "azure_openai":
                return _call_azure_openai(prompt, config)
            elif provider == "openai":
                return _call_openai(prompt, config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            if provider == "local":
                log.error(f"Local model call failed: {e}")
                return "Error: local model call failed"
            wait = 2 ** attempt
            log.warning(f"API call failed (attempt {attempt+1}): {e}. Retry in {wait}s …")
            time.sleep(wait)

    log.error("All retries exhausted.")
    return "Error: LLM call failed"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

SS_PROMPT_TEMPLATE = """\
You are a clinical coding assistant mapping conversation text to the Omaha System.

Task: Identify ALL signs/symptoms mentioned in the CURRENT TURN (not just one).

Context (surrounding conversation for reference only):
{context}

CURRENT TURN to classify:
{query}

Available Omaha classifications (retrieved by relevance):
{options}

RULES:
1. Classify ONLY explicit abnormal symptoms, distress, or dysfunction in the CURRENT TURN.
2. Normal readings, questions, greetings, administrative talk → respond with NONE.
3. Use EXACT wording from the options (domain, problem, signs/symptoms). No synonyms.
4. If multiple signs/symptoms are present, list each on a separate numbered line.
5. Map common phrases: "shortness of breath" / "SOB" → "abnormal breath patterns" (Respiration); \
"swollen"/"edema" → "edema" (Circulation); "high BP"/"145/92" → "abnormal blood pressure reading" \
(Circulation); "tired"/"fatigue" → "somatic complaints/fatigue" (Mental health).
6. Maximum 3 classifications.

OUTPUT FORMAT (use EXACTLY this format, or output NONE):
1. Domain: [exact] | Problem: [exact] | Signs/Symptoms: [exact]
2. Domain: [exact] | Problem: [exact] | Signs/Symptoms: [exact]
...

NONE
"""

INTERVENTION_PROMPT_TEMPLATE = """\
You are a clinical coding assistant mapping conversation text to Omaha System interventions.

Task: Identify ALL nursing interventions or care actions performed in the CURRENT TURN.

Context (surrounding conversation for reference only):
{context}

CURRENT TURN to classify:
{query}

Available Omaha intervention options (retrieved by relevance):
{options}

RULES:
1. Classify ONLY actual clinician actions: teaching, assessing, monitoring, treating, case-managing.
2. Patient statements, greetings, administrative logistics → respond with NONE.
3. Use EXACT wording from the options (category and target). No paraphrasing.
4. If multiple interventions are present, list each on a separate numbered line.
5. Common mappings: assessing vital signs/symptoms → Surveillance_signs/symptoms-physical; \
asking about emotions/mental state → Surveillance_signs/symptoms-mental/emotional; \
explaining medications → Teaching, Guidance, and Counseling_medication action/side effects; \
reviewing history → Surveillance_signs/symptoms-physical.
6. Maximum 3 classifications.

OUTPUT FORMAT (use EXACTLY this format, or output NONE):
1. Category: [exact] | Target: [exact]
2. Category: [exact] | Target: [exact]
...

NONE
"""


def build_ss_prompt(query: str, context: str, retrieved_docs: list[dict]) -> str:
    options_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        options_text += (
            f"{i}. Domain: {doc['domain']} | Problem: {doc['problem']} "
            f"| Signs/Symptoms: {doc['ss']}\n"
        )
    return SS_PROMPT_TEMPLATE.format(
        context=context, query=query, options=options_text.strip()
    )


def build_intervention_prompt(query: str, context: str,
                              retrieved_docs: list[dict]) -> str:
    options_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        options_text += (
            f"{i}. Category: {doc['category']} | Target: {doc['target']}\n"
        )
    return INTERVENTION_PROMPT_TEMPLATE.format(
        context=context, query=query, options=options_text.strip()
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OUTPUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_ss_output(text: str) -> list[dict]:
    """
    Parse LLM output for sign/symptom.
    Returns list of {'domain', 'problem', 'ss', 'label'} dicts.
    Returns [] if output is NONE / no healthcare problem.
    """
    if not text or re.search(r"\bNONE\b", text, re.IGNORECASE):
        # Extra check: if the model said "No sufficient information" etc.
        if re.search(r"no sufficient|no health|not applicable|none", text, re.IGNORECASE):
            return []
        # If it starts with numbered items, continue parsing
        if not re.search(r"^\d+\.", text.strip(), re.MULTILINE):
            return []

    results = []
    # Pattern: "Domain: X | Problem: Y | Signs/Symptoms: Z"
    # or without leading number
    pattern = re.compile(
        r"Domain\s*:\s*(?P<domain>[^|]+)\|\s*Problem\s*:\s*(?P<problem>[^|]+)\|"
        r"\s*Signs/Symptoms\s*:\s*(?P<ss>[^\n]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        domain  = m.group("domain").strip().rstrip("|").strip()
        problem = m.group("problem").strip().rstrip("|").strip()
        ss      = m.group("ss").strip().rstrip("|").strip()
        # Remove markdown artifacts
        for artifact in ["**", "__", "  "]:
            domain  = domain.replace(artifact, "")
            problem = problem.replace(artifact, "")
            ss      = ss.replace(artifact, "")
        if problem and ss:
            results.append({
                "domain":  domain,
                "problem": problem,
                "ss":      ss,
                "label":   f"{problem}_{ss}",
            })
    return results


def parse_intervention_output(text: str) -> list[dict]:
    """
    Parse LLM output for interventions.
    Returns list of {'category', 'target', 'label'} dicts.
    """
    if not text or re.search(r"\bNONE\b", text, re.IGNORECASE):
        if re.search(r"no sufficient|no intervention|not applicable|none", text, re.IGNORECASE):
            return []
        if not re.search(r"^\d+\.", text.strip(), re.MULTILINE):
            return []

    results = []
    # Pattern: "Category: X | Target: Y"
    pattern = re.compile(
        r"Category\s*:\s*(?P<category>[^|]+)\|\s*Target\s*:\s*(?P<target>[^\n]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        category = m.group("category").strip().rstrip("|").strip()
        target   = m.group("target").strip().rstrip("|").strip()
        for artifact in ["**", "__"]:
            category = category.replace(artifact, "")
            target   = target.replace(artifact, "")
        if category and target:
            results.append({
                "category": category,
                "target":   target,
                "label":    f"{category}_{target}",
            })
    return results


def parse_human_labels(row: pd.Series, col_prefix: str,
                       n_cols: int = 3) -> list[str]:
    """
    Extract non-empty human annotation labels from OS_SS_1/2/3 or OS_I_1/2/3.
    Returns list of raw label strings (format: Problem_SignSymptom or Category_Target).
    """
    labels = []
    for i in range(1, n_cols + 1):
        col = f"{col_prefix}_{i}"
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                labels.append(str(val).strip())
    return labels

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CONVERSATION PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def get_context_text(df: pd.DataFrame, idx: int, window: int = CONTEXT_WINDOW) -> str:
    """Build context string from ±window rows around idx."""
    start = max(0, idx - window)
    end   = min(len(df), idx + window + 1)
    lines = []
    for i in range(start, end):
        if i == idx:
            continue
        spk  = str(df.iloc[i].get("Spk", "")).strip()
        conv = str(df.iloc[i].get("Conversation", "")).strip()
        if conv and conv != "nan":
            lines.append(f"[{spk}]: {conv}")
    return "\n".join(lines) if lines else "(no context)"


def process_sheet(
    df: pd.DataFrame,
    ss_docs: list[dict], ss_embeddings: np.ndarray,
    int_docs: list[dict], int_embeddings: np.ndarray,
    embed_model: SentenceTransformer,
    llm_name: str,
) -> pd.DataFrame:
    """
    Process all rows in one conversation sheet.
    Returns a DataFrame with original columns + LLM predictions.
    """
    ss_preds_all  = []
    int_preds_all = []
    ss_raw_all    = []
    int_raw_all   = []

    # Determine meaningful column (skip rows flagged as Negative if desired)
    meaningful_col = next(
        (c for c in df.columns if "meaningful" in c.lower()), None
    )

    for idx in range(len(df)):
        row   = df.iloc[idx]
        query = str(row.get("Conversation", "")).strip()

        # Skip empty turns
        if not query or query == "nan":
            ss_preds_all.append([])
            int_preds_all.append([])
            ss_raw_all.append("")
            int_raw_all.append("")
            continue

        context = get_context_text(df, idx)

        # ── Signs/Symptoms prediction ───────────────────────────────────────
        ss_retrieved  = retrieve(query, ss_docs, ss_embeddings, embed_model)
        ss_prompt     = build_ss_prompt(query, context, ss_retrieved)
        ss_raw        = call_llm(ss_prompt, llm_name)
        ss_preds      = parse_ss_output(ss_raw)

        # ── Intervention prediction ─────────────────────────────────────────
        int_retrieved = retrieve(query, int_docs, int_embeddings, embed_model)
        int_prompt    = build_intervention_prompt(query, context, int_retrieved)
        int_raw       = call_llm(int_prompt, llm_name)
        int_preds     = parse_intervention_output(int_raw)

        ss_preds_all.append(ss_preds)
        int_preds_all.append(int_preds)
        ss_raw_all.append(ss_raw)
        int_raw_all.append(int_raw)

    # ── Flatten predictions into columns ─────────────────────────────────────
    out = df.copy()

    # SS predictions (up to 3)
    for col_i in range(1, 4):
        out[f"Pred_SS_{col_i}"] = [
            p[col_i - 1]["label"] if len(p) >= col_i else ""
            for p in ss_preds_all
        ]
    # Intervention predictions (up to 3)
    for col_i in range(1, 4):
        out[f"Pred_I_{col_i}"] = [
            p[col_i - 1]["label"] if len(p) >= col_i else ""
            for p in int_preds_all
        ]

    # Keep raw LLM output for debugging
    out["LLM_SS_raw"]  = ss_raw_all
    out["LLM_I_raw"]   = int_raw_all

    return out

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def fuzzy_label_match(pred: str, true: str, threshold: int = FUZZY_THRESHOLD) -> bool:
    """True if two label strings are fuzzy-similar above threshold."""
    if not pred or not true:
        return False
    # Normalise: lowercase, strip, collapse whitespace
    p = re.sub(r"\s+", " ", pred.strip().lower())
    t = re.sub(r"\s+", " ", true.strip().lower())
    return fuzz.ratio(p, t) >= threshold


def row_level_match(pred_labels: list[str], true_labels: list[str],
                    threshold: int = FUZZY_THRESHOLD) -> tuple[int, int, int]:
    """
    Bipartite greedy matching of predictions vs ground-truth labels.

    Returns:
        (tp, fp, fn)
    """
    used_true = set()
    tp = 0

    for pred in pred_labels:
        matched = False
        for j, true in enumerate(true_labels):
            if j not in used_true and fuzzy_label_match(pred, true, threshold):
                tp += 1
                used_true.add(j)
                matched = True
                break
        if not matched:
            pass  # will be counted as FP below

    fp = len(pred_labels) - tp
    fn = len(true_labels) - len(used_true)
    return tp, fp, fn


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_sheet(df_out: pd.DataFrame) -> dict:
    """
    Compute P/R/F1 for SS and Interventions from one processed sheet.

    Key design choices:
    - Rows where BOTH human label set AND prediction set are empty → True Negative.
      These are EXCLUDED from P/R/F1 (they don't affect clinical utility).
    - TP counted with greedy bipartite fuzzy matching.
    - Micro-average: accumulate TP/FP/FN across all rows, then compute metrics.
    """
    ss_tp = ss_fp = ss_fn = 0
    i_tp  = i_fp  = i_fn  = 0

    # Rows with at least one human SS label OR at least one prediction
    ss_relevant_rows  = 0
    int_relevant_rows = 0

    for _, row in df_out.iterrows():
        # ── Signs/Symptoms ───────────────────────────────────────────────────
        human_ss  = [row[f"OS_SS_{j}"] for j in range(1, 4)
                     if f"OS_SS_{j}" in df_out.columns
                     and pd.notna(row[f"OS_SS_{j}"])
                     and str(row[f"OS_SS_{j}"]).strip() not in ("", "nan")]
        pred_ss   = [row[f"Pred_SS_{j}"] for j in range(1, 4)
                     if f"Pred_SS_{j}" in df_out.columns
                     and str(row.get(f"Pred_SS_{j}", "")).strip() != ""]

        if human_ss or pred_ss:
            ss_relevant_rows += 1
            tp, fp, fn = row_level_match(pred_ss, human_ss)
            ss_tp += tp; ss_fp += fp; ss_fn += fn

        # ── Interventions ────────────────────────────────────────────────────
        human_int = [row[f"OS_I_{j}"] for j in range(1, 4)
                     if f"OS_I_{j}" in df_out.columns
                     and pd.notna(row[f"OS_I_{j}"])
                     and str(row[f"OS_I_{j}"]).strip() not in ("", "nan")]
        pred_int  = [row[f"Pred_I_{j}"] for j in range(1, 4)
                     if f"Pred_I_{j}" in df_out.columns
                     and str(row.get(f"Pred_I_{j}", "")).strip() != ""]

        if human_int or pred_int:
            int_relevant_rows += 1
            tp, fp, fn = row_level_match(pred_int, human_int)
            i_tp += tp; i_fp += fp; i_fn += fn

    ss_p,  ss_r,  ss_f1  = compute_prf(ss_tp,  ss_fp,  ss_fn)
    int_p, int_r, int_f1 = compute_prf(i_tp,   i_fp,   i_fn)

    return {
        "ss_precision":  ss_p,  "ss_recall":  ss_r,  "ss_f1":  ss_f1,
        "ss_tp": ss_tp, "ss_fp": ss_fp, "ss_fn": ss_fn,
        "ss_relevant_rows": ss_relevant_rows,
        "int_precision": int_p, "int_recall": int_r, "int_f1": int_f1,
        "int_tp": i_tp, "int_fp": i_fp, "int_fn": i_fn,
        "int_relevant_rows": int_relevant_rows,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(llm_name: str = ACTIVE_LLM):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"{'='*60}")
    log.info(f"Starting Omaha Mapping Pipeline | LLM: {llm_name}")
    log.info(f"{'='*60}")

    # ── Load Omaha System ─────────────────────────────────────────────────────
    ss_df, int_df = load_omaha_system(S3_BUCKET, OMAHA_KEY)
    ss_docs       = build_ss_documents(ss_df)
    int_docs      = build_intervention_documents(int_df)

    # ── Build embeddings ──────────────────────────────────────────────────────
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model    = SentenceTransformer(EMBEDDING_MODEL)
    ss_embeddings  = build_index(ss_docs,  embed_model)
    int_embeddings = build_index(int_docs, embed_model)
    log.info("Indices ready.")

    # ── Load annotation data ──────────────────────────────────────────────────
    annotation_sheets = s3_read_excel(S3_BUCKET, ANNOTATION_KEY)
    log.info(f"Annotation sheets: {len(annotation_sheets)}")

    # ── Process each conversation ─────────────────────────────────────────────
    output_sheets   = {}   # sheet_name → processed DataFrame
    sheet_metrics   = []   # list of per-sheet metric dicts
    global_ss_tp = global_ss_fp = global_ss_fn = 0
    global_i_tp  = global_i_fp  = global_i_fn  = 0

    for sheet_name, conv_df in tqdm(annotation_sheets.items(),
                                    desc="Processing conversations"):
        log.info(f"\n--- Sheet: {sheet_name} ---")

        # Standardise column names
        conv_df.columns = [c.strip() for c in conv_df.columns]
        conv_df = conv_df.fillna("")

        try:
            df_out = process_sheet(
                conv_df,
                ss_docs, ss_embeddings,
                int_docs, int_embeddings,
                embed_model,
                llm_name,
            )
            metrics = evaluate_sheet(df_out)

        except Exception as e:
            log.error(f"Sheet '{sheet_name}' failed: {e}")
            df_out  = conv_df.copy()
            metrics = {k: 0 for k in [
                "ss_precision","ss_recall","ss_f1","ss_tp","ss_fp","ss_fn",
                "ss_relevant_rows","int_precision","int_recall","int_f1",
                "int_tp","int_fp","int_fn","int_relevant_rows"]}

        metrics["sheet"] = sheet_name
        sheet_metrics.append(metrics)

        # Accumulate global counts
        global_ss_tp += metrics["ss_tp"];  global_ss_fp += metrics["ss_fp"]
        global_ss_fn += metrics["ss_fn"]
        global_i_tp  += metrics["int_tp"]; global_i_fp  += metrics["int_fp"]
        global_i_fn  += metrics["int_fn"]

        # Store processed sheet (safe sheet name ≤ 31 chars)
        safe = re.sub(r"[\\/*?:\[\]]", "_", sheet_name)[:31]
        output_sheets[safe] = df_out

        log.info(
            f"SS  → P={metrics['ss_precision']:.3f}  R={metrics['ss_recall']:.3f}  "
            f"F1={metrics['ss_f1']:.3f}"
        )
        log.info(
            f"Int → P={metrics['int_precision']:.3f}  R={metrics['int_recall']:.3f}  "
            f"F1={metrics['int_f1']:.3f}"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    global_ss_p,  global_ss_r,  global_ss_f1  = compute_prf(global_ss_tp,  global_ss_fp,  global_ss_fn)
    global_int_p, global_int_r, global_int_f1 = compute_prf(global_i_tp,   global_i_fp,   global_i_fn)

    # Macro averages (mean across sheets, for comparison to prior work)
    macro_ss_p  = np.mean([m["ss_precision"]  for m in sheet_metrics])
    macro_ss_r  = np.mean([m["ss_recall"]     for m in sheet_metrics])
    macro_ss_f1 = np.mean([m["ss_f1"]         for m in sheet_metrics])
    macro_i_p   = np.mean([m["int_precision"] for m in sheet_metrics])
    macro_i_r   = np.mean([m["int_recall"]    for m in sheet_metrics])
    macro_i_f1  = np.mean([m["int_f1"]        for m in sheet_metrics])

    summary = {
        "llm":       llm_name,
        "embedding": EMBEDDING_MODEL,
        "timestamp": timestamp,
        "context_window": CONTEXT_WINDOW,
        "top_k_retrieval": TOP_K_RETRIEVAL,
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "signs_symptoms": {
            "micro": {
                "precision": global_ss_p,
                "recall":    global_ss_r,
                "f1":        global_ss_f1,
                "tp": global_ss_tp, "fp": global_ss_fp, "fn": global_ss_fn,
            },
            "macro": {
                "precision": round(macro_ss_p,  4),
                "recall":    round(macro_ss_r,  4),
                "f1":        round(macro_ss_f1, 4),
            },
        },
        "interventions": {
            "micro": {
                "precision": global_int_p,
                "recall":    global_int_r,
                "f1":        global_int_f1,
                "tp": global_i_tp, "fp": global_i_fp, "fn": global_i_fn,
            },
            "macro": {
                "precision": round(macro_i_p,  4),
                "recall":    round(macro_i_r,  4),
                "f1":        round(macro_i_f1, 4),
            },
        },
        "per_sheet": sheet_metrics,
    }

    # ── Build summary sheet ───────────────────────────────────────────────────
    summary_rows = []
    for m in sheet_metrics:
        summary_rows.append({
            "Sheet":            m["sheet"],
            "SS_Precision":     m["ss_precision"],
            "SS_Recall":        m["ss_recall"],
            "SS_F1":            m["ss_f1"],
            "SS_TP":            m["ss_tp"],
            "SS_FP":            m["ss_fp"],
            "SS_FN":            m["ss_fn"],
            "Int_Precision":    m["int_precision"],
            "Int_Recall":       m["int_recall"],
            "Int_F1":           m["int_f1"],
            "Int_TP":           m["int_tp"],
            "Int_FP":           m["int_fp"],
            "Int_FN":           m["int_fn"],
        })

    # Add aggregate rows
    for row_data in [
        ("MICRO_AGGREGATE", global_ss_p,  global_ss_r,  global_ss_f1,
                            global_int_p, global_int_r, global_int_f1),
        ("MACRO_AVERAGE",   round(macro_ss_p,4),  round(macro_ss_r,4),  round(macro_ss_f1,4),
                            round(macro_i_p,4),   round(macro_i_r,4),   round(macro_i_f1,4)),
    ]:
        label, sp, sr, sf, ip, ir, i_f = row_data
        summary_rows.append({
            "Sheet": label,
            "SS_Precision": sp,  "SS_Recall": sr,  "SS_F1": sf,
            "SS_TP": "",         "SS_FP": "",       "SS_FN": "",
            "Int_Precision": ip, "Int_Recall": ir,  "Int_F1": i_f,
            "Int_TP": "",        "Int_FP": "",       "Int_FN": "",
        })

    output_sheets["SUMMARY"] = pd.DataFrame(summary_rows)

    # ── Save to S3 ────────────────────────────────────────────────────────────
    results_key = f"{OUTPUT_PREFIX}{llm_name}_{timestamp}_results.xlsx"
    summary_key = f"{OUTPUT_PREFIX}{llm_name}_{timestamp}_summary.json"

    s3_write_excel(output_sheets, S3_BUCKET, results_key)
    s3_write_json(summary, S3_BUCKET, summary_key)

    # ── Print final report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS  |  Model: {llm_name}")
    print("=" * 60)
    print(f"{'Metric':<30} {'Signs/Symptoms':>16}  {'Interventions':>14}")
    print("-" * 60)
    print(f"{'Micro Precision':<30} {global_ss_p:>16.4f}  {global_int_p:>14.4f}")
    print(f"{'Micro Recall':<30} {global_ss_r:>16.4f}  {global_int_r:>14.4f}")
    print(f"{'Micro F1':<30} {global_ss_f1:>16.4f}  {global_int_f1:>14.4f}")
    print(f"{'Macro Precision':<30} {macro_ss_p:>16.4f}  {macro_i_p:>14.4f}")
    print(f"{'Macro Recall':<30} {macro_ss_r:>16.4f}  {macro_i_r:>14.4f}")
    print(f"{'Macro F1':<30} {macro_ss_f1:>16.4f}  {macro_i_f1:>14.4f}")
    print("-" * 60)
    print(f"Global TP/FP/FN (SS):  {global_ss_tp}/{global_ss_fp}/{global_ss_fn}")
    print(f"Global TP/FP/FN (Int): {global_i_tp}/{global_i_fp}/{global_i_fn}")
    print("=" * 60)
    print(f"Results saved to: s3://{S3_BUCKET}/{results_key}")
    print(f"Summary saved to: s3://{S3_BUCKET}/{summary_key}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MULTI-MODEL COMPARISON (optional)
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(models_to_compare: list[str] = None):
    """
    Run the full pipeline for each model and produce a side-by-side comparison.
    NOTE: when switching between local models the old model is unloaded first
    to free GPU memory.
    """
    global _local_pipeline
    if models_to_compare is None:
        models_to_compare = MODELS_TO_COMPARE

    all_summaries = {}
    for model_name in models_to_compare:
        log.info(f"\n{'#'*60}\nRunning model: {model_name}\n{'#'*60}")

        # Free previous local model from GPU memory before loading the next one
        if _local_pipeline is not None:
            log.info("Unloading previous local model to free GPU memory …")
            del _local_pipeline
            _local_pipeline = None
            torch.cuda.empty_cache()

        try:
            summary = run_pipeline(llm_name=model_name)
            all_summaries[model_name] = summary
        except Exception as e:
            log.error(f"Model {model_name} failed: {e}")
            all_summaries[model_name] = {"error": str(e)}

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} {'SS_P':>6} {'SS_R':>6} {'SS_F1':>6}  "
          f"{'Int_P':>6} {'Int_R':>6} {'Int_F1':>6}")
    print("-" * 80)
    for mn, s in all_summaries.items():
        if "error" in s:
            print(f"{mn:<25}  ERROR: {s['error']}")
            continue
        ss  = s.get("signs_symptoms", {}).get("micro", {})
        iv  = s.get("interventions",  {}).get("micro", {})
        print(
            f"{mn:<25} {ss.get('precision',0):>6.4f} {ss.get('recall',0):>6.4f} "
            f"{ss.get('f1',0):>6.4f}  {iv.get('precision',0):>6.4f} "
            f"{iv.get('recall',0):>6.4f} {iv.get('f1',0):>6.4f}"
        )
    print("=" * 80)

    # Save comparison to S3
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"{OUTPUT_PREFIX}model_comparison_{ts}.json"
    s3_write_json(all_summaries, S3_BUCKET, key)
    print(f"\nComparison saved: s3://{S3_BUCKET}/{key}")
    return all_summaries


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if COMPARE_ALL_MODELS:
        compare_models()
    else:
        run_pipeline(llm_name=ACTIVE_LLM)
