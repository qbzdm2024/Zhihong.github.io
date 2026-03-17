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

# Suppress noisy third-party warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # stops HF tokenizer fork warnings
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")         # silence TensorFlow INFO/WARNING

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
#EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l"       # higher quality
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

# ── Single-model run ──────────────────────────────────────────────────────────
# Used when COMPARE_ALL_MODELS = False
ACTIVE_LLM = "gpt-4o-mini"

# ── Multi-model comparison ────────────────────────────────────────────────────
# Set COMPARE_ALL_MODELS = True to benchmark several models back-to-back.
# The RAG index is built ONCE and shared across all models (no redundant S3 reads).
# Results per model are saved individually; a side-by-side comparison Excel is
# produced at the end.
#
# Instance guide:
#   OpenAI only              → ml.t3.medium  (~$0.05/hr, CPU is enough)
#   + one local 7/8B model   → ml.g5.xlarge  (~$1.41/hr, 1× A10G 24 GB)
#   + Llama3-70B             → ml.g5.12xlarge (~$5.67/hr, 4× A10G 96 GB)
#
COMPARE_ALL_MODELS = True
MODELS_TO_COMPARE  = [
     "gpt-4o-mini",        # OpenAI — school HIPAA key
    #"mistral-7b-local",   # open-source 7B — local GPU
    #"llama3-8b-local",    # open-source 8B — local GPU
]

# ── Pipeline settings ─────────────────────────────────────────────────────────
CONTEXT_WINDOW  = 2    # rows before/after current turn to include as context
TOP_K_RETRIEVAL = 15   # number of Omaha options sent to the LLM
FUZZY_THRESHOLD = 80   # minimum fuzz.ratio for a match (0–100)

# ── Quick-test mode ───────────────────────────────────────────────────────────
# Set to a small number (e.g. 3) to run only the first N conversation sheets.
# Use this to check prompt quality quickly before running all 23 sheets.
# Set to None (or 0) to run all sheets.
MAX_SHEETS = 1          # ← 1 sheet for local model debugging; change to 3 or None for full run

# ── API / model credentials ────────────────────────────────────────────────────
# School/institution OpenAI key (HIPAA BAA in place — safe for patient data).
# Set as an environment variable in the SageMaker terminal:
#   export OPENAI_API_KEY="sk-..."
# Or paste directly here for quick testing (don't commit the key to git):
OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")

# For local models: HF_TOKEN is only needed to download model weights once.
# Patient data is never sent to HuggingFace when using local models.
HF_TOKEN              = os.environ.get("HF_TOKEN", "")

# For Azure OpenAI (only needed if using "azure_openai" provider):
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY  = os.environ.get("AZURE_OPENAI_API_KEY",  "")
AZURE_API_VERSION     = "2024-08-01-preview"

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


def _find_col(columns: list[str], *keywords: str, exclude: list[str] = None) -> str:
    """Return first column whose name contains ALL keywords (case-insensitive),
    excluding any that contain an exclude keyword. Raises clear error if not found."""
    exclude = [e.lower() for e in (exclude or [])]
    for c in columns:
        cl = c.lower()
        if all(k.lower() in cl for k in keywords):
            if not any(e in cl for e in exclude):
                return c
    raise ValueError(
        f"Could not find column matching keywords={keywords}, exclude={exclude}. "
        f"Available columns: {columns}"
    )


def build_ss_documents(ss_df: pd.DataFrame) -> list[dict]:
    """
    Each document represents one (Problem, Sign/Symptom) pair.
    The text field is used for embedding.
    """
    docs = []
    log.info(f"Sign/symptom sheet columns: {list(ss_df.columns)}")

    # "Problems of the Problem Classification Scheme" — matches "problem" but
    # exclude "domain definition" and "problem definition" (not "classification")
    prob_col = _find_col(ss_df.columns, "problem", exclude=["definition"])
    ss_col   = _find_col(ss_df.columns, "signs")
    # Definition column — prefer the first one found, skip "domain definition"
    def_col  = next(
        (c for c in ss_df.columns
         if "definition" in c.lower() and "domain" not in c.lower()
         and "problem" not in c.lower()),
        None
    )
    log.info(f"Using columns → problem: '{prob_col}' | ss: '{ss_col}' | def: '{def_col}'")

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
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings.astype("float32")


def retrieve(query: str, docs: list[dict], embeddings: np.ndarray,
             model: SentenceTransformer, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Return top-k most relevant documents for the query."""
    q_emb = model.encode([query], normalize_embeddings=True,
                         show_progress_bar=False).astype("float32")
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

    # ── GPU guard ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Local models require a GPU instance "
            "(e.g. ml.g5.xlarge). Current machine has no CUDA device.\n"
            "  • If you are on SageMaker, switch to a GPU instance type.\n"
            "  • If you only want to run API models (gpt-4o-mini, azure), "
            "set COMPARE_ALL_MODELS=False or remove local models from "
            "MODELS_TO_COMPARE."
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_id     = config["model"]
    load_in_4bit = config.get("load_in_4bit", True)

    log.info(f"Loading local model: {model_id}  (4-bit={load_in_4bit})")
    log.info("This takes 2–5 min on first run. Model is cached after that.")

    # Login to HuggingFace for gated models (Llama3 requires accepting licence)
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)

    # Try 4-bit quantisation; fall back to float16 if bitsandbytes is missing
    quant_cfg = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401  — just to verify it's installed
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            log.info("4-bit quantisation enabled (bitsandbytes found)")
        except (ImportError, Exception) as bnb_err:
            log.warning(
                f"bitsandbytes not available ({bnb_err}). "
                "Falling back to float16 (uses ~14 GB VRAM for 7/8B models — "
                "fine on A10G 24 GB). Run: pip install bitsandbytes>=0.43 to enable 4-bit."
            )
            load_in_4bit = False  # use fp16 path below

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
                # Re-raise hard errors (no GPU, OOM) so they surface immediately
                # rather than silently producing empty predictions for every row.
                raise
            wait = 2 ** attempt
            log.warning(f"API call failed (attempt {attempt+1}): {e}. Retry in {wait}s …")
            time.sleep(wait)

    log.error("All retries exhausted.")
    return "Error: LLM call failed"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

# ── Agent 1: Clinical understanding prompt ────────────────────────────────────
#
# Run BEFORE the RAG classifiers.  Produces a structured clinical summary of the
# turn that is injected into both the SS and INT prompts so Agent 2 works with
# a richer description of what is actually being said.
#
UNDERSTAND_PROMPT = """\
You are a clinical content analyst for home healthcare conversations.

Analyze the turn below and extract structured clinical facts to guide
Omaha System coding.

━━━ CONTEXT (surrounding turns) ━━━
{context}

━━━ CURRENT TURN ━━━
{query}

━━━ TASK ━━━
Answer each item concisely based ONLY on what is stated in the current turn.

SPEAKER ROLE
Is the primary speaker a patient/caregiver (reporting symptoms, feelings,
measurements) or a nurse/clinician (assessing, instructing, treating,
coordinating)?

PATIENT HEALTH PROBLEM
Is there a patient-reported symptom, complaint, abnormal measurement, or
finding?
• If yes — describe it in plain clinical terms; note if any value is abnormal.
• If no  — write "none".

NURSE ACTION
Is the nurse/clinician performing or describing an action (assessing, teaching,
treating, or coordinating care)?
• If yes — describe the action type and its clinical target briefly.
• If no  — write "none".

KEY CLINICAL PHRASES
List the exact clinical phrases from the turn verbatim (comma-separated).
Write "none" if there are none.

Respond ONLY in this exact format with no extra text:
SPEAKER: [patient | clinician | unclear]
PATIENT PROBLEM: [description, or "none"]
NURSE ACTION: [description, or "none"]
KEY PHRASES: [phrase1, phrase2, ... or "none"]
"""
Analyze the healthcare query provided below and identify the MOST RELEVANT Omaha System classification.

Instructions:
1. Review the query for any direct mentions of healthcare-related issues across these domains:
   - Environmental Domain (e.g., environment safety, insurance)
   - Psychological Domain (e.g., feelings of fatigue, low mood)
   - Physiological Domain
   - Health-related Behaviors Domain (e.g., nutrition, medication regimen)
2. If the problem pertains to issues affecting others (e.g., family members, children) or describes general environmental or administrative issues (e.g., insurance paperwork, scheduling, date inquiries) that do not directly affect the patient him/herself, ignore it.
3. From the provided options, select the single MOST RELEVANT classification that best matches the query’s domain, problem, and signs/symptoms.
4. IMPORTANT: Your final answer MUST USE THE EXACT WORDING from one of the provided options without any modifications, rephrasing, or synonyms. For example, if the query uses a synonym like "swollen," you MUST map it to "edema" exactly as it appears in the options.
5. EXTRA RULE: If the query contains "swollen" or any variant (e.g., "swelling"), choose the option that lists "edema" with the "Circulation" problem in the Signs/Symptoms field. Do not substitute any other term such as "inflammation."
6. DO NOT OVERINFER. If the query does not clearly describe a healthcare problem, lacks sufficient information to support the presence of a healthcare issue, or if no option closely aligns with the query, respond with "No sufficient information available." For example, vague or ambiguous statements such as "I didn't know. I heard voices." should not be interpreted as a healthcare problem.
7. If the query is phrased as a question, assume that no healthcare problem can be confirmed and respond with "No sufficient information available."
8. If the query includes a blood pressure reading, glucose level, or any other medical measurement, determine whether the value is abnormal. If abnormal, respond with the corresponding problem mapped to the Omaha System; if normal, respond with "No sufficient information available."
9. NORMAL BEHAVIOR RULE: If the query describes typical, expected behavior or general administrative/healthcare information (e.g., exercising, working late, needing rest or sleep, test costs, scheduling, date inquiries, availability, payment details) without any indication of abnormal symptoms, distress, or dysfunction, then no healthcare problem is identified. However, if the query simply states a symptom (e.g., "I am tired today") without additional context indicating normal behavior, treat it as a potential healthcare problem only if supported by corresponding classification options.
10. NEGATIVE QUERY RULE: If the overall tone or content of the query is purely administrative, factual, or routine (e.g., inquiries about dates, insurance paperwork, normal vital sign values, casual remarks about sleep or work schedules) or includes negative phrasing that does not describe a specific abnormal health condition, respond with "No sufficient information available."

11. CHAIN-OF-THOUGHT: In your final output, include a brief chain-of-thought explanation immediately following your classification. This explanation should outline the key reasoning steps without revealing sensitive internal processing details.

Query:
{query}

Options:
{options}

Response format:
If a relevant match is found, provide ONLY ONE classification in this format:

Domain: [Exact match]
Problem: [Exact match]
Signs/Symptoms: [Exact match]

If no healthcare problem is identified or no close match is found:
NONE

Example Queries and Responses:
Example 1:
Query: "when I got back from my walk today; it is 145/92."
Response:
Domain: Physiological Domain
Problem: Circulation
Signs/Symptoms: abnormal blood pressure reading
Chain-of-Thought: The query mentioned a high blood pressure reading (145/92, which is above the normal range), directly mapping to "abnormal blood pressure reading" under the Circulation option.

Example 2:
Query: "You have the shower?"
Response:
No sufficient information available
Chain-of-Thought: The query is phrased as a question, so no specific healthcare-related information can be confirmed.

Example 3:
Query: "I can not breath at night."
Response:
Domain: Physiological Domain
Problem: Circulation
Signs/Symptoms: abnormal breath pattern
Chain-of-Thought: "cannot breath" here is interpreted as an irregular or abnormal breathing pattern rather than a complete inability to breathe independently (which would typically require mechanical assistance). Therefore, the best matching option is "abnormal breath pattern." Do not change the wording.


"""



INTERVENTION_PROMPT_TEMPLATE = """
You are a clinical coding assistant mapping conversation turns to Omaha System interventions.

Context (reference only):
{context}

CURRENT TURN:
{query}

TASK
1. FIRST determine if the query describes a nurse/clincian's behavior related to teaching and guidance, treamtent and procedure, case management, or surveillance:
2. If not, classify it as "None"
3. If yes, Identify Omaha System intervention(s) expressed in the CURRENT TURN.

IMPORTANT RULES
1. Only classify clinician actions in the CURRENT TURN.
2. Patient speech, greetings, filler, acknowledgments, jokes, or casual conversation → NONE.
3. Maximum 3 interventions.
4. Use EXACT Omaha System terminology.

VALID OMAHA INTERVENTION DOMAINS
You must use EXACTLY one of the following:

- Teaching, Guidance, and Counseling
- Treatments and Procedures
- Case Management
- Surveillance

TARGET REQUIREMENT
The Target MUST exactly match one of the official Omaha System targets from the provided list.
Do NOT modify wording, abbreviate, paraphrase, or invent targets.

Examples of INVALID targets:
❌ respiratory therapy care
❌ medication review
❌ wound monitoring

Examples of VALID targets:
✔ respiratory care
✔ medication administration
✔ dressing change/wound care
✔ signs/symptoms-physical

ACTION TYPE DEFINITIONS

SURVEILLANCE
Monitoring, assessing, measuring, reviewing status.

Examples
- checking blood pressure, pulse, oxygen
- assessing heart or lungs
- reviewing medications
- assessing wound

Common mappings
measuring blood pressure / pulse / oxygen / temperature
→ Surveillance | signs/symptoms-physical

checking wound
→ Surveillance | dressing change/wound care

reviewing medications
→ Surveillance | medication administration


TREATMENTS AND PROCEDURES
Hands-on clinical care or procedures.

Examples
- dressing a wound
- applying bandage or saline
- administering medication
- performing wound care

Example mapping
cleaning or dressing wound
→ Treatments and Procedures | dressing change/wound care


TEACHING, GUIDANCE, AND COUNSELING
Providing instruction, explanation, or advice.

Examples
- explaining wound care
- warning about infection
- medication instructions

Example mapping
explaining how to clean a wound
→ Teaching, Guidance, and Counseling | dressing change/wound care


CASE MANAGEMENT
Coordination, referrals, scheduling, contacting providers.

Examples
- calling doctor
- arranging antibiotics
- scheduling visits
- coordinating care

Example mapping
calling doctor about medication
→ Case Management | medication coordination/ordering


MULTI-LABEL RULE
Use multiple interventions only if the CURRENT TURN clearly contains multiple actions.

NONE RULE
Return NONE if the turn contains no clinical intervention.

AVAILABLE OMAHA TARGET OPTIONS
{options}

OUTPUT FORMAT

1. Category: [exact Omaha domain] | Target: [exact Omaha target]
2. Category: [exact Omaha domain] | Target: [exact Omaha target]
3. Category: [exact Omaha domain] | Target: [exact Omaha target]

or

NONE


EXAMPLES

Query:
"I will check your blood pressure and pulse."

1. Category: Surveillance | Target: signs/symptoms-physical

Query:
"Let me clean the wound and place a new bandage."

1. Category: Treatments and Procedures | Target: dressing change/wound care

Query:
"You should clean the wound with saline and watch for redness."

1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query:
"I will call your doctor to arrange the antibiotic prescription."

1. Category: Case Management | Target: medication coordination/ordering

Query: "So over the weekend, I go to take a shower with soap and water."
1. Category: Teaching, Guidance, and Counseling | Target: personal hygiene

Query:
"Okay."

NONE

Query: "Oh, let me see. For five days."
NONE
"""


def understand_turn(query: str, context: str, llm_name: str) -> str:
    """
    Agent 1 — Clinical pre-understanding.

    Calls the LLM to produce a structured analysis of the turn (speaker role,
    patient problem, nurse action, key phrases) BEFORE classification.
    The returned string is injected into both the SS and INT prompts so that
    Agent 2 (the RAG classifier) has a richer description of what is actually
    being said clinically.

    Args:
        query:    The current conversation turn text.
        context:  Surrounding turns from get_context_text().
        llm_name: LLM config key (same model used for classification).

    Returns:
        A multi-line structured string, e.g.
            SPEAKER: patient
            PATIENT PROBLEM: elevated blood pressure reading (145/92, abnormal)
            NURSE ACTION: none
            KEY PHRASES: blood pressure, 145/92
    """
    prompt = UNDERSTAND_PROMPT.format(
        context=context or "(no context)",
        query=query,
    )
    if LLM_CONFIGS[llm_name]["provider"] != "local":
        prompt = re.sub(r"<\|[^|>]+\|>", "", prompt).strip()
    raw = call_llm(prompt, llm_name)
    return raw.strip()


def build_ss_prompt(query: str, context: str, retrieved_docs: list[dict],
                    understanding: str = "") -> str:
    """Build the SS classification prompt.

    If *understanding* (from understand_turn) is provided it is appended to the
    query section so that Agent 2 sees the pre-analysed clinical summary.
    """
    effective_query = (
        f"{query}\n\n[Clinical Pre-Analysis]\n{understanding}"
        if understanding else query
    )
    options_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        options_text += (
            f"{i}. Domain: {doc['domain']} | Problem: {doc['problem']} "
            f"| Signs/Symptoms: {doc['ss']}\n"
        )
    # Build Jinja2-style doc block (replaces {% for doc in documents %}...{% endfor %})
    doc_block = ""
    for doc in retrieved_docs:
        doc_block += (
            f"  Domain: {doc['domain']}\n"
            f"  Problem: {doc['problem']}\n"
            f"  Signs/Symptoms: {doc['ss']}\n"
        )
    prompt = SS_PROMPT_TEMPLATE
    # Replace Jinja2 for-loop block with pre-built doc context
    prompt = re.sub(
        r"\{%-?\s*for\b.*?%\}.*?\{%-?\s*endfor\s*-?%\}",
        doc_block.strip(),
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Replace {{query}} (Jinja2) and {query} (Python format) with actual query
    prompt = re.sub(r"\{\{\s*query\s*\}\}", effective_query, prompt)
    prompt = prompt.replace("{query}", effective_query)
    # Replace {options} (Python format) with options list if template uses it
    prompt = prompt.replace("{options}", options_text.strip())
    # Replace {context} (Python format) with surrounding context
    prompt = prompt.replace("{context}", context)
    return prompt


def build_intervention_prompt(query: str, context: str,
                              retrieved_docs: list[dict],
                              understanding: str = "") -> str:
    """Build the Intervention classification prompt.

    If *understanding* (from understand_turn) is provided it is appended to the
    query section so that Agent 2 sees the pre-analysed clinical summary.
    """
    effective_query = (
        f"{query}\n\n[Clinical Pre-Analysis]\n{understanding}"
        if understanding else query
    )
    options_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        options_text += (
            f"{i}. Category: {doc['category']} | Target: {doc['target']}\n"
        )
    return INTERVENTION_PROMPT_TEMPLATE.format(
        context=context, query=effective_query, options=options_text.strip()
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OUTPUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_answer_section(text: str) -> str:
    """
    Strip verbose preamble that Mistral/Llama sometimes add before the actual answer.
    Looks for SOLUTION:/OUTPUT:/ANSWER: markers and returns the text after them.
    If no marker found, returns the original text.
    """
    marker = re.search(
        r"(?:SOLUTION|OUTPUT|ANSWER)\s*:\s*", text, re.IGNORECASE
    )
    if marker:
        return text[marker.end():]
    return text


def _clean_field(s: str) -> str:
    """Strip markdown artifacts and trailing parenthetical context from a field value."""
    for artifact in ["**", "__", "  "]:
        s = s.replace(artifact, "")
    # Remove trailing parenthetical annotations like "(5 days)" or "(mentioned but not present)"
    s = re.sub(r"\s*\([^)]{0,40}\)\s*$", "", s)
    return s.strip().rstrip("|").strip()


def parse_ss_output(text: str) -> list[dict]:
    """
    Parse LLM output for sign/symptom.
    Returns list of {'domain', 'problem', 'ss', 'label'} dicts.
    Returns [] if output is NONE / no healthcare problem.
    """
    if not text:
        return []

    # Extract the answer section (handles Mistral's "EXPLANATION: ... SOLUTION: ...")
    answer = _extract_answer_section(text)

    # Any form of NONE / "no sufficient information" → empty
    if re.search(r"no sufficient|no health|not applicable", answer, re.IGNORECASE):
        return []
    if re.search(r"\bNONE\b", answer, re.IGNORECASE):
        # Only return empty if there are no numbered classifications following the NONE
        if not re.search(r"^\d+\.\s*Domain", answer, re.IGNORECASE | re.MULTILINE):
            return []

    results = []
    seen = set()  # deduplicate in case both patterns match the same line

    # Pattern 1: pipe-separated  "Domain: X | Problem: Y | Signs/Symptoms: Z"
    pattern_pipe = re.compile(
        r"Domain\s*:\s*(?P<domain>[^|\n]+)\|\s*Problem\s*:\s*(?P<problem>[^|\n]+)\|"
        r"\s*Signs/Symptoms\s*:\s*(?P<ss>[^\n]+)",
        re.IGNORECASE,
    )
    # Pattern 2: multi-line  "Domain: X\nProblem: Y\nSigns/Symptoms: Z"
    pattern_multi = re.compile(
        r"Domain\s*:\s*(?P<domain>[^\n]+)\n\s*Problem\s*:\s*(?P<problem>[^\n]+)\n"
        r"\s*Signs/Symptoms\s*:\s*(?P<ss>[^\n]+)",
        re.IGNORECASE,
    )
    for pat in (pattern_pipe, pattern_multi):
        for m in pat.finditer(answer):
            domain  = _clean_field(m.group("domain"))
            problem = _clean_field(m.group("problem"))
            ss      = _clean_field(m.group("ss"))
            key = (problem.lower(), ss.lower())
            if problem and ss and key not in seen:
                seen.add(key)
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
    if not text:
        return []

    answer = _extract_answer_section(text)

    if re.search(r"no sufficient|no intervention|not applicable", answer, re.IGNORECASE):
        return []
    if re.search(r"\bNONE\b", answer, re.IGNORECASE):
        if not re.search(r"^\d+\.\s*Category", answer, re.IGNORECASE | re.MULTILINE):
            return []

    results = []
    seen = set()
    pattern = re.compile(
        r"Category\s*:\s*(?P<category>[^|\n]+)\|\s*Target\s*:\s*(?P<target>[^\n]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(answer):
        category = _clean_field(m.group("category"))
        target   = _clean_field(m.group("target"))
        key = (category.lower(), target.lower())
        if category and target and key not in seen:
            seen.add(key)
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
    use_understand: bool = False,
) -> pd.DataFrame:
    """
    Process all rows in one conversation sheet.
    Returns a DataFrame with original columns + LLM predictions.

    3-agent pipeline when use_understand=True:
      Agent 1 — understand_turn()   : Extracts speaker role, patient problem,
                                       nurse action, and key clinical phrases.
      Agent 2 — build_ss/int_prompt : RAG classifier enriched with Agent 1 output.
      Agent 3 — verify_sheet()      : (optional, called separately) verifies labels.

    When use_understand=False (default) Agent 1 is skipped (original behaviour).
    The Agent 1 output is stored in LLM_understand_raw for inspection.
    """
    ss_preds_all  = []
    int_preds_all = []
    ss_raw_all    = []
    int_raw_all   = []
    understand_raw_all = []

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
            understand_raw_all.append("")
            continue

        context = get_context_text(df, idx)

        # ── Agent 1: Clinical pre-understanding (optional) ───────────────────
        understanding = ""
        if use_understand:
            log.debug(f"[Agent1] row {idx}: running understand_turn …")
            understanding = understand_turn(query, context, llm_name)
            log.debug(f"[Agent1] row {idx}:\n{understanding}")
        understand_raw_all.append(understanding)

        # ── Agent 2: Signs/Symptoms classification ───────────────────────────
        ss_retrieved  = retrieve(query, ss_docs, ss_embeddings, embed_model)
        ss_prompt     = build_ss_prompt(query, context, ss_retrieved, understanding)
        # Strip Llama3 special tokens for API models (GPT, Azure) — they are
        # literal characters to these models and hurt output format compliance.
        if LLM_CONFIGS[llm_name]["provider"] != "local":
            ss_prompt = re.sub(r"<\|[^|>]+\|>", "", ss_prompt).strip()
        ss_raw        = call_llm(ss_prompt, llm_name)
        ss_preds      = parse_ss_output(ss_raw)

        # ── Agent 2: Intervention classification ─────────────────────────────
        int_retrieved = retrieve(query, int_docs, int_embeddings, embed_model)
        int_prompt    = build_intervention_prompt(query, context, int_retrieved,
                                                  understanding)
        int_raw       = call_llm(int_prompt, llm_name)
        int_preds     = parse_intervention_output(int_raw)

        ss_preds_all.append(ss_preds)
        int_preds_all.append(int_preds)
        ss_raw_all.append(ss_raw)
        int_raw_all.append(int_raw)

    # ── Flatten predictions into columns ─────────────────────────────────────
    out = df.copy()

    # SS predictions (up to 3) — label + individual components
    for col_i in range(1, 4):
        out[f"Pred_SS_{col_i}"]         = [p[col_i-1]["label"]   if len(p) >= col_i else "" for p in ss_preds_all]
        out[f"Pred_SS_{col_i}_domain"]  = [p[col_i-1]["domain"]  if len(p) >= col_i else "" for p in ss_preds_all]
        out[f"Pred_SS_{col_i}_problem"] = [p[col_i-1]["problem"] if len(p) >= col_i else "" for p in ss_preds_all]
        out[f"Pred_SS_{col_i}_ss"]      = [p[col_i-1]["ss"]      if len(p) >= col_i else "" for p in ss_preds_all]

    # Intervention predictions (up to 3) — label + individual components
    for col_i in range(1, 4):
        out[f"Pred_I_{col_i}"]          = [p[col_i-1]["label"]    if len(p) >= col_i else "" for p in int_preds_all]
        out[f"Pred_I_{col_i}_category"] = [p[col_i-1]["category"] if len(p) >= col_i else "" for p in int_preds_all]
        out[f"Pred_I_{col_i}_target"]   = [p[col_i-1]["target"]   if len(p) >= col_i else "" for p in int_preds_all]

    # Keep raw LLM output for debugging
    out["LLM_understand_raw"] = understand_raw_all
    out["LLM_SS_raw"]  = ss_raw_all
    out["LLM_I_raw"]   = int_raw_all

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7b — VERIFICATION AGENT
# ══════════════════════════════════════════════════════════════════════════════
#
# A second LLM pass that reviews each classification and either confirms it
# or corrects it.  Optional — add ~1× extra cost but can close the last few
# F1 points when the primary classifier is ambiguous.
#
# Usage (in eval_local.py or Streamlit):
#     df_out = process_sheet(...)
#     df_out = verify_sheet(df_out, ss_docs, ss_emb, int_docs, int_emb,
#                           embed_model, llm_name)
# ──────────────────────────────────────────────────────────────────────────────

VERIFY_SS_PROMPT = """You are a quality reviewer for Omaha System clinical coding.

A classifier proposed a Signs/Symptoms label for this home healthcare turn.
Your job: confirm it is correct, or correct it.

━━━ TURN ━━━
{query}

━━━ PROPOSED CLASSIFICATION ━━━
{prediction}

━━━ AVAILABLE OPTIONS (top retrieved) ━━━
{options}

━━━ DECISION RULES ━━━
• CONFIRMED  — the proposed classification is correct.
• NONE       — the turn has no patient health problem (nurse action announcement,
               normal finding, greeting, filler, or question with no confirmed finding).
• Better match — if a different option from the list fits better, provide it.

━━━ RESPONSE FORMAT ━━━
CONFIRMED

or

NONE

or

Domain: [Exact]
Problem: [Exact]
Signs/Symptoms: [Exact]
"""

VERIFY_INT_PROMPT = """You are a quality reviewer for Omaha System clinical coding.

A classifier proposed intervention label(s) for this home healthcare turn.
Your job: confirm they are correct, or correct them.

━━━ CONTEXT ━━━
{context}

━━━ TURN ━━━
{query}

━━━ PROPOSED INTERVENTIONS ━━━
{prediction}

━━━ AVAILABLE OPTIONS (top retrieved) ━━━
{options}

━━━ DECISION RULES ━━━
• CONFIRMED  — the proposed interventions are correct.
• NONE       — the turn has no clinical intervention (greeting, filler, purely social).
• Better match — list corrected interventions (up to 3) in the same format as below.

━━━ RESPONSE FORMAT ━━━
CONFIRMED

or

NONE

or

1. Category: [exact] | Target: [exact]
2. Category: [exact] | Target: [exact]
"""


def _verify_ss(query: str, ss_preds: list[dict],
               ss_docs: list[dict], ss_embeddings: np.ndarray,
               embed_model: SentenceTransformer, llm_name: str) -> list[dict]:
    """Run the SS verification agent on one row. Returns (possibly corrected) predictions."""
    if not ss_preds:
        return ss_preds  # Nothing to verify

    prediction_text = "\n".join(
        f"Domain: {p['domain']} | Problem: {p['problem']} | Signs/Symptoms: {p['ss']}"
        for p in ss_preds
    )
    retrieved = retrieve(query, ss_docs, ss_embeddings, embed_model)
    options_text = "\n".join(
        f"{i+1}. Domain: {d['domain']} | Problem: {d['problem']} | Signs/Symptoms: {d['ss']}"
        for i, d in enumerate(retrieved)
    )
    prompt = VERIFY_SS_PROMPT.format(
        query=query,
        prediction=prediction_text,
        options=options_text,
    )
    if LLM_CONFIGS[llm_name]["provider"] != "local":
        prompt = re.sub(r"<\|[^|>]+\|>", "", prompt).strip()

    raw = call_llm(prompt, llm_name)
    raw_stripped = raw.strip()

    if re.search(r"^\s*CONFIRMED\b", raw_stripped, re.IGNORECASE):
        return ss_preds
    if re.search(r"^\s*NONE\b", raw_stripped, re.IGNORECASE):
        return []
    # Otherwise parse as a new classification
    corrected = parse_ss_output(raw_stripped)
    return corrected if corrected else ss_preds  # fall back if parse fails


def _verify_int(query: str, context: str, int_preds: list[dict],
                int_docs: list[dict], int_embeddings: np.ndarray,
                embed_model: SentenceTransformer, llm_name: str) -> list[dict]:
    """Run the INT verification agent on one row. Returns (possibly corrected) predictions."""
    if not int_preds:
        return int_preds

    prediction_text = "\n".join(
        f"{i+1}. Category: {p['category']} | Target: {p['target']}"
        for i, p in enumerate(int_preds)
    )
    retrieved = retrieve(query, int_docs, int_embeddings, embed_model)
    options_text = "\n".join(
        f"{i+1}. Category: {d['category']} | Target: {d['target']}"
        for i, d in enumerate(retrieved)
    )
    prompt = VERIFY_INT_PROMPT.format(
        context=context,
        query=query,
        prediction=prediction_text,
        options=options_text,
    )
    if LLM_CONFIGS[llm_name]["provider"] != "local":
        prompt = re.sub(r"<\|[^|>]+\|>", "", prompt).strip()

    raw = call_llm(prompt, llm_name)
    raw_stripped = raw.strip()

    if re.search(r"^\s*CONFIRMED\b", raw_stripped, re.IGNORECASE):
        return int_preds
    if re.search(r"^\s*NONE\b", raw_stripped, re.IGNORECASE):
        return []
    corrected = parse_intervention_output(raw_stripped)
    return corrected if corrected else int_preds


def verify_sheet(
    df_out: pd.DataFrame,
    ss_docs: list[dict],   ss_embeddings: np.ndarray,
    int_docs: list[dict],  int_embeddings: np.ndarray,
    embed_model: SentenceTransformer,
    llm_name: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> pd.DataFrame:
    """
    Run a verification pass over every row that has at least one SS or INT
    prediction.  Returns an updated DataFrame with corrected Pred_* columns.
    """
    orig_k = TOP_K_RETRIEVAL
    # We access top_k via the retrieve() default arg, so temporarily patch it
    import omaha_sagemaker as _self
    _self.TOP_K_RETRIEVAL = top_k

    out = df_out.copy()

    for idx in range(len(out)):
        row   = out.iloc[idx]
        query = str(row.get("Conversation", "")).strip()
        if not query or query == "nan":
            continue

        context = get_context_text(df_out, idx)

        # Reconstruct current SS predictions
        ss_preds = []
        for j in range(1, 4):
            lbl  = str(row.get(f"Pred_SS_{j}", "")).strip()
            dom  = str(row.get(f"Pred_SS_{j}_domain",  "")).strip()
            prob = str(row.get(f"Pred_SS_{j}_problem", "")).strip()
            ss   = str(row.get(f"Pred_SS_{j}_ss",      "")).strip()
            if lbl and lbl != "nan":
                ss_preds.append({"label": lbl, "domain": dom,
                                 "problem": prob, "ss": ss})

        # Reconstruct current INT predictions
        int_preds = []
        for j in range(1, 4):
            lbl = str(row.get(f"Pred_I_{j}", "")).strip()
            cat = str(row.get(f"Pred_I_{j}_category", "")).strip()
            tgt = str(row.get(f"Pred_I_{j}_target",   "")).strip()
            if lbl and lbl != "nan":
                int_preds.append({"label": lbl, "category": cat, "target": tgt})

        # Verify SS
        if ss_preds:
            ss_preds = _verify_ss(query, ss_preds, ss_docs, ss_embeddings,
                                  embed_model, llm_name)

        # Verify INT
        if int_preds:
            int_preds = _verify_int(query, context, int_preds, int_docs,
                                    int_embeddings, embed_model, llm_name)

        # Write back SS
        for j in range(1, 4):
            p = ss_preds[j-1] if len(ss_preds) >= j else {}
            out.at[idx, f"Pred_SS_{j}"]         = p.get("label",   "")
            out.at[idx, f"Pred_SS_{j}_domain"]  = p.get("domain",  "")
            out.at[idx, f"Pred_SS_{j}_problem"] = p.get("problem", "")
            out.at[idx, f"Pred_SS_{j}_ss"]      = p.get("ss",      "")

        # Write back INT
        for j in range(1, 4):
            p = int_preds[j-1] if len(int_preds) >= j else {}
            out.at[idx, f"Pred_I_{j}"]          = p.get("label",    "")
            out.at[idx, f"Pred_I_{j}_category"] = p.get("category", "")
            out.at[idx, f"Pred_I_{j}_target"]   = p.get("target",   "")

    _self.TOP_K_RETRIEVAL = orig_k
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Lowercase, strip, collapse whitespace, remove common separators."""
    s = re.sub(r"[\s_\-/]+", " ", s.strip().lower())
    return s.strip()


def fuzzy_label_match(pred: str, true: str, threshold: int = FUZZY_THRESHOLD) -> bool:
    """
    True if two label strings are fuzzy-similar above threshold.

    Tries multiple comparisons to be robust against format differences:
    - Full label vs full label          (Problem_SS vs Problem_SS)
    - Last component vs last component  (SS vs SS, or Target vs Target)
    - Partial ratio (substring match)   (catches prefix/suffix differences)
    """
    if not pred or not true:
        return False

    p_full = _norm(pred)
    t_full = _norm(true)

    # 1. Full-label comparison
    if fuzz.ratio(p_full, t_full) >= threshold:
        return True

    # 2. Last component only (e.g. "Circulation_edema" → "edema")
    p_last = p_full.rsplit(" ", 1)[-1]
    t_last = t_full.rsplit(" ", 1)[-1]
    if len(p_last) >= 4 and len(t_last) >= 4:  # avoid trivial single-word noise
        if fuzz.ratio(p_last, t_last) >= threshold:
            return True

    # 3. Token set ratio (order-insensitive, handles extra words)
    if fuzz.token_set_ratio(p_full, t_full) >= threshold:
        return True

    return False


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


def compute_full_metrics(tp: int, fp: int, fn: int, tn: int) -> tuple:
    """
    Compute positive-class, negative-class (None), and macro-averaged P/R/F1.

    For the None class:  TP_none=TN, FP_none=FN, FN_none=FP
    Macro averages the two classes equally.

    Returns:
        (pos_p, pos_r, pos_f1,
         none_p, none_r, none_f1,
         macro_p, macro_r, macro_f1)
    """
    pos_p,  pos_r,  pos_f1  = compute_prf(tp, fp, fn)
    none_p, none_r, none_f1 = compute_prf(tn, fn, fp)   # swap FP↔FN for None class
    macro_p  = round((pos_p  + none_p)  / 2, 4)
    macro_r  = round((pos_r  + none_r)  / 2, 4)
    macro_f1 = round((pos_f1 + none_f1) / 2, 4)
    return pos_p, pos_r, pos_f1, none_p, none_r, none_f1, macro_p, macro_r, macro_f1


def evaluate_sheet(df_out: pd.DataFrame) -> dict:
    """
    Compute P/R/F1 for SS and Interventions from one processed sheet.

    Returns:
    - Combined label metrics (SS full-label, Int full-label)
    - Component metrics:
        SS:  domain, problem, ss (sign/symptom) separately
        Int: category, target separately
    """
    # ── Combined label counters ───────────────────────────────────────────────
    ss_tp = ss_fp = ss_fn = ss_tn = 0
    i_tp  = i_fp  = i_fn  = i_tn  = 0
    ss_relevant_rows  = 0
    int_relevant_rows = 0

    # ── SS component counters (problem / ss-only only — domain omitted because
    #    ground-truth labels are always "Problem_SS", never "Domain_Problem_SS") ──
    prob_tp = prob_fp = prob_fn = prob_tn = 0
    ss_only_tp = ss_only_fp = ss_only_fn = ss_only_tn = 0

    # ── Intervention component counters (category / target) ───────────────────
    cat_tp = cat_fp = cat_fn = cat_tn = 0
    tgt_tp = tgt_fp = tgt_fn = tgt_tn = 0

    # ── Sample GT labels for diagnosis (logged once) ──────────────────────────
    gt_sample_logged = False
    fn_diag_count    = 0          # log first few FN examples for diagnosis
    fp_diag_count    = 0          # log first few FP examples for diagnosis

    for _, row in df_out.iterrows():
        # ── Signs/Symptoms ───────────────────────────────────────────────────
        human_ss  = [str(row[f"OS_SS_{j}"]).strip()
                     for j in range(1, 4)
                     if f"OS_SS_{j}" in df_out.columns
                     and pd.notna(row[f"OS_SS_{j}"])
                     and str(row[f"OS_SS_{j}"]).strip() not in ("", "nan")]

        # Full labels (Problem_SS)
        pred_ss_labels = [str(row.get(f"Pred_SS_{j}", "")).strip()
                          for j in range(1, 4)
                          if str(row.get(f"Pred_SS_{j}", "")).strip() != ""]

        # Component lists for SS
        pred_domains   = [str(row.get(f"Pred_SS_{j}_domain",  "")).strip() for j in range(1, 4) if str(row.get(f"Pred_SS_{j}_domain",  "")).strip()]
        pred_problems  = [str(row.get(f"Pred_SS_{j}_problem", "")).strip() for j in range(1, 4) if str(row.get(f"Pred_SS_{j}_problem", "")).strip()]
        pred_ss_only   = [str(row.get(f"Pred_SS_{j}_ss",      "")).strip() for j in range(1, 4) if str(row.get(f"Pred_SS_{j}_ss",      "")).strip()]

        if not gt_sample_logged and human_ss:
            log.info(f"[DIAG] Sample OS_SS ground-truth labels: {human_ss}")
            log.info(f"[DIAG] Sample Pred_SS labels:            {pred_ss_labels}")
            gt_sample_logged = True

        if human_ss or pred_ss_labels:
            ss_relevant_rows += 1
            tp, fp, fn = row_level_match(pred_ss_labels, human_ss)
            ss_tp += tp; ss_fp += fp; ss_fn += fn

            # ── Diagnostic logging for FN and FP ────────────────────────────
            # FN: GT has label but model predicted nothing (or wrong label)
            if fn > 0 and fn_diag_count < 5:
                raw_out = str(row.get("LLM_SS_raw", "")).strip()
                turn    = str(row.get("Conversation", "")).strip()[:120]
                log.info(
                    f"[FN-SS #{fn_diag_count+1}] Turn: {turn!r}\n"
                    f"           GT:    {human_ss}\n"
                    f"           Pred:  {pred_ss_labels}\n"
                    f"           Raw:   {raw_out[:200]!r}"
                )
                fn_diag_count += 1
            # FP: model predicted something not in GT
            if fp > 0 and fp_diag_count < 3:
                raw_out = str(row.get("LLM_SS_raw", "")).strip()
                turn    = str(row.get("Conversation", "")).strip()[:120]
                log.info(
                    f"[FP-SS #{fp_diag_count+1}] Turn: {turn!r}\n"
                    f"           GT:    {human_ss}\n"
                    f"           Pred:  {pred_ss_labels}\n"
                    f"           Raw:   {raw_out[:200]!r}"
                )
                fp_diag_count += 1

            # Component metrics — GT format is always "Problem_SS" (2 parts):
            # first underscore-separated token = Problem, rest = SS
            human_problems = []
            human_ss_only  = []
            for h in human_ss:
                parts = [p.strip() for p in re.split(r"[_\|]+", h) if p.strip()]
                if len(parts) >= 2:
                    human_problems.append(parts[0])
                    human_ss_only.append("_".join(parts[1:]))
                elif len(parts) == 1:
                    human_ss_only.append(parts[0])

            if human_problems or pred_problems:
                tp2, fp2, fn2 = row_level_match(pred_problems, human_problems)
                prob_tp += tp2; prob_fp += fp2; prob_fn += fn2
            else:
                prob_tn += 1

            if human_ss_only or pred_ss_only:
                tp2, fp2, fn2 = row_level_match(pred_ss_only, human_ss_only)
                ss_only_tp += tp2; ss_only_fp += fp2; ss_only_fn += fn2
            else:
                ss_only_tn += 1

        else:
            ss_tn += 1        # both human and pred are NONE → true negative
            prob_tn += 1      # no SS labels → no problem component either
            ss_only_tn += 1   # no SS labels → no sign/symptom component either

        # ── Interventions ────────────────────────────────────────────────────
        human_int = [str(row[f"OS_I_{j}"]).strip()
                     for j in range(1, 4)
                     if f"OS_I_{j}" in df_out.columns
                     and pd.notna(row[f"OS_I_{j}"])
                     and str(row[f"OS_I_{j}"]).strip() not in ("", "nan")]

        pred_int_labels = [str(row.get(f"Pred_I_{j}", "")).strip()
                           for j in range(1, 4)
                           if str(row.get(f"Pred_I_{j}", "")).strip() != ""]

        pred_cats = [str(row.get(f"Pred_I_{j}_category", "")).strip() for j in range(1, 4) if str(row.get(f"Pred_I_{j}_category", "")).strip()]
        pred_tgts = [str(row.get(f"Pred_I_{j}_target",   "")).strip() for j in range(1, 4) if str(row.get(f"Pred_I_{j}_target",   "")).strip()]

        if human_int or pred_int_labels:
            int_relevant_rows += 1
            tp, fp, fn = row_level_match(pred_int_labels, human_int)
            i_tp += tp; i_fp += fp; i_fn += fn
        else:
            i_tn += 1       # both human and pred are NONE → true negative
            cat_tn += 1     # no INT labels → no category component either
            tgt_tn += 1     # no INT labels → no target component either

        if human_int or pred_int_labels:
            human_cats = []
            human_tgts = []
            for h in human_int:
                parts = [p.strip() for p in re.split(r"[_\|]+", h) if p.strip()]
                if len(parts) >= 2:
                    human_cats.append(parts[0])
                    human_tgts.append("_".join(parts[1:]))
                elif len(parts) == 1:
                    human_tgts.append(parts[0])

            if human_cats or pred_cats:
                tp2, fp2, fn2 = row_level_match(pred_cats, human_cats)
                cat_tp += tp2; cat_fp += fp2; cat_fn += fn2
            else:
                cat_tn += 1

            if human_tgts or pred_tgts:
                tp2, fp2, fn2 = row_level_match(pred_tgts, human_tgts)
                tgt_tp += tp2; tgt_fp += fp2; tgt_fn += fn2
            else:
                tgt_tn += 1

    # ── Positive / None / Macro P·R·F1 for every metric ─────────────────────
    (ss_p,   ss_r,   ss_f1,
     ss_np,  ss_nr,  ss_nf1,
     ss_mp,  ss_mr,  ss_mf1)  = compute_full_metrics(ss_tp,      ss_fp,      ss_fn,      ss_tn)

    (int_p,  int_r,  int_f1,
     int_np, int_nr, int_nf1,
     int_mp, int_mr, int_mf1) = compute_full_metrics(i_tp,       i_fp,       i_fn,       i_tn)

    (prob_p,    prob_r,    prob_f1,
     prob_np,   prob_nr,   prob_nf1,
     prob_mp,   prob_mr,   prob_mf1)  = compute_full_metrics(prob_tp,    prob_fp,    prob_fn,    prob_tn)

    (so_p,   so_r,   so_f1,
     so_np,  so_nr,  so_nf1,
     so_mp,  so_mr,  so_mf1)  = compute_full_metrics(ss_only_tp, ss_only_fp, ss_only_fn, ss_only_tn)

    (cat_p,  cat_r,  cat_f1,
     cat_np, cat_nr, cat_nf1,
     cat_mp, cat_mr, cat_mf1) = compute_full_metrics(cat_tp,     cat_fp,     cat_fn,     cat_tn)

    (tgt_p,  tgt_r,  tgt_f1,
     tgt_np, tgt_nr, tgt_nf1,
     tgt_mp, tgt_mr, tgt_mf1) = compute_full_metrics(tgt_tp,     tgt_fp,     tgt_fn,     tgt_tn)

    # Accuracy: (TP + TN) / (TP + FP + FN + TN)
    # Includes correctly-classified NONE rows as correct predictions.
    ss_total    = ss_tp + ss_fp + ss_fn + ss_tn
    int_total   = i_tp  + i_fp  + i_fn  + i_tn
    ss_accuracy  = round((ss_tp + ss_tn) / ss_total,  4) if ss_total  > 0 else 0.0
    int_accuracy = round((i_tp  + i_tn)  / int_total, 4) if int_total > 0 else 0.0

    return {
        # ── Signs / Symptoms — combined labels ───────────────────────────────
        # Positive class (backward-compatible keys)
        "ss_precision": ss_p,  "ss_recall": ss_r,  "ss_f1": ss_f1,
        "ss_tp": ss_tp, "ss_fp": ss_fp, "ss_fn": ss_fn, "ss_tn": ss_tn,
        "ss_accuracy": ss_accuracy,
        "ss_relevant_rows": ss_relevant_rows,
        # None class
        "ss_none_p": ss_np,  "ss_none_r": ss_nr,  "ss_none_f1": ss_nf1,
        # Macro
        "ss_macro_p": ss_mp, "ss_macro_r": ss_mr, "ss_macro_f1": ss_mf1,

        # ── Interventions — combined labels ──────────────────────────────────
        # Positive class (backward-compatible keys)
        "int_precision": int_p, "int_recall": int_r, "int_f1": int_f1,
        "int_tp": i_tp, "int_fp": i_fp, "int_fn": i_fn, "int_tn": i_tn,
        "int_accuracy": int_accuracy,
        "int_relevant_rows": int_relevant_rows,
        # None class
        "int_none_p": int_np, "int_none_r": int_nr, "int_none_f1": int_nf1,
        # Macro
        "int_macro_p": int_mp, "int_macro_r": int_mr, "int_macro_f1": int_mf1,

        # ── SS components ────────────────────────────────────────────────────
        # Problem (backward-compatible keys + None/Macro)
        "prob_precision": prob_p, "prob_recall": prob_r, "prob_f1": prob_f1,
        "prob_tp": prob_tp, "prob_fp": prob_fp, "prob_fn": prob_fn, "prob_tn": prob_tn,
        "prob_none_p": prob_np, "prob_none_r": prob_nr, "prob_none_f1": prob_nf1,
        "prob_macro_p": prob_mp, "prob_macro_r": prob_mr, "prob_macro_f1": prob_mf1,
        # Sign/Symptom only (backward-compatible keys + None/Macro)
        "ss_only_precision": so_p, "ss_only_recall": so_r, "ss_only_f1": so_f1,
        "ss_only_tp": ss_only_tp, "ss_only_fp": ss_only_fp, "ss_only_fn": ss_only_fn,
        "ss_only_tn": ss_only_tn,
        "ss_only_none_p": so_np, "ss_only_none_r": so_nr, "ss_only_none_f1": so_nf1,
        "ss_only_macro_p": so_mp, "ss_only_macro_r": so_mr, "ss_only_macro_f1": so_mf1,

        # ── Intervention components ───────────────────────────────────────────
        # Category (backward-compatible keys + None/Macro)
        "cat_precision": cat_p, "cat_recall": cat_r, "cat_f1": cat_f1,
        "cat_tp": cat_tp, "cat_fp": cat_fp, "cat_fn": cat_fn, "cat_tn": cat_tn,
        "cat_none_p": cat_np, "cat_none_r": cat_nr, "cat_none_f1": cat_nf1,
        "cat_macro_p": cat_mp, "cat_macro_r": cat_mr, "cat_macro_f1": cat_mf1,
        # Target (backward-compatible keys + None/Macro)
        "tgt_precision": tgt_p, "tgt_recall": tgt_r, "tgt_f1": tgt_f1,
        "tgt_tp": tgt_tp, "tgt_fp": tgt_fp, "tgt_fn": tgt_fn, "tgt_tn": tgt_tn,
        "tgt_none_p": tgt_np, "tgt_none_r": tgt_nr, "tgt_none_f1": tgt_nf1,
        "tgt_macro_p": tgt_mp, "tgt_macro_r": tgt_mr, "tgt_macro_f1": tgt_mf1,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SHARED INDEX BUILDER
# Builds the RAG index and loads annotation data ONCE, shared across all models.
# ══════════════════════════════════════════════════════════════════════════════

def build_shared_resources() -> dict:
    """
    Load Omaha system, build embeddings, and read all annotation sheets.
    Returns a dict of shared resources passed to run_inference().
    Called once per session, even when comparing multiple models.
    """
    log.info("Building shared RAG index (runs once for all models) …")

    ss_df, int_df = load_omaha_system(S3_BUCKET, OMAHA_KEY)
    ss_docs       = build_ss_documents(ss_df)
    int_docs      = build_intervention_documents(int_df)

    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model    = SentenceTransformer(EMBEDDING_MODEL)
    ss_embeddings  = build_index(ss_docs,  embed_model)
    int_embeddings = build_index(int_docs, embed_model)

    log.info("Reading annotation data from S3 …")
    annotation_sheets = s3_read_excel(S3_BUCKET, ANNOTATION_KEY)
    # Standardise column names once
    annotation_sheets = {
        name: df.rename(columns=str.strip).fillna("")
        for name, df in annotation_sheets.items()
    }
    # Quick-test mode: limit to first N sheets
    if MAX_SHEETS:
        annotation_sheets = dict(list(annotation_sheets.items())[:MAX_SHEETS])
        log.info(f"Quick-test mode: using {len(annotation_sheets)} of the available sheets")
    log.info(f"Shared index ready. Annotation sheets: {len(annotation_sheets)}")

    return {
        "ss_docs":          ss_docs,
        "ss_embeddings":    ss_embeddings,
        "int_docs":         int_docs,
        "int_embeddings":   int_embeddings,
        "embed_model":      embed_model,
        "annotation_sheets": annotation_sheets,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — RUN INFERENCE FOR ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(llm_name: str, resources: dict) -> dict:
    """
    Run the full inference + evaluation pipeline for one LLM, using
    pre-built shared resources (no repeated S3 reads or index rebuilds).

    Returns the summary dict and saves results to S3.
    """
    # ── Pre-flight: fail fast before touching any data ─────────────────────────
    config = LLM_CONFIGS[llm_name]
    if config["provider"] == "local":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Skipping '{llm_name}': no CUDA GPU detected. "
                "Switch your SageMaker instance to ml.g5.xlarge (or any GPU type)."
            )
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        if "llama" in config["model"].lower() and not HF_TOKEN:
            log.warning(
                "HF_TOKEN is not set. Llama-3 is a gated model — "
                "you must set HF_TOKEN or the download will fail with 401."
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"\n{'='*60}")
    log.info(f"Running inference  |  LLM: {llm_name}")
    log.info(f"{'='*60}")

    ss_docs          = resources["ss_docs"]
    ss_embeddings    = resources["ss_embeddings"]
    int_docs         = resources["int_docs"]
    int_embeddings   = resources["int_embeddings"]
    embed_model      = resources["embed_model"]
    annotation_sheets = resources["annotation_sheets"]

    output_sheets = {}
    sheet_metrics = []

    # Combined label counters
    global_ss_tp = global_ss_fp = global_ss_fn = global_ss_tn = 0
    global_i_tp  = global_i_fp  = global_i_fn  = global_i_tn  = 0
    # SS component counters (domain omitted — GT is always Problem_SS)
    global_prob_tp  = global_prob_fp  = global_prob_fn  = 0
    global_ssonly_tp= global_ssonly_fp= global_ssonly_fn= 0
    # Intervention component counters
    global_cat_tp   = global_cat_fp   = global_cat_fn   = 0
    global_tgt_tp   = global_tgt_fp   = global_tgt_fn   = 0

    _ZERO_METRICS = {k: 0 for k in [
        "ss_precision","ss_recall","ss_f1","ss_accuracy","ss_tp","ss_fp","ss_fn","ss_tn","ss_relevant_rows",
        "int_precision","int_recall","int_f1","int_accuracy","int_tp","int_fp","int_fn","int_tn","int_relevant_rows",
        "prob_precision","prob_recall","prob_f1","prob_tp","prob_fp","prob_fn",
        "ss_only_precision","ss_only_recall","ss_only_f1","ss_only_tp","ss_only_fp","ss_only_fn",
        "cat_precision","cat_recall","cat_f1","cat_tp","cat_fp","cat_fn",
        "tgt_precision","tgt_recall","tgt_f1","tgt_tp","tgt_fp","tgt_fn",
    ]}

    for sheet_name, conv_df in tqdm(annotation_sheets.items(),
                                    desc=f"[{llm_name}] sheets"):
        try:
            df_out  = process_sheet(conv_df, ss_docs, ss_embeddings,
                                    int_docs, int_embeddings,
                                    embed_model, llm_name)
            metrics = evaluate_sheet(df_out)
        except Exception as e:
            log.error(f"Sheet '{sheet_name}' failed: {e}")
            df_out  = conv_df.copy()
            metrics = dict(_ZERO_METRICS)

        metrics["sheet"] = sheet_name
        sheet_metrics.append(metrics)

        global_ss_tp += metrics["ss_tp"];  global_ss_fp += metrics["ss_fp"];  global_ss_fn += metrics["ss_fn"];  global_ss_tn += metrics["ss_tn"]
        global_i_tp  += metrics["int_tp"]; global_i_fp  += metrics["int_fp"]; global_i_fn  += metrics["int_fn"]; global_i_tn  += metrics["int_tn"]
        global_prob_tp   += metrics["prob_tp"];    global_prob_fp   += metrics["prob_fp"];    global_prob_fn   += metrics["prob_fn"]
        global_ssonly_tp += metrics["ss_only_tp"]; global_ssonly_fp += metrics["ss_only_fp"]; global_ssonly_fn += metrics["ss_only_fn"]
        global_cat_tp    += metrics["cat_tp"];     global_cat_fp    += metrics["cat_fp"];     global_cat_fn    += metrics["cat_fn"]
        global_tgt_tp    += metrics["tgt_tp"];     global_tgt_fp    += metrics["tgt_fp"];     global_tgt_fn    += metrics["tgt_fn"]

        safe = re.sub(r"[\\/*?:\[\]]", "_", sheet_name)[:31]
        output_sheets[safe] = df_out

        log.info(
            f"  SS  P={metrics['ss_precision']:.3f} R={metrics['ss_recall']:.3f} F1={metrics['ss_f1']:.3f}"
            f"  (ss_only F1={metrics['ss_only_f1']:.3f})  |  "
            f"Int P={metrics['int_precision']:.3f} R={metrics['int_recall']:.3f} F1={metrics['int_f1']:.3f}"
            f"  (tgt F1={metrics['tgt_f1']:.3f})"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    gss_p,    gss_r,    gss_f1    = compute_prf(global_ss_tp,     global_ss_fp,     global_ss_fn)
    gi_p,     gi_r,     gi_f1     = compute_prf(global_i_tp,      global_i_fp,      global_i_fn)
    gprob_p,  gprob_r,  gprob_f1  = compute_prf(global_prob_tp,   global_prob_fp,   global_prob_fn)
    gssonly_p,gssonly_r,gssonly_f1= compute_prf(global_ssonly_tp, global_ssonly_fp, global_ssonly_fn)
    gcat_p,   gcat_r,   gcat_f1   = compute_prf(global_cat_tp,    global_cat_fp,    global_cat_fn)
    gtgt_p,   gtgt_r,   gtgt_f1   = compute_prf(global_tgt_tp,    global_tgt_fp,    global_tgt_fn)

    ss_grand_total  = global_ss_tp + global_ss_fp + global_ss_fn + global_ss_tn
    int_grand_total = global_i_tp  + global_i_fp  + global_i_fn  + global_i_tn
    gss_accuracy  = round((global_ss_tp + global_ss_tn) / ss_grand_total,  4) if ss_grand_total  > 0 else 0.0
    gint_accuracy = round((global_i_tp  + global_i_tn)  / int_grand_total, 4) if int_grand_total > 0 else 0.0

    def _macro(key): return np.mean([m.get(key, 0) for m in sheet_metrics])

    summary = {
        "llm": llm_name, "embedding": EMBEDDING_MODEL, "timestamp": timestamp,
        "context_window": CONTEXT_WINDOW, "top_k": TOP_K_RETRIEVAL,
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "signs_symptoms": {
            "micro":    {"precision": gss_p,  "recall": gss_r,  "f1": gss_f1,
                         "accuracy": gss_accuracy,
                         "tp": global_ss_tp, "fp": global_ss_fp, "fn": global_ss_fn, "tn": global_ss_tn},
            "macro":    {"f1": round(_macro("ss_f1"), 4)},
            "problem":  {"micro_p": gprob_p,   "micro_r": gprob_r,   "micro_f1": gprob_f1,
                         "tp": global_prob_tp,  "fp": global_prob_fp,  "fn": global_prob_fn},
            "ss_only":  {"micro_p": gssonly_p, "micro_r": gssonly_r, "micro_f1": gssonly_f1,
                         "tp": global_ssonly_tp,"fp": global_ssonly_fp,"fn": global_ssonly_fn},
        },
        "interventions": {
            "micro":    {"precision": gi_p,   "recall": gi_r,   "f1": gi_f1,
                         "accuracy": gint_accuracy,
                         "tp": global_i_tp,  "fp": global_i_fp,  "fn": global_i_fn,  "tn": global_i_tn},
            "macro":    {"f1": round(_macro("int_f1"), 4)},
            "category": {"micro_p": gcat_p, "micro_r": gcat_r, "micro_f1": gcat_f1,
                         "tp": global_cat_tp, "fp": global_cat_fp, "fn": global_cat_fn},
            "target":   {"micro_p": gtgt_p, "micro_r": gtgt_r, "micro_f1": gtgt_f1,
                         "tp": global_tgt_tp, "fp": global_tgt_fp, "fn": global_tgt_fn},
        },
        "per_sheet": sheet_metrics,
    }

    # ── Build SUMMARY sheet ────────────────────────────────────────────────────
    rows = []
    for m in sheet_metrics:
        rows.append({
            "Sheet":           m["sheet"],
            # Combined labels
            "SS_Precision":    m["ss_precision"],   "SS_Recall":    m["ss_recall"],    "SS_F1":    m["ss_f1"],
            "SS_Accuracy": m["ss_accuracy"],
            "SS_TP": m["ss_tp"], "SS_FP": m["ss_fp"], "SS_FN": m["ss_fn"], "SS_TN": m["ss_tn"],
            # SS components
            "Problem_F1":      m["prob_f1"],
            "SS_Only_F1":      m["ss_only_f1"],
            # Intervention combined
            "Int_Precision":   m["int_precision"],  "Int_Recall":   m["int_recall"],   "Int_F1":   m["int_f1"],
            "Int_Accuracy": m["int_accuracy"],
            "Int_TP": m["int_tp"], "Int_FP": m["int_fp"], "Int_FN": m["int_fn"], "Int_TN": m["int_tn"],
            # Intervention components
            "Category_F1":     m["cat_f1"],
            "Target_F1":       m["tgt_f1"],
        })
    rows.append({
        "Sheet": "MICRO_AGGREGATE",
        "SS_Precision": gss_p,    "SS_Recall": gss_r,    "SS_F1": gss_f1,
        "SS_Accuracy": gss_accuracy,
        "SS_TP": global_ss_tp, "SS_FP": global_ss_fp, "SS_FN": global_ss_fn, "SS_TN": global_ss_tn,
        "Problem_F1": gprob_f1, "SS_Only_F1": gssonly_f1,
        "Int_Precision": gi_p, "Int_Recall": gi_r, "Int_F1": gi_f1,
        "Int_Accuracy": gint_accuracy,
        "Int_TP": global_i_tp, "Int_FP": global_i_fp, "Int_FN": global_i_fn, "Int_TN": global_i_tn,
        "Category_F1": gcat_f1, "Target_F1": gtgt_f1,
    })
    rows.append({
        "Sheet": "MACRO_AVERAGE",
        "SS_F1": round(_macro("ss_f1"),4),
        "SS_Accuracy": round(_macro("ss_accuracy"),4),
        "Problem_F1": round(_macro("prob_f1"),4),
        "SS_Only_F1": round(_macro("ss_only_f1"),4),
        "Int_F1": round(_macro("int_f1"),4),
        "Int_Accuracy": round(_macro("int_accuracy"),4),
        "Category_F1": round(_macro("cat_f1"),4),
        "Target_F1": round(_macro("tgt_f1"),4),
    })
    output_sheets["SUMMARY"] = pd.DataFrame(rows)

    # ── Save per-model results ─────────────────────────────────────────────────
    safe_name   = re.sub(r"[^a-zA-Z0-9_-]", "_", llm_name)
    results_key = f"{OUTPUT_PREFIX}{safe_name}_{timestamp}_results.xlsx"
    s3_write_excel(output_sheets, S3_BUCKET, results_key)

    # ── Print per-model report ─────────────────────────────────────────────────
    W = 30
    print(f"\n{'='*70}")
    print(f"RESULTS  |  {llm_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<{W}} {'Micro P':>8} {'Micro R':>8} {'Micro F1':>9} {'Accuracy':>9}")
    print(f"{'-'*75}")
    for lbl, p, r, f, acc in [
        ("SS (combined label)",    gss_p,    gss_r,    gss_f1,    gss_accuracy),
        ("  └ Problem only",       gprob_p,  gprob_r,  gprob_f1,  float("nan")),
        ("  └ Sign/Symptom only",  gssonly_p,gssonly_r,gssonly_f1,float("nan")),
        ("Intervention (combined)",gi_p,     gi_r,     gi_f1,     gint_accuracy),
        ("  └ Category only",      gcat_p,   gcat_r,   gcat_f1,   float("nan")),
        ("  └ Target only",        gtgt_p,   gtgt_r,   gtgt_f1,   float("nan")),
    ]:
        acc_str = f"{acc:>9.4f}" if acc == acc else "         "  # nan check
        print(f"{lbl:<{W}} {p:>8.4f} {r:>8.4f} {f:>9.4f}{acc_str}")
    print(f"{'-'*75}")
    print(f"Macro SS F1:           {round(_macro('ss_f1'),4):.4f}   "
          f"(ss_only: {round(_macro('ss_only_f1'),4):.4f})")
    print(f"Macro Int F1:          {round(_macro('int_f1'),4):.4f}   "
          f"(target:  {round(_macro('tgt_f1'),4):.4f})")
    print(f"TP/FP/FN/TN (SS):      {global_ss_tp}/{global_ss_fp}/{global_ss_fn}/{global_ss_tn}  Accuracy={gss_accuracy:.4f}")
    print(f"TP/FP/FN/TN (Int):     {global_i_tp}/{global_i_fp}/{global_i_fn}/{global_i_tn}  Accuracy={gint_accuracy:.4f}")
    print(f"{'='*70}")
    print(f"Saved: s3://{S3_BUCKET}/{results_key}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MULTI-MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(models_to_compare: list[str] = None):
    """
    Benchmark multiple models on the same data.

    - RAG index and annotation data are built ONCE (not per model).
    - Local models are unloaded from GPU before the next one loads.
    - Produces per-model Excel results + a side-by-side comparison Excel.
    """
    global _local_pipeline
    if models_to_compare is None:
        models_to_compare = MODELS_TO_COMPARE

    # ── Build shared resources once ────────────────────────────────────────────
    resources     = build_shared_resources()
    all_summaries = {}

    for model_name in models_to_compare:
        # Free previous local model from GPU before loading the next
        if _local_pipeline is not None:
            log.info(f"Unloading previous local model to free GPU memory …")
            del _local_pipeline
            _local_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            summary = run_inference(model_name, resources)
            all_summaries[model_name] = summary
        except Exception as e:
            log.error(f"Model '{model_name}' failed: {e}")
            all_summaries[model_name] = {"error": str(e)}

    # ── Side-by-side comparison table (console) ────────────────────────────────
    W = 22
    print(f"\n{'='*80}\nMODEL COMPARISON — Micro F1 by component\n{'='*80}")
    print(f"{'Model':<{W}}  {'SS_F1':>7} {'Prob_F1':>8} {'SSOnly_F1':>10}"
          f"  {'Int_F1':>7} {'Cat_F1':>7} {'Tgt_F1':>7}")
    print("-" * 80)

    for mn, s in all_summaries.items():
        if "error" in s:
            print(f"{mn:<{W}}  ERROR: {s['error']}")
            continue
        ss  = s.get("signs_symptoms", {})
        iv  = s.get("interventions",  {})
        print(
            f"{mn:<{W}}"
            f"  {ss.get('micro',{}).get('f1',0):>7.4f}"
            f" {ss.get('problem',{}).get('micro_f1',0):>8.4f}"
            f" {ss.get('ss_only',{}).get('micro_f1',0):>10.4f}"
            f"  {iv.get('micro',{}).get('f1',0):>7.4f}"
            f" {iv.get('category',{}).get('micro_f1',0):>7.4f}"
            f" {iv.get('target',{}).get('micro_f1',0):>7.4f}"
        )
    print("=" * 80)

    # ── Side-by-side comparison Excel ─────────────────────────────────────────
    comp_rows      = []
    aggregate_rows = []

    all_sheet_names = []
    for s in all_summaries.values():
        if "per_sheet" in s:
            all_sheet_names = [m["sheet"] for m in s["per_sheet"]]
            break

    for sheet_name in all_sheet_names:
        row = {"Sheet": sheet_name}
        for mn, s in all_summaries.items():
            if "per_sheet" not in s:
                continue
            m = next((x for x in s["per_sheet"] if x["sheet"] == sheet_name), None)
            if m:
                row[f"{mn}_SS_F1"]      = m.get("ss_f1", 0)
                row[f"{mn}_SSOnly_F1"]  = m.get("ss_only_f1", 0)
                row[f"{mn}_Int_F1"]     = m.get("int_f1", 0)
                row[f"{mn}_Target_F1"]  = m.get("tgt_f1", 0)
        comp_rows.append(row)

    for mn, s in all_summaries.items():
        if "error" in s:
            aggregate_rows.append({"Model": mn, "Error": s["error"]})
            continue
        ss_m  = s.get("signs_symptoms", {}).get("micro",    {})
        ss_M  = s.get("signs_symptoms", {}).get("macro",    {})
        ss_p  = s.get("signs_symptoms", {}).get("problem",  {})
        ss_so = s.get("signs_symptoms", {}).get("ss_only",  {})
        iv_m  = s.get("interventions",  {}).get("micro",    {})
        iv_M  = s.get("interventions",  {}).get("macro",    {})
        iv_c  = s.get("interventions",  {}).get("category", {})
        iv_t  = s.get("interventions",  {}).get("target",   {})
        aggregate_rows.append({
            "Model":              mn,
            "SS_Micro_P":         ss_m.get("precision", 0),
            "SS_Micro_R":         ss_m.get("recall",    0),
            "SS_Micro_F1":        ss_m.get("f1",        0),
            "SS_Macro_F1":        ss_M.get("f1",        0),
            "Problem_Micro_F1":   ss_p.get("micro_f1",  0),
            "SSOnly_Micro_F1":    ss_so.get("micro_f1", 0),
            "SS_TP":  ss_m.get("tp", 0), "SS_FP":  ss_m.get("fp", 0), "SS_FN":  ss_m.get("fn", 0), "SS_TN": ss_m.get("tn", 0),
            "Int_Micro_P":        iv_m.get("precision", 0),
            "Int_Micro_R":        iv_m.get("recall",    0),
            "Int_Micro_F1":       iv_m.get("f1",        0),
            "Int_Macro_F1":       iv_M.get("f1",        0),
            "Category_Micro_F1":  iv_c.get("micro_f1",  0),
            "Target_Micro_F1":    iv_t.get("micro_f1",  0),
            "Int_TP": iv_m.get("tp", 0), "Int_FP": iv_m.get("fp", 0), "Int_FN": iv_m.get("fn", 0), "Int_TN": iv_m.get("tn", 0),
        })

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_key = f"{OUTPUT_PREFIX}model_comparison_{ts}.xlsx"
    s3_write_excel(
        {"Aggregate": pd.DataFrame(aggregate_rows),
         "Per_Sheet_F1": pd.DataFrame(comp_rows)},
        S3_BUCKET, comp_key,
    )

    print(f"\nComparison Excel: s3://{S3_BUCKET}/{comp_key}")
    return all_summaries


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if COMPARE_ALL_MODELS:
        compare_models()
    else:
        resources = build_shared_resources()
        run_inference(llm_name=ACTIVE_LLM, resources=resources)
