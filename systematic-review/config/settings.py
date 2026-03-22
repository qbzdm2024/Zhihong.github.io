"""
Central configuration for the Systematic Review Automation System.
All model choices, thresholds, and paths are defined here.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class ModelConfig(BaseSettings):
    """OpenAI model assignments per task. Configurable via environment or UI."""

    # API
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")

    # Model assignments (can be changed per task)
    model_title_screening: str = Field(default="gpt-4o-mini", env="MODEL_TITLE_SCREENING")
    model_fulltext_screening: str = Field(default="gpt-5-mini", env="MODEL_FULLTEXT_SCREENING")
    model_extraction: str = Field(default="gpt-5", env="MODEL_EXTRACTION")
    model_qa_assessment: str = Field(default="gpt-4o", env="MODEL_QA_ASSESSMENT")
    model_synthesis: str = Field(default="gpt-4o", env="MODEL_SYNTHESIS")

    # Second agent (for multi-agent verification)
    model_agent2_screening: str = Field(default="gpt-5", env="MODEL_AGENT2_SCREENING")
    model_agent2_extraction: str = Field(default="gpt-4o-mini", env="MODEL_AGENT2_EXTRACTION")

    # Phase 1 data extraction models
    # model_phase1_extraction: GPT-5 primary open extraction from methods + results
    # model_phase1_verification: smaller model for evidence verification (gpt-4o-mini)
    model_phase1_extraction: str = Field(default="gpt-5", env="MODEL_PHASE1_EXTRACTION")
    model_phase1_verification: str = Field(default="gpt-4.1-mini", env="MODEL_PHASE1_VERIFICATION")
    phase1_output_dir: str = Field(default="data/extracted/phase1", env="PHASE1_OUTPUT_DIR")

    # Decision thresholds
    confidence_threshold: float = Field(default=0.80, env="CONFIDENCE_THRESHOLD")
    agreement_required: bool = Field(default=True, env="AGREEMENT_REQUIRED")

    # Concurrency: number of papers processed in parallel during screening/extraction
    screening_workers: int = Field(default=5, env="SCREENING_WORKERS")

    # Paths
    data_dir: str = Field(default="data", env="DATA_DIR")
    raw_dir: str = Field(default="data/raw", env="RAW_DIR")
    deduped_dir: str = Field(default="data/deduped", env="DEDUPED_DIR")
    screened_dir: str = Field(default="data/screened", env="SCREENED_DIR")
    extracted_dir: str = Field(default="data/extracted", env="EXTRACTED_DIR")
    output_dir: str = Field(default="data/output", env="OUTPUT_DIR")
    pdf_dir: str = Field(default="data/pdfs", env="PDF_DIR")

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton
settings = ModelConfig()


# Decision categories
class Decision:
    INCLUDE = "Included"
    EXCLUDE = "Excluded"
    UNCERTAIN = "Needs Human Verification"
    FULL_TEXT_NEEDED = "Full Text Needed"


# Exclusion codes
EXCLUSION_CODES = {
    "EC1": "LLM used only for non-analytic tasks (transcription, translation, summarization, writing support)",
    "EC2": "No empirical application (framework proposal, theoretical, no results)",
    "EC3": "No qualitative results reported (only quantitative metrics)",
    "EC4": "Restricted publication type (review, editorial, commentary, abstract-only, protocol)",
    "EC5": "Superficial extraction only (keyword/NER/metadata tagging without qualitative interpretation)",
    "EC6": "Non-primary qualitative data (structured/numeric data only)",
    "EC7": "Outside date range (before 2023 or after 2026)",
    "EC8": "Not in English",
    "EC9": "Duplicate",
}

# Round-2 full-text screening exclusion codes (refined criteria for evaluation requirement)
ROUND2_EXCLUSION_CODES = {
    "R2-EC1": "Non-empirical or ineligible publication type (review, editorial, theoretical, framework without evaluation, lecture notes, author correction)",
    "R2-EC2": "No real-world qualitative data (only simulated, synthetic, or hypothetical data used)",
    "R2-EC3": "LLM not used for qualitative analysis (no coding, thematic analysis, content analysis, or theme extraction; analysis conducted entirely by humans)",
    "R2-EC4": "Low-level or non-qualitative NLP task (sentiment analysis, text mining, information extraction, classification, tagging, or prediction without open coding/thematic synthesis)",
    "R2-EC5": "Non-analytic or auxiliary LLM use only (writing/editing/summarization, chatbot responses, conducting interviews without analysis, generating plans or interventions)",
    "R2-EC6": "Focus on LLM evaluation, perception, or interaction rather than analysis (user perceptions, human-AI interaction, chatbot experience, education/learning, system usability, consistency/bias/ethical reasoning evaluation without thematic synthesis)",
    "R2-EC7": "Insufficient analytical depth — coding only without theme development, categorization, or interpretive synthesis",
    "R2-EC8": "Lack of evaluation of LLM outputs (no validation, comparison, or assessment; unclear whether outputs are from LLMs or humans; themes generated without quality evaluation)",
    "R2-EC9": "Methodological unclearity (unclear which LLM used, unclear how LLMs applied, unclear analytic workflow)",
    "R2-EC10": "Platform/tool-only study (introduces platform using LLMs, reports only user feedback or usability, does not evaluate analytic performance)",
    "R2-EC11": "Unclear eligibility (insufficient information to determine whether or how LLMs contributed to qualitative analysis)",
}

# Inclusion criteria codes
INCLUSION_CODES = {
    "IC1": "LLM tool explicitly named and described",
    "IC2": "LLM performs or supports qualitative analytic task",
    "IC3": "Empirical application on real data with reported results",
    "IC4": "Published 2023–2026",
    "IC5": "Published in English",
}

# Analytic tasks taxonomy
ANALYTIC_TASKS = [
    "Inductive coding",
    "Deductive coding",
    "Codebook development",
    "Codebook application",
    "Thematic analysis",
    "Content analysis",
    "Grounded theory coding",
    "Framework analysis",
    "Narrative analysis",
    "Discourse analysis",
    "Interpretive phenomenological analysis",
    "Other",
]

# Workflow structures
WORKFLOW_STRUCTURES = [
    "Human-led (AI assists)",
    "Human-in-the-loop",
    "AI-led (human verifies)",
    "Fully automated",
    "Not specified",
]

# Domains
DOMAINS = [
    "Healthcare",
    "Education",
    "HCI",
    "Psychology",
    "Sociology",
    "Computer Science",
    "Social Science",
    "Other",
]

# Data types
DATA_TYPES = [
    "Interviews",
    "Focus groups",
    "Documents",
    "Social media",
    "Survey responses",
    "Field notes",
    "Mixed",
    "Other",
]
