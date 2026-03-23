"""
Pydantic data models for all pipeline stages.
These models enforce structure and enable full traceability.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import uuid


class DecisionLabel(str, Enum):
    INCLUDE = "Included"
    EXCLUDE = "Excluded"
    UNCERTAIN = "Needs Human Verification"
    FULL_TEXT_NEEDED = "Full Text Needed"


class PipelineStage(str, Enum):
    IMPORT = "import"
    DEDUP = "deduplication"
    TITLE_SCREENING = "title_abstract_screening"
    FULLTEXT_SCREENING = "fulltext_screening"
    SECOND_FULLTEXT_SCREENING = "second_fulltext_screening"
    EXTRACTION = "data_extraction"
    QA = "quality_assessment"
    SYNTHESIS = "synthesis"


class AgentDecision(BaseModel):
    """Single agent's decision at any screening stage."""
    agent_id: str  # e.g., "agent1_gpt-4o-mini"
    model_used: str
    decision: DecisionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    exclusion_code: Optional[str] = None  # EC1–EC9
    flagged_criteria: List[str] = Field(default_factory=list)  # which IC/EC triggered
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScreeningResult(BaseModel):
    """Final screening result after multi-agent comparison."""
    stage: PipelineStage
    final_decision: DecisionLabel
    agent1: AgentDecision
    agent2: AgentDecision
    agents_agree: bool
    consensus_confidence: float
    human_verified: bool = False
    human_decision: Optional[DecisionLabel] = None
    human_rationale: Optional[str] = None
    human_reviewer: Optional[str] = None
    human_timestamp: Optional[datetime] = None


class RawRecord(BaseModel):
    """Raw bibliographic record as imported from search database."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_db: str  # PubMed, Scopus, etc.
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    journal_venue: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    url: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)
    import_timestamp: datetime = Field(default_factory=datetime.utcnow)


class DedupRecord(RawRecord):
    """Record after deduplication."""
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None  # record_id of canonical
    dedup_method: Optional[str] = None  # "doi", "title_fuzzy", "manual"
    dedup_confidence: Optional[float] = None


class ScreenedRecord(DedupRecord):
    """Record after title/abstract screening."""
    title_screening: Optional[ScreeningResult] = None
    second_pass_screening: Optional[ScreeningResult] = None  # stricter second-pass AI screen
    fulltext_screening: Optional[ScreeningResult] = None
    second_fulltext_screening: Optional[ScreeningResult] = None  # round-2 refined criteria screen
    pdf_path: Optional[str] = None
    fulltext_available: bool = False
    current_stage: PipelineStage = PipelineStage.TITLE_SCREENING
    current_decision: Optional[DecisionLabel] = None


class QAScore(BaseModel):
    """Quality assessment checklist scores."""
    qa1_llm_identified: int = Field(ge=0, le=1, description="LLM clearly identified (name + version)")
    qa2_prompts_described: int = Field(ge=0, le=1, description="Prompts or prompt strategy described")
    qa3_process_described: int = Field(ge=0, le=1, description="Analysis process described step-by-step")
    qa4_human_role_defined: int = Field(ge=0, le=1, description="Human role clearly defined")
    qa5_validation_performed: int = Field(ge=0, le=1, description="Validation or quality check performed")
    qa6_results_detailed: int = Field(ge=0, le=1, description="Results reported with sufficient detail")
    qa7_limitations_acknowledged: int = Field(ge=0, le=1, description="Limitations acknowledged")
    qa8_data_adequate: int = Field(ge=0, le=1, description="Data description adequate")
    qa9_reproducibility: int = Field(ge=0, le=1, description="Reproducibility materials available")
    qa10_ethics: int = Field(ge=0, le=1, description="Ethical considerations addressed")

    @property
    def total_score(self) -> int:
        return sum([
            self.qa1_llm_identified, self.qa2_prompts_described,
            self.qa3_process_described, self.qa4_human_role_defined,
            self.qa5_validation_performed, self.qa6_results_detailed,
            self.qa7_limitations_acknowledged, self.qa8_data_adequate,
            self.qa9_reproducibility, self.qa10_ethics
        ])


class ExtractionResult(BaseModel):
    """Extracted data fields from a single study."""
    # Study characteristics
    study_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    journal_venue: Optional[str] = None
    doi: Optional[str] = None
    country: Optional[str] = None
    discipline: Optional[str] = None
    study_aim: Optional[str] = None

    # Data and context
    data_type: Optional[str] = None
    sample_size: Optional[str] = None
    corpus_size: Optional[str] = None
    domain: Optional[str] = None
    data_language: Optional[str] = None

    # LLM characteristics
    model_name: Optional[str] = None
    model_type: Optional[str] = None  # proprietary/open-source
    model_provider: Optional[str] = None
    prompting_strategy: Optional[str] = None
    prompt_provided: Optional[bool] = None
    fine_tuned: Optional[bool] = None
    rag_used: Optional[bool] = None
    temperature: Optional[float] = None

    # Qualitative analysis use
    analytic_task: Optional[List[str]] = None
    analysis_stage: Optional[str] = None
    workflow_structure: Optional[str] = None
    pipeline_type: Optional[str] = None
    human_oversight: Optional[str] = None

    # Methodological framework
    qualitative_approach: Optional[str] = None
    formal_methodology: Optional[bool] = None
    codebook_development: Optional[str] = None
    epistemological_stance: Optional[str] = None

    # Evaluation and validation
    human_comparison: Optional[bool] = None
    agreement_method: Optional[str] = None
    agreement_score: Optional[str] = None
    quantitative_metrics: Optional[str] = None
    qualitative_validation: Optional[bool] = None
    audit_trail: Optional[bool] = None
    reflexivity: Optional[bool] = None

    # Outcomes
    key_findings: Optional[str] = None
    strengths_reported: Optional[str] = None
    limitations_reported: Optional[str] = None
    ethical_considerations: Optional[str] = None
    reproducibility_score: Optional[int] = None  # 1–4

    # Extraction metadata
    not_reported_fields: List[str] = Field(default_factory=list)
    uncertain_fields: List[str] = Field(default_factory=list)
    extraction_notes: Optional[str] = None


class ExtractedRecord(BaseModel):
    """Full extracted record with both agent outputs and final result."""
    record_id: str
    study_id: str
    extraction_agent1: Optional[ExtractionResult] = None
    extraction_agent2: Optional[ExtractionResult] = None
    extraction_final: Optional[ExtractionResult] = None
    qa_score: Optional[QAScore] = None
    qa_agent1: Optional[QAScore] = None
    qa_agent2: Optional[QAScore] = None
    agents_agree_extraction: bool = False
    disagreement_fields: List[str] = Field(default_factory=list)
    human_verified: bool = False
    human_corrections: dict = Field(default_factory=dict)
    human_reviewer: Optional[str] = None
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision: DecisionLabel = DecisionLabel.UNCERTAIN
    token_usage: dict = Field(default_factory=dict)  # {model: {prompt, completion}}


class PipelineRecord(BaseModel):
    """Master record tracking a study through the entire pipeline."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    study_id: Optional[str] = None  # assigned after inclusion confirmed
    raw: Optional[RawRecord] = None
    dedup: Optional[DedupRecord] = None
    screened: Optional[ScreenedRecord] = None
    extracted: Optional[ExtractedRecord] = None
    final_decision: DecisionLabel = DecisionLabel.UNCERTAIN
    pipeline_stage: PipelineStage = PipelineStage.IMPORT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update_stage(self, stage: PipelineStage, decision: DecisionLabel):
        self.pipeline_stage = stage
        self.final_decision = decision
        self.updated_at = datetime.utcnow()
