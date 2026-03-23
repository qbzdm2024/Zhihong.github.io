"""
Multi-agent data extraction module.
Two agents extract independently, then compare field-by-field.
Disagreements on individual fields are flagged, not silently resolved.
"""
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .openai_client import get_client
from .prompts import EXTRACTION_SYSTEM, EXTRACTION_USER, QA_SYSTEM, QA_USER
from config.settings import settings
from pipeline.models import (
    ExtractionResult, QAScore, ExtractedRecord, DecisionLabel
)

logger = logging.getLogger(__name__)

# Fields where minor disagreement is tolerable (e.g., paraphrases of same content)
SOFT_FIELDS = {"study_aim", "key_findings", "strengths_reported", "limitations_reported",
               "human_oversight", "codebook_development", "extraction_notes"}

# Fields where disagreement always flags human review
HARD_FIELDS = {"model_name", "workflow_structure", "analytic_task", "human_comparison",
               "fine_tuned", "rag_used", "formal_methodology", "qualitative_approach"}


def _parse_extraction(raw: dict) -> ExtractionResult:
    """Parse model JSON output into ExtractionResult, handling edge cases."""
    # Ensure not_reported_fields and uncertain_fields are lists
    raw["not_reported_fields"] = raw.get("not_reported_fields") or []
    raw["uncertain_fields"] = raw.get("uncertain_fields") or []

    # analytic_task should be a list
    at = raw.get("analytic_task")
    if isinstance(at, str):
        raw["analytic_task"] = [at]
    elif at is None:
        raw["analytic_task"] = []

    # Coerce year
    year_raw = raw.get("year")
    if year_raw is not None:
        try:
            raw["year"] = int(str(year_raw)[:4])
        except (ValueError, TypeError):
            raw["year"] = None

    # Coerce temperature
    temp = raw.get("temperature")
    if temp is not None:
        try:
            raw["temperature"] = float(temp)
        except (ValueError, TypeError):
            raw["temperature"] = None

    # reproducibility_score: clamp to 1-4
    rs = raw.get("reproducibility_score")
    if rs is not None:
        try:
            raw["reproducibility_score"] = max(1, min(4, int(rs)))
        except (ValueError, TypeError):
            raw["reproducibility_score"] = None

    try:
        return ExtractionResult(**{k: v for k, v in raw.items()
                                   if k in ExtractionResult.model_fields})
    except Exception as e:
        logger.error(f"ExtractionResult parse error: {e}")
        return ExtractionResult(extraction_notes=f"Parse error: {e}")


def _parse_qa(raw: dict) -> QAScore:
    """Parse model JSON output into QAScore."""
    qa_fields = {k: int(raw.get(k, 0)) for k in [
        "qa1_llm_identified", "qa2_prompts_described", "qa3_process_described",
        "qa4_human_role_defined", "qa5_validation_performed", "qa6_results_detailed",
        "qa7_limitations_acknowledged", "qa8_data_adequate", "qa9_reproducibility",
        "qa10_ethics"
    ]}
    return QAScore(**qa_fields)


def _extract_single_agent(agent_id: str, model: str, title: str,
                           fulltext: str) -> Tuple[ExtractionResult, dict]:
    """Run a single extraction agent. Returns (result, usage)."""
    client = get_client()
    truncated = fulltext[:14000] if len(fulltext) > 14000 else fulltext

    user_prompt = EXTRACTION_USER.format(
        title=title,
        authors="",
        year="",
        journal_venue="",
        fulltext=truncated,
    )

    try:
        raw, usage = client.chat_json(
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            model=model,
            max_tokens=6000,
        )
        result = _parse_extraction(raw)
        logger.info(f"[{agent_id}] Extraction complete. Not reported: {result.not_reported_fields}")
        return result, usage or {}
    except Exception as e:
        logger.error(f"[{agent_id}] Extraction failed: {e}")
        return ExtractionResult(
            uncertain_fields=["ALL"],
            extraction_notes=f"Agent error: {e}"
        ), {}


def _qa_single_agent(agent_id: str, model: str, title: str, fulltext: str) -> Tuple[QAScore, dict]:
    """Run a single QA assessment agent. Returns (score, usage)."""
    client = get_client()
    truncated = fulltext[:10000] if len(fulltext) > 10000 else fulltext
    user_prompt = QA_USER.format(title=title, fulltext=truncated)

    try:
        raw, usage = client.chat_json(
            system_prompt=QA_SYSTEM,
            user_prompt=user_prompt,
            model=model,
        )
        return _parse_qa(raw), usage or {}
    except Exception as e:
        logger.error(f"[{agent_id}] QA failed: {e}")
        return QAScore(
            qa1_llm_identified=0, qa2_prompts_described=0, qa3_process_described=0,
            qa4_human_role_defined=0, qa5_validation_performed=0, qa6_results_detailed=0,
            qa7_limitations_acknowledged=0, qa8_data_adequate=0, qa9_reproducibility=0,
            qa10_ethics=0
        ), {}


def _compare_extractions(
    e1: ExtractionResult,
    e2: ExtractionResult
) -> Tuple[ExtractionResult, List[str], bool]:
    """
    Compare two extraction results field by field.
    Returns:
        - merged ExtractionResult (agent1 values where they agree, null where they disagree)
        - disagreement_fields: list of field names that differ significantly
        - needs_human: True if any HARD_FIELD disagrees
    """
    disagreement_fields = []
    merged = {}

    # Get all defined fields
    all_fields = ExtractionResult.model_fields.keys()

    for field in all_fields:
        if field in ("not_reported_fields", "uncertain_fields", "extraction_notes"):
            # Merge lists
            list1 = getattr(e1, field, []) or []
            list2 = getattr(e2, field, []) or []
            merged[field] = list(set(list1 + list2))
            continue

        v1 = getattr(e1, field, None)
        v2 = getattr(e2, field, None)

        # Both null → keep null
        if v1 is None and v2 is None:
            merged[field] = None
            continue

        # One null, one not → take non-null but flag uncertainty
        if v1 is None or v2 is None:
            non_null = v1 if v1 is not None else v2
            merged[field] = non_null
            if field in HARD_FIELDS:
                disagreement_fields.append(field)
                merged["uncertain_fields"] = merged.get("uncertain_fields", []) + [field]
            continue

        # Both have values — compare
        if isinstance(v1, list) and isinstance(v2, list):
            if set(v1) != set(v2) and field in HARD_FIELDS:
                disagreement_fields.append(field)
                # Take union as merged value; human will decide
                merged[field] = list(set(v1) | set(v2))
                merged["uncertain_fields"] = merged.get("uncertain_fields", []) + [field]
            else:
                merged[field] = v1  # default to agent1
        elif isinstance(v1, bool) and isinstance(v2, bool):
            if v1 != v2:
                disagreement_fields.append(field)
                merged[field] = None  # ambiguous → null, flag
                merged["uncertain_fields"] = merged.get("uncertain_fields", []) + [field]
            else:
                merged[field] = v1
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(v1 - v2) > 0.5 and field in HARD_FIELDS:
                disagreement_fields.append(field)
                merged[field] = None
                merged["uncertain_fields"] = merged.get("uncertain_fields", []) + [field]
            else:
                merged[field] = v1
        else:
            # String comparison — normalize
            s1 = str(v1).strip().lower()
            s2 = str(v2).strip().lower()
            if s1 != s2 and field in HARD_FIELDS:
                disagreement_fields.append(field)
                merged[field] = v1  # keep agent1 but flag
                merged["uncertain_fields"] = merged.get("uncertain_fields", []) + [field]
            else:
                merged[field] = v1

    needs_human = any(f in HARD_FIELDS for f in disagreement_fields)

    # Ensure uncertain_fields is unique
    merged["uncertain_fields"] = list(set(merged.get("uncertain_fields", [])))
    merged["not_reported_fields"] = list(set(merged.get("not_reported_fields", [])))

    return ExtractionResult(**{k: v for k, v in merged.items()
                                if k in ExtractionResult.model_fields}), disagreement_fields, needs_human


def extract_and_assess(
    record_id: str,
    study_id: str,
    title: str,
    fulltext: str,
) -> ExtractedRecord:
    """
    Full extraction + QA pipeline for one included study.
    Runs two extraction agents and two QA agents independently.
    """
    def _add_usage(acc: dict, model: str, usage: dict) -> None:
        """Accumulate token counts per model."""
        if not usage:
            return
        entry = acc.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})
        entry["prompt_tokens"]     += usage.get("prompt_tokens", 0)
        entry["completion_tokens"] += usage.get("completion_tokens", 0)

    token_usage: Dict[str, Any] = {}

    # --- Extraction ---
    e1, u1 = _extract_single_agent(
        agent_id=f"extract_agent1_{settings.model_extraction}",
        model=settings.model_extraction,
        title=title,
        fulltext=fulltext,
    )
    e1.study_id = study_id
    _add_usage(token_usage, settings.model_extraction, u1)

    e2, u2 = _extract_single_agent(
        agent_id=f"extract_agent2_{settings.model_agent2_extraction}",
        model=settings.model_agent2_extraction,
        title=title,
        fulltext=fulltext,
    )
    e2.study_id = study_id
    _add_usage(token_usage, settings.model_agent2_extraction, u2)

    merged, disagreement_fields, needs_human = _compare_extractions(e1, e2)

    # --- QA Assessment ---
    qa1, uq1 = _qa_single_agent(
        agent_id=f"qa_agent1_{settings.model_qa_assessment}",
        model=settings.model_qa_assessment,
        title=title,
        fulltext=fulltext,
    )
    _add_usage(token_usage, settings.model_qa_assessment, uq1)

    qa2, uq2 = _qa_single_agent(
        agent_id=f"qa_agent2_{settings.model_agent2_extraction}",
        model=settings.model_agent2_extraction,
        title=title,
        fulltext=fulltext,
    )
    _add_usage(token_usage, settings.model_agent2_extraction, uq2)

    # Average QA scores
    qa_final = QAScore(
        qa1_llm_identified=round((qa1.qa1_llm_identified + qa2.qa1_llm_identified) / 2),
        qa2_prompts_described=round((qa1.qa2_prompts_described + qa2.qa2_prompts_described) / 2),
        qa3_process_described=round((qa1.qa3_process_described + qa2.qa3_process_described) / 2),
        qa4_human_role_defined=round((qa1.qa4_human_role_defined + qa2.qa4_human_role_defined) / 2),
        qa5_validation_performed=round((qa1.qa5_validation_performed + qa2.qa5_validation_performed) / 2),
        qa6_results_detailed=round((qa1.qa6_results_detailed + qa2.qa6_results_detailed) / 2),
        qa7_limitations_acknowledged=round((qa1.qa7_limitations_acknowledged + qa2.qa7_limitations_acknowledged) / 2),
        qa8_data_adequate=round((qa1.qa8_data_adequate + qa2.qa8_data_adequate) / 2),
        qa9_reproducibility=round((qa1.qa9_reproducibility + qa2.qa9_reproducibility) / 2),
        qa10_ethics=round((qa1.qa10_ethics + qa2.qa10_ethics) / 2),
    )

    final_decision = DecisionLabel.UNCERTAIN if needs_human else DecisionLabel.INCLUDE

    return ExtractedRecord(
        record_id=record_id,
        study_id=study_id,
        extraction_agent1=e1,
        extraction_agent2=e2,
        extraction_final=merged,
        qa_score=qa_final,
        qa_agent1=qa1,
        qa_agent2=qa2,
        agents_agree_extraction=not needs_human,
        disagreement_fields=disagreement_fields,
        human_verified=False,
        decision=final_decision,
        extraction_timestamp=datetime.utcnow(),
        token_usage=token_usage,
    )
