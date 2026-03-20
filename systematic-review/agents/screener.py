"""
Multi-agent screening module.
Two independent agents screen each record.
If they agree → finalize.
If they disagree → flag for human review.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Tuple

from .openai_client import get_client
from .prompts import (
    TITLE_SCREENING_SYSTEM, TITLE_SCREENING_USER,
    FULLTEXT_SCREENING_SYSTEM, FULLTEXT_SCREENING_USER,
    COMPARISON_SYSTEM,
    SECOND_PASS_SCREENING_SYSTEM, SECOND_PASS_SCREENING_USER,
    SECOND_PASS_COMPARISON_SYSTEM,
)
from config.settings import settings
from pipeline.models import (
    AgentDecision, ScreeningResult, DecisionLabel, PipelineStage,
    DedupRecord, ScreenedRecord
)

logger = logging.getLogger(__name__)

# Tracks agent errors so we can surface them to the user
_agent_error_log: list = []


def get_agent_errors(n: int = 20) -> list:
    """Return the last n agent error messages."""
    return _agent_error_log[-n:]


COMPARISON_USER = """Agent 1 decision:
{agent1_json}

Agent 2 decision:
{agent2_json}

Compare and produce consensus. Return ONLY valid JSON."""


def _parse_agent_response(raw: dict, agent_id: str, model: str) -> AgentDecision:
    """Parse raw JSON dict from model into AgentDecision."""
    decision_str = raw.get("decision", "Needs Human Verification")
    try:
        decision = DecisionLabel(decision_str)
    except ValueError:
        decision = DecisionLabel.UNCERTAIN

    return AgentDecision(
        agent_id=agent_id,
        model_used=model,
        decision=decision,
        confidence=float(raw.get("confidence", 0.5)),
        rationale=raw.get("rationale", ""),
        exclusion_code=raw.get("exclusion_code"),
        flagged_criteria=raw.get("flagged_criteria", []),
        timestamp=datetime.utcnow(),
    )


def _screen_single_agent(
    agent_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> AgentDecision:
    """Run a single agent's screening decision."""
    client = get_client()
    try:
        raw, usage = client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )
        agent_dec = _parse_agent_response(raw, agent_id, model)
        logger.info(f"[{agent_id}] Decision: {agent_dec.decision} (conf={agent_dec.confidence:.2f})")
        return agent_dec

    except Exception as e:
        err_msg = f"[{agent_id}] model={model} error={type(e).__name__}: {e}"
        logger.error(err_msg)
        _agent_error_log.append(err_msg)
        if len(_agent_error_log) > 100:
            _agent_error_log.pop(0)
        # Return uncertain decision on error — never silently exclude
        return AgentDecision(
            agent_id=agent_id,
            model_used=model,
            decision=DecisionLabel.UNCERTAIN,
            confidence=0.0,
            rationale=f"Agent error: {e}",
            exclusion_code=None,
            flagged_criteria=["AGENT_ERROR"],
        )


def _compare_agents(agent1: AgentDecision, agent2: AgentDecision) -> ScreeningResult:
    """Compare two agents' decisions and produce a ScreeningResult."""
    client = get_client()

    # Quick path: either agent errored out → send to human
    if (agent1.decision == DecisionLabel.UNCERTAIN and "AGENT_ERROR" in (agent1.flagged_criteria or [])) or \
       (agent2.decision == DecisionLabel.UNCERTAIN and "AGENT_ERROR" in (agent2.flagged_criteria or [])):
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.UNCERTAIN,
            agent1=agent1,
            agent2=agent2,
            agents_agree=False,
            consensus_confidence=0.0,
            human_verified=False,
        )

    # Quick path: both agree with sufficient confidence — auto-accept
    same_decision = agent1.decision == agent2.decision
    conf_diff = abs(agent1.confidence - agent2.confidence)
    min_conf = min(agent1.confidence, agent2.confidence)

    if same_decision and conf_diff < 0.15 and min_conf >= settings.confidence_threshold:
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=agent1.decision,
            agent1=agent1,
            agent2=agent2,
            agents_agree=True,
            consensus_confidence=(agent1.confidence + agent2.confidence) / 2,
            human_verified=False,
        )

    # Quick path: both say Exclude with reasonable confidence (≥ 0.65) — auto-exclude
    # Liberal inclusion principle: lower bar for exclusion than inclusion.
    if (same_decision and agent1.decision == DecisionLabel.EXCLUDE
            and min_conf >= 0.65 and conf_diff < 0.20):
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.EXCLUDE,
            agent1=agent1,
            agent2=agent2,
            agents_agree=True,
            consensus_confidence=(agent1.confidence + agent2.confidence) / 2,
            human_verified=False,
        )

    # Use comparison model for borderline cases
    try:
        comparison_prompt = COMPARISON_USER.format(
            agent1_json=json.dumps(agent1.model_dump(), default=str),
            agent2_json=json.dumps(agent2.model_dump(), default=str),
        )
        raw, _ = client.chat_json(
            system_prompt=COMPARISON_SYSTEM,
            user_prompt=comparison_prompt,
            model=settings.model_fulltext_screening,
        )

        consensus_dec_str = raw.get("consensus_decision", "Needs Human Verification")
        try:
            consensus_dec = DecisionLabel(consensus_dec_str)
        except ValueError:
            consensus_dec = DecisionLabel.UNCERTAIN

        agents_agree = raw.get("agents_agree", False)
        consensus_conf = float(raw.get("consensus_confidence", 0.5))
        # Default "recommendation" to "accept_consensus" — only force human
        # when the model EXPLICITLY says to send to human.
        recommendation = raw.get("recommendation", "accept_consensus")

        # Force human review only when explicitly recommended or decision is truly unclear
        if recommendation == "send_to_human":
            consensus_dec = DecisionLabel.UNCERTAIN

        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=consensus_dec,
            agent1=agent1,
            agent2=agent2,
            agents_agree=agents_agree,
            consensus_confidence=consensus_conf,
            human_verified=False,
        )

    except Exception as e:
        logger.error(f"Comparison agent failed: {e}")
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.UNCERTAIN,
            agent1=agent1,
            agent2=agent2,
            agents_agree=False,
            consensus_confidence=0.0,
            human_verified=False,
        )


def screen_title_abstract(record: DedupRecord) -> ScreeningResult:
    """
    Run two-agent title/abstract screening on a single record.
    Returns ScreeningResult with final_decision set.
    """
    user_prompt = TITLE_SCREENING_USER.format(
        title=record.title or "",
        authors=record.authors or "Unknown",
        year=record.year or "Unknown",
        journal_venue=record.journal_venue or "Unknown",
        abstract=record.abstract or "[No abstract available]",
    )

    # Run agent1 and agent2 in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(
            _screen_single_agent,
            f"agent1_{settings.model_title_screening}",
            settings.model_title_screening,
            TITLE_SCREENING_SYSTEM,
            user_prompt,
        )
        f2 = ex.submit(
            _screen_single_agent,
            f"agent2_{settings.model_agent2_screening}",
            settings.model_agent2_screening,
            TITLE_SCREENING_SYSTEM,
            user_prompt,
        )
        agent1 = f1.result()
        agent2 = f2.result()

    result = _compare_agents(agent1, agent2)
    result.stage = PipelineStage.TITLE_SCREENING
    return result


def _compare_agents_strict(agent1: AgentDecision, agent2: AgentDecision) -> ScreeningResult:
    """Strict comparison for second-pass: uncertain cases go to human, not auto-included."""
    client = get_client()

    # Either agent errored → human review
    if (agent1.decision == DecisionLabel.UNCERTAIN and "AGENT_ERROR" in (agent1.flagged_criteria or [])) or \
       (agent2.decision == DecisionLabel.UNCERTAIN and "AGENT_ERROR" in (agent2.flagged_criteria or [])):
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.UNCERTAIN,
            agent1=agent1,
            agent2=agent2,
            agents_agree=False,
            consensus_confidence=0.0,
            human_verified=False,
        )

    # Both confidently agree → auto-decide
    same_decision = agent1.decision == agent2.decision
    min_conf = min(agent1.confidence, agent2.confidence)
    conf_diff = abs(agent1.confidence - agent2.confidence)

    if same_decision and agent1.decision == DecisionLabel.EXCLUDE and min_conf >= 0.60:
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.EXCLUDE,
            agent1=agent1,
            agent2=agent2,
            agents_agree=True,
            consensus_confidence=(agent1.confidence + agent2.confidence) / 2,
            human_verified=False,
        )

    if same_decision and agent1.decision == DecisionLabel.INCLUDE and min_conf >= settings.confidence_threshold and conf_diff < 0.15:
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.INCLUDE,
            agent1=agent1,
            agent2=agent2,
            agents_agree=True,
            consensus_confidence=(agent1.confidence + agent2.confidence) / 2,
            human_verified=False,
        )

    # All other cases: use comparison model with strict prompt
    try:
        comparison_prompt = COMPARISON_USER.format(
            agent1_json=json.dumps(agent1.model_dump(), default=str),
            agent2_json=json.dumps(agent2.model_dump(), default=str),
        )
        raw, _ = client.chat_json(
            system_prompt=SECOND_PASS_COMPARISON_SYSTEM,
            user_prompt=comparison_prompt,
            model=settings.model_fulltext_screening,
        )

        consensus_dec_str = raw.get("consensus_decision", "Needs Human Verification")
        try:
            consensus_dec = DecisionLabel(consensus_dec_str)
        except ValueError:
            consensus_dec = DecisionLabel.UNCERTAIN

        recommendation = raw.get("recommendation", "send_to_human")
        if recommendation == "send_to_human":
            consensus_dec = DecisionLabel.UNCERTAIN

        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=consensus_dec,
            agent1=agent1,
            agent2=agent2,
            agents_agree=raw.get("agents_agree", False),
            consensus_confidence=float(raw.get("consensus_confidence", 0.5)),
            human_verified=False,
        )

    except Exception as e:
        logger.error(f"Second-pass comparison agent failed: {e}")
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.UNCERTAIN,
            agent1=agent1,
            agent2=agent2,
            agents_agree=False,
            consensus_confidence=0.0,
            human_verified=False,
        )


def screen_second_pass(record: DedupRecord, agent1_rationale: str = "", agent2_rationale: str = "") -> ScreeningResult:
    """
    Run strict two-agent second-pass screening on a record that passed first-pass.
    Passes the first-pass agent rationale as context so agents can see why it was included.
    Returns ScreeningResult; uncertain cases are flagged for human review (not auto-included).
    """
    user_prompt = SECOND_PASS_SCREENING_USER.format(
        title=record.title or "",
        authors=record.authors or "Unknown",
        year=record.year or "Unknown",
        journal_venue=record.journal_venue or "Unknown",
        abstract=record.abstract or "[No abstract available]",
        agent1_rationale=agent1_rationale or "Not available",
        agent2_rationale=agent2_rationale or "Not available",
    )

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(
            _screen_single_agent,
            f"sp_agent1_{settings.model_title_screening}",
            settings.model_title_screening,
            SECOND_PASS_SCREENING_SYSTEM,
            user_prompt,
        )
        f2 = ex.submit(
            _screen_single_agent,
            f"sp_agent2_{settings.model_agent2_screening}",
            settings.model_agent2_screening,
            SECOND_PASS_SCREENING_SYSTEM,
            user_prompt,
        )
        agent1 = f1.result()
        agent2 = f2.result()

    result = _compare_agents_strict(agent1, agent2)
    result.stage = PipelineStage.TITLE_SCREENING
    return result


def screen_fulltext(record: DedupRecord, fulltext: str) -> ScreeningResult:
    """
    Run two-agent full-text screening on a single record.
    Requires the extracted text from PDF or HTML.
    """
    # Truncate to avoid token limits (keep first 12000 chars)
    truncated = fulltext[:12000] if len(fulltext) > 12000 else fulltext

    user_prompt = FULLTEXT_SCREENING_USER.format(
        title=record.title or "",
        authors=record.authors or "Unknown",
        year=record.year or "Unknown",
        journal_venue=record.journal_venue or "Unknown",
        fulltext=truncated,
    )

    # Run agent1 and agent2 in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(
            _screen_single_agent,
            f"agent1_{settings.model_fulltext_screening}",
            settings.model_fulltext_screening,
            FULLTEXT_SCREENING_SYSTEM,
            user_prompt,
        )
        f2 = ex.submit(
            _screen_single_agent,
            f"agent2_{settings.model_agent2_screening}",
            settings.model_agent2_screening,
            FULLTEXT_SCREENING_SYSTEM,
            user_prompt,
        )
        agent1 = f1.result()
        agent2 = f2.result()

    result = _compare_agents(agent1, agent2)
    result.stage = PipelineStage.FULLTEXT_SCREENING
    return result
