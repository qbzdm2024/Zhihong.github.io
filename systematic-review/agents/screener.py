"""
Multi-agent screening module.
Two independent agents screen each record.
If they agree → finalize.
If they disagree → flag for human review.
"""
import json
import logging
from datetime import datetime
from typing import Optional, Tuple

from .openai_client import get_client
from .prompts import (
    TITLE_SCREENING_SYSTEM, TITLE_SCREENING_USER,
    FULLTEXT_SCREENING_SYSTEM, FULLTEXT_SCREENING_USER,
    COMPARISON_SYSTEM,
)
from config.settings import settings
from pipeline.models import (
    AgentDecision, ScreeningResult, DecisionLabel, PipelineStage,
    DedupRecord, ScreenedRecord
)

logger = logging.getLogger(__name__)

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
        logger.error(f"[{agent_id}] Screening failed: {e}")
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

    # Quick path: either agent is uncertain → always go to human
    if (agent1.decision == DecisionLabel.UNCERTAIN or
            agent2.decision == DecisionLabel.UNCERTAIN):
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=DecisionLabel.UNCERTAIN,
            agent1=agent1,
            agent2=agent2,
            agents_agree=False,
            consensus_confidence=min(agent1.confidence, agent2.confidence),
            human_verified=False,
        )

    # Quick path: both agree with high confidence
    same_decision = agent1.decision == agent2.decision
    conf_diff = abs(agent1.confidence - agent2.confidence)

    if same_decision and conf_diff < 0.15 and min(agent1.confidence, agent2.confidence) >= settings.confidence_threshold:
        return ScreeningResult(
            stage=PipelineStage.TITLE_SCREENING,
            final_decision=agent1.decision,
            agent1=agent1,
            agent2=agent2,
            agents_agree=True,
            consensus_confidence=(agent1.confidence + agent2.confidence) / 2,
            human_verified=False,
        )

    # Use model to compare
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
        recommendation = raw.get("recommendation", "send_to_human")

        # If recommendation is send_to_human, force uncertain
        if recommendation == "send_to_human" or not agents_agree:
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
        # On comparison failure → always go to human
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

    # Agent 1
    agent1 = _screen_single_agent(
        agent_id=f"agent1_{settings.model_title_screening}",
        model=settings.model_title_screening,
        system_prompt=TITLE_SCREENING_SYSTEM,
        user_prompt=user_prompt,
    )

    # Agent 2 (different model if configured)
    agent2 = _screen_single_agent(
        agent_id=f"agent2_{settings.model_agent2_screening}",
        model=settings.model_agent2_screening,
        system_prompt=TITLE_SCREENING_SYSTEM,
        user_prompt=user_prompt,
    )

    result = _compare_agents(agent1, agent2)
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

    agent1 = _screen_single_agent(
        agent_id=f"agent1_{settings.model_fulltext_screening}",
        model=settings.model_fulltext_screening,
        system_prompt=FULLTEXT_SCREENING_SYSTEM,
        user_prompt=user_prompt,
    )

    agent2 = _screen_single_agent(
        agent_id=f"agent2_{settings.model_agent2_screening}",
        model=settings.model_agent2_screening,
        system_prompt=FULLTEXT_SCREENING_SYSTEM,
        user_prompt=user_prompt,
    )

    result = _compare_agents(agent1, agent2)
    result.stage = PipelineStage.FULLTEXT_SCREENING
    return result
