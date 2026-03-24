Extract detailed, evidence-based information about LLM use in qualitative analysis from the paper below.

TITLE: {title}

FULL TEXT:
{fulltext}

Return JSON:
{{
  "llm_usage_overview": "1-3 sentence summary of what the LLM was used for in the analysis",

  "analysis_stage": {{
    "stages_involved": [
      "list each qualitative analysis stage the LLM was used for, in the paper's own terms",
      "e.g. initial/open coding, focused coding, codebook development, theme generation, theme refinement, interpretation"
    ],
    "stage_description": "brief narrative of which stage(s) and in what order",
    "covers_full_analysis": "end-to-end — LLM handled all stages / partial — LLM assisted specific stages / not reported"
  }},

  "llm_details": {{
    "model_name": "exact model name(s) as stated, e.g. GPT-4, Claude 3, Llama 2",
    "model_version": "version or checkpoint if reported, else not reported",
    "prompting_strategy": "brief description — zero-shot, few-shot, chain-of-thought, custom system prompt, etc.",
    "input_data_type": "what was fed to the LLM — interview transcripts, survey responses, field notes, etc.",
    "unit_of_analysis": "what the LLM processed per call — full transcript, paragraph, sentence, code segment, etc.",
    "temperature_or_params": "any reported generation parameters, else not reported"
  }},

  "analysis_process": {{
    "step_by_step_description": [
      "Step 1: ...",
      "Step 2: ..."
    ],
    "coding_process_description": "how codes or categories were generated or applied",
    "theme_or_pattern_generation": "how themes or higher-level patterns were derived, if applicable",
    "iteration_or_refinement": "any iterative or multi-pass processes described",
    "human_involvement": "what humans did in the loop — reviewing, correcting, validating, prompting, etc."
  }},

  "multi_agent_details": {{
    "used_multiple_agents": "yes / no / not reported",
    "agent_descriptions": [
      {{"agent_name_or_role": "e.g. Coder Agent", "model_used": "model name if specified", "task": "what this agent does"}}
    ],
    "agent_workflow": "how agents interact or pass outputs to each other, else not reported"
  }},

  "evaluation": {{
    "description": "brief summary of how the LLM analysis outputs were evaluated",
    "comparison": "compared to human coders or another baseline? describe briefly, else not reported",
    "metrics": "exact metric names reported — e.g. Cohen's kappa, percent agreement, F1, IRR, else not reported",
    "performance": "actual scores or values — e.g. kappa=0.82, 91% agreement, else not reported",
    "qualitative_validation": "any qualitative or interpretive validation described beyond numbers, else not reported"
  }},

  "study_context": {{
    "domain": "research domain — e.g. healthcare, education, social media",
    "sample_size": "number of participants or texts analyzed",
    "data_resources_or_type": "description of the qualitative data corpus",
    "is_preprint_arxiv": "yes / no / not reported"
  }},

  "key_phrases": [
    "short verbatim phrases from the paper that best characterize the LLM role or method"
  ],

  "evidence_quotes": [
    "direct quotes from the paper supporting the extraction above"
  ],

  "not_reported_fields": ["field names not found in the paper"],
  "uncertain_fields": ["field names with weak or ambiguous evidence"],
  "extraction_notes": "any notes about difficult or borderline extractions"
}}
