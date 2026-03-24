Analyze the following paper sections and extract detailed descriptions of LLM use in qualitative analysis.

=== METHODS ===
{methods_text}

=== RESULTS ===
{results_text}

Return JSON:
{{
  "paper_id": "{paper_id}",

  "llm_usage_overview": "1-3 sentence summary of what the LLM was used for in the analysis",

  "llm_details": {{
    "model_name": "exact model name(s) as stated in the paper, e.g. GPT-4, Claude 3, Llama 2",
    "model_version": "version or checkpoint if reported, else not reported",
    "prompting_strategy": "brief description of how the model was prompted — zero-shot, few-shot with examples, chain-of-thought, custom system prompt, etc.",
    "input_data_type": "what was fed to the LLM — interview transcripts, survey responses, field notes, etc.",
    "unit_of_analysis": "what the LLM processed per call — full transcript, paragraph, sentence, code segment, etc.",
    "temperature_or_params": "any reported generation parameters"
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
      {{
        "agent_name_or_role": "e.g. Extraction Agent, Verification Agent",
        "model_used": "model name if different per agent",
        "task": "brief description of what this agent does"
      }}
    ],
    "agent_workflow": "brief description of how agents interact or pass outputs to each other"
  }},

  "evaluation": {{
    "description": "brief summary of how the LLM analysis outputs were evaluated",
    "comparison": "was it compared to human coders or another baseline? describe briefly",
    "metrics": "exact metric names reported — e.g. Cohen's kappa, percent agreement, F1, IRR",
    "performance": "actual scores or values reported — e.g. kappa=0.82, 91% agreement",
    "qualitative_validation": "any qualitative or interpretive validation described beyond numbers"
  }},

  "study_context": {{
    "domain": "research domain or field — e.g. healthcare, education, social media",
    "sample_size": "number of participants or texts analyzed",
    "data_resources_or_type": "description of the qualitative data corpus",
    "is_preprint_arxiv": "yes / no / not reported"
  }},

  "key_phrases": [
    "short verbatim phrases that best characterize the LLM role or method"
  ],

  "evidence_quotes": [
    "direct quotes from the paper that best support the extraction above"
  ]
}}
