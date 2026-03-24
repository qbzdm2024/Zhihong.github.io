You are given an extraction produced by another model. Verify it against the original paper sections.

=== EXTRACTION ===
{gpt5_output}

=== METHODS ===
{methods_text}

=== RESULTS ===
{results_text}

Return JSON:
{{
  "paper_id": "{paper_id}",
  "verification_results": {{
    "llm_usage": {{
      "supported": true,
      "evidence_quote": "",
      "issue": ""
    }},
    "analysis_stage": {{
      "supported": true,
      "evidence_quote": "",
      "issue": ""
    }},
    "analysis_process": {{
      "supported": true,
      "evidence_quote": "",
      "issue": ""
    }},
    "human_involvement": {{
      "supported": true,
      "evidence_quote": "",
      "issue": ""
    }},
    "multi_agent_details": {{
      "supported": true,
      "evidence_quote": "",
      "issue": ""
    }},
    "evaluation": {{
      "supported": true,
      "evidence_quote": "",
      "issue": ""
    }}
  }},
  "missing_information": [
    "information claimed in extraction but not found in text",
    "important details not reported in the paper"
  ],
  "potential_overinterpretation": [
    "cases where the extraction inferred too much beyond what is stated"
  ],
  "confidence_assessment": {{
    "overall": "high / medium / low",
    "reason": ""
  }}
}}
