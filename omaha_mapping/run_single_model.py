"""
Run inference for a single model.

Usage:
    python run_single_model.py <model_name> [max_sheets]

Examples:
    python run_single_model.py gpt-4o-mini
    python run_single_model.py mistral-7b-local
    python run_single_model.py llama3-8b-local
    python run_single_model.py gpt-4o-mini 3     # only first 3 sheets
    python run_single_model.py mistral-7b-local 1 # only 1 sheet (debug)

Available model names (from omaha_sagemaker.LLM_CONFIGS):
    gpt-4o-mini, azure-gpt4o-mini, mistral-7b-local, llama3-8b-local, llama3-70b-local
"""

import sys
import omaha_sagemaker as om

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in om.LLM_CONFIGS:
        print(f"Unknown model: '{model_name}'")
        print(f"Available: {list(om.LLM_CONFIGS.keys())}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        om.MAX_SHEETS = int(sys.argv[2])
        print(f"MAX_SHEETS set to {om.MAX_SHEETS}")
    else:
        # Use whatever is set in omaha_sagemaker.py
        pass

    print(f"Running model: {model_name}")
    resources = om.build_shared_resources()
    om.run_inference(model_name, resources)

if __name__ == "__main__":
    main()
