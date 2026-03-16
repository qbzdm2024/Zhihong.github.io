"""
Single-turn tester for the Omaha mapping pipeline.

Lets you test ONE conversation turn in isolation — see the exact prompt
sent to the LLM, the raw response, and the parsed result — without
loading S3 or running the whole annotation sheet.

Usage (SageMaker terminal):
    python test_single_turn.py
    python test_single_turn.py --turn "I can't breathe at night."
    python test_single_turn.py --turn "I'll check your blood pressure." --task intervention
    python test_single_turn.py --turn "My ankles are swollen." --task both
    python test_single_turn.py --turn "Okay." --context "Nurse: How are you feeling? Patient: Okay."

Options:
    --turn TEXT       The conversation turn to classify (prompted if omitted)
    --context TEXT    Optional surrounding context (rows before/after)
    --task            ss | intervention | both   (default: both)
    --model           LLM config name from omaha_sagemaker.LLM_CONFIGS (default: gpt-4o-mini)
    --top-k           Number of Omaha options to retrieve (default: 15)
    --omaha-file      Local path to Omaha Excel (auto-detected if omitted)
    --show-prompt     Print the full prompt sent to the LLM (default: True)
"""

import argparse
import os
import sys

# ── Make sure omaha_sagemaker is importable from same directory ────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Lazy import: don't crash if sentence-transformers isn't installed ──────────
try:
    import omaha_sagemaker as om
except ImportError as e:
    sys.exit(f"Cannot import omaha_sagemaker: {e}\n"
             f"Run: pip install -r {os.path.join(os.path.dirname(__file__), 'requirements.txt')}")

import pandas as pd
from io import BytesIO


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours (disabled automatically if stdout is not a tty)
# ─────────────────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)
CYAN   = lambda t: _c("96",   t)
GREEN  = lambda t: _c("92",   t)
YELLOW = lambda t: _c("93",   t)
RED    = lambda t: _c("91",   t)
BLUE   = lambda t: _c("94",   t)

def divider(title="", char="─", width=72):
    if title:
        pad = max(0, width - len(title) - 3)
        print(CYAN(f"{char*2} {BOLD(title)} {char*pad}"))
    else:
        print(DIM(char * width))


# ─────────────────────────────────────────────────────────────────────────────
# Load Omaha Excel from local file (bypasses S3 for quick testing)
# ─────────────────────────────────────────────────────────────────────────────

def _find_omaha_excel(hint: str = None) -> str:
    """Return a path to the Omaha Excel, checking several candidate locations."""
    candidates = []
    if hint:
        candidates.append(hint)

    # Repo root (one level up from omaha_mapping/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates += [
        os.path.join(repo_root, "Omaha_system list with definition.xlsx"),
        os.path.join(os.path.dirname(__file__), "Omaha_system list with definition.xlsx"),
        os.path.join(os.path.dirname(__file__), "omaha_system.xlsx"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def load_omaha_local(path: str):
    """Read the Omaha Excel from a local file, mirroring load_omaha_system()."""
    print(DIM(f"  Loading Omaha definitions from: {path}"))
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    sheet_names = list(sheets.keys())

    ss_df  = sheets[sheet_names[0]].copy()
    int_df = sheets[sheet_names[1]].copy()

    ss_df.columns  = [c.strip() for c in ss_df.columns]
    int_df.columns = [c.strip() for c in int_df.columns]

    ss_df  = ss_df.dropna(how="all").reset_index(drop=True)
    int_df = int_df.dropna(how="all").reset_index(drop=True)
    return ss_df, int_df


# ─────────────────────────────────────────────────────────────────────────────
# Main test function
# ─────────────────────────────────────────────────────────────────────────────

def test_turn(
    turn: str,
    context: str = "",
    task: str = "both",           # "ss" | "intervention" | "both"
    model_name: str = "gpt-4o-mini",
    top_k: int = 15,
    omaha_file: str = None,
    show_prompt: bool = True,
):
    # ── Load Omaha definitions ─────────────────────────────────────────────────
    omaha_path = _find_omaha_excel(omaha_file)
    if omaha_path is None:
        print(RED("ERROR: Omaha Excel file not found."))
        print("  Place 'Omaha_system list with definition.xlsx' in the repo root or pass --omaha-file.")
        sys.exit(1)

    ss_df, int_df = load_omaha_local(omaha_path)

    # ── Build embedding index (once) ──────────────────────────────────────────
    print(DIM("  Loading embedding model …"))
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(om.EMBEDDING_MODEL)

    ss_docs  = om.build_ss_documents(ss_df)
    int_docs = om.build_intervention_documents(int_df)

    ss_embeddings  = om.build_index(ss_docs,  embed_model)
    int_embeddings = om.build_index(int_docs, embed_model)

    # ── Print header ──────────────────────────────────────────────────────────
    print()
    divider("SINGLE-TURN TEST", "═")
    print(f"  {BOLD('Model:')}   {model_name}")
    print(f"  {BOLD('Task:')}    {task}")
    print(f"  {BOLD('Top-K:')}   {top_k}")
    print()

    # ── Input ─────────────────────────────────────────────────────────────────
    divider("CONVERSATION TURN")
    print(YELLOW(f"  {turn}"))
    if context.strip():
        print()
        print(DIM("  Context:"))
        for line in context.strip().splitlines():
            print(DIM(f"    {line}"))

    # ── Signs/Symptoms ────────────────────────────────────────────────────────
    if task in ("ss", "both"):
        print()
        divider("SIGNS / SYMPTOMS MAPPING")

        retrieved_ss = om.retrieve(turn, ss_docs, ss_embeddings, embed_model, top_k)
        ss_prompt    = om.build_ss_prompt(turn, context, retrieved_ss)

        print(DIM(f"  Retrieved {len(retrieved_ss)} SS options."))

        if show_prompt:
            print()
            divider(f"SS PROMPT ({len(ss_prompt)} chars)", "·")
            for line in ss_prompt.splitlines():
                print(DIM(f"  {line}"))

        print()
        divider("SS RAW RESPONSE", "·")
        ss_raw = om.call_llm(ss_prompt, model_name)
        print(GREEN(f"  {ss_raw}"))

        print()
        divider("SS PARSED RESULT", "·")
        ss_parsed = om.parse_ss_output(ss_raw)
        if ss_parsed:
            for i, r in enumerate(ss_parsed, 1):
                print(GREEN(f"  [{i}] Domain: {r['domain']}"))
                print(GREEN(f"       Problem: {r['problem']}"))
                print(GREEN(f"       Signs/Symptoms: {r['ss']}"))
        else:
            print(YELLOW("  → None (no classification)"))

    # ── Interventions ─────────────────────────────────────────────────────────
    if task in ("intervention", "both"):
        print()
        divider("INTERVENTION MAPPING")

        retrieved_int = om.retrieve(turn, int_docs, int_embeddings, embed_model, top_k)
        int_prompt    = om.build_intervention_prompt(turn, context, retrieved_int)

        print(DIM(f"  Retrieved {len(retrieved_int)} intervention options."))

        if show_prompt:
            print()
            divider(f"INTERVENTION PROMPT ({len(int_prompt)} chars)", "·")
            for line in int_prompt.splitlines():
                print(DIM(f"  {line}"))

        print()
        divider("INTERVENTION RAW RESPONSE", "·")
        int_raw = om.call_llm(int_prompt, model_name)
        print(GREEN(f"  {int_raw}"))

        print()
        divider("INTERVENTION PARSED RESULT", "·")
        int_parsed = om.parse_intervention_output(int_raw)
        if int_parsed:
            for i, r in enumerate(int_parsed, 1):
                print(GREEN(f"  [{i}] Category: {r['category']} | Target: {r['target']}"))
        else:
            print(YELLOW("  → None (no intervention)"))

    divider("", "═")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test a single Omaha mapping turn without running the full pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--turn",       default=None,
                        help="Conversation turn to classify")
    parser.add_argument("--context",    default="",
                        help="Surrounding context text (optional)")
    parser.add_argument("--task",       default="both",
                        choices=["ss", "intervention", "both"],
                        help="Which mapping to run (default: both)")
    parser.add_argument("--model",      default="gpt-4o-mini",
                        help="LLM config name (default: gpt-4o-mini)")
    parser.add_argument("--top-k",      type=int, default=15,
                        help="Omaha options to retrieve (default: 15)")
    parser.add_argument("--omaha-file", default=None,
                        help="Local path to Omaha Excel file")
    parser.add_argument("--no-prompt",  action="store_true",
                        help="Hide the full prompt (only show response)")
    args = parser.parse_args()

    # Validate model name
    if args.model not in om.LLM_CONFIGS:
        print(RED(f"Unknown model '{args.model}'."))
        print(f"Available: {list(om.LLM_CONFIGS.keys())}")
        sys.exit(1)

    # Get turn interactively if not passed
    turn = args.turn
    if not turn:
        print("Enter the conversation turn to test (press Enter twice when done):")
        lines = []
        try:
            while True:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
        except EOFError:
            pass
        turn = "\n".join(lines).strip()
    if not turn:
        sys.exit("No turn provided.")

    test_turn(
        turn=turn,
        context=args.context,
        task=args.task,
        model_name=args.model,
        top_k=args.top_k,
        omaha_file=args.omaha_file,
        show_prompt=not args.no_prompt,
    )


if __name__ == "__main__":
    main()
