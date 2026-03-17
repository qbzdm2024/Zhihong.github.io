"""
Single-turn tester for the Omaha mapping pipeline.

Lets you test ONE conversation turn (with optional preceding context turns)
without loading S3 or running the whole annotation sheet.

Usage (SageMaker terminal):
    python test_single_turn.py --turn "I can't breathe at night."
    python test_single_turn.py --turn "My ankles are swollen." --task both --verbose
    python test_single_turn.py \\
        --context-turn "Nurse: How have you been feeling?" \\
        --context-turn "Patient: Pretty tired still." \\
        --turn "My legs feel really heavy today." \\
        --context-window 3

    # 3-agent pipeline (understand → classify → verify):
    python test_single_turn.py --turn "..." --understand --verify

    # Custom prompts from files:
    python test_single_turn.py --turn "..." --ss-prompt-file my_ss_prompt.txt

Notebook usage (inline):
    import test_single_turn as tst
    tst.test_turn(
        turn="My blood pressure machine says 158 over 92.",
        context_turns=[
            {"spk": "Nurse", "text": "Can you check your blood pressure for me?"},
        ],
        context_window=3,
        task="both",
        model_name="gpt-4o-mini",
        top_k=15,
        use_understand=True,
        use_verify=True,
    )

    # Shorthand: pass a list as `turn` (last = current, rest = preceding context)
    tst.test_turn(
        turn=[
            "Nurse: Can you check your blood pressure?",
            "My blood pressure machine says 158 over 92.",
        ],
        context_window=5,
    )

Options:
    --turn TEXT              Current turn to classify (prompted if omitted)
    --context-turn TEXT      Preceding turn (repeatable, builds conversation context)
    --context TEXT           Surrounding context as a single raw string (legacy)
    --context-window INT     Max preceding turns to include (default: om.CONTEXT_WINDOW=2)
    --task                   ss | intervention | both   (default: both)
    --model TEXT             LLM config name (default: gpt-4o-mini)
    --top-k INT              Number of Omaha options to retrieve (default: 15)
    --embedding-model TEXT   SentenceTransformer model name
    --omaha-file TEXT        Local path to Omaha Excel (auto-detected if omitted)
    --ss-prompt-file TEXT    Path to .txt file with custom SS prompt template
    --int-prompt-file TEXT   Path to .txt file with custom intervention prompt
    --understand             Enable Agent 1 (clinical pre-analysis before classification)
    --verify                 Enable Agent 3 (verification after classification)
    --verbose                Show retrieved options, full prompt, and raw LLM response
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    import omaha_sagemaker as om
except ImportError as e:
    sys.exit(f"Cannot import omaha_sagemaker: {e}\n"
             f"Run: pip install -r {os.path.join(os.path.dirname(__file__), 'requirements.txt')}")

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Module-level index cache — avoids rebuilding embeddings on repeated calls
# ─────────────────────────────────────────────────────────────────────────────

_INDEX_CACHE: dict = {}  # (omaha_path, embed_model_name) → (ss_docs, ss_emb, int_docs, int_emb, model)


def _get_index(omaha_path: str, embed_model_name: str):
    """Load and cache the Omaha embedding index.  Returns cached copy if available."""
    key = (omaha_path, embed_model_name)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    from sentence_transformers import SentenceTransformer

    sheets = pd.read_excel(omaha_path, sheet_name=None, engine="openpyxl")
    names  = list(sheets.keys())
    ss_df  = sheets[names[0]].dropna(how="all").reset_index(drop=True)
    int_df = sheets[names[1]].dropna(how="all").reset_index(drop=True)
    ss_df.columns  = [c.strip() for c in ss_df.columns]
    int_df.columns = [c.strip() for c in int_df.columns]

    embed_model    = SentenceTransformer(embed_model_name, device="cpu")
    ss_docs        = om.build_ss_documents(ss_df)
    int_docs       = om.build_intervention_documents(int_df)
    ss_embeddings  = om.build_index(ss_docs,  embed_model)
    int_embeddings = om.build_index(int_docs, embed_model)

    _INDEX_CACHE[key] = (ss_docs, ss_embeddings, int_docs, int_embeddings, embed_model)
    return _INDEX_CACHE[key]


def clear_index_cache():
    """Call this if you change the Omaha Excel or embedding model mid-session."""
    _INDEX_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
CYAN   = lambda t: _c("96", t)
GREEN  = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
RED    = lambda t: _c("91", t)

def divider(title="", char="─", width=72):
    if title:
        pad = max(0, width - len(title) - 3)
        print(CYAN(f"{char*2} {BOLD(title)} {char*pad}"))
    else:
        print(DIM(char * width))


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_turn_item(item) -> tuple[str, str]:
    """Extract (speaker, text) from a context_turns item.

    Accepts:
      - dict with "spk"/"speaker" and "text"/"Conversation"
      - str  optionally prefixed with "Speaker: " (e.g. "Nurse: How are you?")
    """
    if isinstance(item, dict):
        spk = (item.get("spk") or item.get("Spk") or
               item.get("speaker") or item.get("Speaker") or "?")
        txt = (item.get("text") or item.get("Conversation") or
               item.get("turn") or "")
        return str(spk).strip(), str(txt).strip()
    else:
        txt = str(item).strip()
        m = re.match(r'^([A-Za-z /]+?)\s*:\s*(.+)$', txt, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return "?", txt


def _build_context_str(context_turns: list, window: int) -> str:
    """Build a context string from a list of preceding turns.

    Takes at most the last *window* turns so the context doesn't exceed the
    model's attention span.
    """
    if not context_turns:
        return "(no context)"
    windowed = context_turns[-window:] if window > 0 else context_turns
    lines = []
    for item in windowed:
        spk, txt = _parse_turn_item(item)
        if txt:
            lines.append(f"[{spk}]: {txt}")
    return "\n".join(lines) if lines else "(no context)"


def _find_omaha_excel(hint: str = None) -> str:
    candidates = []
    if hint:
        candidates.append(hint)
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


# ─────────────────────────────────────────────────────────────────────────────
# Main test function
# ─────────────────────────────────────────────────────────────────────────────

def test_turn(
    turn,                               # str OR list (last = current, rest = context)
    context: str = "",                  # legacy raw context string (overridden by context_turns)
    context_turns: list = None,         # list of preceding turns: str or {"spk":…, "text":…}
    context_window: int = None,         # max preceding turns to use (default: om.CONTEXT_WINDOW)
    task: str = "both",                 # "ss" | "intervention" | "both"
    model_name: str = "gpt-4o-mini",
    top_k: int = 15,
    embedding_model: str = None,        # None → om.EMBEDDING_MODEL
    omaha_file: str = None,
    ss_prompt_template: str = None,     # None → om.SS_PROMPT_TEMPLATE
    int_prompt_template: str = None,    # None → om.INTERVENTION_PROMPT_TEMPLATE
    use_understand: bool = False,       # Agent 1: clinical pre-analysis
    use_verify: bool = False,           # Agent 3: post-classification verification
    verbose: bool = False,
):
    """Run the Omaha mapping pipeline on a single conversation turn.

    Args:
        turn:            The current turn to classify. If a **list** is passed,
                         the last element is treated as the current turn and all
                         preceding elements become the context turns.
        context:         Legacy: raw context string. Ignored when context_turns
                         or a list-form turn is provided.
        context_turns:   Ordered list of preceding turns. Each item is either a
                         plain string (optionally prefixed "Speaker: text") or a
                         dict {"spk": "Nurse", "text": "…"}.
        context_window:  Maximum number of preceding turns to include as context.
                         Defaults to om.CONTEXT_WINDOW (currently 2).
        task:            "ss" | "intervention" | "both"
        model_name:      LLM config key in om.LLM_CONFIGS.
        top_k:           How many Omaha options to retrieve for the RAG step.
        embedding_model: SentenceTransformer model name (cached across calls).
        omaha_file:      Path to the Omaha Excel (auto-detected if omitted).
        ss_prompt_template:  Custom SS prompt (overrides om.SS_PROMPT_TEMPLATE).
        int_prompt_template: Custom INT prompt (overrides om.INTERVENTION_PROMPT_TEMPLATE).
        use_understand:  Run Agent 1 (clinical pre-analysis) before classification.
        use_verify:      Run Agent 3 (verification) after classification.
        verbose:         Print retrieved options, full prompt, and raw LLM output.

    Returns:
        dict with keys "ss" and/or "int", each a list of prediction dicts.
    """
    import logging
    if not verbose:
        logging.getLogger("omaha_sagemaker").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # ── Resolve list-form turn ────────────────────────────────────────────────
    if isinstance(turn, list):
        if len(turn) == 0:
            raise ValueError("turn list must not be empty")
        if context_turns is None:
            context_turns = turn[:-1]  # all but last are preceding turns
        turn = turn[-1]               # last element is the target turn

    # ── Resolve context ───────────────────────────────────────────────────────
    window = context_window if context_window is not None else om.CONTEXT_WINDOW
    if context_turns:
        ctx = _build_context_str(context_turns, window)
    elif context.strip():
        ctx = context.strip()         # legacy: plain string
    else:
        ctx = "(no context)"

    # ── Resolve Omaha Excel ───────────────────────────────────────────────────
    omaha_path = _find_omaha_excel(omaha_file)
    if omaha_path is None:
        print(RED("ERROR: Omaha Excel file not found."))
        print("  Place 'Omaha_system list with definition.xlsx' in the repo root "
              "or pass omaha_file=.")
        raise FileNotFoundError("Omaha Excel not found")

    # ── Override prompt templates ─────────────────────────────────────────────
    if ss_prompt_template:
        om.SS_PROMPT_TEMPLATE = ss_prompt_template
    if int_prompt_template:
        om.INTERVENTION_PROMPT_TEMPLATE = int_prompt_template

    # ── Load embedding index (cached) ─────────────────────────────────────────
    _embed_name = embedding_model or om.EMBEDDING_MODEL
    print(DIM(f"  [index] loading from cache or building ({_embed_name}) …"))
    ss_docs, ss_emb, int_docs, int_emb, embed_model = _get_index(omaha_path, _embed_name)

    # ── Header ────────────────────────────────────────────────────────────────
    agents = ["Agent2:classify"]
    if use_understand: agents.insert(0, "Agent1:understand")
    if use_verify:     agents.append("Agent3:verify")

    print()
    divider("SINGLE-TURN TEST", "═")
    print(f"  {BOLD('Model:')}           {model_name}")
    print(f"  {BOLD('Task:')}            {task}")
    print(f"  {BOLD('Top-K:')}           {top_k}")
    print(f"  {BOLD('Context window:')}  {window} preceding turns")
    print(f"  {BOLD('Pipeline:')}        {' → '.join(agents)}")
    if ss_prompt_template:  print(f"  {BOLD('SS prompt:')}       (custom)")
    if int_prompt_template: print(f"  {BOLD('Int prompt:')}      (custom)")
    print()

    divider("CONVERSATION CONTEXT")
    if ctx != "(no context)":
        for line in ctx.splitlines():
            print(DIM(f"  {line}"))
    else:
        print(DIM("  (no preceding context)"))

    print()
    divider("CURRENT TURN")
    print(YELLOW(f"  {turn}"))

    # ── Agent 1: Understand ───────────────────────────────────────────────────
    understanding = ""
    if use_understand:
        print()
        divider("AGENT 1 — Clinical Pre-Analysis")
        understanding = om.understand_turn(turn, ctx, model_name)
        for line in understanding.splitlines():
            print(GREEN(f"  {line}"))

    # ── Agent 2: Signs/Symptoms ───────────────────────────────────────────────
    ss_parsed = []
    if task in ("ss", "both"):
        print()
        divider("AGENT 2 — Signs/Symptoms Classification")

        retrieved_ss = om.retrieve(turn, ss_docs, ss_emb, embed_model, top_k)
        ss_prompt    = om.build_ss_prompt(turn, ctx, retrieved_ss, understanding)
        if om.LLM_CONFIGS[model_name]["provider"] != "local":
            import re as _re
            ss_prompt = _re.sub(r"<\|[^|>]+\|>", "", ss_prompt).strip()

        if verbose:
            print(DIM(f"  Retrieved {len(retrieved_ss)} SS options."))
            print()
            divider(f"SS PROMPT ({len(ss_prompt)} chars)", "·")
            for line in ss_prompt.splitlines():
                print(DIM(f"  {line}"))
            print()
            divider("SS RAW RESPONSE", "·")

        ss_raw    = om.call_llm(ss_prompt, model_name)
        ss_parsed = om.parse_ss_output(ss_raw)

        if verbose:
            print(GREEN(f"  {ss_raw}"))
        else:
            print(DIM(f"  Raw: {ss_raw[:120].replace(chr(10), ' ')}{'…' if len(ss_raw)>120 else ''}"))

        print()
        print(BOLD("  Parsed SS:"))
        if ss_parsed:
            for i, r in enumerate(ss_parsed, 1):
                print(GREEN(f"    [{i}] {r['domain']} | {r['problem']} | {r['ss']}"))
        else:
            print(YELLOW("    → NONE"))

    # ── Agent 2: Interventions ────────────────────────────────────────────────
    int_parsed = []
    if task in ("intervention", "both"):
        print()
        divider("AGENT 2 — Intervention Classification")

        retrieved_int = om.retrieve(turn, int_docs, int_emb, embed_model, top_k)
        int_prompt    = om.build_intervention_prompt(turn, ctx, retrieved_int, understanding)

        if verbose:
            print(DIM(f"  Retrieved {len(retrieved_int)} intervention options."))
            print()
            divider(f"INT PROMPT ({len(int_prompt)} chars)", "·")
            for line in int_prompt.splitlines():
                print(DIM(f"  {line}"))
            print()
            divider("INT RAW RESPONSE", "·")

        int_raw    = om.call_llm(int_prompt, model_name)
        int_parsed = om.parse_intervention_output(int_raw)

        if verbose:
            print(GREEN(f"  {int_raw}"))
        else:
            print(DIM(f"  Raw: {int_raw[:120].replace(chr(10), ' ')}{'…' if len(int_raw)>120 else ''}"))

        print()
        print(BOLD("  Parsed INT:"))
        if int_parsed:
            for i, r in enumerate(int_parsed, 1):
                print(GREEN(f"    [{i}] {r['category']} | {r['target']}"))
        else:
            print(YELLOW("    → NONE"))

    # ── Agent 3: Verify ───────────────────────────────────────────────────────
    if use_verify:
        print()
        divider("AGENT 3 — Verification")

        if task in ("ss", "both") and ss_parsed:
            ss_v = om._verify_ss(turn, ss_parsed, ss_docs, ss_emb, embed_model, model_name)
            if ss_v != ss_parsed:
                print(YELLOW("  SS corrected →"))
                for r in ss_v:
                    print(GREEN(f"    {r['domain']} | {r['problem']} | {r['ss']}"))
            else:
                print(DIM("  SS: CONFIRMED (no change)"))
            ss_parsed = ss_v
        elif task in ("ss", "both"):
            print(DIM("  SS: skipped (no predictions to verify)"))

        if task in ("intervention", "both") and int_parsed:
            int_v = om._verify_int(turn, ctx, int_parsed, int_docs, int_emb,
                                   embed_model, model_name)
            if int_v != int_parsed:
                print(YELLOW("  INT corrected →"))
                for r in int_v:
                    print(GREEN(f"    {r['category']} | {r['target']}"))
            else:
                print(DIM("  INT: CONFIRMED (no change)"))
            int_parsed = int_v
        elif task in ("intervention", "both"):
            print(DIM("  INT: skipped (no predictions to verify)"))

    # ── Final result ──────────────────────────────────────────────────────────
    print()
    divider("FINAL RESULT", "═")
    print(YELLOW(f"  Turn: {turn}"))
    print()

    if task in ("ss", "both"):
        print(BOLD("  Signs / Symptoms:"))
        if ss_parsed:
            for i, r in enumerate(ss_parsed, 1):
                print(GREEN(f"    [{i}] Domain:         {r['domain']}"))
                print(GREEN(f"         Problem:        {r['problem']}"))
                print(GREEN(f"         Signs/Symptoms: {r['ss']}"))
        else:
            print(YELLOW("    → None"))

    if task == "both":
        print()

    if task in ("intervention", "both"):
        print(BOLD("  Intervention:"))
        if int_parsed:
            for i, r in enumerate(int_parsed, 1):
                print(GREEN(f"    [{i}] Category: {r['category']} | Target: {r['target']}"))
        else:
            print(YELLOW("    → None"))

    print()
    divider("", "═")
    print()

    result = {}
    if task in ("ss", "both"):         result["ss"]  = ss_parsed
    if task in ("intervention", "both"): result["int"] = int_parsed
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test a single Omaha mapping turn without running the full pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--turn",             default=None,
                        help="Current conversation turn to classify")
    parser.add_argument("--context-turn",     action="append", default=[],
                        dest="context_turns", metavar="TEXT",
                        help="Preceding turn (repeatable; ordered oldest→newest). "
                             "Format: 'Speaker: text' or plain text.")
    parser.add_argument("--context",          default="",
                        help="Surrounding context as a single raw string (legacy; "
                             "ignored when --context-turn is provided)")
    parser.add_argument("--context-window",   type=int, default=None,
                        dest="context_window",
                        help=f"Max preceding turns to include as context "
                             f"(default: om.CONTEXT_WINDOW={om.CONTEXT_WINDOW})")
    parser.add_argument("--task",             default="both",
                        choices=["ss", "intervention", "both"])
    parser.add_argument("--model",            default="gpt-4o-mini")
    parser.add_argument("--top-k",            type=int, default=15)
    parser.add_argument("--embedding-model",  default=None)
    parser.add_argument("--omaha-file",       default=None)
    parser.add_argument("--ss-prompt-file",   default=None)
    parser.add_argument("--int-prompt-file",  default=None)
    parser.add_argument("--understand",       action="store_true",
                        help="Enable Agent 1 (clinical pre-analysis)")
    parser.add_argument("--verify",           action="store_true",
                        help="Enable Agent 3 (post-classification verification)")
    parser.add_argument("--verbose",          action="store_true")
    args = parser.parse_args()

    if args.model not in om.LLM_CONFIGS:
        print(RED(f"Unknown model '{args.model}'."))
        print(f"Available: {list(om.LLM_CONFIGS.keys())}")
        sys.exit(1)

    turn = args.turn
    if not turn:
        print("Enter the current conversation turn (press Enter twice when done):")
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

    ss_prompt_template = None
    if args.ss_prompt_file:
        if not os.path.isfile(args.ss_prompt_file):
            sys.exit(RED(f"ERROR: SS prompt file not found: {args.ss_prompt_file}"))
        with open(args.ss_prompt_file) as f:
            ss_prompt_template = f.read()

    int_prompt_template = None
    if args.int_prompt_file:
        if not os.path.isfile(args.int_prompt_file):
            sys.exit(RED(f"ERROR: INT prompt file not found: {args.int_prompt_file}"))
        with open(args.int_prompt_file) as f:
            int_prompt_template = f.read()

    test_turn(
        turn=turn,
        context=args.context,
        context_turns=args.context_turns or None,
        context_window=args.context_window,
        task=args.task,
        model_name=args.model,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        omaha_file=args.omaha_file,
        ss_prompt_template=ss_prompt_template,
        int_prompt_template=int_prompt_template,
        use_understand=args.understand,
        use_verify=args.verify,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
