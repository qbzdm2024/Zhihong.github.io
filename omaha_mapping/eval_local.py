"""
Local evaluation script — runs the full pipeline on the example TSV file
and reports SS / Intervention F1 scores with detailed error analysis.

Usage:
    python eval_local.py                          # baseline prompts
    python eval_local.py --prompts refined        # prompts from refined_prompts.py
    python eval_local.py --prompts refined --verbose   # + row-by-row diff

Options:
    --example FILE   path to annotated TSV (default: ./example)
    --omaha FILE     path to Omaha Excel  (default: auto-detect)
    --model NAME     LLM config name      (default: gpt-4o-mini)
    --top-k INT      retrieval K          (default: 15)
    --prompts        baseline | refined   (default: baseline)
    --verbose        print every FP/FN row
"""

import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("omaha_sagemaker").setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import omaha_sagemaker as om

# ── helpers ────────────────────────────────────────────────────────────────

def find_omaha_excel(hint=None):
    candidates = []
    if hint:
        candidates.append(hint)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates += [
        os.path.join(repo_root, "Omaha_system list with definition.xlsx"),
        os.path.join(os.path.dirname(__file__), "Omaha_system list with definition.xlsx"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def load_example(path):
    """Load the annotated TSV example file into a DataFrame."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df


def build_resources(omaha_path, top_k, embed_model_name):
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
    return ss_docs, ss_embeddings, int_docs, int_embeddings, embed_model


# ── evaluation ─────────────────────────────────────────────────────────────

def run_and_eval(df, ss_docs, ss_embeddings, int_docs, int_embeddings,
                 embed_model, llm_name, top_k, verbose,
                 use_understand=False):
    """Run process_sheet + evaluate_sheet and return (metrics, df_out).

    Pipeline:
      Agent 1 (understand)  — enabled by use_understand=True
      Agent 2 (classify)    — always runs (RAG + LLM)
      Agent 3 (verify)      — called separately via verify_sheet() if needed
    """
    orig_k = om.TOP_K_RETRIEVAL
    om.TOP_K_RETRIEVAL = top_k

    df_out = om.process_sheet(df, ss_docs, ss_embeddings,
                              int_docs, int_embeddings,
                              embed_model, llm_name,
                              use_understand=use_understand)
    om.TOP_K_RETRIEVAL = orig_k

    metrics = om.evaluate_sheet(df_out)

    if verbose:
        _print_errors(df_out)

    return metrics, df_out


def _print_errors(df_out):
    sep = "─" * 80
    print(f"\n{sep}")
    print("ROW-BY-ROW ERRORS")
    print(sep)
    for _, row in df_out.iterrows():
        turn = str(row.get("Conversation", ""))[:90]

        human_ss  = [str(row.get(f"OS_SS_{j}","")).strip() for j in range(1,4)
                     if str(row.get(f"OS_SS_{j}","")).strip() not in ("","nan")]
        pred_ss   = [str(row.get(f"Pred_SS_{j}","")).strip() for j in range(1,4)
                     if str(row.get(f"Pred_SS_{j}","")).strip() not in ("","nan")]
        human_int = [str(row.get(f"OS_I_{j}","")).strip() for j in range(1,4)
                     if str(row.get(f"OS_I_{j}","")).strip() not in ("","nan")]
        pred_int  = [str(row.get(f"Pred_I_{j}","")).strip() for j in range(1,4)
                     if str(row.get(f"Pred_I_{j}","")).strip() not in ("","nan")]

        ss_wrong  = (human_ss != [] or pred_ss != []) and (
            not any(om.fuzzy_label_match(p, h) for p in pred_ss for h in human_ss)
            and (human_ss or pred_ss)
        )
        # More precise: compute tp/fp/fn
        ss_tp, ss_fp, ss_fn = om.row_level_match(pred_ss, human_ss)
        int_tp, int_fp, int_fn = om.row_level_match(pred_int, human_int)

        if ss_fp > 0 or ss_fn > 0 or int_fp > 0 or int_fn > 0:
            print(f"\nTURN : {turn!r}")
            if ss_fn > 0:
                print(f"  SS  FN (missed)  : {[h for h in human_ss]}")
                print(f"  SS  predicted    : {pred_ss}")
            if ss_fp > 0:
                print(f"  SS  FP (wrong)   : {pred_ss}")
                print(f"  SS  expected     : {human_ss}")
            if int_fn > 0:
                print(f"  INT FN (missed)  : {[h for h in human_int]}")
                print(f"  INT predicted    : {pred_int}")
            if int_fp > 0:
                print(f"  INT FP (wrong)   : {pred_int}")
                print(f"  INT expected     : {human_int}")
    print(sep)


def _prf_row(prefix, p, r, f1, tp=None, fp=None, fn=None, tn=None):
    counts = ""
    if tp is not None: counts += f"  TP={tp} FP={fp} FN={fn}"
    if tn is not None: counts += f"  TN={tn}"
    return f"  {prefix:<18}│ P={p:.3f}  R={r:.3f}  F1={f1:.3f}{counts}"


def _sub_section(title, pos_p, pos_r, pos_f1, none_p, none_r, none_f1,
                 macro_p, macro_r, macro_f1, tp, fp, fn, tn):
    lines = [f"  {title}"]
    lines.append(f"    Positive      │ P={pos_p:.3f}  R={pos_r:.3f}  F1={pos_f1:.3f}"
                 f"  TP={tp} FP={fp} FN={fn}")
    lines.append(f"    None class    │ P={none_p:.3f}  R={none_r:.3f}  F1={none_f1:.3f}"
                 f"  TN={tn}")
    lines.append(f"    ★ Macro       │ P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")
    return "\n".join(lines)


def print_metrics(metrics, label):
    w = 68
    print(f"\n{'═'*w}")
    print(f"  {label}")

    # ── SIGNS / SYMPTOMS ─────────────────────────────────────────────────────
    print(f"\n  SIGNS / SYMPTOMS")
    print(_prf_row("Positive class",
                   metrics["ss_precision"], metrics["ss_recall"], metrics["ss_f1"],
                   tp=metrics["ss_tp"], fp=metrics["ss_fp"], fn=metrics["ss_fn"]))
    print(_prf_row("None class",
                   metrics["ss_none_p"], metrics["ss_none_r"], metrics["ss_none_f1"],
                   tn=metrics["ss_tn"]))
    print(_prf_row("★ Macro",
                   metrics["ss_macro_p"], metrics["ss_macro_r"], metrics["ss_macro_f1"]))
    print(f"  {'Accuracy':<18}│ {metrics['ss_accuracy']:.3f}"
          f"  ({metrics['ss_tp']+metrics['ss_tn']} / "
          f"{metrics['ss_tp']+metrics['ss_fp']+metrics['ss_fn']+metrics['ss_tn']} turns)")

    print()
    print(_sub_section("Problem only",
                       metrics["prob_precision"],    metrics["prob_recall"],    metrics["prob_f1"],
                       metrics["prob_none_p"],        metrics["prob_none_r"],    metrics["prob_none_f1"],
                       metrics["prob_macro_p"],       metrics["prob_macro_r"],   metrics["prob_macro_f1"],
                       metrics["prob_tp"], metrics["prob_fp"], metrics["prob_fn"], metrics["prob_tn"]))

    print()
    print(_sub_section("Sign/Symptom only",
                       metrics["ss_only_precision"], metrics["ss_only_recall"], metrics["ss_only_f1"],
                       metrics["ss_only_none_p"],     metrics["ss_only_none_r"], metrics["ss_only_none_f1"],
                       metrics["ss_only_macro_p"],    metrics["ss_only_macro_r"],metrics["ss_only_macro_f1"],
                       metrics["ss_only_tp"], metrics["ss_only_fp"], metrics["ss_only_fn"], metrics["ss_only_tn"]))

    # ── INTERVENTIONS ────────────────────────────────────────────────────────
    print(f"\n  INTERVENTIONS")
    print(_prf_row("Positive class",
                   metrics["int_precision"], metrics["int_recall"], metrics["int_f1"],
                   tp=metrics["int_tp"], fp=metrics["int_fp"], fn=metrics["int_fn"]))
    print(_prf_row("None class",
                   metrics["int_none_p"], metrics["int_none_r"], metrics["int_none_f1"],
                   tn=metrics["int_tn"]))
    print(_prf_row("★ Macro",
                   metrics["int_macro_p"], metrics["int_macro_r"], metrics["int_macro_f1"]))
    print(f"  {'Accuracy':<18}│ {metrics['int_accuracy']:.3f}"
          f"  ({metrics['int_tp']+metrics['int_tn']} / "
          f"{metrics['int_tp']+metrics['int_fp']+metrics['int_fn']+metrics['int_tn']} turns)")

    print()
    print(_sub_section("Category only",
                       metrics["cat_precision"], metrics["cat_recall"], metrics["cat_f1"],
                       metrics["cat_none_p"],     metrics["cat_none_r"], metrics["cat_none_f1"],
                       metrics["cat_macro_p"],    metrics["cat_macro_r"],metrics["cat_macro_f1"],
                       metrics["cat_tp"], metrics["cat_fp"], metrics["cat_fn"], metrics["cat_tn"]))

    print()
    print(_sub_section("Target only",
                       metrics["tgt_precision"], metrics["tgt_recall"], metrics["tgt_f1"],
                       metrics["tgt_none_p"],     metrics["tgt_none_r"], metrics["tgt_none_f1"],
                       metrics["tgt_macro_p"],    metrics["tgt_macro_r"],metrics["tgt_macro_f1"],
                       metrics["tgt_tp"], metrics["tgt_fp"], metrics["tgt_fn"], metrics["tgt_tn"]))

    print(f"\n{'═'*w}")
    return metrics["ss_f1"], metrics["int_f1"]


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local F1 eval on example TSV")
    parser.add_argument("--example",  default=os.path.join(os.path.dirname(__file__), "example"))
    parser.add_argument("--omaha",    default=None)
    parser.add_argument("--model",    default="gpt-4o-mini")
    parser.add_argument("--top-k",    type=int, default=15)
    parser.add_argument("--prompts",    default="baseline", choices=["baseline","refined"])
    parser.add_argument("--understand", action="store_true",
                        help="Enable Agent 1 (clinical pre-analysis) before classification")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.model not in om.LLM_CONFIGS:
        sys.exit(f"Unknown model '{args.model}'. Available: {list(om.LLM_CONFIGS.keys())}")

    omaha_path = find_omaha_excel(args.omaha)
    if not omaha_path:
        sys.exit("Omaha Excel not found. Pass --omaha <path>.")

    if not os.path.isfile(args.example):
        sys.exit(f"Example file not found: {args.example}")

    # ── optionally swap in refined prompts ─────────────────────────────────
    if args.prompts == "refined":
        try:
            from refined_prompts import MY_SS_PROMPT, MY_INT_PROMPT
            om.SS_PROMPT_TEMPLATE            = MY_SS_PROMPT
            om.INTERVENTION_PROMPT_TEMPLATE  = MY_INT_PROMPT
            print("Using REFINED prompts from refined_prompts.py")
        except ImportError:
            sys.exit("refined_prompts.py not found. Run from omaha_mapping/ directory.")
    else:
        print("Using BASELINE prompts from omaha_sagemaker.py")

    print(f"Loading Omaha Excel: {omaha_path}")
    print(f"Building embedding index …", flush=True)
    ss_docs, ss_emb, int_docs, int_emb, embed_model = build_resources(
        omaha_path, args.top_k, om.EMBEDDING_MODEL
    )

    print(f"Loading example: {args.example}")
    df = load_example(args.example)
    print(f"  {len(df)} rows loaded")

    agents = ['Agent2:classify']
    if args.understand: agents.insert(0, 'Agent1:understand')
    print(f"Running inference with {args.model} (top-k={args.top_k})  "
          f"pipeline=[{' → '.join(agents)}] …", flush=True)
    metrics, df_out = run_and_eval(
        df, ss_docs, ss_emb, int_docs, int_emb,
        embed_model, args.model, args.top_k, args.verbose,
        use_understand=args.understand,
    )

    ss_f1, int_f1 = print_metrics(metrics, f"prompts={args.prompts}  model={args.model}")

    target = 0.9
    gap_ss  = max(0, target - ss_f1)
    gap_int = max(0, target - int_f1)
    if gap_ss > 0 or gap_int > 0:
        print(f"\n  Gap to F1=0.90 → SS: {gap_ss:+.3f}   INT: {gap_int:+.3f}")
    else:
        print(f"\n  ✓ Both SS F1 and INT F1 ≥ 0.90")


if __name__ == "__main__":
    main()
