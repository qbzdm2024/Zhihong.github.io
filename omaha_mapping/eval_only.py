"""
Re-run evaluation on an existing results xlsx without re-running the LLM.

Useful when you want to:
  - Try a different FUZZY_THRESHOLD
  - Fix a bug in the evaluation logic
  - Inspect predictions vs ground truth

Usage:
    python eval_only.py <s3_results_key_or_local_path> [fuzzy_threshold]

Examples:
    # Re-evaluate from S3 with default threshold (80)
    python eval_only.py Fake example_Sandy/omaha-mapping/output/gpt-4o-mini_20240101_results.xlsx

    # Try a lower fuzzy threshold (70)
    python eval_only.py path/to/results.xlsx 70

    # From local file
    python eval_only.py /tmp/gpt-4o-mini_results.xlsx 75
"""

import sys
import boto3
import pandas as pd
from io import BytesIO
import omaha_sagemaker as om

S3_BUCKET = om.S3_BUCKET

def load_results(path: str) -> dict[str, pd.DataFrame]:
    """Load results xlsx from S3 key or local path. Returns dict of sheet→DataFrame."""
    if path.startswith("s3://"):
        path = path[len(f"s3://{S3_BUCKET}/"):]

    if path.startswith("/") or path.startswith("."):
        # Local file
        return pd.read_excel(path, sheet_name=None)
    else:
        # S3 key
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=path)
        return pd.read_excel(BytesIO(obj["Body"].read()), sheet_name=None)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]

    if len(sys.argv) >= 3:
        om.FUZZY_THRESHOLD = int(sys.argv[2])
        print(f"FUZZY_THRESHOLD set to {om.FUZZY_THRESHOLD}")

    print(f"Loading results from: {path}")
    all_sheets = load_results(path)

    # Skip the SUMMARY sheet (it has no predictions)
    conv_sheets = {k: v for k, v in all_sheets.items() if k != "SUMMARY"}
    print(f"Found {len(conv_sheets)} conversation sheets")

    sheet_metrics = []
    for sheet_name, df in conv_sheets.items():
        metrics = om.evaluate_sheet(df)
        metrics["sheet"] = sheet_name
        sheet_metrics.append(metrics)
        print(
            f"  {sheet_name:<30}  SS_F1={metrics['ss_f1']:.3f}  "
            f"SSOnly_F1={metrics['ss_only_f1']:.3f}  "
            f"Int_F1={metrics['int_f1']:.3f}  "
            f"Tgt_F1={metrics['tgt_f1']:.3f}"
        )

    # Aggregate across all sheets
    from omaha_sagemaker import compute_full_metrics

    def _agg(key):
        return sum(m[key] for m in sheet_metrics)

    def _full(tp_key, fp_key, fn_key, tn_key):
        return compute_full_metrics(_agg(tp_key), _agg(fp_key), _agg(fn_key), _agg(tn_key))

    def _prf_row(prefix, pos_p, pos_r, pos_f1, none_p, none_r, none_f1,
                 macro_p, macro_r, macro_f1, tp=None, fp=None, fn=None, tn=None):
        counts = ""
        if tp is not None: counts = f"  TP={tp} FP={fp} FN={fn} TN={tn}"
        return (f"  {prefix:<20}│ Pos P={pos_p:.3f} R={pos_r:.3f} F1={pos_f1:.3f}"
                f"  None P={none_p:.3f} R={none_r:.3f} F1={none_f1:.3f}"
                f"  ★Macro F1={macro_f1:.3f}{counts}")

    (ss_pp, ss_pr, ss_pf, ss_np, ss_nr, ss_nf, ss_mp, ss_mr, ss_mf) = \
        _full("ss_tp",      "ss_fp",      "ss_fn",      "ss_tn")
    (so_pp, so_pr, so_pf, so_np, so_nr, so_nf, so_mp, so_mr, so_mf) = \
        _full("ss_only_tp", "ss_only_fp", "ss_only_fn", "ss_only_tn")
    (pb_pp, pb_pr, pb_pf, pb_np, pb_nr, pb_nf, pb_mp, pb_mr, pb_mf) = \
        _full("prob_tp",    "prob_fp",    "prob_fn",    "prob_tn")
    (i_pp,  i_pr,  i_pf,  i_np,  i_nr,  i_nf,  i_mp,  i_mr,  i_mf)  = \
        _full("int_tp",     "int_fp",     "int_fn",     "int_tn")
    (ca_pp, ca_pr, ca_pf, ca_np, ca_nr, ca_nf, ca_mp, ca_mr, ca_mf) = \
        _full("cat_tp",     "cat_fp",     "cat_fn",     "cat_tn")
    (t_pp,  t_pr,  t_pf,  t_np,  t_nr,  t_nf,  t_mp,  t_mr,  t_mf)  = \
        _full("tgt_tp",     "tgt_fp",     "tgt_fn",     "tgt_tn")

    w = 100
    print(f"\n{'='*w}")
    print(f"AGGREGATE  (threshold={om.FUZZY_THRESHOLD})")
    print(f"{'='*w}")

    print(f"\n  SIGNS / SYMPTOMS")
    print(_prf_row("Combined labels",
                   ss_pp, ss_pr, ss_pf, ss_np, ss_nr, ss_nf, ss_mp, ss_mr, ss_mf,
                   tp=_agg("ss_tp"), fp=_agg("ss_fp"), fn=_agg("ss_fn"), tn=_agg("ss_tn")))
    print(_prf_row("Problem only",
                   pb_pp, pb_pr, pb_pf, pb_np, pb_nr, pb_nf, pb_mp, pb_mr, pb_mf,
                   tp=_agg("prob_tp"), fp=_agg("prob_fp"), fn=_agg("prob_fn"), tn=_agg("prob_tn")))
    print(_prf_row("Sign/Symptom only",
                   so_pp, so_pr, so_pf, so_np, so_nr, so_nf, so_mp, so_mr, so_mf,
                   tp=_agg("ss_only_tp"), fp=_agg("ss_only_fp"), fn=_agg("ss_only_fn"), tn=_agg("ss_only_tn")))

    print(f"\n  INTERVENTIONS")
    print(_prf_row("Combined labels",
                   i_pp,  i_pr,  i_pf,  i_np,  i_nr,  i_nf,  i_mp,  i_mr,  i_mf,
                   tp=_agg("int_tp"), fp=_agg("int_fp"), fn=_agg("int_fn"), tn=_agg("int_tn")))
    print(_prf_row("Category only",
                   ca_pp, ca_pr, ca_pf, ca_np, ca_nr, ca_nf, ca_mp, ca_mr, ca_mf,
                   tp=_agg("cat_tp"), fp=_agg("cat_fp"), fn=_agg("cat_fn"), tn=_agg("cat_tn")))
    print(_prf_row("Target only",
                   t_pp,  t_pr,  t_pf,  t_np,  t_nr,  t_nf,  t_mp,  t_mr,  t_mf,
                   tp=_agg("tgt_tp"), fp=_agg("tgt_fp"), fn=_agg("tgt_fn"), tn=_agg("tgt_tn")))
    print(f"\n{'='*w}")

if __name__ == "__main__":
    main()
