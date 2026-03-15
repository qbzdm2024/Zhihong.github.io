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

    # Aggregate
    import numpy as np
    from omaha_sagemaker import compute_prf

    def _agg(key):
        return sum(m[key] for m in sheet_metrics)

    ss_p, ss_r, ss_f1     = compute_prf(_agg("ss_tp"),      _agg("ss_fp"),      _agg("ss_fn"))
    so_p, so_r, so_f1     = compute_prf(_agg("ss_only_tp"), _agg("ss_only_fp"), _agg("ss_only_fn"))
    i_p,  i_r,  i_f1      = compute_prf(_agg("int_tp"),     _agg("int_fp"),     _agg("int_fn"))
    t_p,  t_r,  t_f1      = compute_prf(_agg("tgt_tp"),     _agg("tgt_fp"),     _agg("tgt_fn"))

    print(f"\n{'='*60}")
    print(f"AGGREGATE  (threshold={om.FUZZY_THRESHOLD})")
    print(f"{'='*60}")
    print(f"SS combined   P={ss_p:.4f}  R={ss_r:.4f}  F1={ss_f1:.4f}")
    print(f"SS only       P={so_p:.4f}  R={so_r:.4f}  F1={so_f1:.4f}")
    print(f"Int combined  P={i_p:.4f}   R={i_r:.4f}   F1={i_f1:.4f}")
    print(f"Int target    P={t_p:.4f}   R={t_r:.4f}   F1={t_f1:.4f}")
    print(f"{'='*60}")
    print(f"TP/FP/FN (SS):   {_agg('ss_tp')}/{_agg('ss_fp')}/{_agg('ss_fn')}")
    print(f"TP/FP/FN (Int):  {_agg('int_tp')}/{_agg('int_fp')}/{_agg('int_fn')}")

if __name__ == "__main__":
    main()
