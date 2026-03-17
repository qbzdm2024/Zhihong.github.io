"""
Row-Level Time Distribution — standalone runner
================================================
Runs compute_time_distribution() on either:

  (A) The ground-truth annotation file (Completed_annotation.xlsx)
      Uses OS_I_1/2/3 labels + Spk/Speaker column.
      No LLM or GPU needed.

  (B) A previously saved LLM results Excel from S3
      Uses Pred_I_1_category + LLM_understand_raw column.

Usage (SageMaker notebook):

  # Option A — run on ground-truth annotation file (default)
  !python omaha_mapping/run_time_dist.py

  # Option A — explicit annotation file path on S3
  !python omaha_mapping/run_time_dist.py --annotation

  # Option B — run on a specific LLM results file
  !python omaha_mapping/run_time_dist.py --results gpt-4o_20240315_120000_results.xlsx

  # Option B — run on the most recent LLM results file automatically
  !python omaha_mapping/run_time_dist.py --results latest

Output is printed to console and saved to S3 as:
  <original_key>_time_dist.xlsx  (TIME_OVERVIEW + TIME_BY_CATEGORY sheets)
"""

import argparse
import logging
import sys
from datetime import datetime
from io import BytesIO

import boto3
import pandas as pd

from omaha_sagemaker import (
    S3_BUCKET,
    OUTPUT_PREFIX,
    ANNOTATION_KEY,
    compute_time_distribution,
    s3_write_excel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

s3 = boto3.client("s3")

STRATEGIES = ("label_only", "next_nurse", "fixed_2", "full_exchange")
STRAT_DESC = {
    "label_only":    "labeled turn only (lower bound)",
    "next_nurse":    "labeled + patient replies until next nurse turn",
    "fixed_2":       "labeled ± 2 rows  (matches CONTEXT_WINDOW)",
    "full_exchange": "labeled + all contiguous non-nurse rows both ways",
}
SKIP_SHEETS = {"SUMMARY", "TIME_OVERVIEW", "TIME_BY_CATEGORY"}


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _read_excel(key: str) -> dict[str, pd.DataFrame]:
    log.info(f"Reading s3://{S3_BUCKET}/{key}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    buf = BytesIO(obj["Body"].read())
    sheets = pd.read_excel(buf, sheet_name=None, engine="openpyxl")
    return {k: v.fillna("") for k, v in sheets.items()}


def _latest_results_key() -> str:
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=OUTPUT_PREFIX)
    keys = sorted(
        [o["Key"] for o in resp.get("Contents", [])
         if o["Key"].endswith("_results.xlsx")],
        reverse=True,
    )
    if not keys:
        log.error(f"No *_results.xlsx found under s3://{S3_BUCKET}/{OUTPUT_PREFIX}")
        sys.exit(1)
    return keys[0]


# ── Core ─────────────────────────────────────────────────────────────────────

def run(source_key: str):
    sheets = _read_excel(source_key)
    conv_sheets = {k: v for k, v in sheets.items() if k not in SKIP_SHEETS}
    log.info(f"Processing {len(conv_sheets)} sheet(s) from {source_key}")

    all_td        = []
    overview_rows = []
    cat_rows      = []

    for sheet_name, df in conv_sheets.items():
        td = compute_time_distribution(df)
        if not td:
            log.warning(f"  Sheet '{sheet_name}': empty or missing required columns — skipped")
            continue
        all_td.append(td)

        ov = {
            "Sheet":       sheet_name,
            "DataSource":  td.get("data_source", "?"),
            "Total_Rows":  td["total_rows"],
            "Nurse_Rows":  td["nurse_rows"],
            "Patient_Rows":td["patient_rows"],
            "Unclear_Rows":td["unclear_rows"],
            "Nurse_Pct":   td["nurse_pct"],
        }
        for strat in STRATEGIES:
            s = td.get(strat, {})
            ov[f"{strat}_IntRows"]     = s.get("total_int_rows",   0)
            ov[f"{strat}_IntPctNurse"] = s.get("int_pct_of_nurse", 0.0)
            for cat, cnt in s.get("by_category", {}).items():
                cat_rows.append({
                    "Sheet":    sheet_name,
                    "Strategy": strat,
                    "Category": cat,
                    "Rows":     cnt,
                })
        overview_rows.append(ov)

    if not all_td:
        log.error(
            "No sheets produced time data.\n"
            "  Ground-truth mode needs: OS_I_1 + Spk/Speaker columns\n"
            "  LLM results mode needs:  Pred_I_1_category + LLM_understand_raw columns"
        )
        sys.exit(1)

    # ── Console summary ───────────────────────────────────────────────────────
    source_label = all_td[0].get("data_source", "?")
    tot = sum(t["total_rows"]   for t in all_td)
    nur = sum(t["nurse_rows"]   for t in all_td)
    pat = sum(t["patient_rows"] for t in all_td)
    unc = sum(t["unclear_rows"] for t in all_td)

    print(f"\nROW-LEVEL TIME DISTRIBUTION  |  source={source_label}  ({len(all_td)} visits)")
    print(f"{'─'*68}")
    print(f"  Total rows   : {tot}")
    print(f"  Nurse rows   : {nur}  ({nur/tot*100:.1f}%  includes unclear/close turns)")
    print(f"  Patient rows : {pat}  ({pat/tot*100:.1f}%)")
    print(f"  Unclear rows : {unc}  ({unc/tot*100:.1f}%  counted as nurse time)")

    print(f"\n  {'Strategy':<16}  {'Int Rows':>9}  {'% Nurse':>7}  Rationale")
    print(f"  {'─'*74}")
    for strat in STRATEGIES:
        int_rows = sum(t.get(strat, {}).get("total_int_rows",   0) for t in all_td)
        pct      = round(int_rows / nur * 100, 1) if nur else 0.0
        print(f"  {strat:<16}  {int_rows:>9}  {pct:>6.1f}%  {STRAT_DESC[strat]}")

    print(f"\n  Category breakdown  (full_exchange strategy, all visits):")
    cat_agg: dict[str, int] = {}
    for t in all_td:
        for cat, cnt in t.get("full_exchange", {}).get("by_category", {}).items():
            cat_agg[cat] = cat_agg.get(cat, 0) + cnt
    for cat, cnt in sorted(cat_agg.items(), key=lambda x: -x[1]):
        pct = round(cnt / nur * 100, 1) if nur else 0.0
        print(f"    {cat:<40}  {cnt:>6} rows  ({pct:.1f}% of nurse time)")
    print(f"{'─'*68}")

    # ── Save to S3 ────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = "gt" if source_label == "groundtruth" else "llm"
    out_key = f"{OUTPUT_PREFIX}time_dist_{tag}_{ts}.xlsx"
    s3_write_excel(
        {
            "TIME_OVERVIEW":    pd.DataFrame(overview_rows),
            "TIME_BY_CATEGORY": pd.DataFrame(cat_rows) if cat_rows else pd.DataFrame(),
        },
        S3_BUCKET,
        out_key,
    )
    print(f"\nSaved: s3://{S3_BUCKET}/{out_key}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Row-level time distribution on annotation or LLM results file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--annotation", action="store_true",
        help="Run on ground-truth annotation file (default)",
    )
    group.add_argument(
        "--results", metavar="KEY",
        help='S3 key or filename of LLM results Excel, or "latest" for most recent',
    )
    args = parser.parse_args()

    if args.results:
        key = args.results
        if key == "latest":
            key = _latest_results_key()
            log.info(f"Using most recent results: {key}")
        elif not key.startswith(OUTPUT_PREFIX):
            key = OUTPUT_PREFIX + key
    else:
        # Default: ground-truth annotation file
        key = ANNOTATION_KEY

    run(key)


if __name__ == "__main__":
    main()
