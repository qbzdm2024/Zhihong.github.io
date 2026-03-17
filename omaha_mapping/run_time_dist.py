"""
Time Distribution Analysis — standalone runner
===============================================
Two modes, one script:

  (A) Ground-truth annotation file  (default / --annotation)
      Uses actual SONY timestamps → real seconds per category and target.
      Aggregates mean ± SD across all 23 visits.
      No LLM or GPU needed.

  (B) LLM results file  (--results)
      Uses row-count proxy (each row = 1 time step) because results files
      carry no SONY timestamps.
      Uses Pred_I_1_category + LLM_understand_raw for speaker/label.

Usage (SageMaker notebook):

  # Option A — ground-truth annotation file (default)
  !python omaha_mapping/run_time_dist.py

  # Option B — most recent LLM results file
  !python omaha_mapping/run_time_dist.py --results latest

  # Option B — specific LLM results file
  !python omaha_mapping/run_time_dist.py --results gpt-4o_20240315_results.xlsx

Output saved to S3 under output/:
  time_dist_gt_<ts>.xlsx   — ground-truth mode
    sheets: TRANSCRIPT_SUMMARY, CATEGORY_AGG, TARGET_AGG, TARGET_PER_VISIT
  time_dist_llm_<ts>.xlsx  — LLM row-count mode
    sheets: TIME_OVERVIEW, TIME_BY_CATEGORY
"""

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime
from io import BytesIO

import boto3
import pandas as pd
import numpy as np

from omaha_sagemaker import (
    S3_BUCKET,
    OUTPUT_PREFIX,
    ANNOTATION_KEY,
    compute_time_distribution,
    s3_write_excel,
)

# Import timestamp-based analysis from the standalone time_analysis module
sys.path.insert(0, os.path.dirname(__file__))
from time_analysis import analyze_all, aggregate, print_summary, TranscriptResult, AggregatedResult

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
SKIP_SHEETS = {"SUMMARY", "TIME_OVERVIEW", "TIME_BY_CATEGORY",
               "TRANSCRIPT_SUMMARY", "CATEGORY_AGG", "TARGET_AGG", "TARGET_PER_VISIT"}


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _read_excel_sheets(key: str) -> dict[str, pd.DataFrame]:
    log.info(f"Reading s3://{S3_BUCKET}/{key}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    buf = BytesIO(obj["Body"].read())
    sheets = pd.read_excel(buf, sheet_name=None, engine="openpyxl")
    return {k: v.fillna("") for k, v in sheets.items()}


def _download_to_tmp(key: str) -> str:
    """Download an S3 object to a temp file and return its local path."""
    log.info(f"Downloading s3://{S3_BUCKET}/{key} to temp file")
    obj  = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read()
    _, ext = os.path.splitext(key)
    tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(data)
    tmp.close()
    return tmp.name


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


# ── Ground-truth mode (real timestamps) ───────────────────────────────────────

def _results_to_transcript_df(results: list[TranscriptResult]) -> pd.DataFrame:
    """Per-transcript summary: labeled/nurse seconds + category breakdown."""
    rows = []
    for r in results:
        row = {
            "Visit":             r.name,
            "Nurse_sec":         round(r.total_nurse_sec,   1),
            "Labeled_sec":       round(r.total_labeled_sec, 1),
            "Labeled_pct_nurse": round(
                100.0 * r.total_labeled_sec / r.total_nurse_sec
                if r.total_nurse_sec > 0 else 0.0, 1),
        }
        for cat, sec in sorted(r.category_sec.items()):
            pct = 100.0 * sec / r.total_labeled_sec if r.total_labeled_sec > 0 else 0.0
            row[f"{cat}_sec"] = round(sec, 1)
            row[f"{cat}_pct"] = round(pct, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _agg_to_category_df(agg: AggregatedResult) -> pd.DataFrame:
    """Aggregate category table: mean ± SD in seconds and % of labeled nurse time."""
    rows = []
    for cat in sorted(agg.category_mean, key=lambda c: -agg.category_mean[c]):
        rows.append({
            "Category":    cat,
            "Mean_sec":    round(agg.category_mean[cat],     1),
            "SD_sec":      round(agg.category_std[cat],      1),
            "Mean_pct":    round(agg.category_pct_mean[cat], 1),
            "SD_pct":      round(agg.category_pct_std[cat],  1),
            "N_visits":    agg.n_transcripts,
        })
    return pd.DataFrame(rows)


def _agg_to_target_df(agg: AggregatedResult) -> pd.DataFrame:
    """Aggregate target table: mean ± SD in seconds and %."""
    rows = []
    for (cat, tgt) in sorted(agg.target_mean, key=lambda k: -agg.target_mean[k]):
        rows.append({
            "Category":  cat,
            "Target":    tgt,
            "Mean_sec":  round(agg.target_mean[(cat, tgt)],     1),
            "SD_sec":    round(agg.target_std[(cat, tgt)],      1),
            "Mean_pct":  round(agg.target_pct_mean[(cat, tgt)], 1),
            "SD_pct":    round(agg.target_pct_std[(cat, tgt)],  1),
            "N_visits":  agg.n_transcripts,
        })
    return pd.DataFrame(rows)


def _per_visit_target_df(results: list[TranscriptResult],
                          agg: AggregatedResult) -> pd.DataFrame:
    """Per-visit × target seconds, wide format (one row per target, one col per visit)."""
    all_targets = sorted(agg.target_mean, key=lambda k: -agg.target_mean[k])
    rows = []
    for (cat, tgt) in all_targets:
        row = {
            "Category": cat,
            "Target":   tgt,
            "Mean_sec": round(agg.target_mean[(cat, tgt)], 1),
        }
        for r in results:
            row[r.name] = round(r.target_sec.get((cat, tgt), 0.0), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def run_groundtruth(annotation_key: str):
    """Download annotation file, run real-timestamp analysis, save to S3."""
    tmp_path = _download_to_tmp(annotation_key)
    try:
        results = analyze_all(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not results:
        log.error("No transcripts with labeled nurse turns found in annotation file.")
        sys.exit(1)

    agg = aggregate(results)
    print_summary(results, agg)

    # ── Save Excel to S3 ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_key = f"{OUTPUT_PREFIX}time_dist_gt_{ts}.xlsx"
    s3_write_excel(
        {
            "TRANSCRIPT_SUMMARY": _results_to_transcript_df(results),
            "CATEGORY_AGG":       _agg_to_category_df(agg),
            "TARGET_AGG":         _agg_to_target_df(agg),
            "TARGET_PER_VISIT":   _per_visit_target_df(results, agg),
        },
        S3_BUCKET,
        out_key,
    )
    print(f"\nSaved: s3://{S3_BUCKET}/{out_key}")
    print("  Sheets: TRANSCRIPT_SUMMARY, CATEGORY_AGG, TARGET_AGG, TARGET_PER_VISIT")


# ── LLM results mode (row-count proxy) ────────────────────────────────────────

def run_llm_results(results_key: str):
    """Run row-count proxy analysis on a saved LLM results Excel."""
    sheets = _read_excel_sheets(results_key)
    conv_sheets = {k: v for k, v in sheets.items() if k not in SKIP_SHEETS}
    log.info(f"Processing {len(conv_sheets)} sheet(s)")

    all_td        = []
    overview_rows = []
    cat_rows      = []

    for sheet_name, df in conv_sheets.items():
        td = compute_time_distribution(df)
        if not td:
            log.warning(f"  Sheet '{sheet_name}': empty or missing columns — skipped")
            continue
        all_td.append(td)

        ov = {
            "Sheet":        sheet_name,
            "Total_Rows":   td["total_rows"],
            "Nurse_Rows":   td["nurse_rows"],
            "Patient_Rows": td["patient_rows"],
            "Unclear_Rows": td["unclear_rows"],
            "Nurse_Pct":    td["nurse_pct"],
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
            "No sheets produced data. Results file needs "
            "Pred_I_1_category + LLM_understand_raw columns."
        )
        sys.exit(1)

    # ── Console summary ───────────────────────────────────────────────────────
    tot = sum(t["total_rows"]   for t in all_td)
    nur = sum(t["nurse_rows"]   for t in all_td)
    pat = sum(t["patient_rows"] for t in all_td)
    unc = sum(t["unclear_rows"] for t in all_td)

    print(f"\nROW-COUNT PROXY  (LLM results, {len(all_td)} visits)")
    print(f"  NOTE: no timestamps in results file — each row = 1 time step")
    print(f"{'─'*68}")
    print(f"  Total rows   : {tot}")
    print(f"  Nurse rows   : {nur}  ({nur/tot*100:.1f}%)")
    print(f"  Patient rows : {pat}  ({pat/tot*100:.1f}%)")
    print(f"  Unclear rows : {unc}  ({unc/tot*100:.1f}%  counted as nurse)")

    print(f"\n  {'Strategy':<16}  {'Int Rows':>9}  {'% Nurse':>7}  Rationale")
    print(f"  {'─'*74}")
    for strat in STRATEGIES:
        int_rows = sum(t.get(strat, {}).get("total_int_rows", 0) for t in all_td)
        pct      = round(int_rows / nur * 100, 1) if nur else 0.0
        print(f"  {strat:<16}  {int_rows:>9}  {pct:>6.1f}%  {STRAT_DESC[strat]}")

    print(f"\n  Category breakdown (full_exchange, all visits):")
    cat_agg: dict[str, int] = {}
    for t in all_td:
        for cat, cnt in t.get("full_exchange", {}).get("by_category", {}).items():
            cat_agg[cat] = cat_agg.get(cat, 0) + cnt
    for cat, cnt in sorted(cat_agg.items(), key=lambda x: -x[1]):
        pct = round(cnt / nur * 100, 1) if nur else 0.0
        print(f"    {cat:<40}  {cnt:>6} rows  ({pct:.1f}% nurse)")
    print(f"{'─'*68}")

    # ── Save to S3 ────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_key = f"{OUTPUT_PREFIX}time_dist_llm_{ts}.xlsx"
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
        description=(
            "Time distribution analysis.\n"
            "Default (no flag): ground-truth annotation file with real timestamps.\n"
            "--results: row-count proxy on a saved LLM results Excel."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--annotation", action="store_true",
        help="Run on ground-truth annotation file using real SONY timestamps (default)",
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
        run_llm_results(key)
    else:
        # Default / --annotation: ground-truth with real timestamps
        run_groundtruth(ANNOTATION_KEY)


if __name__ == "__main__":
    main()
