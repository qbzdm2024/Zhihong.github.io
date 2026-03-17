#!/usr/bin/env python3
"""
Nurse Intervention Time Analysis
=================================
Computes how much time a nurse spends on each Omaha System intervention
category and target, using human-annotated timestamps from conversation
transcripts.

For each transcript sheet:
  1. Detect the timestamp column (e.g. "SONY_RB_21921 [2]")
  2. Parse turn start times (MM:SS or HH:MM:SS, with speaker-code prefix stripped)
  3. Assign each turn a duration = gap to next turn's start timestamp
  4. For every nurse turn with OS_I_1/OS_I_2/OS_I_3 labels:
       - Attribute the time window to each label
  5. Merge overlapping / contiguous windows per label so no time is counted twice
  6. Compute category-level and target-level durations per transcript
  7. Aggregate across all 23 transcripts → mean ± SD

Visualisations produced:
  Fig 1 – Stacked bar (% time by category) per transcript
  Fig 2 – Mean category distribution (bar + error bars)
  Fig 3 – Mean target duration by category (horizontal grouped bar)
  Fig 4 – Top-N target distribution heatmap across transcripts

Usage (CLI):
    python time_analysis.py                          # auto-detect annotation file
    python time_analysis.py --file path/to/file.xlsx [--sheet-name Sheet1]
    python time_analysis.py --file example           # single-sheet TSV

Usage (notebook / import):
    from time_analysis import run_analysis
    results, agg = run_analysis("path/to/Completed_annotation.xlsx")
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# ── Configuration ─────────────────────────────────────────────────────────────

# Default duration (seconds) assigned to the LAST turn of a transcript
# (no following turn to derive an end-time from).
LAST_TURN_DEFAULT_SEC: float = 30.0

# Nurse speaker codes (case-insensitive); extend if your data differs.
NURSE_SPK_CODES = {"s2", "n", "nurse", "rn", "clinician"}

# Omaha category display names and colour palette
CATEGORY_COLOURS = {
    "Surveillance":                            "#4C72B0",
    "Treatments and Procedures":               "#DD8452",
    "Teaching, Guidance, and Counseling":      "#55A868",
    "Health Teaching, Guidance, and Counseling": "#55A868",  # alias
    "Case Management":                         "#C44E52",
}

# Canonical display name (collapse the two "Teaching" variants)
CATEGORY_CANONICAL = {
    "Health Teaching, Guidance, and Counseling": "Teaching, Guidance, and Counseling",
}

# How many top targets to show in Fig 3 / Fig 4
TOP_N_TARGETS = 15


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptResult:
    name: str                                        # sheet / file name
    category_sec: Dict[str, float] = field(default_factory=dict)
    target_sec:   Dict[Tuple[str, str], float] = field(default_factory=dict)
    total_labeled_sec: float = 0.0                   # nurse time with ≥1 label
    total_nurse_sec:   float = 0.0                   # all nurse time


@dataclass
class AggregatedResult:
    category_mean: Dict[str, float] = field(default_factory=dict)
    category_std:  Dict[str, float] = field(default_factory=dict)
    category_pct_mean: Dict[str, float] = field(default_factory=dict)
    category_pct_std:  Dict[str, float] = field(default_factory=dict)
    target_mean:   Dict[Tuple[str, str], float] = field(default_factory=dict)
    target_std:    Dict[Tuple[str, str], float] = field(default_factory=dict)
    target_pct_mean: Dict[Tuple[str, str], float] = field(default_factory=dict)
    target_pct_std:  Dict[Tuple[str, str], float] = field(default_factory=dict)
    n_transcripts: int = 0


# ── Utility functions ─────────────────────────────────────────────────────────

def parse_timestamp(ts_str) -> Optional[float]:
    """'S2 01:38'  →  98.0 seconds.  Handles MM:SS and HH:MM:SS."""
    if ts_str is None or (isinstance(ts_str, float) and np.isnan(ts_str)):
        return None
    ts = str(ts_str).strip()
    # Strip leading speaker-code prefix like 'S1 ', 'S2 ', 'N '
    ts = re.sub(r'^[A-Za-z][A-Za-z0-9]*\s+', '', ts).strip()
    parts = ts.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def detect_ts_column(df: pd.DataFrame) -> Optional[str]:
    """Return the column name that holds 'Speaker HH:MM:SS' timestamps."""
    for col in df.columns:
        if re.search(r'sony|SONY|timestamp|time_?stamp', col, re.IGNORECASE):
            return col
    # Fallback: first column whose values match the pattern
    pat = re.compile(r'^[A-Za-z][A-Za-z0-9]*\s+\d+:\d+')
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(10)
        if sample.apply(lambda v: bool(pat.match(v))).any():
            return col
    return None


def split_label(label: str) -> Tuple[str, str]:
    """'Surveillance_signs/symptoms-physical' → ('Surveillance', 'signs/symptoms-physical')."""
    known = [
        "Health Teaching, Guidance, and Counseling",
        "Teaching, Guidance, and Counseling",
        "Treatments and Procedures",
        "Case Management",
        "Surveillance",
    ]
    for cat in known:
        if label.startswith(cat + "_"):
            tgt = label[len(cat) + 1:]
            cat = CATEGORY_CANONICAL.get(cat, cat)
            return cat, tgt
    # Fallback: first underscore
    parts = label.split("_", 1)
    cat = CATEGORY_CANONICAL.get(parts[0], parts[0])
    return (cat, parts[1]) if len(parts) == 2 else (label, "unknown")


def get_intervention_labels(row) -> List[str]:
    """Return non-empty OS_I_1/2/3 values from a row."""
    labels = []
    for col in ("OS_I_1", "OS_I_2", "OS_I_3"):
        val = row.get(col, "")
        if isinstance(val, float) and np.isnan(val):
            continue
        val = str(val).strip()
        if val and val.lower() not in ("nan", "none", ""):
            labels.append(val)
    return labels


def merge_intervals(ivs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping/adjacent [start, end) intervals; returns sorted list."""
    if not ivs:
        return []
    ivs = sorted(ivs)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:          # overlapping or touching
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def total_seconds(ivs: List[Tuple[float, float]]) -> float:
    return sum(e - s for s, e in ivs)


def is_nurse(spk) -> bool:
    return str(spk).strip().lower() in NURSE_SPK_CODES


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_transcript(df: pd.DataFrame, name: str = "?") -> TranscriptResult:
    """Compute time distribution for one transcript DataFrame."""
    result = TranscriptResult(name=name)

    # ── Normalise column names ────────────────────────────────────────────
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = detect_ts_column(df)
    if ts_col is None:
        print(f"  [WARN] {name}: no timestamp column found — skipping", file=sys.stderr)
        return result

    # ── Parse timestamps ─────────────────────────────────────────────────
    df["_ts"] = df[ts_col].apply(parse_timestamp)

    # ── Compute per-row end time = start of next row's timestamp ─────────
    ts_arr = df["_ts"].values.astype(float)   # NaN where missing
    n = len(ts_arr)
    end_arr = np.empty(n)
    end_arr[:] = np.nan

    for i in range(n):
        if np.isnan(ts_arr[i]):
            continue
        # Find the next row (any speaker) that has a valid timestamp
        found = False
        for j in range(i + 1, n):
            if not np.isnan(ts_arr[j]):
                end_arr[i] = ts_arr[j]
                found = True
                break
        if not found:
            end_arr[i] = ts_arr[i] + LAST_TURN_DEFAULT_SEC

    df["_end"] = end_arr

    # ── Collect windows ───────────────────────────────────────────────────
    spk_col = "Spk" if "Spk" in df.columns else "Speaker"

    # All windows keyed by (category, target) and by category alone
    label_windows: Dict[Tuple[str, str], List] = defaultdict(list)
    cat_windows:   Dict[str, List]              = defaultdict(list)
    nurse_windows: List = []
    labeled_windows: List = []   # nurse turns with ≥1 label

    for _, row in df.iterrows():
        s = row["_ts"]
        e = row["_end"]
        if np.isnan(s) or np.isnan(e) or e <= s:
            continue

        spk_val = row.get(spk_col, "")
        if not is_nurse(spk_val):
            continue

        nurse_windows.append((s, e))
        labels = get_intervention_labels(row)

        if labels:
            labeled_windows.append((s, e))
            for lbl in labels:
                cat, tgt = split_label(lbl)
                label_windows[(cat, tgt)].append((s, e))
                cat_windows[cat].append((s, e))

    # ── Merge and sum ─────────────────────────────────────────────────────
    result.total_nurse_sec   = total_seconds(merge_intervals(nurse_windows))
    result.total_labeled_sec = total_seconds(merge_intervals(labeled_windows))

    for (cat, tgt), ivs in label_windows.items():
        result.target_sec[(cat, tgt)] = total_seconds(merge_intervals(ivs))

    for cat, ivs in cat_windows.items():
        result.category_sec[cat] = total_seconds(merge_intervals(ivs))

    return result


def load_sheets(path: str) -> Dict[str, pd.DataFrame]:
    """Load one or more transcript DataFrames from a file.

    Supports:
      - Multi-sheet Excel (.xlsx) → one df per sheet
      - Single-sheet TSV / CSV    → one df named after the file
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path, sheet_name=None, engine="openpyxl", dtype=str)
        sheets = {}
        for name, df in raw.items():
            df = df.fillna("").copy()
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        return sheets
    else:
        # TSV or CSV
        sep = "\t" if ext in (".tsv", "") else ","
        df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        return {os.path.basename(path): df}


def analyze_all(path: str) -> List[TranscriptResult]:
    """Run analysis on every sheet in the file."""
    sheets = load_sheets(path)
    results = []
    for name, df in sheets.items():
        r = analyze_transcript(df, name=name)
        if r.total_labeled_sec > 0:
            results.append(r)
        else:
            print(f"  [INFO] {name}: no labeled nurse turns found — excluded",
                  file=sys.stderr)
    print(f"  Analysed {len(results)} transcript(s) with intervention labels.")
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def _pct(d: dict, total: float) -> dict:
    if total <= 0:
        return {k: 0.0 for k in d}
    return {k: 100.0 * v / total for k, v in d.items()}


def aggregate(results: List[TranscriptResult]) -> AggregatedResult:
    """Compute mean ± SD across transcripts (% of labeled-nurse time)."""
    agg = AggregatedResult(n_transcripts=len(results))

    # Collect all unique categories and targets
    all_cats  = sorted({c for r in results for c in r.category_sec})
    all_tgts  = sorted({k for r in results for k in r.target_sec},
                       key=lambda x: x[0] + "_" + x[1])

    # ── Categories ────────────────────────────────────────────────────────
    cat_sec_mat  = {c: [] for c in all_cats}
    cat_pct_mat  = {c: [] for c in all_cats}

    for r in results:
        for c in all_cats:
            sec = r.category_sec.get(c, 0.0)
            pct = 100.0 * sec / r.total_labeled_sec if r.total_labeled_sec > 0 else 0.0
            cat_sec_mat[c].append(sec)
            cat_pct_mat[c].append(pct)

    agg.category_mean = {c: float(np.mean(v)) for c, v in cat_sec_mat.items()}
    agg.category_std  = {c: float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                         for c, v in cat_sec_mat.items()}
    agg.category_pct_mean = {c: float(np.mean(v)) for c, v in cat_pct_mat.items()}
    agg.category_pct_std  = {c: float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                              for c, v in cat_pct_mat.items()}

    # ── Targets ───────────────────────────────────────────────────────────
    tgt_sec_mat  = {t: [] for t in all_tgts}
    tgt_pct_mat  = {t: [] for t in all_tgts}

    for r in results:
        for t in all_tgts:
            sec = r.target_sec.get(t, 0.0)
            pct = 100.0 * sec / r.total_labeled_sec if r.total_labeled_sec > 0 else 0.0
            tgt_sec_mat[t].append(sec)
            tgt_pct_mat[t].append(pct)

    agg.target_mean = {t: float(np.mean(v)) for t, v in tgt_sec_mat.items()}
    agg.target_std  = {t: float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                       for t, v in tgt_sec_mat.items()}
    agg.target_pct_mean = {t: float(np.mean(v)) for t, v in tgt_pct_mat.items()}
    agg.target_pct_std  = {t: float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                            for t, v in tgt_pct_mat.items()}

    return agg


# ── Visualisation ─────────────────────────────────────────────────────────────

def _cat_colour(cat: str) -> str:
    return CATEGORY_COLOURS.get(cat, "#888888")


def _short_name(name: str, max_len: int = 20) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def plot_all(results: List[TranscriptResult],
             agg: AggregatedResult,
             output_dir: str = ".",
             show: bool = True):
    """Generate and save all four figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Canonical category order for stacking
    cat_order = [
        "Surveillance",
        "Treatments and Procedures",
        "Teaching, Guidance, and Counseling",
        "Case Management",
    ]
    # Keep only categories that actually appear
    cat_order = [c for c in cat_order if c in agg.category_mean]
    # Append any unexpected categories
    for c in sorted(agg.category_mean):
        if c not in cat_order:
            cat_order.append(c)

    # ──────────────────────────────────────────────────────────────────────
    # Fig 1 — Stacked bar: % time by category, one bar per transcript
    # ──────────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(max(10, len(results) * 0.55 + 2), 5))

    labels_x = [_short_name(r.name) for r in results]
    bottoms  = np.zeros(len(results))
    handles  = []

    for cat in cat_order:
        col  = _cat_colour(cat)
        vals = np.array([
            100.0 * r.category_sec.get(cat, 0.0) / r.total_labeled_sec
            if r.total_labeled_sec > 0 else 0.0
            for r in results
        ])
        ax1.bar(labels_x, vals, bottom=bottoms, color=col, label=cat, width=0.6)
        handles.append(mpatches.Patch(color=col, label=cat))
        bottoms += vals

    ax1.set_xlabel("Transcript")
    ax1.set_ylabel("% of labeled nurse time")
    ax1.set_title("Intervention Category Distribution per Transcript")
    ax1.set_ylim(0, 110)
    ax1.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
               fontsize=8, framealpha=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "fig1_category_per_transcript.png")
    fig1.savefig(p1, bbox_inches="tight")
    print(f"  Saved: {p1}")

    # ──────────────────────────────────────────────────────────────────────
    # Fig 2 — Mean category distribution (bar + error bars + pie inset)
    # ──────────────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart with SD error bars
    ax_bar = axes2[0]
    cats   = cat_order
    means  = [agg.category_pct_mean.get(c, 0.0) for c in cats]
    stds   = [agg.category_pct_std.get(c, 0.0)  for c in cats]
    colours = [_cat_colour(c) for c in cats]

    bars = ax_bar.bar(cats, means, yerr=stds, capsize=4, color=colours,
                      edgecolor="white", linewidth=0.5)
    ax_bar.set_ylabel("Mean % of labeled nurse time")
    ax_bar.set_title(f"Mean Category Distribution\n(N={agg.n_transcripts} transcripts)")
    ax_bar.set_ylim(0, max(means) * 1.35 + 5)
    ax_bar.set_xticklabels(
        [c.replace(", and Counseling", ",\nand Counseling")
          .replace("Treatments and", "Treatments\nand")
         for c in cats],
        rotation=0, ha="center", fontsize=8
    )
    for bar, mean, std in zip(bars, means, stds):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.5,
                    f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)

    # Right: pie chart (mean %)
    ax_pie = axes2[1]
    pie_means = [max(m, 0) for m in means]
    if sum(pie_means) > 0:
        wedges, texts, autotexts = ax_pie.pie(
            pie_means,
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            colors=colours,
            startangle=90,
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax_pie.legend(wedges, cats, loc="lower center",
                      bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8)
    ax_pie.set_title("Mean Category Distribution (Pie)")

    plt.tight_layout()
    p2 = os.path.join(output_dir, "fig2_mean_category.png")
    fig2.savefig(p2, bbox_inches="tight")
    print(f"  Saved: {p2}")

    # ──────────────────────────────────────────────────────────────────────
    # Fig 3 — Top-N target mean duration (horizontal grouped bar by category)
    # ──────────────────────────────────────────────────────────────────────
    # Sort targets by mean seconds descending, take top N
    tgt_sorted = sorted(agg.target_mean.items(), key=lambda x: -x[1])
    top_tgts   = tgt_sorted[:TOP_N_TARGETS]

    if top_tgts:
        tgt_keys  = [t for t, _ in top_tgts]
        tgt_means = [agg.target_mean[t] for t in tgt_keys]
        tgt_stds  = [agg.target_std[t]  for t in tgt_keys]
        tgt_cols  = [_cat_colour(t[0])  for t in tgt_keys]
        tgt_labels = [f"{t[1]}\n({t[0][:4]}…)" if len(t[0]) > 12
                      else f"{t[1]}\n({t[0]})"
                      for t in tgt_keys]

        fig3, ax3 = plt.subplots(figsize=(8, max(5, len(top_tgts) * 0.5 + 2)))
        y_pos = np.arange(len(top_tgts))

        ax3.barh(y_pos, tgt_means, xerr=tgt_stds, color=tgt_cols,
                 capsize=3, edgecolor="white", linewidth=0.5, height=0.6)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(tgt_labels, fontsize=8)
        ax3.invert_yaxis()
        ax3.set_xlabel("Mean seconds per transcript")
        ax3.set_title(f"Top-{len(top_tgts)} Targets — Mean Time\n(N={agg.n_transcripts} transcripts)")

        # Category legend
        seen_cats = []
        leg_handles = []
        for (cat, _) in tgt_keys:
            if cat not in seen_cats:
                seen_cats.append(cat)
                leg_handles.append(mpatches.Patch(color=_cat_colour(cat), label=cat))
        ax3.legend(handles=leg_handles, loc="lower right", fontsize=8)

        plt.tight_layout()
        p3 = os.path.join(output_dir, "fig3_top_targets.png")
        fig3.savefig(p3, bbox_inches="tight")
        print(f"  Saved: {p3}")

    # ──────────────────────────────────────────────────────────────────────
    # Fig 4 — Heatmap: target × transcript (% of labeled time)
    # ──────────────────────────────────────────────────────────────────────
    if top_tgts and len(results) > 1:
        matrix = np.zeros((len(top_tgts), len(results)))
        for j, r in enumerate(results):
            for i, (tgt_key, _) in enumerate(top_tgts):
                sec = r.target_sec.get(tgt_key, 0.0)
                pct = 100.0 * sec / r.total_labeled_sec if r.total_labeled_sec > 0 else 0.0
                matrix[i, j] = pct

        fig4, ax4 = plt.subplots(figsize=(max(10, len(results) * 0.55 + 3),
                                          max(5, len(top_tgts) * 0.4 + 2)))
        im = ax4.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax4.set_xticks(np.arange(len(results)))
        ax4.set_xticklabels([_short_name(r.name, 14) for r in results],
                            rotation=45, ha="right", fontsize=7)
        ax4.set_yticks(np.arange(len(top_tgts)))
        ax4.set_yticklabels([f"{t[1]} ({t[0][:4]})" for t, _ in top_tgts], fontsize=8)
        plt.colorbar(im, ax=ax4, label="% labeled nurse time")
        ax4.set_title(f"Intervention Target Distribution across Transcripts\n"
                      f"(top {len(top_tgts)} targets by mean time)")
        plt.tight_layout()
        p4 = os.path.join(output_dir, "fig4_target_heatmap.png")
        fig4.savefig(p4, bbox_inches="tight")
        print(f"  Saved: {p4}")

    if show:
        plt.show()
    else:
        plt.close("all")


# ── Summary tables ────────────────────────────────────────────────────────────

def print_summary(results: List[TranscriptResult], agg: AggregatedResult):
    sep = "─" * 72

    print(f"\n{'═'*72}")
    print(f"  INTERVENTION TIME ANALYSIS  —  {agg.n_transcripts} transcript(s)")
    print(f"{'═'*72}")

    # Per-transcript table
    print(f"\n  {'Transcript':<26} {'Labeled(s)':>10} {'Total nurse(s)':>14}", end="")
    for c in ("Surveillance", "Treatments and Procedures",
              "Teaching, Guidance, and Counseling", "Case Management"):
        if c in agg.category_mean:
            print(f"  {c[:10]+'.':>12}", end="")
    print()
    print(f"  {sep}")

    for r in results:
        row_s = f"  {_short_name(r.name, 26):<26} {r.total_labeled_sec:>10.1f} {r.total_nurse_sec:>14.1f}"
        for c in ("Surveillance", "Treatments and Procedures",
                  "Teaching, Guidance, and Counseling", "Case Management"):
            if c in agg.category_mean:
                sec = r.category_sec.get(c, 0.0)
                pct = 100.0 * sec / r.total_labeled_sec if r.total_labeled_sec > 0 else 0
                row_s += f"  {pct:>10.1f}%"
        print(row_s)

    # Aggregated category summary
    print(f"\n{'═'*72}")
    print("  MEAN CATEGORY DISTRIBUTION (% of labeled nurse time)")
    print(f"  {sep}")
    for c in sorted(agg.category_pct_mean, key=lambda x: -agg.category_pct_mean[x]):
        m = agg.category_pct_mean[c]
        s = agg.category_pct_std[c]
        bar = "█" * int(m / 2)
        print(f"  {c:<45}  {m:5.1f}% ± {s:4.1f}%  {bar}")

    # Aggregated target summary
    print(f"\n{'═'*72}")
    print("  MEAN TARGET DISTRIBUTION (seconds)")
    print(f"  {sep}")
    for (cat, tgt), mean_sec in sorted(agg.target_mean.items(),
                                       key=lambda x: -x[1])[:TOP_N_TARGETS]:
        std = agg.target_std[(cat, tgt)]
        bar = "█" * int(mean_sec / 10)
        print(f"  [{cat[:4]}] {tgt:<40}  {mean_sec:6.1f}s ± {std:5.1f}s  {bar}")
    print(f"{'═'*72}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def _find_annotation_file(hint: str = None) -> Optional[str]:
    candidates = []
    if hint:
        candidates.append(hint)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(root, "Completed_annotation.xlsx"),
        os.path.join(here, "Completed_annotation.xlsx"),
        os.path.join(here, "example"),
        os.path.join(root, "example"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def run_analysis(
    annotation_path: str = None,
    output_dir: str = "time_analysis_output",
    show_plots: bool = True,
) -> Tuple[List[TranscriptResult], AggregatedResult]:
    """
    Full pipeline: load → analyse → aggregate → visualise.

    Args:
        annotation_path: Path to Completed_annotation.xlsx (multi-sheet) or
                         a single TSV file. Auto-detected if None.
        output_dir:      Directory where figures are saved.
        show_plots:      Display figures interactively (set False for CI/batch).

    Returns:
        (results, agg) — list of per-transcript results and the aggregate summary.
    """
    path = _find_annotation_file(annotation_path)
    if path is None:
        raise FileNotFoundError(
            "Cannot locate annotation file. "
            "Pass annotation_path= or place Completed_annotation.xlsx in the repo root."
        )
    print(f"  Loading: {path}")

    results = analyze_all(path)
    if not results:
        raise ValueError("No usable transcripts found in the annotation file.")

    agg = aggregate(results)
    print_summary(results, agg)
    plot_all(results, agg, output_dir=output_dir, show=show_plots)
    return results, agg


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualise nurse intervention time distributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file", default=None,
                        help="Annotation Excel / TSV (auto-detected if omitted)")
    parser.add_argument("--output-dir", default="time_analysis_output",
                        help="Directory to save figures (default: time_analysis_output/)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive plot windows (just save files)")
    args = parser.parse_args()

    run_analysis(
        annotation_path=args.file,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    main()
