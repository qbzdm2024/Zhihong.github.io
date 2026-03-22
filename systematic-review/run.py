#!/usr/bin/env python3
"""
Quick-start script for the Systematic Review Automation System.
Usage:
    python run.py                    # start API server
    python run.py --stage import     # run a pipeline stage
    python run.py --stage dedup
    python run.py --stage title_screening [--limit 50]
    python run.py --stage fulltext_screening [--limit 20]
    python run.py --stage extraction [--limit 10]
    python run.py --status           # print pipeline status
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def run_server():
    import uvicorn
    print("\n🚀 Starting Systematic Review API server...")
    print("   API docs:  http://localhost:8000/docs")
    print("   UI:        http://localhost:8000/app")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "pipeline", "agents", "config"],
    )


def run_stage(stage: str, limit=None, force=False):
    from pipeline.pipeline import PipelineRunner
    runner = PipelineRunner()
    runner.load_state()

    stage_map = {
        "import": runner.run_import,
        "dedup": runner.run_deduplication,
        "title_screening": lambda: runner.run_title_screening(limit),
        "fulltext_screening": lambda: runner.run_fulltext_screening(limit),
        "second_fulltext_screening": lambda: runner.run_second_fulltext_screening(limit),
        "extraction": lambda: runner.run_extraction(limit),
        "reset_fulltext_screening": lambda: runner.reset_fulltext_screening(force=force),
        "reset_second_fulltext_screening": lambda: runner.reset_second_fulltext_screening(force=force),
        "restore_bulk_excluded": runner.restore_bulk_excluded,
        "reset_failed_screenings": runner.reset_failed_screenings,
    }

    if stage not in stage_map:
        print(f"Unknown stage: {stage}. Options: {list(stage_map.keys())}")
        sys.exit(1)

    print(f"\n▶ Running stage: {stage}" + (f" (limit={limit})" if limit else ""))
    result = stage_map[stage]()
    print(f"✓ {stage} complete.\n  {result}")


def print_status():
    from pipeline.pipeline import PipelineRunner
    runner = PipelineRunner()
    runner.load_state()
    report = runner.get_full_prisma_report()
    p = report["prisma_counts"]

    # ── Sum checks ────────────────────────────────────────────────
    ta   = p.get('title_abstract_included', 0)
    r1p  = p.get('fulltext_r1_passed', 0)
    ft_sum = p.get('fulltext_retrieved', 0) + p.get('full_text_needed', 0)
    r1_sum = (p.get('fulltext_r1_excluded', 0) + p.get('fulltext_r1_passed', 0)
              + p.get('fulltext_r1_uncertain', 0) + p.get('full_text_needed', 0))
    r2_sum = (p.get('fulltext_r2_excluded', 0) + p.get('fulltext_r2_included', 0)
              + p.get('fulltext_r2_uncertain', 0))

    def chk(s, e): return "✓" if (e == 0 or s == e) else f"✗ (got {s}, expected {e})"

    # ── Source databases ──────────────────────────────────────────
    print("\n── Source Databases ─────────────────────────────────────")
    for db, n in sorted(report["source_databases_all"].items(), key=lambda x: -x[1]):
        uniq = report["source_databases_unique"].get(db, 0)
        print(f"  {db:<35} total={n:>5}  after dedup={uniq:>5}")

    # ── PRISMA Flow ───────────────────────────────────────────────
    print("\n── PRISMA Flow ──────────────────────────────────────────")
    print(f"  {'Identified (all imports)':<45} {p.get('identified', '—')}")
    print(f"  {'Duplicates removed':<45} {p.get('duplicates_removed', '—')}")
    print(f"  {'After dedup (unique)':<45} {p.get('after_dedup', '—')}")
    print(f"  {'Title/abstract excluded':<45} {p.get('title_abstract_excluded', '—')}")
    print(f"  {'Title/abstract included':<45} {ta}")
    print(f"  {'─' * 55}")
    print(f"  {'Full text retrieved':<45} {p.get('fulltext_retrieved', '—')}")
    print(f"  {'Full text not retrieved (manual upload)':<45} {p.get('full_text_needed', '—')}")
    print(f"  {'  [check] retrieved + not-retrieved = title incl':<45} {chk(ft_sum, ta)}")
    print(f"  {'─' * 55}")
    print(f"  [ROUND-1 full-text screening]")
    print(f"  {'  Excluded':<45} {p.get('fulltext_r1_excluded', '—')}")
    print(f"  {'  Passed → Round-2':<45} {r1p}")
    print(f"  {'  [check] excl + passed + not-ret = title incl':<45} {chk(r1_sum, ta)}")
    print(f"  {'─' * 55}")
    print(f"  [ROUND-2 full-text screening]")
    print(f"  {'  Excluded':<45} {p.get('fulltext_r2_excluded', '—')}")
    if p.get('fulltext_r2_uncertain', 0):
        print(f"  {'  Resolved via human verification':<45} {p.get('fulltext_r2_uncertain', 0)}")
    print(f"  {'  Included (final)':<45} {p.get('final_included', '—')}")
    if r1p > 0:
        print(f"  {'  [check] excl + incl + unc = r1-passed':<45} {chk(r2_sum, r1p)}")
    print(f"  {'─' * 55}")
    print(f"  {'FINAL INCLUDED':<45} {p.get('final_included', '—')}")

    # ── Round-2 exclusion reasons ─────────────────────────────────
    if report.get("round2_exclusion_by_reason"):
        print(f"\n── Round-2 Exclusion Reasons (n={report['round2_total_excluded']}) ──")
        for r in report["round2_exclusion_by_reason"]:
            print(f"  [{r['code']}] {r['label']:<55} n={r['count']}")

    # ── Round-1 exclusion reasons ─────────────────────────────────
    if report.get("round1_exclusion_by_reason"):
        print(f"\n── Round-1 Exclusion Reasons (n={report['round1_total_excluded']}) ──")
        for r in report["round1_exclusion_by_reason"]:
            print(f"  [{r['code']}] {r['label']:<55} n={r['count']}")

    print("\n── Formatted PRISMA Flowchart ───────────────────────────")
    print(report["prisma_flowchart_text"])
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic Review Automation System")
    parser.add_argument("--stage", help="Pipeline stage to run")
    parser.add_argument("--limit", type=int, help="Max records to process")
    parser.add_argument("--status", action="store_true", help="Print pipeline status")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.stage:
        run_stage(args.stage, args.limit)
    else:
        run_server()
