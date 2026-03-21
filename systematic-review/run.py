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
    p = runner.get_prisma_counts()
    ta = p.get('title_abstract_included', 0)
    r1p = p.get('fulltext_r1_passed', 0)
    ft_sum = p.get('fulltext_retrieved', 0) + p.get('full_text_needed', 0)
    r1_sum = (p.get('fulltext_r1_excluded', 0) + p.get('fulltext_r1_passed', 0)
              + p.get('fulltext_r1_uncertain', 0) + p.get('full_text_needed', 0))
    r2_sum = (p.get('fulltext_r2_excluded', 0) + p.get('fulltext_r2_included', 0)
              + p.get('fulltext_r2_uncertain', 0))

    def chk(s, e): return f"✓" if (e == 0 or s == e) else f"✗ (got {s}, expected {e})"

    print("\n── PRISMA Flow ──────────────────────────────────────────")
    print(f"  {'Identified (all imports)':<45} {p.get('identified', '—')}")
    print(f"  {'Duplicates removed':<45} {p.get('duplicates_removed', '—')}")
    print(f"  {'After dedup (unique)':<45} {p.get('after_dedup', '—')}")
    print(f"  {'Awaiting title screening':<45} {p.get('awaiting_title_screening', '—')}")
    print(f"  {'Title/abstract excluded':<45} {p.get('title_abstract_excluded', '—')}")
    print(f"  {'Title/abstract included':<45} {ta}")
    print(f"  {'─' * 55}")
    print(f"  {'Full text retrieved':<45} {p.get('fulltext_retrieved', '—')}")
    print(f"  {'Full text NOT retrieved (manual upload)':<45} {p.get('full_text_needed', '—')}")
    print(f"  {'  [check] retrieved + not-retrieved':<45} {chk(ft_sum, ta)}")
    print(f"  {'─' * 55}")
    print(f"  {'[ROUND-1 full-text screening]':<45}")
    print(f"  {'  Round-1 excluded':<45} {p.get('fulltext_r1_excluded', '—')}")
    print(f"  {'  Round-1 uncertain (needs human)':<45} {p.get('fulltext_r1_uncertain', '—')}")
    print(f"  {'  Round-1 passed → sent to Round-2':<45} {r1p}")
    print(f"  {'  [check] r1-excl + r1-pass + r1-unc + not-ret':<45} {chk(r1_sum, ta)}")
    print(f"  {'─' * 55}")
    print(f"  {'[ROUND-2 full-text screening]':<45}")
    print(f"  {'  Round-2 excluded':<45} {p.get('fulltext_r2_excluded', '—')}")
    print(f"  {'  Round-2 uncertain (needs human)':<45} {p.get('fulltext_r2_uncertain', '—')}")
    print(f"  {'  Round-2 included (final)':<45} {p.get('fulltext_r2_included', '—')}")
    if r1p > 0:
        print(f"  {'  [check] r2-excl + r2-incl + r2-unc':<45} {chk(r2_sum, r1p)}")
    print(f"  {'─' * 55}")
    print(f"  {'Final included':<45} {p.get('final_included', '—')}")
    print(f"  {'Needs human (total)':<45} {p.get('needs_human_verification', '—')}")
    print(f"    title: {p.get('needs_human_title_screening',0)}  "
          f"round-1 FT: {p.get('needs_human_fulltext_screening',0)}  "
          f"round-2 FT: {p.get('needs_human_round2_screening',0)}  "
          f"extraction: {p.get('needs_human_extraction',0)}")
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
