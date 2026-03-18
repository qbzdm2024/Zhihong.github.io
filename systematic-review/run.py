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


def run_stage(stage: str, limit=None):
    from pipeline.pipeline import PipelineRunner
    runner = PipelineRunner()
    runner.load_state()

    stage_map = {
        "import": runner.run_import,
        "dedup": runner.run_deduplication,
        "title_screening": lambda: runner.run_title_screening(limit),
        "fulltext_screening": lambda: runner.run_fulltext_screening(limit),
        "extraction": lambda: runner.run_extraction(limit),
    }

    if stage not in stage_map:
        print(f"Unknown stage: {stage}. Options: {list(stage_map.keys())}")
        sys.exit(1)

    print(f"\n▶ Running stage: {stage}" + (f" (limit={limit})" if limit else ""))
    result = stage_map[stage]()
    print(f"✓ Stage complete: {result}")


def print_status():
    from pipeline.pipeline import PipelineRunner
    runner = PipelineRunner()
    runner.load_state()
    counts = runner.get_prisma_counts()
    print("\n── PRISMA Flow ─────────────────────────────")
    for k, v in counts.items():
        print(f"  {k:40s} {v}")
    print("────────────────────────────────────────────\n")


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
