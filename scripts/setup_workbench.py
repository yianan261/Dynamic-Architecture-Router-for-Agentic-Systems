#!/usr/bin/env python3
"""
Clone the WorkBench benchmark repo (olly-styles/WorkBench) into vendor/WorkBench.

WorkBench ships sandbox databases, task–outcome CSVs, and LangChain-based inference.
This project keeps PCAB as the default harness; use a separate Python environment
for WorkBench (its requirements pin older LangChain than DynamicRoutingAgents).

Usage (from project root):
    python scripts/setup_workbench.py
    python scripts/setup_workbench.py --build-slice   # also build benchmarks/workbench_50_queries.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/olly-styles/WorkBench.git"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone WorkBench into vendor/WorkBench.")
    parser.add_argument(
        "--build-slice",
        action="store_true",
        help="Run scripts/build_workbench_50_slice.py after clone.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    vendor = root / "vendor" / "WorkBench"

    if vendor.exists():
        print(f"WorkBench already present: {vendor}")
    else:
        vendor.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(vendor)],
            check=True,
        )
        print(f"Cloned WorkBench to {vendor}")

    qa = vendor / "data" / "processed" / "queries_and_answers"
    if not qa.is_dir():
        print(f"ERROR: Missing {qa}", file=sys.stderr)
        sys.exit(1)

    if args.build_slice:
        slice_script = root / "scripts" / "build_workbench_50_slice.py"
        subprocess.run([sys.executable, str(slice_script)], check=True)

    print()
    print("Next steps (WorkBench uses its own venv; versions differ from this project):")
    print("  cd vendor/WorkBench && python3 -m venv venv && source venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  # API keys — see vendor/WorkBench/README.md (only if using upstream generate_results.py)")
    print("  # This repo: python run_workbench_benchmark.py  (Llama via vLLM, SAS vs CMAS)")
    print("  # 50-task slice (benchmarks/workbench_50_queries.csv after --build-slice):")
    p = (root / "benchmarks" / "workbench_50_queries.csv").resolve()
    print(f"  cd vendor/WorkBench")
    print(
        "  python scripts/inference/generate_results.py \\",
    )
    print("      --model_name gpt-4 \\")
    print(f"      --queries_path {p}")


if __name__ == "__main__":
    main()
