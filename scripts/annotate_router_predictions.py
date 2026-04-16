#!/usr/bin/env python3
"""
Fill router_prediction on each task using the Dynamic Router (metadata + thresholds).

Does not re-run SAS/CMAS — only classifies the task text. Use after a benchmark JSON
exists, then run evaluate_regret.py for regret metrics.

Usage (from project root):
  python scripts/annotate_router_predictions.py benchmark_workbench_results.json
  python scripts/annotate_router_predictions.py benchmark_workbench_results.json -o out.json

Requires the same vLLM/env as the router (USE_LLM_ROUTER, VLLM_BASE_URL on :8000, etc.)
or keyword fallback if vLLM is down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv 

load_project_root_dotenv()

from dynamic_routing.router import predict_routed_architecture 


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate benchmark JSON with router_prediction")
    parser.add_argument("input_json", type=Path, help="Benchmark results (e.g. benchmark_workbench_results.json)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here (default: overwrite input)",
    )
    args = parser.parse_args()
    inp = args.input_json
    if not inp.is_file():
        print(f"ERROR: {inp} not found", file=sys.stderr)
        sys.exit(1)
    out = args.output or inp

    with inp.open(encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    ok = 0
    fail = 0
    for item in tasks:
        q = item.get("description") or ""
        try:
            item["router_prediction"] = predict_routed_architecture(q)
            ok += 1
        except Exception as e:
            print(f"  [router-annotate] {item.get('task_id', '?')}: {str(e)[:120]}; defaulting to Single-Agent System")
            item["router_prediction"] = "Single-Agent System"
            fail += 1

    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out.resolve()} with router_prediction on {len(tasks)} tasks ({ok} ok, {fail} fallback)")
    print("Next: python evaluate_regret.py", out)


if __name__ == "__main__":
    main()
