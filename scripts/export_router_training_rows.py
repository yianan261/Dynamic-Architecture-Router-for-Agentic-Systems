#!/usr/bin/env python3
"""
Build a CSV of (routing features, oracle architecture) from a benchmark JSON.

Uses ``RegretEvaluator`` oracle (same weights as ``evaluate_regret.py`` defaults)
and ``predict_routing_metadata`` for LLM/keyword features.

Usage:
  python scripts/export_router_training_rows.py results/benchmark_workbench_results_TS.json \\
      -o results/router_train_001.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()

from dynamic_routing.vllm_integration import predict_routing_metadata  # noqa: E402
from evaluate_regret import ExecutionResult, RegretEvaluator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_json", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    ev = RegretEvaluator()
    fieldnames = [
        "task_id",
        "description",
        "oracle_architecture",
        "estimated_sequential_depth",
        "parallelization_factor",
        "estimated_tool_count",
        "num_subgoals",
        "entity_count",
        "constraint_tightness",
        "open_endedness",
        "aggregation_required",
        "expected_retrieval_fanout",
        "domain_span",
        "expected_context_expansion",
        "final_synthesis_complexity",
        "cross_branch_dependency",
        "communication_load_estimate",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in data.get("tasks", []):
            desc = item.get("description") or ""
            results = [
                ExecutionResult(
                    architecture=r["architecture"],
                    accuracy_score=float(r["accuracy_score"]),
                    latency_sec=float(r["latency_sec"]),
                    total_tokens=int(r["total_tokens"]),
                    error_type=r.get("error_type"),
                )
                for r in item.get("results", [])
            ]
            if not results:
                continue
            oracle = ev.determine_oracle(results)
            meta = predict_routing_metadata(desc)
            w.writerow({
                "task_id": item.get("task_id", ""),
                "description": desc[:2000],
                "oracle_architecture": oracle.architecture,
                "estimated_sequential_depth": meta.get("estimated_sequential_depth", ""),
                "parallelization_factor": meta.get("parallelization_factor", ""),
                "estimated_tool_count": meta.get("estimated_tool_count", ""),
                "num_subgoals": meta.get("num_subgoals", ""),
                "entity_count": meta.get("entity_count", ""),
                "constraint_tightness": meta.get("constraint_tightness", ""),
                "open_endedness": meta.get("open_endedness", ""),
                "aggregation_required": meta.get("aggregation_required", ""),
                "expected_retrieval_fanout": meta.get("expected_retrieval_fanout", ""),
                "domain_span": meta.get("domain_span", ""),
                "expected_context_expansion": meta.get("expected_context_expansion", ""),
                "final_synthesis_complexity": meta.get("final_synthesis_complexity", ""),
                "cross_branch_dependency": meta.get("cross_branch_dependency", ""),
                "communication_load_estimate": meta.get("communication_load_estimate", ""),
            })
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
