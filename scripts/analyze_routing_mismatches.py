#!/usr/bin/env python3
"""
Post-hoc mismatch analysis for a WorkBench benchmark run.

Given a benchmark results JSON (produced by ``run_workbench_benchmark.py`` with
``--annotate-router``), recompute the oracle architecture per task and export
every task where ``router_prediction != oracle``, with enough context to do
qualitative follow-up analysis (query text, gold answer, per-architecture
execution trace, error strings, accuracy/latency/token regret).

Each mismatch is tagged with a ``mismatch_category``:

* ``oracle_only_correct``   — oracle got acc=1.0, router's choice got acc<1.0
                              (this is where the router actually costs you points)
* ``router_only_correct``   — shouldn't normally happen (oracle definition picks
                              the best composite score), but we keep the bucket
                              so we notice if the scorer misbehaves.
* ``both_correct_efficiency`` — both correct; oracle wins on latency/tokens.
* ``both_fail_efficiency``  — neither correct; oracle chosen as cheaper/faster.
* ``other``                 — partial-credit or anything not covered above.

Outputs (always under ``results/``):
  * ``<stem>_YYYYMMDD_HHMMSS.csv``   — flat per-task mismatch table
  * ``<stem>_YYYYMMDD_HHMMSS.json``  — same data + summary + category counts

Run from project root:
    python scripts/analyze_routing_mismatches.py results/<sweep>.json \\
        --export mismatches

Optional flags:
    --include-matches  also include perfect-routing rows (handy if you want a
                       single unified CSV for pandas analysis).
    --min-acc-regret   only export rows whose acc_regret >= this float (default 0.0).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from evaluate_regret import ExecutionResult, RegretEvaluator  # noqa: E402

RESULTS_DIR = project_root / "results"


def _categorize(
    oracle_arch: str,
    router_arch: str,
    oracle_acc: float,
    router_acc: float,
) -> str:
    if oracle_arch == router_arch:
        return "match"
    oracle_correct = oracle_acc >= 1.0
    router_correct = router_acc >= 1.0
    if oracle_correct and not router_correct:
        return "oracle_only_correct"
    if router_correct and not oracle_correct:
        return "router_only_correct"
    if oracle_correct and router_correct:
        return "both_correct_efficiency"
    if oracle_acc <= 0.0 and router_acc <= 0.0:
        return "both_fail_efficiency"
    return "other"


def _arch_row(results: list[dict], name: str) -> dict[str, Any]:
    for r in results:
        if r.get("architecture") == name:
            return r
    return {}


def _timestamped(stem: Path, suffix: str, ts: str) -> Path:
    return RESULTS_DIR / f"{stem.stem}_{ts}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export router↔oracle mismatches from a WorkBench benchmark JSON."
    )
    parser.add_argument("results_json", type=Path, help="Benchmark results JSON.")
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("router_mismatches"),
        metavar="STEM",
        help="Filename stem (under results/). CSV + JSON are written with a shared timestamp.",
    )
    parser.add_argument(
        "--include-matches",
        action="store_true",
        help="Also include perfect-routing tasks in the export (category=match).",
    )
    parser.add_argument(
        "--min-acc-regret",
        type=float,
        default=0.0,
        help="Only export rows whose accuracy_regret >= this (default 0.0 = include all mismatches).",
    )
    args = parser.parse_args()

    if not args.results_json.is_file():
        print(f"ERROR: {args.results_json} not found", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    metadata = data.get("metadata", {})

    evaluator = RegretEvaluator()

    rows: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    tasks_with_router = 0

    for item in tasks:
        router_pred = item.get("router_prediction")
        raw_results = item.get("results", [])
        if not raw_results:
            continue

        exec_results = [
            ExecutionResult(
                architecture=r["architecture"],
                accuracy_score=float(r["accuracy_score"]),
                latency_sec=float(r["latency_sec"]),
                total_tokens=int(r["total_tokens"]),
                error_type=r.get("error_type"),
            )
            for r in raw_results
        ]
        oracle = evaluator.determine_oracle(exec_results)

        if router_pred is None:
            # Can't classify a mismatch without a router prediction; skip.
            continue
        tasks_with_router += 1

        metrics = evaluator.calculate_regret(router_pred, exec_results)
        router_choice = next(
            (r for r in exec_results if r.architecture == router_pred),
            exec_results[0],
        )
        category = _categorize(
            oracle.architecture,
            router_pred,
            oracle.accuracy_score,
            router_choice.accuracy_score,
        )
        category_counts[category] = category_counts.get(category, 0) + 1

        is_match = category == "match"
        if is_match and not args.include_matches:
            continue
        if float(metrics["accuracy_regret"]) < args.min_acc_regret:
            continue

        sas_r = _arch_row(raw_results, "Single-Agent System")
        cmas_r = _arch_row(raw_results, "Centralized MAS")

        rows.append({
            "task_id": item.get("task_id", ""),
            "description": item.get("description", ""),
            "domains": json.dumps(item.get("domains", [])),
            "gold_answer": json.dumps(item.get("gold_answer", [])),
            "router_prediction": router_pred,
            "oracle_architecture": oracle.architecture,
            "mismatch_category": category,
            "accuracy_regret": metrics["accuracy_regret"],
            "latency_regret_sec": metrics["latency_regret_sec"],
            "token_regret": metrics["token_regret"],
            "oracle_accuracy": oracle.accuracy_score,
            "router_accuracy": router_choice.accuracy_score,
            "sas_accuracy": sas_r.get("accuracy_score", ""),
            "sas_latency_sec": sas_r.get("latency_sec", ""),
            "sas_total_tokens": sas_r.get("total_tokens", ""),
            "sas_num_tool_calls": len(sas_r.get("execution_path") or []),
            "sas_error": sas_r.get("error_type") or "",
            "sas_execution_path": json.dumps(sas_r.get("execution_path") or []),
            "cmas_accuracy": cmas_r.get("accuracy_score", ""),
            "cmas_latency_sec": cmas_r.get("latency_sec", ""),
            "cmas_total_tokens": cmas_r.get("total_tokens", ""),
            "cmas_num_tool_calls": len(cmas_r.get("execution_path") or []),
            "cmas_error": cmas_r.get("error_type") or "",
            "cmas_execution_path": json.dumps(cmas_r.get("execution_path") or []),
        })

    summary = {
        "source": str(args.results_json.resolve()),
        "phase": metadata.get("phase"),
        "worker_backend": metadata.get("worker_backend"),
        "worker_model": metadata.get("worker_model"),
        "router_backend": metadata.get("router_backend"),
        "router_model": metadata.get("router_model"),
        "tasks_total": len(tasks),
        "tasks_with_router_prediction": tasks_with_router,
        "exported_rows": len(rows),
        "category_counts": category_counts,
        "filters": {
            "include_matches": bool(args.include_matches),
            "min_acc_regret": args.min_acc_regret,
        },
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _timestamped(args.export, ".csv", ts)
    json_path = _timestamped(args.export, ".json", ts)

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "tasks": rows}, f, indent=2)

    print("Mismatch analysis complete.")
    print(f"  Source        : {args.results_json}")
    print(f"  Tasks total   : {summary['tasks_total']}")
    print(f"  With router   : {summary['tasks_with_router_prediction']}")
    print(f"  Categories    : {category_counts}")
    print(f"  Exported rows : {summary['exported_rows']}")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")


if __name__ == "__main__":
    main()
