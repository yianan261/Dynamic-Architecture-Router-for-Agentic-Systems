#!/usr/bin/env python3
"""
Benchmark Sweep Executor.

Runs the PCAB tasks through the Single-Agent and Centralized MAS architectures,
records telemetry (latency, trajectory accuracy, tokens), and saves results to
benchmark_results.json for evaluate_regret.py.

Prerequisites:
    python scripts/setup_pcab.py   # Seed the PCAB database

Run from project root:
    python run_benchmark_sweep.py
"""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.centralized_mas import centralized_mas_app
from dynamic_routing.pcab import get_db_path
from dynamic_routing.pcab_tasks import get_pcab_tasks
from dynamic_routing.single_agent import single_agent_app

# Map PCAB required_tools to the internal context tags emitted by CMAS workers
_TOOL_TO_CMAS_TAG = {
    "get_calendar_events": "[calendar]",
    "search_drive_docs": "[drive]",
    "estimate_commute": "[commute]",
    "get_contact_preferences": "[contacts]",
}

# Map PCAB required_tools to the internal tool names executed by SAS
_TOOL_TO_SAS_CALL = {
    "get_calendar_events": "call_get_calendar_events",
    "search_drive_docs": "call_search_drive_docs",
    "estimate_commute": "call_estimate_commute",
    "get_contact_preferences": "call_get_contact_preferences",
}


def calculate_cmas_accuracy(required_tools: list[str], aggregated_context: list) -> float:
    """Trajectory accuracy: fraction of required tools whose context was aggregated."""
    if not required_tools:
        return 1.0
    context_str = str(aggregated_context).lower()
    hits = sum(1 for t in required_tools if _TOOL_TO_CMAS_TAG.get(t, "") in context_str)
    return hits / len(required_tools)


def calculate_sas_accuracy(required_tools: list[str], executed_tools: list) -> float:
    """Trajectory accuracy: fraction of required tools that were successfully executed."""
    if not required_tools:
        return 1.0
    executed_set = set(executed_tools or [])
    hits = sum(
        1
        for t in required_tools
        if _TOOL_TO_SAS_CALL.get(t, "") in executed_set
    )
    return hits / len(required_tools)


def _ensure_db() -> None:
    """Ensure PCAB database exists and is populated."""
    db_path = get_db_path()
    if not db_path.exists():
        print("ERROR: PCAB database not found. Run: python scripts/setup_pcab.py")
        sys.exit(1)


def main() -> None:
    _ensure_db()

    tasks = get_pcab_tasks()
    print("=" * 70)
    print(f"Benchmark Sweep: {len(tasks)} PCAB Tasks")
    print("(Rule-based execution — no LLM; proves pipeline before adding Llama)")
    print("=" * 70)

    all_sas = []
    all_cmas = []

    for task in tasks:
        print(f"\n[Task] {task.id} ({task.category.upper()})")
        print(f"  Goal: {task.description}")

        overrides = task.extraction_params.as_override_dict()

        # --- 1. Single-Agent System ---
        sas_state = {
            "task": task.description,
            "required_tools": task.required_tools,
            "extraction_overrides": overrides,
            "messages": [],
            "executed_tools": [],
            "pending_tool": "",
            "final_response": "",
        }
        start = time.perf_counter()
        sas_result = single_agent_app.invoke(sas_state)
        sas_latency = time.perf_counter() - start
        sas_executed = sas_result.get("executed_tools") or []
        sas_accuracy = calculate_sas_accuracy(task.required_tools, sas_executed)
        # Mock tokens (replace with LLM metadata later)
        sas_tokens = len(sas_executed) * 400

        print(f"  SAS   | Accuracy: {sas_accuracy:.2f} | Latency: {sas_latency:.4f}s | Tokens: {sas_tokens}")

        # --- 2. Centralized MAS ---
        cmas_state = {
            "task": task.description,
            "required_tools": task.required_tools,
            "extraction_overrides": overrides,
            "aggregated_context": [],
            "next_action": "",
            "final_synthesis": "",
        }
        start = time.perf_counter()
        cmas_result = centralized_mas_app.invoke(cmas_state)
        cmas_latency = time.perf_counter() - start
        cmas_context = cmas_result.get("aggregated_context") or []
        cmas_accuracy = calculate_cmas_accuracy(task.required_tools, cmas_context)
        cmas_tokens = len(cmas_context) * 400 + 800  # Overhead for supervisor

        print(f"  CMAS  | Accuracy: {cmas_accuracy:.2f} | Latency: {cmas_latency:.4f}s | Tokens: {cmas_tokens}")

        all_sas.append((task.id, sas_accuracy, sas_latency, sas_tokens))
        all_cmas.append((task.id, cmas_accuracy, cmas_latency, cmas_tokens))

    # --- Save to benchmark_results.json ---
    results_path = project_root / "benchmark_results.json"
    task_results = []
    for i, task in enumerate(tasks):
        sid, sa, sl, st = all_sas[i]
        cid, ca, cl, ct = all_cmas[i]
        task_results.append({
            "task_id": task.id,
            "description": task.description,
            "category": task.category,
            "router_prediction": None,  # Populate when running full router
            "results": [
                {
                    "architecture": "Single-Agent System",
                    "accuracy_score": round(sa, 2),
                    "latency_sec": round(sl, 4),
                    "total_tokens": st,
                    "error_type": None,
                },
                {
                    "architecture": "Centralized MAS",
                    "accuracy_score": round(ca, 2),
                    "latency_sec": round(cl, 4),
                    "total_tokens": ct,
                    "error_type": None,
                },
            ],
        })
    output = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "rule_based",
            "note": "total_tokens from mock formula; use LLM response_metadata when wired.",
        },
        "tasks": task_results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Task':<25} {'SAS Acc':>8} {'SAS Lat':>10} {'CMAS Acc':>8} {'CMAS Lat':>10}")
    print("-" * 70)
    for i, task in enumerate(tasks):
        sid, sa, sl, st = all_sas[i]
        cid, ca, cl, ct = all_cmas[i]
        print(f"{sid:<25} {sa:>8.2f} {sl:>9.4f}s {ca:>8.2f} {cl:>9.4f}s")
    print("-" * 70)
    avg_sas_acc = sum(r[1] for r in all_sas) / len(all_sas)
    avg_cmas_acc = sum(r[1] for r in all_cmas) / len(all_cmas)
    print(f"{'AVERAGE':<25} {avg_sas_acc:>8.2f} {'':>10} {avg_cmas_acc:>8.2f}")
    print("\nPipeline verified. Run: python evaluate_regret.py (loads benchmark_results.json)")


if __name__ == "__main__":
    main()
