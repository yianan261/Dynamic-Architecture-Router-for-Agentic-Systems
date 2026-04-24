#!/usr/bin/env python3
"""
Benchmark Sweep Executor.

Runs the PCAB tasks through SAS, Centralized MAS, and Decentralized MAS,
records telemetry (latency, trajectory accuracy, tokens), and saves results to
benchmark_results.json for evaluate_regret.py.

Prerequisites:
    python scripts/setup_pcab.py   # Seed the PCAB database

Run from project root:
    python run_benchmark_sweep.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import cast

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv

load_project_root_dotenv()

from dynamic_routing.centralized_mas import centralized_mas_app
from dynamic_routing.decentralized_mas import decentralized_mas_app
from dynamic_routing.pcab import get_db_path
from dynamic_routing.pcab_tasks import get_pcab_tasks
from dynamic_routing.single_agent import single_agent_app
from dynamic_routing.state import CentralizedState, SingleAgentState

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
    use_llm = os.environ.get("USE_LLM_WORKERS", "false").lower() in ("true", "1", "yes")
    phase = "llm" if use_llm else "rule_based"
    print("=" * 70)
    print(f"Benchmark Sweep: {len(tasks)} PCAB Tasks")
    print(f"Mode: {'LLM (Llama-3.1 via vLLM)' if use_llm else 'Rule-based (no LLM)'}")
    print("=" * 70)

    all_sas = []
    all_cmas = []
    all_dmas = []

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
        try:
            start = time.perf_counter()
            sas_result = single_agent_app.invoke(cast(SingleAgentState, sas_state))
            sas_latency = time.perf_counter() - start
            sas_executed = sas_result.get("executed_tools") or []
            sas_accuracy = calculate_sas_accuracy(task.required_tools, sas_executed)
            sas_tokens = sas_result.get("total_tokens") or len(sas_executed) * 400
            sas_taxonomy = sas_result.get("failure_taxonomy")
            sas_path = sas_result.get("execution_path", [])
        except Exception as e:
            sas_latency = time.perf_counter() - start
            sas_accuracy, sas_tokens = 0.0, 0
            sas_taxonomy = f"Unhandled Exception: {str(e)[:100]}"
            sas_path = []
            print(f"  SAS   | ERROR: {sas_taxonomy}")

        if sas_taxonomy:
            print(f"  SAS   | TAXONOMY: {sas_taxonomy}")
            print(f"  SAS   | Accuracy: {sas_accuracy:.2f} | Latency: {sas_latency:.4f}s | Tokens: {sas_tokens}")
        else:
            print(f"  SAS   | Accuracy: {sas_accuracy:.2f} | Latency: {sas_latency:.4f}s | Tokens: {sas_tokens}")
            print(f"          Path: {' → '.join(sas_path) if sas_path else 'n/a'}")

        # --- 2. Centralized MAS ---
        cmas_state = {
            "task": task.description,
            "required_tools": task.required_tools,
            "extraction_overrides": overrides,
            "aggregated_context": [],
            "next_action": "",
            "final_synthesis": "",
        }
        try:
            start = time.perf_counter()
            cmas_result = centralized_mas_app.invoke(cast(CentralizedState, cmas_state))
            cmas_latency = time.perf_counter() - start
            cmas_context = cmas_result.get("aggregated_context") or []
            cmas_accuracy = calculate_cmas_accuracy(task.required_tools, cmas_context)
            cmas_tokens = cmas_result.get("total_tokens") or (len(cmas_context) * 400 + 800)
            cmas_taxonomy = cmas_result.get("failure_taxonomy")
            cmas_path = cmas_result.get("execution_path", [])
        except Exception as e:
            cmas_latency = time.perf_counter() - start
            cmas_accuracy, cmas_tokens = 0.0, 0
            cmas_taxonomy = f"Unhandled Exception: {str(e)[:100]}"
            cmas_path = []
            print(f"  CMAS  | ERROR: {cmas_taxonomy}")

        if cmas_taxonomy:
            print(f"  CMAS  | TAXONOMY: {cmas_taxonomy}")
            print(f"  CMAS  | Accuracy: {cmas_accuracy:.2f} | Latency: {cmas_latency:.4f}s | Tokens: {cmas_tokens}")
        else:
            print(f"  CMAS  | Accuracy: {cmas_accuracy:.2f} | Latency: {cmas_latency:.4f}s | Tokens: {cmas_tokens}")
            print(f"          Path: {' → '.join(cmas_path) if cmas_path else 'n/a'}")

        # --- 3. Decentralized MAS (parallel peers + merge) ---
        dmas_state = {
            "task": task.description,
            "required_tools": task.required_tools,
            "extraction_overrides": overrides,
            "aggregated_context": [],
            "next_action": "",
            "final_synthesis": "",
        }
        try:
            start = time.perf_counter()
            dmas_result = decentralized_mas_app.invoke(cast(CentralizedState, dmas_state))
            dmas_latency = time.perf_counter() - start
            dmas_context = dmas_result.get("aggregated_context") or []
            dmas_accuracy = calculate_cmas_accuracy(task.required_tools, dmas_context)
            dmas_tokens = dmas_result.get("total_tokens") or (len(dmas_context) * 400 + 600)
            dmas_taxonomy = dmas_result.get("failure_taxonomy")
            dmas_path = dmas_result.get("execution_path") or []
        except Exception as e:
            dmas_latency = time.perf_counter() - start
            dmas_accuracy, dmas_tokens = 0.0, 0
            dmas_taxonomy = f"Unhandled Exception: {str(e)[:100]}"
            dmas_path = []
            print(f"  DMAS  | ERROR: {dmas_taxonomy}")

        if dmas_taxonomy:
            print(f"  DMAS  | TAXONOMY: {dmas_taxonomy}")
            print(f"  DMAS  | Accuracy: {dmas_accuracy:.2f} | Latency: {dmas_latency:.4f}s | Tokens: {dmas_tokens}")
        else:
            print(f"  DMAS  | Accuracy: {dmas_accuracy:.2f} | Latency: {dmas_latency:.4f}s | Tokens: {dmas_tokens}")
            print(f"          Path: {' → '.join(str(p) for p in dmas_path) if dmas_path else 'n/a'}")

        all_sas.append((task.id, sas_accuracy, sas_latency, sas_tokens, sas_taxonomy, sas_path))
        all_cmas.append((task.id, cmas_accuracy, cmas_latency, cmas_tokens, cmas_taxonomy, cmas_path))
        all_dmas.append((task.id, dmas_accuracy, dmas_latency, dmas_tokens, dmas_taxonomy, dmas_path))

    # --- Save to benchmark_results.json ---
    results_path = project_root / "benchmark_results.json"
    task_results = []
    for i, task in enumerate(tasks):
        sid, sa, sl, st, se, sp = all_sas[i]
        cid, ca, cl, ct, ce, cp = all_cmas[i]
        did, da, dl, dt, de, dp = all_dmas[i]
        task_results.append({
            "task_id": task.id,
            "description": task.description,
            "category": task.category,
            "router_prediction": None,
            "results": [
                {
                    "architecture": "Single-Agent System",
                    "accuracy_score": round(sa, 2),
                    "latency_sec": round(sl, 4),
                    "total_tokens": st,
                    "error_type": se,
                    "execution_path": sp,
                },
                {
                    "architecture": "Centralized MAS",
                    "accuracy_score": round(ca, 2),
                    "latency_sec": round(cl, 4),
                    "total_tokens": ct,
                    "error_type": ce,
                    "execution_path": cp,
                },
                {
                    "architecture": "Decentralized MAS",
                    "accuracy_score": round(da, 2),
                    "latency_sec": round(dl, 4),
                    "total_tokens": dt,
                    "error_type": de,
                    "execution_path": dp,
                },
            ],
        })
    output = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "note": "total_tokens from LLM response_metadata when USE_LLM_WORKERS=true, mock formula otherwise.",
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
    print(
        f"{'Task':<25} {'SAS Acc':>8} {'SAS Lat':>10} {'CMAS Acc':>8} {'CMAS Lat':>10} "
        f"{'DMAS Acc':>8} {'DMAS Lat':>10}"
    )
    print("-" * 96)
    for i, task in enumerate(tasks):
        sid, sa, sl, st, se, sp = all_sas[i]
        cid, ca, cl, ct, ce, cp = all_cmas[i]
        did, da, dl, dt, de, dp = all_dmas[i]
        sas_col = f"{sa:>8.2f}" if se is None else "   FAIL "
        cmas_col = f"{ca:>8.2f}" if ce is None else "   FAIL "
        dmas_col = f"{da:>8.2f}" if de is None else "   FAIL "
        print(f"{sid:<25} {sas_col} {sl:>9.4f}s {cmas_col} {cl:>9.4f}s {dmas_col} {dl:>9.4f}s")
    print("-" * 96)
    avg_sas_acc = sum(r[1] for r in all_sas) / len(all_sas)
    avg_cmas_acc = sum(r[1] for r in all_cmas) / len(all_cmas)
    avg_dmas_acc = sum(r[1] for r in all_dmas) / len(all_dmas)
    print(f"{'AVERAGE':<25} {avg_sas_acc:>8.2f} {'':>10} {avg_cmas_acc:>8.2f} {'':>10} {avg_dmas_acc:>8.2f}")
    print("\nPipeline verified. Run: python evaluate_regret.py (loads benchmark_results.json)")


if __name__ == "__main__":
    main()
