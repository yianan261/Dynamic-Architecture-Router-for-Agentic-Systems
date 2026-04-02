#!/usr/bin/env python3
"""
WorkBench 50-task sweep: SAS vs CMAS using WorkBench sandbox tools and Llama (vLLM).

Reads benchmarks/workbench_50_queries.csv, runs each architecture on fresh DB state,
grades with outcome-centric equality (same idea as WorkBench is_correct).

Prerequisites:
  - Clone WorkBench: python scripts/setup_workbench.py
  - vLLM worker: set VLLM_WORKER_URL (default http://localhost:8001/v1)

Outputs benchmark_workbench_results.json by default (does not overwrite PCAB results).

Run from project root:
  python run_workbench_benchmark.py
  python run_workbench_benchmark.py path/to/custom.csv --output my_results.json
  python run_workbench_benchmark.py --write-csv   # also writes my_results.csv (same stem as -o)
  python run_workbench_benchmark.py --annotate-router   # fills router_prediction (routing LLM :8000)
  python evaluate_regret.py benchmark_workbench_results.json --export-json regret.json --export-csv regret.csv

Results: JSON is canonical; optional flat CSV is for paper tables (input CSV is not modified).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.workbench_env import (  # noqa: E402
    get_tools_for_domains,
    parse_domains_cell,
    workbench_root,
)
from dynamic_routing.workbench_grade import workbench_accuracy_score  # noqa: E402
from dynamic_routing.workbench_runner import run_workbench_cmas, run_workbench_sas  # noqa: E402


DEFAULT_CSV = project_root / "benchmarks" / "workbench_50_queries.csv"
DEFAULT_OUTPUT = project_root / "benchmark_workbench_results.json"


def _require_workbench() -> None:
    root = workbench_root()
    if not root.is_dir():
        print(f"ERROR: WorkBench not found at {root}. Run: python scripts/setup_workbench.py", file=sys.stderr)
        sys.exit(1)


def _write_flat_table_csv(path: Path, task_results: list[dict]) -> None:
    """One row per task; flat columns for SAS vs CMAS (for LaTeX/Excel)."""
    fieldnames = [
        "task_id",
        "description",
        "domains",
        "sas_accuracy",
        "sas_latency_sec",
        "sas_total_tokens",
        "sas_num_tool_calls",
        "sas_outcome_match",
        "sas_exact_side_effect_match",
        "sas_error",
        "cmas_accuracy",
        "cmas_latency_sec",
        "cmas_total_tokens",
        "cmas_num_tool_calls",
        "cmas_outcome_match",
        "cmas_exact_side_effect_match",
        "cmas_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in task_results:
            r = {x["architecture"]: x for x in item.get("results", [])}
            sas = r.get("Single-Agent System", {})
            cmas = r.get("Centralized MAS", {})
            sg = sas.get("grading") or {}
            cg = cmas.get("grading") or {}
            dom = item.get("domains", [])
            dom_s = json.dumps(dom) if not isinstance(dom, str) else dom
            w.writerow({
                "task_id": item.get("task_id", ""),
                "description": item.get("description", ""),
                "domains": dom_s,
                "sas_accuracy": sas.get("accuracy_score", ""),
                "sas_latency_sec": sas.get("latency_sec", ""),
                "sas_total_tokens": sas.get("total_tokens", ""),
                "sas_num_tool_calls": len(sas.get("execution_path") or []),
                "sas_outcome_match": sg.get("outcome_match", ""),
                "sas_exact_side_effect_match": sg.get("exact_side_effect_match", ""),
                "sas_error": sas.get("error_type") or "",
                "cmas_accuracy": cmas.get("accuracy_score", ""),
                "cmas_latency_sec": cmas.get("latency_sec", ""),
                "cmas_total_tokens": cmas.get("total_tokens", ""),
                "cmas_num_tool_calls": len(cmas.get("execution_path") or []),
                "cmas_outcome_match": cg.get("outcome_match", ""),
                "cmas_exact_side_effect_match": cg.get("exact_side_effect_match", ""),
                "cmas_error": cmas.get("error_type") or "",
            })


def _require_llm_env() -> None:
    """Inform if worker URL is default (user may need vLLM)."""
    url = os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1")
    print(f"Using VLLM_WORKER_URL={url}  model={os.environ.get('VLLM_WORKER_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WorkBench SAS vs CMAS benchmark sweep")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"Queries CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a flat table next to the JSON (same path, .csv extension).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_out",
        default="",
        metavar="PATH",
        help="Explicit path for the flat results table CSV (overrides --write-csv stem).",
    )
    parser.add_argument(
        "--annotate-router",
        action="store_true",
        help=(
            "After sweep, set router_prediction per task via predict_routed_architecture "
            "(Mistral @ VLLM_BASE_URL :8000 or keyword fallback). Skips separate annotate script."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only first N tasks (0 = all)")
    args = parser.parse_args()

    _require_workbench()
    _require_llm_env()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]

    task_results: list[dict] = []
    phase = "workbench_llama_vllm"

    for i, row in enumerate(rows):
        q = row["query"]
        domains = parse_domains_cell(row["domains"])
        tools = get_tools_for_domains(domains)
        gold = ast.literal_eval(row["answer"])
        tid = f"WB-{i:03d}"

        print(f"\n[{tid}] domains={domains}")
        print(f"  Q: {q[:100]}{'...' if len(q) > 100 else ''}")

        sas = run_workbench_sas(q, tools)
        sas_score, sas_outcome, sas_exact = workbench_accuracy_score(sas["function_calls"], gold, sas["error"])
        sas_tok = sas["total_tokens"] or (len(sas["function_calls"]) * 400)
        if sas["error"]:
            print(f"  SAS   | ERROR: {sas['error'][:120]}")
        print(
            f"  SAS   | acc={sas_score:.2f} outcome={sas_outcome} exact={sas_exact} "
            f"lat={sas['latency_sec']:.3f}s tok={sas_tok} calls={len(sas['function_calls'])}"
        )

        cmas = run_workbench_cmas(q, domains)
        cmas_err = cmas.get("error") or ""
        cmas_score, cmas_outcome, cmas_exact = workbench_accuracy_score(
            cmas["function_calls"], gold, cmas_err
        )
        cmas_tok = cmas["total_tokens"] or (len(cmas["function_calls"]) * 400 + 800)
        tax = cmas.get("failure_taxonomy")
        if cmas_err:
            print(f"  CMAS  | ERROR: {cmas_err[:120]}")
        if tax:
            print(f"  CMAS  | TAXONOMY: {tax}")
        print(
            f"  CMAS  | acc={cmas_score:.2f} outcome={cmas_outcome} exact={cmas_exact} "
            f"lat={cmas['latency_sec']:.3f}s tok={cmas_tok} calls={len(cmas['function_calls'])}"
        )

        err_sas = sas["error"][:200] if sas["error"] else None
        _cmas_msg = (cmas_err or tax or "").strip()
        err_cmas = _cmas_msg[:200] if _cmas_msg else None

        task_results.append({
            "task_id": tid,
            "description": q,
            "category": "workbench",
            "domains": domains,
            "gold_answer": gold,
            "router_prediction": None,  # filled if --annotate-router
            "results": [
                {
                    "architecture": "Single-Agent System",
                    "accuracy_score": round(sas_score, 2),
                    "latency_sec": round(sas["latency_sec"], 4),
                    "total_tokens": int(sas_tok),
                    "error_type": err_sas,
                    "execution_path": sas["function_calls"],
                    "grading": {"outcome_match": sas_outcome, "exact_side_effect_match": sas_exact},
                },
                {
                    "architecture": "Centralized MAS",
                    "accuracy_score": round(cmas_score, 2),
                    "latency_sec": round(cmas["latency_sec"], 4),
                    "total_tokens": int(cmas_tok),
                    "error_type": err_cmas,
                    "execution_path": cmas["function_calls"],
                    "grading": {"outcome_match": cmas_outcome, "exact_side_effect_match": cmas_exact},
                },
            ],
        })

    if args.annotate_router:
        from dynamic_routing.router import predict_routed_architecture  # noqa: E402

        print("\nAnnotating router_prediction (routing meta-LLM / fallback)...")
        for item in task_results:
            item["router_prediction"] = predict_routed_architecture(item.get("description") or "")

    out_path = Path(args.output)
    payload = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "csv": str(csv_path.resolve()),
            "note": (
                "accuracy_score: WorkBench outcome match (DB state after predicted calls vs gold). "
                "execution_path: WorkBench-style .func(...) strings. "
                "Tokens from LLM usage when available."
            ),
        },
        "tasks": task_results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path.resolve()}")

    if args.csv_out:
        csv_table = Path(args.csv_out)
    elif args.write_csv:
        csv_table = out_path.with_suffix(".csv")
    else:
        csv_table = None
    if csv_table is not None:
        _write_flat_table_csv(csv_table, task_results)
        print(f"Wrote table CSV {csv_table.resolve()}")

    print("Oracle / regret: python evaluate_regret.py", out_path)


if __name__ == "__main__":
    main()
