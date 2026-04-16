#!/usr/bin/env python3
"""
WorkBench 50-task sweep: SAS vs CMAS using WorkBench sandbox tools and Llama (vLLM).

Reads benchmarks/workbench_50_queries.csv, runs each architecture on fresh DB state,
grades with outcome-centric equality (same idea as WorkBench is_correct).

Prerequisites:
  - Clone WorkBench: python scripts/setup_workbench.py
  - LLM_BACKEND=vllm (default): VLLM_WORKER_URL + VLLM_WORKER_MODEL
  - LLM_BACKEND=openai: OPENAI_API_KEY, OPENAI_WORKER_MODEL=gpt-5.4-mini (optional OPENAI_BASE_URL)
  - LLM_BACKEND=google: GOOGLE_API_KEY, GOOGLE_WORKER_MODEL=gemini-3.1-flash-lite-preview
  - Optional ROUTER_LLM_BACKEND to use a different provider for routing metadata

Writes timestamped files under results/ (e.g. results/benchmark_workbench_results_YYYYMMDD_HHMMSS.json).

Run from project root:
  python run_workbench_benchmark.py
  python run_workbench_benchmark.py path/to/custom.csv --output my_results.json
  python run_workbench_benchmark.py --write-csv   # same run id: results/<stem>_TS.json + .csv
  python run_workbench_benchmark.py --annotate-router   # fills router_prediction (routing LLM :8000)
  python evaluate_regret.py results/<latest>.json --export-json regret.json --export-csv regret.csv

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

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()

from dynamic_routing.workbench_env import (  # noqa: E402
    get_tools_for_domains,
    parse_domains_cell,
    workbench_root,
)
from dynamic_routing.workbench_grade import workbench_accuracy_score  # noqa: E402
from dynamic_routing.workbench_runner import run_workbench_cmas, run_workbench_sas  # noqa: E402


DEFAULT_CSV = project_root / "benchmarks" / "workbench_50_queries.csv"
DEFAULT_OUTPUT = project_root / "benchmark_workbench_results.json"
RESULTS_DIR = project_root / "results"


def _timestamped_results_path(user_path: Path, ts: str) -> Path:
    """results/<stem>_YYYYMMDD_HHMMSS<suffix> (suffix from user_path, default .json)."""
    stem = user_path.stem
    suffix = user_path.suffix if user_path.suffix else ".json"
    return RESULTS_DIR / f"{stem}_{ts}{suffix}"


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
    """Log active LLM backend and main model env (see dynamic_routing.chat_models)."""
    backend = os.environ.get("LLM_BACKEND", "vllm").strip().lower()
    if backend == "openai":
        print(
            "Using LLM_BACKEND=openai "
            f"model={os.environ.get('OPENAI_WORKER_MODEL', 'gpt-5.4-mini')} "
            f"base={os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')}"
        )
    elif backend == "google":
        print(
            "Using LLM_BACKEND=google "
            f"model={os.environ.get('GOOGLE_WORKER_MODEL', 'gemini-3.1-flash-lite-preview')}"
        )
    else:
        url = os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1")
        print(
            f"Using LLM_BACKEND=vllm VLLM_WORKER_URL={url} "
            f"model={os.environ.get('VLLM_WORKER_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="WorkBench SAS vs CMAS benchmark sweep")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"Queries CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT.name),
        help=f"JSON filename stem (default: {DEFAULT_OUTPUT.name}); written under results/ with timestamp",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write results/<same_stem_as_json>_TS.csv (same timestamp as JSON).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_out",
        default="",
        metavar="STEM",
        help="CSV filename stem under results/ with same timestamp (overrides default table name).",
    )
    parser.add_argument(
        "--annotate-router",
        dest="annotate_router",
        action="store_true",
        default=True,
        help=(
            "(default ON) After sweep, set router_prediction per task via "
            "predict_routed_architecture. Each call is wrapped in try/except and falls "
            "back to 'Single-Agent System' if the routing LLM fails."
        ),
    )
    parser.add_argument(
        "--no-annotate-router",
        dest="annotate_router",
        action="store_false",
        help="Leave router_prediction as None (you can still run scripts/annotate_router_predictions.py later).",
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
        ok = 0
        fail = 0
        for item in task_results:
            desc = item.get("description") or ""
            try:
                item["router_prediction"] = predict_routed_architecture(desc)
                ok += 1
            except Exception as e:
                print(f"  [router-annotate] {item.get('task_id', '?')}: {str(e)[:120]}; defaulting to Single-Agent System")
                item["router_prediction"] = "Single-Agent System"
                fail += 1
        print(f"  router annotations: {ok} ok, {fail} fell back to SAS default")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = _timestamped_results_path(Path(args.output), run_ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        csv_table = _timestamped_results_path(Path(args.csv_out), run_ts)
    elif args.write_csv:
        csv_table = out_path.with_suffix(".csv")
    else:
        csv_table = None
    if csv_table is not None:
        csv_table.parent.mkdir(parents=True, exist_ok=True)
        _write_flat_table_csv(csv_table, task_results)
        print(f"Wrote table CSV {csv_table.resolve()}")

    print("Oracle / regret: python evaluate_regret.py", out_path)


if __name__ == "__main__":
    main()
