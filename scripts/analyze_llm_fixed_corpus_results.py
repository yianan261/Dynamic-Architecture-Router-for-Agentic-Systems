#!/usr/bin/env python3
"""Aggregate LLM-backed fixed-corpus benchmark results.

This script computes architecture-level performance, oracle counts, and router
policy regret for BrowseComp-Plus and Fin-RATE result JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()


ACCURACY_WEIGHT = 0.50
LATENCY_WEIGHT = 0.40
TOKEN_WEIGHT = 0.10
ARCH_ORDER = ["Single-Agent System", "Centralized MAS", "Decentralized MAS"]
ARCH_SHORT = {
    "Single-Agent System": "SAS",
    "Centralized MAS": "CMAS",
    "Decentralized MAS": "DMAS",
}


def _oracle_for(results: list[dict[str, Any]]) -> dict[str, Any]:
    max_lat = max(max(float(r["latency_sec"]) for r in results), 1.0)
    max_tok = max(max(int(r["total_tokens"]) for r in results), 1)

    def score(run: dict[str, Any]) -> float:
        return (
            ACCURACY_WEIGHT * float(run["accuracy_score"])
            - LATENCY_WEIGHT * (float(run["latency_sec"]) / max_lat)
            - TOKEN_WEIGHT * (int(run["total_tokens"]) / max_tok)
        )

    return max(results, key=score)


def _result_map(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in task.get("results", []):
        out[str(raw["architecture"])] = {
            "architecture": str(raw["architecture"]),
            "accuracy_score": float(raw.get("accuracy_score") or 0.0),
            "latency_sec": float(raw.get("latency_sec") or 0.0),
            "total_tokens": int(raw.get("total_tokens") or 0),
            "semantic_failure": raw.get("semantic_failure") or None,
            "error_type": raw.get("error_type"),
        }
    return out


def _regret(
    prediction: str,
    rmap: dict[str, dict[str, Any]],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    chosen = rmap.get(prediction) or next(iter(rmap.values()))
    return {
        "perfect": prediction == oracle["architecture"],
        "accuracy_regret": max(
            0.0,
            float(oracle["accuracy_score"]) - float(chosen["accuracy_score"]),
        ),
        "latency_regret_sec": max(
            0.0,
            float(chosen["latency_sec"]) - float(oracle["latency_sec"]),
        ),
        "token_regret": max(0, int(chosen["total_tokens"]) - int(oracle["total_tokens"])),
    }


def _learned_predictions(
    tasks: list[dict[str, Any]],
    model_path: Path,
) -> dict[str, str]:
    import joblib

    from dynamic_routing.router_policy import predict_learned_display_architecture
    from dynamic_routing.vllm_integration import predict_routing_metadata

    old = os.environ.get("ROUTER_LEARNED_MODEL_PATH")
    os.environ["ROUTER_LEARNED_MODEL_PATH"] = str(model_path)
    predictions: dict[str, str] = {}
    try:
        joblib.load(model_path)
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            desc = str(task.get("description") or "")
            meta = predict_routing_metadata(desc)
            predictions[task_id] = predict_learned_display_architecture(meta) or "Single-Agent System"
    finally:
        if old is None:
            os.environ.pop("ROUTER_LEARNED_MODEL_PATH", None)
        else:
            os.environ["ROUTER_LEARNED_MODEL_PATH"] = old
    return predictions


def _pct(count: int, total: int) -> str:
    return f"{(100 * count / max(total, 1)):.1f}\\%"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latex_tables(perf_rows: list[dict[str, Any]], router_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\caption{LLM-Backed Fixed-Corpus Performance \& Primary Failure Modes}",
            r"\begin{center}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{|l|l|c|c|p{4cm}|}",
            r"\hline",
            r"\textbf{Benchmark} & \textbf{Arch.} & \textbf{Avg. Acc.} & \textbf{Oracle Wins} & \textbf{Primary Failure Mode / Note} \\",
            r"\hline",
        ]
    )
    for benchmark in ("BrowseComp-Plus", "Fin-RATE"):
        rows = [r for r in perf_rows if r["benchmark"] == benchmark]
        for idx, row in enumerate(rows):
            bench_cell = rf"\multirow{{3}}{{*}}{{\textbf{{{benchmark}}}}}" if idx == 0 else ""
            wins = f"{row['oracle_wins']} ({_pct(int(row['oracle_wins']), int(row['tasks']))})"
            lines.append(
                f"{bench_cell} & {row['architecture']} & {float(row['avg_accuracy']):.4f} & "
                f"{wins} & {row['primary_failure_mode_note']} \\\\"
            )
        lines.append(r"\hline")
    lines.extend(
        [
            r"\end{tabular}%",
            "}",
            r"\label{tab:llm_architecture_performance}",
            r"\end{center}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\caption{LLM-Backed Fixed-Corpus Router Evaluation (Regret Metrics)}",
            r"\begin{center}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{|l|l|c|c|c|c|}",
            r"\hline",
            r"\textbf{Dataset} & \textbf{Router Policy} & \textbf{Perfect Routing} & \textbf{Acc. Regret} & \textbf{Lat. Regret(s)} & \textbf{Tok. Regret} \\",
            r"\hline",
        ]
    )
    for benchmark in ("BrowseComp-Plus", "Fin-RATE"):
        rows = [r for r in router_rows if r["dataset"] == benchmark]
        for idx, row in enumerate(rows):
            bench_cell = rf"\multirow{{3}}{{*}}{{\textbf{{{benchmark}}}}}" if idx == 0 else ""
            perfect = (
                f"{row['perfect_routing_count']} / {row['tasks']} "
                f"({_pct(int(row['perfect_routing_count']), int(row['tasks']))})"
            )
            lines.append(
                f"{bench_cell} & {row['router_policy']} & {perfect} & "
                f"{float(row['avg_accuracy_regret']):.4f} & "
                f"{float(row['avg_latency_regret_sec']):.4f} & "
                f"{float(row['avg_token_regret']):.2f} \\\\"
            )
        lines.append(r"\hline")
    lines.extend(
        [
            r"\end{tabular}%",
            "}",
            r"\label{tab:llm_router_regret}",
            r"\end{center}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LLM fixed-corpus benchmark results")
    parser.add_argument("--browsecomp", type=Path, required=True)
    parser.add_argument("--finrate", type=Path, required=True)
    parser.add_argument("--balanced-model", type=Path, required=True)
    parser.add_argument("--unweighted-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=project_root / "results" / "llm_fixed_corpus_analysis")
    args = parser.parse_args()

    sources = {
        "BrowseComp-Plus": args.browsecomp,
        "Fin-RATE": args.finrate,
    }
    models = {
        "LR (Balanced)": args.balanced_model,
        "LR (Unweighted)": args.unweighted_model,
    }

    perf_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y%m%d_%H%M%S"),
        "sources": {k: str(v) for k, v in sources.items()},
        "benchmarks": {},
    }

    for benchmark, path in sources.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = data.get("tasks", [])
        task_infos: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]] = []
        by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
        failure_modes: dict[str, Counter[str]] = defaultdict(Counter)
        oracle_counts: Counter[str] = Counter()

        for task in tasks:
            rmap = _result_map(task)
            if not rmap:
                continue
            results = [rmap[arch] for arch in ARCH_ORDER if arch in rmap]
            oracle = _oracle_for(results)
            oracle_counts[str(oracle["architecture"])] += 1
            task_infos.append((task, rmap, oracle))
            oracle_rows.append(
                {
                    "benchmark": benchmark,
                    "task_id": task.get("task_id"),
                    "oracle_architecture": oracle["architecture"],
                    "oracle_accuracy": oracle["accuracy_score"],
                    "oracle_latency_sec": round(float(oracle["latency_sec"]), 4),
                    "oracle_total_tokens": oracle["total_tokens"],
                }
            )
            for arch, result in rmap.items():
                by_arch[arch].append(result)
                semantic_failure = result.get("semantic_failure")
                if semantic_failure:
                    failure_modes[arch][semantic_failure.get("label") or "Semantic failure"] += 1
                elif float(result["accuracy_score"]) < 1.0:
                    failure_modes[arch]["Imperfect scores (unlabeled)"] += 1

        benchmark_report: dict[str, Any] = {
            "task_count": len(task_infos),
            "oracle_counts": dict(oracle_counts),
            "architectures": {},
        }
        for arch in ARCH_ORDER:
            rows = by_arch.get(arch, [])
            if not rows:
                continue
            modes = failure_modes[arch]
            if modes:
                mode, count = modes.most_common(1)[0]
                note = f"{mode} ({count})"
            else:
                note = "No primary failures observed"
            row = {
                "benchmark": benchmark,
                "architecture": ARCH_SHORT[arch],
                "architecture_full": arch,
                "tasks": len(rows),
                "avg_accuracy": round(sum(float(r["accuracy_score"]) for r in rows) / len(rows), 4),
                "avg_latency_sec": round(sum(float(r["latency_sec"]) for r in rows) / len(rows), 4),
                "avg_tokens": round(sum(int(r["total_tokens"]) for r in rows) / len(rows), 2),
                "oracle_wins": oracle_counts[arch],
                "oracle_win_rate": round(oracle_counts[arch] / max(len(rows), 1), 4),
                "perfect_count": sum(1 for r in rows if float(r["accuracy_score"]) >= 1.0),
                "primary_failure_mode_note": note,
            }
            perf_rows.append(row)
            benchmark_report["architectures"][arch] = row

        policies: dict[str, dict[str, str]] = {
            "Rule/SAS (Base)": {
                str(task.get("task_id") or ""): "Single-Agent System"
                for task, _, _ in task_infos
            }
        }
        for label, model_path in models.items():
            policies[label] = _learned_predictions([task for task, _, _ in task_infos], model_path)

        for policy_name, predictions in policies.items():
            perfect = 0
            acc_regret = 0.0
            lat_regret = 0.0
            token_regret = 0.0
            pred_counts: Counter[str] = Counter()
            for task, rmap, oracle in task_infos:
                pred = predictions.get(str(task.get("task_id") or ""), "Single-Agent System")
                pred_counts[pred] += 1
                metrics = _regret(pred, rmap, oracle)
                perfect += int(metrics["perfect"])
                acc_regret += float(metrics["accuracy_regret"])
                lat_regret += float(metrics["latency_regret_sec"])
                token_regret += int(metrics["token_regret"])
            n = len(task_infos)
            router_rows.append(
                {
                    "dataset": benchmark,
                    "router_policy": policy_name,
                    "tasks": n,
                    "perfect_routing_count": perfect,
                    "perfect_routing_rate": round(perfect / max(n, 1), 4),
                    "avg_accuracy_regret": round(acc_regret / max(n, 1), 4),
                    "avg_latency_regret_sec": round(lat_regret / max(n, 1), 4),
                    "avg_token_regret": round(token_regret / max(n, 1), 2),
                    "prediction_counts": json.dumps(dict(pred_counts), sort_keys=True),
                }
            )
        report["benchmarks"][benchmark] = benchmark_report

    stamp = report["generated_at"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"llm_fixed_corpus_30_summary_{stamp}.json"
    perf_csv = args.output_dir / f"llm_fixed_corpus_30_architecture_performance_{stamp}.csv"
    router_csv = args.output_dir / f"llm_fixed_corpus_30_router_regret_{stamp}.csv"
    oracle_csv = args.output_dir / f"llm_fixed_corpus_30_oracle_tasks_{stamp}.csv"
    latex_path = args.output_dir / f"llm_fixed_corpus_30_tables_{stamp}.tex"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(perf_csv, perf_rows)
    _write_csv(router_csv, router_rows)
    _write_csv(oracle_csv, oracle_rows)
    latex_path.write_text(_latex_tables(perf_rows, router_rows), encoding="utf-8")

    for path in (json_path, perf_csv, router_csv, oracle_csv, latex_path):
        print(path.resolve())
    print(latex_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
