#!/usr/bin/env python3
"""
Oracle Evaluation Harness (v2) for the Dynamic Architecture Router.

Calculates the mathematical ground truth (Oracle Baseline) using partial
completion scores (Trajectory Accuracy) to fairly evaluate agent runs.

Composite reward: Score = (α * Accuracy) - (β * Latency_norm) - (γ * Tokens_norm)

Data source: Loads benchmark_results.json produced by run_benchmark_sweep.py.
Tokens come from mock formula (rule-based) or LLM response_metadata (when wired).

Run from project root:
    python run_benchmark_sweep.py   # Generate results
    python evaluate_regret.py       # Load JSON and compute oracle/regret
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

DEFAULT_RESULTS_PATH = project_root / "benchmark_results.json"


@dataclass
class ExecutionResult:
    """Stores the telemetry from a single architecture's attempt at a task."""

    architecture: str
    accuracy_score: float  # 0.0 to 1.0 (e.g., 0.66 if 2/3 tools succeeded)
    latency_sec: float
    total_tokens: int
    error_type: str | None = None


class RegretEvaluator:
    """
    Calculates the Oracle Baseline by maximizing a composite reward score.
    Accuracy is heavily weighted because a fast, cheap, but wrong answer is useless.
    """

    def __init__(
        self,
        accuracy_weight: float = 0.6,
        latency_weight: float = 0.25,
        token_weight: float = 0.15,
    ):
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.token_weight = token_weight

    def determine_oracle(self, results: list[ExecutionResult]) -> ExecutionResult:
        """
        Calculates the Oracle Baseline by maximizing the composite reward score.
        """
        if not results:
            raise ValueError("Cannot determine oracle from empty results")

        max_lat = max(r.latency_sec for r in results)
        max_tok = max(r.total_tokens for r in results)

        # Prevent division by zero if a run takes 0 seconds/tokens
        max_lat = max_lat if max_lat > 0 else 1.0
        max_tok = max_tok if max_tok > 0 else 1

        def oracle_score(run: ExecutionResult) -> float:
            norm_lat = run.latency_sec / max_lat
            norm_tok = run.total_tokens / max_tok
            reward = (
                (self.accuracy_weight * run.accuracy_score)
                - (self.latency_weight * norm_lat)
                - (self.token_weight * norm_tok)
            )
            return reward

        return max(results, key=oracle_score)

    def calculate_regret(
        self,
        predicted_arch: str,
        results: list[ExecutionResult],
    ) -> dict[str, Any]:
        """
        Calculates the "Regret" suffered by trusting the Router instead of the Oracle.
        """
        oracle = self.determine_oracle(results)
        router_choice = next(
            (r for r in results if r.architecture == predicted_arch),
            results[0],
        )

        accuracy_regret = max(0.0, oracle.accuracy_score - router_choice.accuracy_score)
        latency_regret = max(0.0, router_choice.latency_sec - oracle.latency_sec)
        token_regret = max(0, router_choice.total_tokens - oracle.total_tokens)

        return {
            "router_prediction": predicted_arch,
            "oracle_baseline": oracle.architecture,
            "perfect_routing": predicted_arch == oracle.architecture,
            "accuracy_regret": round(accuracy_regret, 2),
            "latency_regret_sec": round(latency_regret, 2),
            "token_regret": token_regret,
        }


def compute_trajectory_accuracy(
    required_tools: list[str],
    executed_tools: list[str],
) -> float:
    """
    Compute Trajectory Accuracy (Tool Recall) as fraction of required tools
    that were successfully executed.

    required_tools: e.g. ["get_contact_preferences", "get_calendar_events", "estimate_commute"]
    executed_tools: tools that were actually called (from agent logs)
    """
    if not required_tools:
        return 1.0
    executed_set = {t.lower() for t in executed_tools}
    hits = sum(1 for t in required_tools if t.lower() in executed_set)
    return hits / len(required_tools)


def load_results(path: Path | None = None) -> dict[str, Any]:
    """Load benchmark results from JSON. Returns the parsed object."""
    p = path or DEFAULT_RESULTS_PATH
    if not p.exists():
        print(f"ERROR: {p} not found. Run: python run_benchmark_sweep.py")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def main(results_path: Path | None = None) -> None:
    data = load_results(results_path)
    metadata = data.get("metadata", {})
    tasks = data.get("tasks", [])

    print("=" * 70)
    print("Oracle Evaluation Harness (v2)")
    print(f"Data: benchmark_results.json | Phase: {metadata.get('phase', '?')}")
    print("=" * 70)

    evaluator = RegretEvaluator()
    perfect_count = 0
    total_regret_acc = 0.0
    tasks_with_router = 0

    for item in tasks:
        task_id = item.get("task_id", "?")
        description = item.get("description", "")[:60]
        router_pred = item.get("router_prediction")

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

        oracle = evaluator.determine_oracle(results)
        print(f"\n[{task_id}] {description}...")
        print(f"  Oracle: {oracle.architecture} (acc={oracle.accuracy_score:.2f}, lat={oracle.latency_sec:.4f}s, tok={oracle.total_tokens})")

        if router_pred is not None:
            metrics = evaluator.calculate_regret(router_pred, results)
            print(f"  Router: {router_pred} | Perfect: {metrics['perfect_routing']}")
            if not metrics["perfect_routing"]:
                print(f"  Regret: acc={metrics['accuracy_regret']}, lat={metrics['latency_regret_sec']}s, tok={metrics['token_regret']}")
            perfect_count += int(metrics["perfect_routing"])
            total_regret_acc += metrics["accuracy_regret"]
            tasks_with_router += 1
        else:
            print("  (No router_prediction in sweep; run full router for regret)")

    if tasks_with_router > 0:
        print("\n" + "-" * 70)
        print(f"Perfect routing: {perfect_count}/{tasks_with_router} | Avg accuracy regret: {total_regret_acc / tasks_with_router:.2f}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(path)
