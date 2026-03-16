#!/usr/bin/env python3
"""
Oracle Evaluation Harness (v2) for the Dynamic Architecture Router.

Calculates the mathematical ground truth (Oracle Baseline) using partial
completion scores (Trajectory Accuracy) to fairly evaluate agent runs.

Composite reward: Score = (α * Accuracy) - (β * Latency_norm) - (γ * Tokens_norm)

Run from project root (after scripts/setup_pcab.py):
    python evaluate_regret.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))


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


def main() -> None:
    evaluator = RegretEvaluator()

    print("--- Benchmark Task: PCAB-Exploratory-03 ---")
    print("Task: 'Cross-reference 3 different data sources to find schedule anomalies.'")

    test_results = [
        ExecutionResult(
            "Single-Agent System",
            accuracy_score=0.33,
            latency_sec=60.0,
            total_tokens=4200,
            error_type="Timeout",
        ),
        ExecutionResult(
            "Centralized MAS",
            accuracy_score=0.66,
            latency_sec=18.0,
            total_tokens=12500,
            error_type="Synthesis Drift",
        ),
        ExecutionResult(
            "Decentralized MAS",
            accuracy_score=1.0,
            latency_sec=45.0,
            total_tokens=35000,
        ),
    ]

    router_prediction = "Centralized MAS"
    metrics = evaluator.calculate_regret(router_prediction, test_results)

    print(f"Oracle Determined Best Architecture: {metrics['oracle_baseline']}")
    print(f"Router Selected: {metrics['router_prediction']}")
    print(f"Perfect Routing: {metrics['perfect_routing']}")
    print(f"Accuracy Regret: {metrics['accuracy_regret']}")
    print(f"Latency Regret: {metrics['latency_regret_sec']}s")
    print(f"Token Regret: {metrics['token_regret']} tokens")

    print("\n--- Benchmark Task: PCAB-Seq-01 (Heavy Sequential) ---")
    print("Task: 'Find my advisor's next available slot, check conflicts, plan commute.'")

    seq_results = [
        ExecutionResult("Single-Agent System", 1.0, 12.5, 4200),
        ExecutionResult("Centralized MAS", 1.0, 28.0, 14500),
        ExecutionResult(
            "Decentralized MAS",
            0.0,
            45.0,
            22000,
            error_type="Infinite Debate Loop",
        ),
    ]

    seq_metrics = evaluator.calculate_regret("Single-Agent System", seq_results)
    print(f"Oracle: {seq_metrics['oracle_baseline']}")
    print(f"Perfect Routing: {seq_metrics['perfect_routing']}")

    print("\n--- Benchmark Task: PCAB-Par-14 (Heavy Parallel) ---")
    print("(Router mistakenly picks SAS for a highly parallel task)")

    parallel_results = [
        ExecutionResult("Single-Agent System", 1.0, 45.0, 8000),
        ExecutionResult("Centralized MAS", 1.0, 14.2, 9500),
    ]

    par_metrics = evaluator.calculate_regret("Single-Agent System", parallel_results)
    print(f"Oracle: {par_metrics['oracle_baseline']}")
    print(f"Latency Regret: {par_metrics['latency_regret_sec']}s penalty paid by the user!")


if __name__ == "__main__":
    main()
