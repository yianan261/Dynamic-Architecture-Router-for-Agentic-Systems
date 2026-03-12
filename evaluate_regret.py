#!/usr/bin/env python3
"""
Oracle Evaluation Harness for the Dynamic Architecture Router.

Defines the mathematical ground truth for the PCAB benchmark. Evaluates a task
across all three topologies (SAS, CMAS, DMAS) to find the "Oracle Best"
architecture, and calculates the Routing Regret of the Dynamic Router's prediction.

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
    success: bool
    latency_sec: float
    total_tokens: int
    error_type: str | None = None


class RegretEvaluator:
    """
    Calculates Oracle Baseline and Routing Regret.
    Hyperparameters control the trade-off between latency and token cost.
    """

    def __init__(self, latency_weight: float = 0.6, token_weight: float = 0.4):
        self.latency_weight = latency_weight
        self.token_weight = token_weight

    def determine_oracle(self, results: list[ExecutionResult]) -> ExecutionResult:
        """
        Calculates the mathematical ground truth (Oracle Baseline).
        Rule 1: Success is paramount. Failed runs are disqualified.
        Rule 2: If multiple succeed, rank by weighted combo of latency and tokens.
        """
        successful_runs = [r for r in results if r.success]

        if not successful_runs:
            return min(results, key=lambda x: (x.latency_sec, x.total_tokens))

        if len(successful_runs) == 1:
            return successful_runs[0]

        max_lat = max(r.latency_sec for r in successful_runs)
        max_tok = max(r.total_tokens for r in successful_runs)

        def oracle_score(run: ExecutionResult) -> float:
            norm_lat = run.latency_sec / max_lat if max_lat > 0 else 0
            norm_tok = run.total_tokens / max_tok if max_tok > 0 else 0
            return (self.latency_weight * norm_lat) + (self.token_weight * norm_tok)

        return min(successful_runs, key=oracle_score)

    def calculate_regret(
        self,
        predicted_arch: str,
        results: list[ExecutionResult],
    ) -> dict[str, Any]:
        """
        Calculates how much Regret the system suffered by trusting the Router
        instead of the perfect Oracle.
        """
        oracle = self.determine_oracle(results)
        router_choice = next(
            (r for r in results if r.architecture == predicted_arch),
            results[0],
        )

        success_regret = int(oracle.success) - int(router_choice.success)
        latency_regret = max(0, router_choice.latency_sec - oracle.latency_sec)
        token_regret = max(0, router_choice.total_tokens - oracle.total_tokens)

        return {
            "router_prediction": predicted_arch,
            "oracle_baseline": oracle.architecture,
            "success_regret": success_regret,
            "latency_regret_sec": round(latency_regret, 2),
            "token_regret": token_regret,
            "perfect_routing": predicted_arch == oracle.architecture,
        }


def main() -> None:
    evaluator = RegretEvaluator()

    print("--- Benchmark Task: PCAB-Seq-01 (Heavy Sequential) ---")
    print("Task: 'Find my advisor's next available slot, check conflicts, plan commute.'")

    test_results = [
        ExecutionResult("Single-Agent System", success=True, latency_sec=12.5, total_tokens=4200),
        ExecutionResult(
            "Centralized MAS",
            success=True,
            latency_sec=28.0,
            total_tokens=14500,
        ),
        ExecutionResult(
            "Decentralized MAS",
            success=False,
            latency_sec=45.0,
            total_tokens=22000,
            error_type="Infinite Debate Loop",
        ),
    ]

    router_prediction = "Single-Agent System"
    metrics = evaluator.calculate_regret(router_prediction, test_results)

    print(f"Oracle Determined Best Architecture: {metrics['oracle_baseline']}")
    print(f"Router Selected: {metrics['router_prediction']}")
    print(f"Perfect Routing: {metrics['perfect_routing']}")
    print(f"Latency Regret: {metrics['latency_regret_sec']}s")
    print(f"Token Regret: {metrics['token_regret']} tokens")

    print("\n--- Benchmark Task: PCAB-Par-14 (Heavy Parallel) ---")
    print("(Router mistakenly picks SAS for a highly parallel task)")

    bad_prediction = "Single-Agent System"
    parallel_results = [
        ExecutionResult(
            "Single-Agent System",
            success=True,
            latency_sec=45.0,
            total_tokens=8000,
        ),
        ExecutionResult(
            "Centralized MAS",
            success=True,
            latency_sec=14.2,
            total_tokens=9500,
        ),
    ]

    bad_metrics = evaluator.calculate_regret(bad_prediction, parallel_results)
    print(f"Oracle Determined Best Architecture: {bad_metrics['oracle_baseline']}")
    print(f"Router Selected: {bad_metrics['router_prediction']} (MISTAKE)")
    print(f"Latency Regret: {bad_metrics['latency_regret_sec']}s penalty paid by the user!")


if __name__ == "__main__":
    main()
