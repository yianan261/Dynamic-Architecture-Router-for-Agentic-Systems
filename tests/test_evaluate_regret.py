"""Tests for the Oracle Evaluation Harness (v2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_regret import ExecutionResult, RegretEvaluator, compute_trajectory_accuracy


def test_regret_evaluator_oracle_selects_highest_accuracy() -> None:
    """Oracle should prefer high-accuracy run over fast/cheap but low-accuracy run."""
    evaluator = RegretEvaluator(accuracy_weight=0.6, latency_weight=0.25, token_weight=0.15)

    # Use scores where Decentralized (1.0 acc) clearly beats Centralized (0.5 acc)
    # when accuracy is weighted heavily
    results = [
        ExecutionResult("Centralized MAS", accuracy_score=0.5, latency_sec=18.0, total_tokens=10000),
        ExecutionResult("Decentralized MAS", accuracy_score=1.0, latency_sec=30.0, total_tokens=20000),
    ]

    oracle = evaluator.determine_oracle(results)
    assert oracle.architecture == "Decentralized MAS"
    assert oracle.accuracy_score == 1.0


def test_regret_evaluator_accuracy_regret() -> None:
    """When router picks lower-accuracy arch, accuracy_regret > 0."""
    evaluator = RegretEvaluator(accuracy_weight=0.6, latency_weight=0.25, token_weight=0.15)

    results = [
        ExecutionResult("Centralized MAS", accuracy_score=0.5, latency_sec=18.0, total_tokens=10000),
        ExecutionResult("Decentralized MAS", accuracy_score=1.0, latency_sec=30.0, total_tokens=20000),
    ]

    metrics = evaluator.calculate_regret("Centralized MAS", results)
    assert metrics["oracle_baseline"] == "Decentralized MAS"
    assert metrics["accuracy_regret"] == 0.5
    assert metrics["perfect_routing"] is False


def test_trajectory_accuracy_full_match() -> None:
    """All required tools executed -> 1.0 accuracy."""
    acc = compute_trajectory_accuracy(
        required_tools=["get_contact_preferences", "get_calendar_events", "estimate_commute"],
        executed_tools=["get_contact_preferences", "get_calendar_events", "estimate_commute"],
    )
    assert acc == 1.0


def test_trajectory_accuracy_partial() -> None:
    """2 of 3 tools -> 0.66... accuracy."""
    acc = compute_trajectory_accuracy(
        required_tools=["get_contact_preferences", "get_calendar_events", "estimate_commute"],
        executed_tools=["get_contact_preferences", "get_calendar_events"],
    )
    assert abs(acc - 0.666) < 0.01


def test_trajectory_accuracy_empty_required() -> None:
    """No required tools -> 1.0 (vacuous success)."""
    acc = compute_trajectory_accuracy(required_tools=[], executed_tools=[])
    assert acc == 1.0
