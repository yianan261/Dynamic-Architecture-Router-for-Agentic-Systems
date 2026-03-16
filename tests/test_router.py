"""Tests for the Dynamic Architecture Router."""

import os
import re
import sys
from pathlib import Path

# Use keyword fallback for fast, deterministic tests (no vLLM required)
os.environ.setdefault("USE_LLM_ROUTER", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_routing.router import app


def test_routes_tool_heavy_to_sas() -> None:
    """Tool-heavy, sequential tasks should route to Single-Agent System."""
    result = app.invoke({
        "user_query": "Execute this complex workflow step-by-step using all available tools."
    })
    assert result["selected_architecture"] == "Single-Agent System"


def test_routes_parallel_aggregation_to_centralized_mas() -> None:
    """High-parallelization aggregation should route to Centralized MAS."""
    result = app.invoke({
        "user_query": "Please aggregate my Calendar and Maps data simultaneously."
    })
    assert result["selected_architecture"] == "Centralized MAS"


def test_routes_open_ended_to_decentralized_mas() -> None:
    """Open-ended exploratory tasks should route to Decentralized MAS."""
    result = app.invoke({
        "user_query": "What are the pros and cons of different approaches?"
    })
    assert result["selected_architecture"] == "Decentralized MAS"


def test_centralized_mas_aggregates_pcab_sources() -> None:
    """Centralized MAS should aggregate PCAB sources (calendar, drive, commute) when task mentions them."""
    result = app.invoke({
        "user_query": "Cross-reference my calendar with maps for hiking and check my drive notes."
    })
    assert result["selected_architecture"] == "Centralized MAS"
    assert "Synthesis Complete" in result["final_response"]
    # Robust: assert we aggregated multiple sources (not brittle string match)
    match = re.search(r"Aggregated (\d+) sources", result["final_response"])
    assert match is not None
    assert int(match.group(1)) >= 2


def test_pcab_task_with_extraction_overrides() -> None:
    """When PCAB task provides extraction_overrides and required_tools, agents use them (no hard-coded keywords)."""
    from dynamic_routing.pcab_tasks import get_pcab_task

    task = get_pcab_task("PCAB-Par-01")
    assert task is not None
    overrides = task.extraction_params.as_override_dict()

    result = app.invoke({
        "user_query": task.description,
        "extraction_overrides": overrides,
        "required_tools": task.required_tools,
    })
    assert result["selected_architecture"] == "Centralized MAS"
    assert "Synthesis Complete" in result["final_response"]
    # Should have aggregated 3 sources (calendar, drive, commute) per required_tools
    match = re.search(r"Aggregated (\d+) sources", result["final_response"])
    assert match is not None
    assert int(match.group(1)) >= 3
