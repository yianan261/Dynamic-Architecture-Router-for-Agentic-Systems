"""Tests for the Dynamic Architecture Router."""

import sys
from pathlib import Path

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


def test_centralized_mas_aggregates_sandbox_sources() -> None:
    """Centralized MAS should aggregate Calendar, Maps, Drive when task mentions them."""
    result = app.invoke({
        "user_query": "Cross-reference my calendar with maps for hiking and check my drive notes."
    })
    assert result["selected_architecture"] == "Centralized MAS"
    assert "Synthesis Complete" in result["final_response"]
    assert "3 sources" in result["final_response"]
