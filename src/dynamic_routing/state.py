"""
State schemas for the Dynamic Architecture Router and sub-topologies.
"""

import operator
from typing import Annotated, TypedDict


class RouterState(TypedDict, total=False):
    """Global state passed through the main routing graph."""

    user_query: str
    estimated_sequential_depth: int
    parallelization_factor: float  # Scale of 0.0 to 1.0
    estimated_tool_count: int
    selected_architecture: str
    final_response: str
    extraction_overrides: dict
    required_tools: list
    total_tokens: int


class CentralizedState(TypedDict, total=False):
    """State for the Centralized MAS (hub-and-spoke) workflow."""

    task: str
    aggregated_context: Annotated[list, operator.add]
    next_action: str
    final_synthesis: str
    extraction_overrides: dict
    required_tools: list
    total_tokens: Annotated[int, operator.add]


class SingleAgentState(TypedDict, total=False):
    """State for the Single-Agent System (unified memory, ReAct loop)."""

    task: str
    messages: Annotated[list, operator.add]
    executed_tools: Annotated[list[str], operator.add]
    pending_tool: str
    final_response: str
    extraction_overrides: dict
    required_tools: list
    total_tokens: Annotated[int, operator.add]
