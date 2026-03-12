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


class CentralizedState(TypedDict, total=False):
    """State for the Centralized MAS (hub-and-spoke) workflow."""

    task: str
    aggregated_context: Annotated[list, operator.add]
    next_action: str
    final_synthesis: str
