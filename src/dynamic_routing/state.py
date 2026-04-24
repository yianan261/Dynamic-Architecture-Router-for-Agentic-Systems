"""
State schemas for the Dynamic Architecture Router and sub-topologies.
"""

import operator
from typing import Annotated, TypedDict
from typing_extensions import NotRequired


class RouterState(TypedDict, total=False):
    """Global state passed through the main routing graph."""

    user_query: str
    estimated_sequential_depth: int
    parallelization_factor: float  # Scale of 0.0 to 1.0
    estimated_tool_count: int
    # Optional structured features for learned routing (see vllm_integration.RoutingMetadata)
    num_subgoals: NotRequired[int]
    entity_count: NotRequired[int]
    constraint_tightness: NotRequired[int]
    open_endedness: NotRequired[int]
    aggregation_required: NotRequired[bool]
    expected_retrieval_fanout: NotRequired[int]
    domain_span: NotRequired[int]
    expected_context_expansion: NotRequired[int]
    final_synthesis_complexity: NotRequired[int]
    cross_branch_dependency: NotRequired[int]
    communication_load_estimate: NotRequired[int]
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
    execution_path: Annotated[list[str], operator.add]
    failure_taxonomy: str


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
    execution_path: Annotated[list[str], operator.add]
    failure_taxonomy: str
