"""
Dynamic Architecture Router: Meta-agent that classifies tasks and routes
to the optimal topology (SAS, Centralized MAS, or Decentralized MAS).
"""

from typing import Literal, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dynamic_routing.centralized_mas import centralized_mas_app
from dynamic_routing.single_agent import single_agent_app
from dynamic_routing.state import CentralizedState, RouterState, SingleAgentState
from dynamic_routing.vllm_integration import predict_routing_metadata


# --- Dynamic Router Node (Meta-Agent) ---


def dynamic_router_node(state: RouterState) -> dict:
    """
    Analyzes the prompt and outputs structural metadata using the Mistral-7B
    classifier via vLLM. Falls back to keyword heuristics if vLLM is unavailable.
    """
    query = state.get("user_query", "")
    return predict_routing_metadata(query)


# --- Routing Logic (Conditional Edge) ---

def route_task(
    state: RouterState,
) -> Literal["single_agent_node", "centralized_mas_node", "decentralized_mas_node"]:
    """
    Applies quantitative scaling thresholds from the literature:
    - Tool-coordination trade-off (β=-0.267): tool-heavy → SAS
    - Parallelization benefit: decomposable tasks → Centralized MAS
    - Exploratory / open-ended tasks → Decentralized MAS (placeholder today)

    Note: Decentralized MAS is an unimplemented placeholder, so the default
    must NOT be DMAS (it would guarantee 0 accuracy). Default is SAS, which
    is also the Science-of-Scaling matched-compute baseline.
    """
    tools = state.get("estimated_tool_count", 0)
    depth = state.get("estimated_sequential_depth", 0)
    parallelism = state.get("parallelization_factor", 0.0)

    if tools >= 12 or depth > 5:
        return "single_agent_node"

    if parallelism > 0.6 and tools < 12:
        return "centralized_mas_node"

    if parallelism > 0.85:
        return "decentralized_mas_node"

    return "single_agent_node"


_ROUTE_TO_DISPLAY: dict[str, str] = {
    "single_agent_node": "Single-Agent System",
    "centralized_mas_node": "Centralized MAS",
    "decentralized_mas_node": "Decentralized MAS",
}


def predict_routed_architecture(user_query: str) -> str:
    """
    Routing-only: metadata + thresholds → display name (does not execute SAS/CMAS).
    Used to populate router_prediction in benchmark JSON.
    """
    base: RouterState = {"user_query": user_query}
    meta = dynamic_router_node(base)
    state = cast(RouterState, {**base, **meta})
    dest = route_task(state)
    return _ROUTE_TO_DISPLAY[dest]


# --- Architecture Execution Nodes ---


def single_agent_node(state: RouterState) -> dict:
    """Single-Agent System: invokes the ReAct-loop subgraph (unified memory)."""
    task = state.get("user_query", "")
    extras = {}
    extraction_overrides = state.get("extraction_overrides")
    if extraction_overrides:
        extras["extraction_overrides"] = extraction_overrides
    required_tools = state.get("required_tools")
    if required_tools:
        extras["required_tools"] = required_tools

    sas_result = single_agent_app.invoke(
        cast(
            SingleAgentState,
            {
                "task": task,
                "messages": [],
                "executed_tools": [],
                "pending_tool": "",
                "final_response": "",
                "total_tokens": 0,
                **extras,
            },
        )
    )

    return {
        "selected_architecture": "Single-Agent System",
        "final_response": sas_result.get("final_response", "Single-Agent execution complete."),
        "total_tokens": sas_result.get("total_tokens", 0),
    }


def centralized_mas_node(state: RouterState) -> dict:
    """Centralized MAS: invokes the hub-and-spoke subgraph."""
    task = state.get("user_query", "")
    extras = {}
    extraction_overrides = state.get("extraction_overrides")
    if extraction_overrides:
        extras["extraction_overrides"] = extraction_overrides
    required_tools = state.get("required_tools")
    if required_tools:
        extras["required_tools"] = required_tools

    mas_result = centralized_mas_app.invoke(
        cast(
            CentralizedState,
            {
                "task": task,
                "aggregated_context": [],
                "next_action": "",
                "final_synthesis": "",
                "total_tokens": 0,
                **extras,
            },
        )
    )

    return {
        "selected_architecture": "Centralized MAS",
        "final_response": mas_result.get("final_synthesis", "Executed in parallel via supervisor orchestration."),
        "total_tokens": mas_result.get("total_tokens", 0),
    }


def decentralized_mas_node(state: RouterState) -> dict:
    """Decentralized MAS: peer-to-peer consensus (placeholder)."""
    return {
        "selected_architecture": "Decentralized MAS",
        "final_response": "Executed via peer-to-peer consensus.",
    }


# --- Build the Main Router Graph ---


def build_router_graph() -> CompiledStateGraph[
    RouterState, None, RouterState, RouterState
]:
    """Build and compile the Dynamic Router graph."""
    workflow = StateGraph(RouterState)

    workflow.add_node("router", dynamic_router_node)
    workflow.add_node("single_agent_node", single_agent_node)
    workflow.add_node("centralized_mas_node", centralized_mas_node)
    workflow.add_node("decentralized_mas_node", decentralized_mas_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_task)

    workflow.add_edge("single_agent_node", END)
    workflow.add_edge("centralized_mas_node", END)
    workflow.add_edge("decentralized_mas_node", END)

    return workflow.compile()


app = build_router_graph()
