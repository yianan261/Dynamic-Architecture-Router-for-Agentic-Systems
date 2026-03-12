"""
Dynamic Architecture Router: Meta-agent that classifies tasks and routes
to the optimal topology (SAS, Centralized MAS, or Decentralized MAS).
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from dynamic_routing.centralized_mas import centralized_mas_app
from dynamic_routing.state import RouterState


# --- Dynamic Router Node (Meta-Agent) ---


def dynamic_router_node(state: RouterState) -> dict:
    """
    Analyzes the prompt and outputs structural metadata.
    In production: Mistral-7B classifier enforcing JSON output.
    """
    query = state.get("user_query", "").lower()

    # Multiple data sources (Calendar, Maps, Drive) → parallel aggregation
    sources = sum(1 for s in ("calendar", "maps", "drive") if s in query)

    if "simultaneously" in query or "aggregate" in query or sources >= 2:
        # e.g., Fetching Drive, Calendar, and Maps data at the same time
        return {
            "estimated_sequential_depth": 2,
            "parallelization_factor": 0.9,
            "estimated_tool_count": 5,
        }
    elif "step-by-step" in query or "complex workflow" in query:
        # e.g., Deep reasoning task requiring heavy tool use
        return {
            "estimated_sequential_depth": 8,
            "parallelization_factor": 0.1,
            "estimated_tool_count": 14,
        }
    else:
        # Open-ended exploration
        return {
            "estimated_sequential_depth": 3,
            "parallelization_factor": 0.4,
            "estimated_tool_count": 3,
        }


# --- Routing Logic (Conditional Edge) ---


def route_task(
    state: RouterState,
) -> Literal["single_agent_node", "centralized_mas_node", "decentralized_mas_node"]:
    """
    Applies quantitative scaling thresholds from the literature:
    - Tool-coordination trade-off (β=-0.267): tool-heavy → SAS
    - Parallelization benefit: decomposable tasks → Centralized MAS
    - Open-ended consensus: exploratory tasks → Decentralized MAS
    """
    tools = state.get("estimated_tool_count", 0)
    depth = state.get("estimated_sequential_depth", 0)
    parallelism = state.get("parallelization_factor", 0.0)

    # Threshold A: Tool-heavy or deep sequential constraints → SAS
    if tools >= 12 or depth > 5:
        return "single_agent_node"

    # Threshold B: High parallelization, moderate tools → Centralized MAS
    if parallelism > 0.6 and tools < 12:
        return "centralized_mas_node"

    # Threshold C: Default → Decentralized MAS for peer debate
    return "decentralized_mas_node"


# --- Architecture Execution Nodes ---


def single_agent_node(state: RouterState) -> dict:
    """Single-Agent System: unified memory, zero inter-agent overhead."""
    return {
        "selected_architecture": "Single-Agent System",
        "final_response": "Executed sequentially to save token overhead.",
    }


def centralized_mas_node(state: RouterState) -> dict:
    """Centralized MAS: invokes the hub-and-spoke subgraph."""
    task = state.get("user_query", "")

    mas_result = centralized_mas_app.invoke(
        {
            "task": task,
            "aggregated_context": [],
            "next_action": "",
            "final_synthesis": "",
        }
    )

    return {
        "selected_architecture": "Centralized MAS",
        "final_response": mas_result.get("final_synthesis", "Executed in parallel via supervisor orchestration."),
    }


def decentralized_mas_node(state: RouterState) -> dict:
    """Decentralized MAS: peer-to-peer consensus (placeholder)."""
    return {
        "selected_architecture": "Decentralized MAS",
        "final_response": "Executed via peer-to-peer consensus.",
    }


# --- Build the Main Router Graph ---


def build_router_graph() -> StateGraph:
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
