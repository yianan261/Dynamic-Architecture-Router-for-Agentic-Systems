"""
Dynamic Architecture Router: Meta-agent that classifies tasks and routes
to the optimal topology (SAS, Centralized MAS, or Decentralized MAS).
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from dynamic_routing.centralized_mas import centralized_mas_app
from dynamic_routing.single_agent import single_agent_app
from dynamic_routing.state import RouterState
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
    """Single-Agent System: invokes the ReAct-loop subgraph (unified memory)."""
    task = state.get("user_query", "")
    extras = {}
    if state.get("extraction_overrides"):
        extras["extraction_overrides"] = state["extraction_overrides"]
    if state.get("required_tools"):
        extras["required_tools"] = state["required_tools"]

    sas_result = single_agent_app.invoke({
        "task": task,
        "messages": [],
        "pending_tool": "",
        "final_response": "",
        **extras,
    })

    return {
        "selected_architecture": "Single-Agent System",
        "final_response": sas_result.get("final_response", "Single-Agent execution complete."),
    }


def centralized_mas_node(state: RouterState) -> dict:
    """Centralized MAS: invokes the hub-and-spoke subgraph."""
    task = state.get("user_query", "")
    extras = {}
    if state.get("extraction_overrides"):
        extras["extraction_overrides"] = state["extraction_overrides"]
    if state.get("required_tools"):
        extras["required_tools"] = state["required_tools"]

    mas_result = centralized_mas_app.invoke({
        "task": task,
        "aggregated_context": [],
        "next_action": "",
        "final_synthesis": "",
        **extras,
    })

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
