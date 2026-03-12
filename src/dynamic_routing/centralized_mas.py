"""
Centralized Multi-Agent System (MAS): Hub-and-spoke topology.

The Supervisor orchestrates worker agents (Calendar, Maps, Drive). Workers
cannot communicate directly—they report to the Supervisor, which acts as
a validation bottleneck to restrict error amplification (~4.4x).
"""

from langgraph.graph import END, StateGraph

from dynamic_routing.state import CentralizedState


# --- Mock APIs for the Personalized Context Sandbox ---


def fetch_calendar_data(query: str) -> str:
    """Mock API to simulate fetching user calendar events."""
    return (
        "[Calendar API] 10:00 AM: Meeting with Data Eng Team. "
        "3:00 PM: Flight to Albany, NY."
    )


def fetch_maps_data(query: str) -> str:
    """Mock API to simulate fetching saved locations."""
    return (
        "[Maps API] Saved Locations in Upstate NY: "
        "Sleepy Hollow Cemetery, Cold Spring hiking trails."
    )


def fetch_drive_data(query: str) -> str:
    """Mock API to simulate fetching Drive documents."""
    return (
        "[Drive API] Doc 'Fall Trip Notes': "
        "'Looking for cute towns with easy hiking routes and charming main streets.'"
    )


# --- Worker Agents ---


def calendar_agent(state: CentralizedState) -> dict:
    """Worker: fetches calendar data."""
    data = fetch_calendar_data(state["task"])
    return {"aggregated_context": [data]}


def maps_agent(state: CentralizedState) -> dict:
    """Worker: fetches maps data."""
    data = fetch_maps_data(state["task"])
    return {"aggregated_context": [data]}


def drive_agent(state: CentralizedState) -> dict:
    """Worker: fetches drive data."""
    data = fetch_drive_data(state["task"])
    return {"aggregated_context": [data]}


# --- Supervisor Orchestrator ---


def supervisor_node(state: CentralizedState) -> dict:
    """
    Supervisor analyzes aggregated context and decides which worker
    to dispatch next, or if synthesis is complete.
    """
    context = state.get("aggregated_context", [])
    task = state["task"].lower()
    context_str = str(context).lower()

    # In production, this logic is handled by prompting Llama-3-8B.
    if "calendar api" not in context_str and "calendar" in task:
        next_worker = "calendar_agent"
    elif "maps api" not in context_str and "maps" in task:
        next_worker = "maps_agent"
    elif "drive api" not in context_str and "drive" in task:
        next_worker = "drive_agent"
    else:
        next_worker = "FINISH"

    if next_worker == "FINISH":
        synthesis = (
            f"Synthesis Complete. Aggregated {len(context)} sources "
            "to build personalized context."
        )
        return {"next_action": next_worker, "final_synthesis": synthesis}

    return {"next_action": next_worker}


def supervisor_router(state: CentralizedState):
    """Routes the graph based on supervisor decision."""
    if state.get("next_action") == "FINISH":
        return END
    return state["next_action"]


# --- Build the Centralized Graph ---


def build_centralized_mas_graph() -> StateGraph:
    """Build and return the compiled Centralized MAS graph."""
    builder = StateGraph(CentralizedState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("calendar_agent", calendar_agent)
    builder.add_node("maps_agent", maps_agent)
    builder.add_node("drive_agent", drive_agent)

    builder.set_entry_point("supervisor")

    # Hub-and-spoke: workers always return to Supervisor
    builder.add_edge("calendar_agent", "supervisor")
    builder.add_edge("maps_agent", "supervisor")
    builder.add_edge("drive_agent", "supervisor")

    builder.add_conditional_edges("supervisor", supervisor_router)

    return builder.compile()


# Pre-compiled app for use by the main router
centralized_mas_app = build_centralized_mas_graph()
