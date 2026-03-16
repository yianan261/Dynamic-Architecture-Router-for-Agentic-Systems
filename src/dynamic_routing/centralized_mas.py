"""
Centralized Multi-Agent System (MAS): Hub-and-spoke topology.

The Supervisor orchestrates worker agents (Calendar, Drive, Commute, Contacts)
backed by the PCAB (Personalized Context Assembly Benchmark) SQLite database.
Workers cannot communicate directly—they report to the Supervisor, which acts
as a validation bottleneck to restrict error amplification (~4.4x).
"""

from langgraph.graph import END, StateGraph

# Map required_tools registry names to worker agent names
_REQUIRED_TOOL_TO_WORKER: dict[str, str] = {
    "get_contact_preferences": "contacts_agent",
    "get_calendar_events": "calendar_agent",
    "search_drive_docs": "drive_agent",
    "estimate_commute": "commute_agent",
}

from dynamic_routing.pcab import (
    estimate_commute,
    extract_commute_pair,
    extract_contact_name,
    extract_date,
    extract_drive_query,
    get_calendar_events,
    get_contact_preferences,
    get_db_path,
    search_drive_docs,
)
from dynamic_routing.state import CentralizedState


# --- Worker Agents (PCAB-backed) ---


def calendar_agent(state: CentralizedState) -> dict:
    """Worker: fetches calendar events from PCAB."""
    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    date = extract_date(task, overrides)
    result = get_calendar_events(date, get_db_path())
    return {"aggregated_context": [f"[CALENDAR]{result}"]}


def drive_agent(state: CentralizedState) -> dict:
    """Worker: searches Drive docs from PCAB."""
    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    query = extract_drive_query(task, overrides)
    result = search_drive_docs(query, get_db_path())
    return {"aggregated_context": [f"[DRIVE]{result}"]}


def commute_agent(state: CentralizedState) -> dict:
    """Worker: estimates commute between locations from PCAB Maps/Commute data."""
    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    pair = extract_commute_pair(task, overrides)
    if pair is None:
        result = '{"status": "skipped", "message": "No commute query detected."}'
    else:
        result = estimate_commute(pair[0], pair[1], get_db_path())
    return {"aggregated_context": [f"[COMMUTE]{result}"]}


def contacts_agent(state: CentralizedState) -> dict:
    """Worker: fetches contact preferences from PCAB."""
    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    name = extract_contact_name(task, overrides)
    if name is None:
        result = '{"status": "skipped", "message": "No contact query detected."}'
    else:
        result = get_contact_preferences(name, get_db_path())
    return {"aggregated_context": [f"[CONTACTS]{result}"]}


# --- Supervisor Orchestrator ---


def supervisor_node(state: CentralizedState) -> dict:
    """
    Supervisor analyzes aggregated context and decides which worker
    to dispatch next, or if synthesis is complete.
    When required_tools is provided (e.g. from PCAB task registry), uses that
    instead of hard-coded keyword matching.
    """
    context = state.get("aggregated_context", [])
    task = state["task"].lower()
    context_str = str(context).lower()
    required_tools = state.get("required_tools") or []

    # Check which PCAB sources have been queried
    has_calendar = "[calendar]" in context_str
    has_drive = "[drive]" in context_str
    has_commute = "[commute]" in context_str
    has_contacts = "[contacts]" in context_str

    # When task registry provides required_tools, use data-driven dispatch
    if required_tools:
        for tool_name in required_tools:
            worker = _REQUIRED_TOOL_TO_WORKER.get(tool_name)
            if not worker:
                continue
            # Workers emit e.g. "[CALENDAR]{result}"; context_str is lowercased
            context_key = f"[{worker.replace('_agent', '').lower()}]"
            if context_key not in context_str:
                return {"next_action": worker}
        synthesis = (
            f"Synthesis Complete. Aggregated {len(context)} sources "
            "to build personalized context."
        )
        return {"next_action": "FINISH", "final_synthesis": synthesis}

    # Fallback: keyword-based worker selection
    if not has_calendar and ("calendar" in task or "schedule" in task or "event" in task or "meeting" in task):
        next_worker = "calendar_agent"
    elif not has_drive and ("drive" in task or "notes" in task or "doc" in task or "capstone" in task):
        next_worker = "drive_agent"
    elif not has_commute and ("commute" in task or "walk" in task or "travel" in task or "maps" in task or "location" in task):
        next_worker = "commute_agent"
    elif not has_contacts and ("advisor" in task or "contact" in task or "dr." in task or "dr " in task):
        next_worker = "contacts_agent"
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
    builder.add_node("drive_agent", drive_agent)
    builder.add_node("commute_agent", commute_agent)
    builder.add_node("contacts_agent", contacts_agent)

    builder.set_entry_point("supervisor")

    builder.add_edge("calendar_agent", "supervisor")
    builder.add_edge("drive_agent", "supervisor")
    builder.add_edge("commute_agent", "supervisor")
    builder.add_edge("contacts_agent", "supervisor")

    builder.add_conditional_edges("supervisor", supervisor_router)

    return builder.compile()


centralized_mas_app = build_centralized_mas_graph()
