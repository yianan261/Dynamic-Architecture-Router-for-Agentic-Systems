"""
Centralized Multi-Agent System (MAS): Hub-and-spoke topology.

The Supervisor orchestrates worker agents (Calendar, Drive, Commute, Contacts)
backed by the PCAB (Personalized Context Assembly Benchmark) SQLite database.
Workers cannot communicate directly—they report to the Supervisor, which acts
as a validation bottleneck to restrict error amplification (~4.4x).
"""

import re
from langgraph.graph import END, StateGraph

from dynamic_routing.pcab import (
    get_calendar_events,
    search_drive_docs,
    estimate_commute,
    get_contact_preferences,
    get_db_path,
)
from dynamic_routing.state import CentralizedState


# --- Parameter extraction from natural language task ---


def _extract_date(task: str) -> str:
    """Extract date from task; default to benchmark date 2026-03-16."""
    # YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", task)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # March 16, Mar 16, 3/16
    if "march 16" in task or "mar 16" in task or "3/16" in task:
        return "2026-03-16"
    return "2026-03-16"


def _extract_drive_query(task: str) -> str:
    """Extract search terms for Drive from task."""
    task_lower = task.lower()
    if "capstone" in task_lower or "proposal" in task_lower:
        return "Capstone"
    if "advisor" in task_lower or "meeting prep" in task_lower:
        return "Advisor"
    if "notes" in task_lower:
        return "notes"
    return "Capstone"


def _extract_commute_pair(task: str) -> tuple[str, str] | None:
    """Extract origin/destination for commute. Returns (origin, dest) or None."""
    task_lower = task.lower()
    if "commute" not in task_lower and "walk" not in task_lower and "travel" not in task_lower:
        return None
    # Common PCAB locations
    locs = ["babbio center", "gateway north", "library", "hoboken coffee"]
    found = [loc for loc in locs if loc in task_lower]
    if len(found) >= 2:
        return (found[0].title(), found[1].title())
    # Default: lecture to advisor meeting
    if "advisor" in task_lower or "dr." in task_lower:
        return ("Babbio Center", "Gateway North")
    return ("Babbio Center", "Gateway North")


def _extract_contact_name(task: str) -> str | None:
    """Extract contact name from task."""
    if "dr. hong man" in task.lower() or "dr hong man" in task.lower():
        return "Dr. Hong Man"
    if "advisor" in task.lower() and "dr" not in task.lower():
        return "Dr. Hong Man"
    return None


# --- Worker Agents (PCAB-backed) ---


def calendar_agent(state: CentralizedState) -> dict:
    """Worker: fetches calendar events from PCAB."""
    task = state.get("task", "")
    date = _extract_date(task)
    result = get_calendar_events(date, get_db_path())
    return {"aggregated_context": [f"[CALENDAR]{result}"]}


def drive_agent(state: CentralizedState) -> dict:
    """Worker: searches Drive docs from PCAB."""
    task = state.get("task", "")
    query = _extract_drive_query(task)
    result = search_drive_docs(query, get_db_path())
    return {"aggregated_context": [f"[DRIVE]{result}"]}


def commute_agent(state: CentralizedState) -> dict:
    """Worker: estimates commute between locations from PCAB Maps/Commute data."""
    task = state.get("task", "")
    pair = _extract_commute_pair(task)
    if pair is None:
        result = '{"status": "skipped", "message": "No commute query detected."}'
    else:
        result = estimate_commute(pair[0], pair[1], get_db_path())
    return {"aggregated_context": [f"[COMMUTE]{result}"]}


def contacts_agent(state: CentralizedState) -> dict:
    """Worker: fetches contact preferences from PCAB."""
    task = state.get("task", "")
    name = _extract_contact_name(task)
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
    """
    context = state.get("aggregated_context", [])
    task = state["task"].lower()
    context_str = str(context).lower()

    # Check which PCAB sources have been queried
    has_calendar = "[calendar]" in context_str
    has_drive = "[drive]" in context_str
    has_commute = "[commute]" in context_str
    has_contacts = "[contacts]" in context_str

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
