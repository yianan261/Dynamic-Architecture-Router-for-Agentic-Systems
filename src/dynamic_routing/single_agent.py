"""
Single-Agent System (SAS): Unified memory topology.

A single agent executes tasks sequentially using a rule-based ReAct-style loop
(reason node + tool node). Tool completion is tracked explicitly via executed_tools
state—not by scanning message strings, which would misclassify "I need to call X"
as X already executed.
"""

import logging

from langgraph.graph import END, StateGraph

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
from dynamic_routing.state import SingleAgentState


# Tool name constants (parsed from thought messages)
_TOOL_PREFIX = "call_"
_TOOLS = frozenset({
    "call_get_contact_preferences",
    "call_get_calendar_events",
    "call_search_drive_docs",
    "call_estimate_commute",
})

# Map required_tools registry names to internal tool names
_REQUIRED_TOOL_TO_CALL: dict[str, str] = {
    "get_contact_preferences": "call_get_contact_preferences",
    "get_calendar_events": "call_get_calendar_events",
    "search_drive_docs": "call_search_drive_docs",
    "estimate_commute": "call_estimate_commute",
}


def _parse_pending_tool(messages: list) -> str | None:
    """Extract the tool name from the last thought message."""
    if not messages:
        return None
    last = str(messages[-1])
    for tool in _TOOLS:
        if tool in last:
            return tool
    return None


# --- ReAct: Reasoning Node ---


def sas_reasoning_node(state: SingleAgentState) -> dict:
    """
    Decides: call another tool, or synthesize final answer.
    Uses executed_tools (not message-string scan) to avoid misclassifying
    "Thought: I need to execute X" as X already run.
    """
    task = state["task"]
    task_lower = task.lower()
    required_tools = state.get("required_tools") or []
    executed = set(state.get("executed_tools") or [])

    # When task registry provides required_tools, use data-driven dispatch
    if required_tools:
        for tool_name in required_tools:
            call_name = _REQUIRED_TOOL_TO_CALL.get(tool_name, f"call_{tool_name}")
            if call_name not in _TOOLS:
                logging.warning(
                    "SAS: skipping unsupported tool '%s' (mapped to '%s'); "
                    "not in _TOOLS. Check PCAB task config.",
                    tool_name,
                    call_name,
                )
                continue
            if call_name not in executed:
                return {
                    "messages": [f"Thought: I need to execute {call_name} next."],
                    "pending_tool": call_name,
                }
        return {
            "final_response": (
                "Single-Agent synthesis complete. Processed all steps sequentially."
            ),
        }

    # Fallback: keyword-based tool selection (also use executed_tools)
    has_contact = "call_get_contact_preferences" in executed
    has_calendar = "call_get_calendar_events" in executed
    has_drive = "call_search_drive_docs" in executed
    has_commute = "call_estimate_commute" in executed

    if not has_contact and (
        "advisor" in task_lower or "contact" in task_lower or "dr." in task_lower
    ):
        action = "call_get_contact_preferences"
    elif not has_calendar and (
        "calendar" in task_lower or "schedule" in task_lower or "event" in task_lower
        or "meeting" in task_lower or "meet" in task_lower or "lecture" in task_lower
    ):
        action = "call_get_calendar_events"
    elif not has_drive and (
        "drive" in task_lower or "notes" in task_lower or "doc" in task_lower or "capstone" in task_lower
    ):
        action = "call_search_drive_docs"
    elif not has_commute and (
        "commute" in task_lower or "walk" in task_lower or "travel" in task_lower or "maps" in task_lower or "location" in task_lower
    ):
        action = "call_estimate_commute"
    else:
        action = "FINISH"

    if action == "FINISH":
        return {
            "final_response": (
                "Single-Agent synthesis complete. Processed all steps sequentially."
            ),
        }

    return {
        "messages": [f"Thought: I need to execute {action} next."],
        "pending_tool": action,
    }


# --- ReAct: Tool Execution Node ---


def sas_tool_node(state: SingleAgentState) -> dict:
    """Executes the pending tool and appends the result to the unified memory."""
    task = state.get("task", "")
    pending = state.get("pending_tool", "")
    overrides = state.get("extraction_overrides") or {}

    if not pending or pending not in _TOOLS:
        # Fallback: try to parse from last message
        pending = _parse_pending_tool(state.get("messages", [])) or pending

    db = get_db_path()
    result: str

    if pending == "call_get_contact_preferences":
        name = extract_contact_name(task, overrides) or "Dr. Hong Man"
        result = get_contact_preferences(name, db)
    elif pending == "call_get_calendar_events":
        date = extract_date(task, overrides)
        result = get_calendar_events(date, db)
    elif pending == "call_search_drive_docs":
        query = extract_drive_query(task, overrides)
        result = search_drive_docs(query, db)
    elif pending == "call_estimate_commute":
        pair = extract_commute_pair(task, overrides)
        if pair:
            result = estimate_commute(pair[0], pair[1], db)
        else:
            result = '{"status": "skipped", "message": "No commute query detected."}'
    else:
        result = f'{{"error": "Unknown tool: {pending}"}}'

    return {
        "messages": [f"[{pending}] {result}"],
        "executed_tools": [pending],
        "pending_tool": "",
    }


# --- ReAct: Routing Edge ---


def sas_router(state: SingleAgentState):
    """If synthesis complete, end. Otherwise, execute the pending tool."""
    if state.get("final_response"):
        return END
    return "tools"


# --- Build the SAS Graph ---


def build_single_agent_graph() -> StateGraph:
    """Build and return the compiled Single-Agent System graph."""
    builder = StateGraph(SingleAgentState)

    builder.add_node("reason", sas_reasoning_node)
    builder.add_node("tools", sas_tool_node)

    builder.set_entry_point("reason")
    builder.add_conditional_edges("reason", sas_router)
    builder.add_edge("tools", "reason")

    return builder.compile()


single_agent_app = build_single_agent_graph()
