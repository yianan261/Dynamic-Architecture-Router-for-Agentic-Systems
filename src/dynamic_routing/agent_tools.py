"""
LangChain Tool Bindings for PCAB.

Converts PCAB functions into strict JSON schemas that Llama-3 (or any LLM
via vLLM) can read and execute dynamically via create_react_agent.

Telemetry: When using ChatOpenAI against vLLM, AIMessage.response_metadata
contains usage info (e.g. total_tokens) for accurate token tracking.
"""

from langchain_core.tools import tool

from dynamic_routing.pcab import (
    estimate_commute,
    get_calendar_events,
    get_contact_preferences,
    get_db_path,
    search_drive_docs,
)


@tool
def calendar_tool(date: str) -> str:
    """Fetch all calendar events for a specific date in YYYY-MM-DD format."""
    return get_calendar_events(date, get_db_path())


@tool
def drive_tool(query: str) -> str:
    """Search Drive documents for a keyword or phrase."""
    return search_drive_docs(query, get_db_path())


@tool
def commute_tool(origin: str, destination: str) -> str:
    """Get estimated travel time between two locations."""
    return estimate_commute(origin, destination, get_db_path())


@tool
def contact_tool(name: str) -> str:
    """Look up a contact's email and meeting preferences by name."""
    return get_contact_preferences(name, get_db_path())


PCAB_TOOLS = [calendar_tool, drive_tool, commute_tool, contact_tool]

# Map LLM tool call names back to the benchmark's internal tracking names
LLM_TOOL_TO_SAS_CALL = {
    "calendar_tool": "call_get_calendar_events",
    "drive_tool": "call_search_drive_docs",
    "commute_tool": "call_estimate_commute",
    "contact_tool": "call_get_contact_preferences",
}

LLM_TOOL_TO_CMAS_TAG = {
    "calendar_tool": "CALENDAR",
    "drive_tool": "DRIVE",
    "commute_tool": "COMMUTE",
    "contact_tool": "CONTACTS",
}
