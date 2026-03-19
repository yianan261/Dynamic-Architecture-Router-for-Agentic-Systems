"""
LangChain Tool Bindings for PCAB.

Converts PCAB functions into strict JSON schemas that Llama-3 (or any LLM
via vLLM) can read and execute dynamically. Wire these into single_agent and
centralized_mas when replacing rule-based dispatching with LLM tool calls.

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
def get_calendar_events_tool(date: str) -> str:
    """Fetch all calendar events for a specific date in YYYY-MM-DD format."""
    return get_calendar_events(date, get_db_path())


@tool
def search_drive_docs_tool(query: str) -> str:
    """Search Drive documents for a keyword or phrase."""
    return search_drive_docs(query, get_db_path())


@tool
def estimate_commute_tool(origin: str, destination: str) -> str:
    """Get estimated travel time between two locations."""
    return estimate_commute(origin, destination, get_db_path())


@tool
def get_contact_preferences_tool(name: str) -> str:
    """Look up a contact's email and meeting preferences by name."""
    return get_contact_preferences(name, get_db_path())


# Bundle for LLM binding (e.g. llm.bind_tools([...]))
PCAB_TOOLS = [
    get_calendar_events_tool,
    search_drive_docs_tool,
    estimate_commute_tool,
    get_contact_preferences_tool,
]
