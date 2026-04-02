"""
Single-Agent System (SAS): Unified memory topology.

Two modes controlled by USE_LLM_WORKERS env var:
  - LLM mode: Llama-3.1-8B via vLLM + create_react_agent (full ReAct loop)
  - Rule-based mode (default): deterministic tool dispatch for CI/benchmarking
"""

import logging
import os

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dynamic_routing.state import SingleAgentState
from pydantic import SecretStr

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TOOLS = frozenset({
    "call_get_contact_preferences",
    "call_get_calendar_events",
    "call_search_drive_docs",
    "call_estimate_commute",
})

_REQUIRED_TOOL_TO_CALL: dict[str, str] = {
    "get_contact_preferences": "call_get_contact_preferences",
    "get_calendar_events": "call_get_calendar_events",
    "search_drive_docs": "call_search_drive_docs",
    "estimate_commute": "call_estimate_commute",
}

_USE_LLM = os.environ.get("USE_LLM_WORKERS", "false").lower() in ("true", "1", "yes")

MAX_REACT_ITERATIONS = 15

# ===========================================================================
# LLM Mode — Llama-3.1 via vLLM with create_react_agent
# ===========================================================================


def _build_llm_sas_graph() -> CompiledStateGraph[
    SingleAgentState, None, SingleAgentState, SingleAgentState
]:
    """Build SAS graph powered by Llama-3.1 ReAct agent."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from dynamic_routing.agent_tools import LLM_TOOL_TO_SAS_CALL, PCAB_TOOLS

    worker_llm = ChatOpenAI(
        model=os.environ.get("VLLM_WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "EMPTY")),
        base_url=os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1"),
        temperature=0.1,
    ).bind_tools(PCAB_TOOLS, parallel_tool_calls=False)

    react_agent = create_react_agent(
        worker_llm,
        PCAB_TOOLS,
        prompt="You are an efficient personal assistant. Use your tools to fetch data and solve the user's task.",
    )

    def sas_llm_node(state: SingleAgentState) -> dict:
        try:
            result = react_agent.invoke(
                {"messages": [("user", state.get("task", ""))]},
                config={"recursion_limit": MAX_REACT_ITERATIONS},
            )
        except Exception as e:
            err = str(e).lower()
            if "recursion" in err:
                taxonomy = "System Design Issue: ReAct Loop Exhaustion"
            elif "context length" in err or "input_tokens" in err:
                taxonomy = "System Design Issue: Context Window Overflow"
            else:
                taxonomy = f"Task Verification Failure: {str(e)[:80]}"
            logging.warning("SAS LLM circuit breaker: %s", taxonomy)
            return {
                "final_response": f"HALTED: {taxonomy}",
                "total_tokens": 0,
                "executed_tools": [],
                "failure_taxonomy": taxonomy,
            }

        tokens = 0
        executed: list[str] = []
        path: list[str] = []
        tool_signatures: list[str] = []
        for msg in result.get("messages", []):
            if hasattr(msg, "response_metadata"):
                usage = msg.response_metadata.get("token_usage") or {}
                tokens += usage.get("total_tokens", 0)
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    tool_name = str(call.get("name") or "")
                    mapped = LLM_TOOL_TO_SAS_CALL.get(tool_name, tool_name)
                    executed.append(mapped)
                    path.append(tool_name)
                    sig = f"{mapped}({call.get('args', {})})"
                    tool_signatures.append(sig)

        taxonomy = None
        if len(tool_signatures) >= 2 and tool_signatures[-1] == tool_signatures[-2]:
            taxonomy = f"System Design Issue: Tool Explosion ({tool_signatures[-1]})"
            logging.warning("SAS parameter hash: %s", taxonomy)

        final = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                final = msg.content
                break

        out: dict = {
            "final_response": final or "Single-Agent execution complete.",
            "total_tokens": tokens,
            "executed_tools": executed,
            "execution_path": path,
        }
        if taxonomy:
            out["failure_taxonomy"] = taxonomy
        return out

    builder = StateGraph(SingleAgentState)
    builder.add_node("sas", sas_llm_node)
    builder.set_entry_point("sas")
    builder.add_edge("sas", END)
    return builder.compile()


# ===========================================================================
# Rule-Based Mode — deterministic dispatch (no LLM required)
# ===========================================================================


def _parse_pending_tool(messages: list) -> str | None:
    if not messages:
        return None
    last = str(messages[-1])
    for tool in _TOOLS:
        if tool in last:
            return tool
    return None


def _sas_reasoning_node(state: SingleAgentState) -> dict:
    from dynamic_routing.pcab import (
        extract_commute_pair,
        extract_contact_name,
        extract_date,
        extract_drive_query,
    )

    task = state.get("task", "")
    task_lower = task.lower()
    required_tools = state.get("required_tools") or []
    executed = set(state.get("executed_tools") or [])

    if required_tools:
        for tool_name in required_tools:
            call_name = _REQUIRED_TOOL_TO_CALL.get(tool_name, f"call_{tool_name}")
            if call_name not in _TOOLS:
                logging.warning(
                    "SAS: skipping unsupported tool '%s' (mapped to '%s')",
                    tool_name, call_name,
                )
                continue
            if call_name not in executed:
                return {
                    "messages": [f"Thought: I need to execute {call_name} next."],
                    "pending_tool": call_name,
                    "execution_path": [call_name],
                }
        return {
            "final_response": "Single-Agent synthesis complete. Processed all steps sequentially.",
            "execution_path": ["FINISH"],
        }

    has_contact = "call_get_contact_preferences" in executed
    has_calendar = "call_get_calendar_events" in executed
    has_drive = "call_search_drive_docs" in executed
    has_commute = "call_estimate_commute" in executed

    if not has_contact and ("advisor" in task_lower or "contact" in task_lower or "dr." in task_lower):
        action = "call_get_contact_preferences"
    elif not has_calendar and ("calendar" in task_lower or "schedule" in task_lower or "event" in task_lower or "meeting" in task_lower or "meet" in task_lower or "lecture" in task_lower):
        action = "call_get_calendar_events"
    elif not has_drive and ("drive" in task_lower or "notes" in task_lower or "doc" in task_lower or "capstone" in task_lower):
        action = "call_search_drive_docs"
    elif not has_commute and ("commute" in task_lower or "walk" in task_lower or "travel" in task_lower or "maps" in task_lower or "location" in task_lower):
        action = "call_estimate_commute"
    else:
        action = "FINISH"

    if action == "FINISH":
        return {
            "final_response": "Single-Agent synthesis complete. Processed all steps sequentially.",
            "execution_path": ["FINISH"],
        }

    return {
        "messages": [f"Thought: I need to execute {action} next."],
        "pending_tool": action,
        "execution_path": [action],
    }


def _sas_tool_node(state: SingleAgentState) -> dict:
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

    task = state.get("task", "")
    pending = state.get("pending_tool", "")
    overrides = state.get("extraction_overrides") or {}

    if not pending or pending not in _TOOLS:
        pending = _parse_pending_tool(state.get("messages", [])) or pending

    db = get_db_path()

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


def _sas_router(state: SingleAgentState):
    if state.get("final_response"):
        return END
    return "tools"


def _build_rule_based_sas_graph() -> CompiledStateGraph[
    SingleAgentState, None, SingleAgentState, SingleAgentState
]:
    builder = StateGraph(SingleAgentState)
    builder.add_node("reason", _sas_reasoning_node)
    builder.add_node("tools", _sas_tool_node)
    builder.set_entry_point("reason")
    builder.add_conditional_edges("reason", _sas_router)
    builder.add_edge("tools", "reason")
    return builder.compile()


# ===========================================================================
# Export the correct graph based on mode
# ===========================================================================


def build_single_agent_graph() -> CompiledStateGraph[
    SingleAgentState, None, SingleAgentState, SingleAgentState
]:
    if _USE_LLM:
        return _build_llm_sas_graph()
    return _build_rule_based_sas_graph()


single_agent_app = build_single_agent_graph()
