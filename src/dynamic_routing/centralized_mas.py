"""
Centralized Multi-Agent System (MAS): Hub-and-spoke topology.

Two modes controlled by USE_LLM_WORKERS env var:
  - LLM mode: Each worker is a restricted Llama-3.1 ReAct agent with a single tool.
    The Supervisor uses Llama-3.1 for final synthesis.
  - Rule-based mode (default): deterministic dispatch for CI/benchmarking.
"""

import os

from langgraph.graph import END, StateGraph

from dynamic_routing.state import CentralizedState

_REQUIRED_TOOL_TO_WORKER: dict[str, str] = {
    "get_contact_preferences": "contacts_agent",
    "get_calendar_events": "calendar_agent",
    "search_drive_docs": "drive_agent",
    "estimate_commute": "commute_agent",
}

_USE_LLM = os.environ.get("USE_LLM_WORKERS", "false").lower() in ("true", "1", "yes")

# ===========================================================================
# LLM Mode — Llama-3.1 restricted workers + LLM synthesis
# ===========================================================================


def _build_llm_cmas_graph() -> StateGraph:
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from dynamic_routing.agent_tools import (
        calendar_tool,
        commute_tool,
        contact_tool,
        drive_tool,
    )

    worker_llm = ChatOpenAI(
        model=os.environ.get("VLLM_WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1"),
        temperature=0.1,
    )

    cal_react = create_react_agent(worker_llm, [calendar_tool], prompt="You are a Calendar Agent. Fetch events and return a concise summary.")
    drv_react = create_react_agent(worker_llm, [drive_tool], prompt="You are a Drive Agent. Search notes and summarize.")
    com_react = create_react_agent(worker_llm, [commute_tool], prompt="You are a Commute Agent. Estimate travel times.")
    con_react = create_react_agent(worker_llm, [contact_tool], prompt="You are a Contacts Agent. Get meeting preferences.")

    def _run_llm_worker(react_agent, task: str, prefix: str) -> dict:
        result = react_agent.invoke({"messages": [("user", task)]})
        tokens = 0
        for msg in result.get("messages", []):
            if hasattr(msg, "response_metadata"):
                usage = msg.response_metadata.get("token_usage") or {}
                tokens += usage.get("total_tokens", 0)
        final = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                final = msg.content
                break
        return {
            "aggregated_context": [f"[{prefix}]{final}"],
            "total_tokens": tokens,
        }

    def llm_calendar_agent(state: CentralizedState) -> dict:
        return _run_llm_worker(cal_react, state["task"], "CALENDAR")

    def llm_drive_agent(state: CentralizedState) -> dict:
        return _run_llm_worker(drv_react, state["task"], "DRIVE")

    def llm_commute_agent(state: CentralizedState) -> dict:
        return _run_llm_worker(com_react, state["task"], "COMMUTE")

    def llm_contacts_agent(state: CentralizedState) -> dict:
        return _run_llm_worker(con_react, state["task"], "CONTACTS")

    def llm_supervisor_node(state: CentralizedState) -> dict:
        context = state.get("aggregated_context", [])
        context_str = str(context).lower()
        req_tools = state.get("required_tools") or []
        task = state["task"]

        # Data-driven dispatch for remaining workers
        if req_tools:
            for tool_name in req_tools:
                worker = _REQUIRED_TOOL_TO_WORKER.get(tool_name)
                if worker and f"[{worker.replace('_agent', '').lower()}]" not in context_str:
                    return {"next_action": worker}

        # All workers done — LLM synthesis
        prompt = f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize this data into a clear final answer."
        response = worker_llm.invoke(prompt)

        tokens = 0
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage") or {}
            tokens = usage.get("total_tokens", 0)

        return {
            "next_action": "FINISH",
            "final_synthesis": response.content,
            "total_tokens": tokens,
        }

    def supervisor_router(state: CentralizedState):
        return END if state.get("next_action") == "FINISH" else state["next_action"]

    builder = StateGraph(CentralizedState)
    builder.add_node("supervisor", llm_supervisor_node)
    builder.add_node("calendar_agent", llm_calendar_agent)
    builder.add_node("drive_agent", llm_drive_agent)
    builder.add_node("commute_agent", llm_commute_agent)
    builder.add_node("contacts_agent", llm_contacts_agent)
    builder.set_entry_point("supervisor")
    builder.add_edge("calendar_agent", "supervisor")
    builder.add_edge("drive_agent", "supervisor")
    builder.add_edge("commute_agent", "supervisor")
    builder.add_edge("contacts_agent", "supervisor")
    builder.add_conditional_edges("supervisor", supervisor_router)
    return builder.compile()


# ===========================================================================
# Rule-Based Mode — deterministic dispatch (no LLM required)
# ===========================================================================


def _rule_calendar_agent(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import extract_date, get_calendar_events, get_db_path

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    date = extract_date(task, overrides)
    result = get_calendar_events(date, get_db_path())
    return {"aggregated_context": [f"[CALENDAR]{result}"]}


def _rule_drive_agent(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import extract_drive_query, get_db_path, search_drive_docs

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    query = extract_drive_query(task, overrides)
    result = search_drive_docs(query, get_db_path())
    return {"aggregated_context": [f"[DRIVE]{result}"]}


def _rule_commute_agent(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import estimate_commute, extract_commute_pair, get_db_path

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    pair = extract_commute_pair(task, overrides)
    if pair is None:
        result = '{"status": "skipped", "message": "No commute query detected."}'
    else:
        result = estimate_commute(pair[0], pair[1], get_db_path())
    return {"aggregated_context": [f"[COMMUTE]{result}"]}


def _rule_contacts_agent(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import extract_contact_name, get_contact_preferences, get_db_path

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    name = extract_contact_name(task, overrides)
    if name is None:
        result = '{"status": "skipped", "message": "No contact query detected."}'
    else:
        result = get_contact_preferences(name, get_db_path())
    return {"aggregated_context": [f"[CONTACTS]{result}"]}


def _rule_supervisor_node(state: CentralizedState) -> dict:
    context = state.get("aggregated_context", [])
    task = state["task"].lower()
    context_str = str(context).lower()
    required_tools = state.get("required_tools") or []

    if required_tools:
        for tool_name in required_tools:
            worker = _REQUIRED_TOOL_TO_WORKER.get(tool_name)
            if not worker:
                continue
            context_key = f"[{worker.replace('_agent', '').lower()}]"
            if context_key not in context_str:
                return {"next_action": worker}
        synthesis = f"Synthesis Complete. Aggregated {len(context)} sources to build personalized context."
        return {"next_action": "FINISH", "final_synthesis": synthesis}

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
        synthesis = f"Synthesis Complete. Aggregated {len(context)} sources to build personalized context."
        return {"next_action": next_worker, "final_synthesis": synthesis}

    return {"next_action": next_worker}


def _rule_supervisor_router(state: CentralizedState):
    return END if state.get("next_action") == "FINISH" else state["next_action"]


def _build_rule_based_cmas_graph() -> StateGraph:
    builder = StateGraph(CentralizedState)
    builder.add_node("supervisor", _rule_supervisor_node)
    builder.add_node("calendar_agent", _rule_calendar_agent)
    builder.add_node("drive_agent", _rule_drive_agent)
    builder.add_node("commute_agent", _rule_commute_agent)
    builder.add_node("contacts_agent", _rule_contacts_agent)
    builder.set_entry_point("supervisor")
    builder.add_edge("calendar_agent", "supervisor")
    builder.add_edge("drive_agent", "supervisor")
    builder.add_edge("commute_agent", "supervisor")
    builder.add_edge("contacts_agent", "supervisor")
    builder.add_conditional_edges("supervisor", _rule_supervisor_router)
    return builder.compile()


# ===========================================================================
# Export the correct graph based on mode
# ===========================================================================


def build_centralized_mas_graph():
    if _USE_LLM:
        return _build_llm_cmas_graph()
    return _build_rule_based_cmas_graph()


centralized_mas_app = build_centralized_mas_graph()
