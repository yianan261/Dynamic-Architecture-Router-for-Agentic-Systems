"""
Centralized Multi-Agent System (MAS): Hub-and-spoke topology.

Two modes controlled by USE_LLM_WORKERS env var:
  - LLM mode: Each worker is a restricted Llama-3.1 ReAct agent with a single tool.
    The Supervisor uses Llama-3.1 for final synthesis.
  - Rule-based mode (default): deterministic dispatch for CI/benchmarking.
"""

import logging
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

MAX_SUPERVISOR_TURNS = 10

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

    base_llm = ChatOpenAI(
        model=os.environ.get("VLLM_WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1"),
        temperature=0.1,
    )

    cal_react = create_react_agent(base_llm.bind_tools([calendar_tool], parallel_tool_calls=False), [calendar_tool], prompt="You are a Calendar Agent. Fetch events and return a concise summary.")
    drv_react = create_react_agent(base_llm.bind_tools([drive_tool], parallel_tool_calls=False), [drive_tool], prompt="You are a Drive Agent. Search notes and summarize.")
    com_react = create_react_agent(base_llm.bind_tools([commute_tool], parallel_tool_calls=False), [commute_tool], prompt="You are a Commute Agent. Estimate travel times.")
    con_react = create_react_agent(base_llm.bind_tools([contact_tool], parallel_tool_calls=False), [contact_tool], prompt="You are a Contacts Agent. Get meeting preferences.")

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

    _WORKER_DESCRIPTIONS = {
        "calendar_agent": "Fetches calendar events and schedules",
        "drive_agent": "Searches documents and notes",
        "commute_agent": "Estimates travel/commute times between locations",
        "contacts_agent": "Gets contact preferences and meeting details",
    }

    def _extract_tokens(response) -> int:
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage") or {}
            return usage.get("total_tokens", 0)
        return 0

    def _detect_cycle(history: list[str]) -> str | None:
        """Return a taxonomy tag if the execution path shows a cyclic failure."""
        if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
            return "Inter-agent Misalignment: Cyclic Repetition"
        if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
            return "Inter-agent Misalignment: Oscillating Dispatch"
        return None

    def llm_supervisor_node(state: CentralizedState) -> dict:
        context = state.get("aggregated_context", [])
        context_str = str(context).lower()
        task = state["task"]
        history = state.get("execution_path", [])

        # --- Circuit Breaker: Total Turn Limit ---
        if len(history) > MAX_SUPERVISOR_TURNS:
            logging.warning("CMAS circuit breaker: max turns (%d) exceeded", MAX_SUPERVISOR_TURNS)
            return {
                "next_action": "FINISH",
                "final_synthesis": "HALTED: Maximum coordination turns exceeded.",
                "failure_taxonomy": "System Design Issue: Infinite Loop / Token Exhaustion",
                "total_tokens": 0,
            }

        # --- Circuit Breaker: Cyclic Pattern ---
        cycle_tag = _detect_cycle(history)
        if cycle_tag:
            logging.warning("CMAS circuit breaker: %s detected in path %s", cycle_tag, history[-4:])
            return {
                "next_action": "FINISH",
                "final_synthesis": f"HALTED: {cycle_tag} detected.",
                "failure_taxonomy": cycle_tag,
                "total_tokens": 0,
            }

        done_workers = set()
        for tag, worker in [("calendar", "calendar_agent"), ("drive", "drive_agent"),
                            ("commute", "commute_agent"), ("contacts", "contacts_agent")]:
            if f"[{tag}]" in context_str:
                done_workers.add(worker)

        available = [w for w in _WORKER_DESCRIPTIONS if w not in done_workers]

        if not available:
            prompt = f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize this data into a clear final answer."
            response = base_llm.invoke(prompt)
            return {
                "next_action": "FINISH",
                "final_synthesis": response.content,
                "total_tokens": _extract_tokens(response),
                "execution_path": ["FINISH"],
            }

        available_desc = "\n".join(
            f"- {w}: {_WORKER_DESCRIPTIONS[w]}" for w in available
        )
        dispatch_prompt = (
            f"You are a supervisor coordinating workers to answer a user's task.\n\n"
            f"Task: {task}\n\n"
            f"Data collected so far:\n{context if context else 'Nothing yet.'}\n\n"
            f"Available workers you can dispatch:\n{available_desc}\n\n"
            f"Analyze the data collected so far. Is it sufficient to fully answer the task?\n"
            f"If NO: Explain exactly WHY the data is insufficient or what is missing.\n"
            f"If YES: Explain that you have all required data.\n\n"
            f"Respond strictly in this two-line format:\n"
            f"REASONING: <your explanation>\n"
            f"ACTION: <worker_name or FINISH>"
        )
        response = base_llm.invoke(dispatch_prompt)
        tokens = _extract_tokens(response)
        text = response.content.strip()

        reasoning = ""
        action_str = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("REASONING:"):
                reasoning = stripped[10:].strip()
            elif stripped.upper().startswith("ACTION:"):
                action_str = stripped[7:].strip().lower()

        if not action_str:
            action_str = text.lower()

        chosen_worker = None
        for w in available:
            if w.replace("_agent", "") in action_str:
                chosen_worker = w
                break

        if chosen_worker:
            is_redispatch = chosen_worker in done_workers
            path_entry = f"{chosen_worker}|{reasoning}" if reasoning else chosen_worker

            if is_redispatch:
                taxonomy = (
                    f"Inter-agent Misalignment: Synthesis Drift. "
                    f"Supervisor re-dispatched {chosen_worker}: '{reasoning}'"
                )
                logging.warning("CMAS rejection: %s", taxonomy)
                return {
                    "next_action": "FINISH",
                    "final_synthesis": f"HALTED: {taxonomy}",
                    "failure_taxonomy": taxonomy,
                    "total_tokens": tokens,
                    "execution_path": [path_entry],
                }

            return {
                "next_action": chosen_worker,
                "total_tokens": tokens,
                "execution_path": [path_entry],
            }

        prompt = f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize this data into a clear final answer."
        synth_response = base_llm.invoke(prompt)
        return {
            "next_action": "FINISH",
            "final_synthesis": synth_response.content,
            "total_tokens": tokens + _extract_tokens(synth_response),
            "execution_path": [f"FINISH|{reasoning}" if reasoning else "FINISH"],
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
                return {"next_action": worker, "execution_path": [worker]}
        synthesis = f"Synthesis Complete. Aggregated {len(context)} sources to build personalized context."
        return {"next_action": "FINISH", "final_synthesis": synthesis, "execution_path": ["FINISH"]}

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
        return {"next_action": next_worker, "final_synthesis": synthesis, "execution_path": ["FINISH"]}

    return {"next_action": next_worker, "execution_path": [next_worker]}


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
