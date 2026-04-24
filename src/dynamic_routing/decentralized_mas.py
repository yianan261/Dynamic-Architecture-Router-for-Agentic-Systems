"""
Decentralized Multi-Agent System (MAS): parallel specialist peers + consensus merge.

Contrasts with Centralized MAS (hub-and-spoke): there is no supervisor loop that
re-plans the next worker after each observation. All peers run in one batch (PCAB:
parallel threads over read-mostly DB access; WorkBench: sequential fixed-order
peer passes on shared sandbox to avoid tool races), then a single merge step
produces the final answer.

Modes (same env as CMAS):
  USE_LLM_WORKERS=true  — LLM workers + LLM consensus
  USE_LLM_WORKERS=false — deterministic peer tools + template consensus
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dynamic_routing.chat_models import bind_tools_safely, get_worker_chat_model
from dynamic_routing.state import CentralizedState

_USE_LLM = os.environ.get("USE_LLM_WORKERS", "false").lower() in ("true", "1", "yes")

MAX_WORKER_REACT_ITERATIONS = 12

# Peer order for deterministic DMAS (WorkBench sequential pass; PCAB parallel batch)
_PEER_ORDER = ("calendar_agent", "drive_agent", "commute_agent", "contacts_agent")


# --- Rule-based peers (same data as CMAS rule workers) ---


def _rule_calendar_peer(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import extract_date, get_calendar_events, get_db_path

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    date = extract_date(task, overrides)
    result = get_calendar_events(date, get_db_path())
    return {"aggregated_context": [f"[CALENDAR]{result}"]}


def _rule_drive_peer(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import extract_drive_query, get_db_path, search_drive_docs

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    query = extract_drive_query(task, overrides)
    result = search_drive_docs(query, get_db_path())
    return {"aggregated_context": [f"[DRIVE]{result}"]}


def _rule_commute_peer(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import estimate_commute, extract_commute_pair, get_db_path

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    pair = extract_commute_pair(task, overrides)
    if pair is None:
        result = '{"status": "skipped", "message": "No commute query detected."}'
    else:
        result = estimate_commute(pair[0], pair[1], get_db_path())
    return {"aggregated_context": [f"[COMMUTE]{result}"]}


def _rule_contacts_peer(state: CentralizedState) -> dict:
    from dynamic_routing.pcab import extract_contact_name, get_contact_preferences, get_db_path

    task = state.get("task", "")
    overrides = state.get("extraction_overrides") or {}
    name = extract_contact_name(task, overrides)
    if name is None:
        result = '{"status": "skipped", "message": "No contact query detected."}'
    else:
        result = get_contact_preferences(name, get_db_path())
    return {"aggregated_context": [f"[CONTACTS]{result}"]}


_RULE_PEER_FUNCS: dict[str, Any] = {
    "calendar_agent": _rule_calendar_peer,
    "drive_agent": _rule_drive_peer,
    "commute_agent": _rule_commute_peer,
    "contacts_agent": _rule_contacts_peer,
}


def _rule_merge_synthesis(aggregated: list) -> str:
    n = len(aggregated)
    return (
        f"DMAS consensus merge complete. Integrated {n} peer reports into a unified view."
    )


def _rule_parallel_peers_node(state: CentralizedState) -> dict:
    """Run all PCAB peers in parallel (read-heavy); then template merge."""

    def _run(name: str) -> dict:
        fn = _RULE_PEER_FUNCS[name]
        return fn(state)

    chunks: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run, name): name for name in _PEER_ORDER}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                out = fut.result()
                chunks.extend(out.get("aggregated_context") or [])
            except Exception as e:
                logging.warning("DMAS peer %s failed: %s", name, e)
                chunks.append(f"[{name.replace('_agent', '').upper()}] ERROR: {str(e)[:80]}")

    def _peer_sort_key(s: str) -> int:
        u = s.upper()
        for i, p in enumerate(("[CALENDAR]", "[DRIVE]", "[COMMUTE]", "[CONTACTS]")):
            if u.startswith(p):
                return i
        return 99

    ordered = sorted((c for c in chunks if isinstance(c, str)), key=_peer_sort_key)
    synthesis = _rule_merge_synthesis(ordered)
    return {
        "aggregated_context": ordered,
        "final_synthesis": synthesis,
        "next_action": "FINISH",
        "execution_path": ["parallel_peers", "consensus_merge"],
    }


def _build_rule_dmas_graph() -> CompiledStateGraph[
    CentralizedState, None, CentralizedState, CentralizedState
]:
    builder = StateGraph(CentralizedState)
    builder.add_node("dmas_run", _rule_parallel_peers_node)
    builder.set_entry_point("dmas_run")
    builder.add_edge("dmas_run", END)
    return builder.compile()


# --- LLM mode: parallel ReAct peers + single LLM merge ---


def _build_llm_dmas_graph() -> CompiledStateGraph[
    CentralizedState, None, CentralizedState, CentralizedState
]:
    from langchain.agents import create_agent

    from dynamic_routing.agent_tools import (
        calendar_tool,
        commute_tool,
        contact_tool,
        drive_tool,
    )

    base_llm = get_worker_chat_model(temperature=0.1)

    cal_react = create_agent(
        model=bind_tools_safely(base_llm, [calendar_tool]),
        tools=[calendar_tool],
        system_prompt="You are a Calendar peer agent. Fetch events and return a concise summary.",
    )
    drv_react = create_agent(
        model=bind_tools_safely(base_llm, [drive_tool]),
        tools=[drive_tool],
        system_prompt="You are a Drive peer agent. Search notes and summarize.",
    )
    com_react = create_agent(
        model=bind_tools_safely(base_llm, [commute_tool]),
        tools=[commute_tool],
        system_prompt="You are a Commute peer agent. Estimate travel times.",
    )
    con_react = create_agent(
        model=bind_tools_safely(base_llm, [contact_tool]),
        tools=[contact_tool],
        system_prompt="You are a Contacts peer agent. Get meeting preferences.",
    )

    def _extract_tokens(response: Any) -> int:
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage") or {}
            return int(usage.get("total_tokens", 0) or 0)
        return 0

    def _llm_content_str(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(c) for c in content)
        return str(content) if content is not None else ""

    def _run_llm_peer(react_agent: Any, task: str, prefix: str) -> dict:
        try:
            result = react_agent.invoke(
                {"messages": [("user", task)]},
                config={"recursion_limit": MAX_WORKER_REACT_ITERATIONS},
            )
        except Exception as e:
            err = str(e).lower()
            if "recursion" in err:
                tag = "Worker ReAct Loop Exhaustion"
            elif "context length" in err or "input_tokens" in err:
                tag = "Worker Context Window Overflow"
            else:
                tag = f"Worker Error: {str(e)[:80]}"
            logging.warning("DMAS LLM peer [%s] failed: %s", prefix, tag)
            return {
                "aggregated_context": [f"[{prefix}] ERROR: {tag}"],
                "total_tokens": 0,
                "failure_taxonomy": f"System Design Issue: {tag}",
            }

        tokens = 0
        for msg in result.get("messages", []):
            if hasattr(msg, "response_metadata"):
                usage = msg.response_metadata.get("token_usage") or {}
                tokens += int(usage.get("total_tokens", 0) or 0)
        final = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                final = str(msg.content)
                break
        return {
            "aggregated_context": [f"[{prefix}]{final}"],
            "total_tokens": tokens,
        }

    def _llm_parallel_peers_node(state: CentralizedState) -> dict:
        task = state.get("task", "")
        peers = [
            (cal_react, task, "CALENDAR"),
            (drv_react, task, "DRIVE"),
            (com_react, task, "COMMUTE"),
            (con_react, task, "CONTACTS"),
        ]
        contexts: list[str] = []
        total_tok = 0
        tax: str | None = None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(_run_llm_peer, r, t, p) for r, t, p in peers]
            for fut in futs:
                out = fut.result()
                total_tok += int(out.get("total_tokens") or 0)
                if out.get("failure_taxonomy"):
                    tax = out.get("failure_taxonomy")
                contexts.extend(out.get("aggregated_context") or [])

        prompt = (
            f"Task: {task}\n\nPeer agent reports (no further tool use):\n{contexts}\n\n"
            "Synthesize a single clear answer for the user from these reports."
        )
        response = base_llm.invoke(prompt)
        merge_tok = _extract_tokens(response)
        synthesis = _llm_content_str(response.content)

        return {
            "aggregated_context": contexts,
            "final_synthesis": synthesis,
            "next_action": "FINISH",
            "total_tokens": total_tok + merge_tok,
            "execution_path": ["parallel_llm_peers", "consensus_merge"],
            "failure_taxonomy": tax,
        }

    builder = StateGraph(CentralizedState)
    builder.add_node("dmas_run", _llm_parallel_peers_node)
    builder.set_entry_point("dmas_run")
    builder.add_edge("dmas_run", END)
    return builder.compile()


def build_decentralized_mas_graph() -> CompiledStateGraph[
    CentralizedState, None, CentralizedState, CentralizedState
]:
    if _USE_LLM:
        return _build_llm_dmas_graph()
    return _build_rule_dmas_graph()


decentralized_mas_app = build_decentralized_mas_graph()
