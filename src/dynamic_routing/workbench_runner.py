"""
Run Single-Agent and Centralized MAS on WorkBench tools + sandbox (Llama via vLLM).

Uses LangGraph create_react_agent with WorkBench LangChain tools. Grading is
outcome-centric (DataFrame equality after executing predicted vs gold calls).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import SecretStr

from dynamic_routing.state import CentralizedState
from dynamic_routing.workbench_env import (
    get_tools_for_domains,
    reset_workbench_state,
    workbench_system_prompt_prefix,
)

MAX_REACT_ITERATIONS = 22
MAX_CMAS_SUPERVISOR_TURNS = 12
MAX_WORKER_REACT = 16
LOOP_SCORE_THRESHOLD = 0.5


def _worker_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("VLLM_WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "EMPTY")),
        base_url=os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1"),
        temperature=0.1,
    )


def tool_calls_to_strings(messages: list) -> list[str]:
    """Build WorkBench-style call strings from LangGraph message list."""
    out: list[str] = []
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None) or []
        for call in tcs:
            name = call["name"]
            args = call.get("args") or {}
            if not isinstance(args, dict):
                continue
            parts = [f'{k}="{v}"' for k, v in sorted(args.items())]
            out.append(f"{name}.func(" + ", ".join(parts) + ")")
    return out


def run_workbench_sas(task: str, tools: list) -> dict[str, Any]:
    """Single ReAct agent with full tool set for this task's domains."""
    from langgraph.prebuilt import create_react_agent

    reset_workbench_state()
    base_prompt = (
        workbench_system_prompt_prefix()
        + "You are an efficient workplace assistant. Use tools to complete the task. "
        "Call only tools that are necessary."
    )
    llm = _worker_llm().bind_tools(tools, parallel_tool_calls=False)
    agent = create_react_agent(llm, tools, prompt=base_prompt)
    err = ""
    t0 = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [("user", task)]},
            config={"recursion_limit": MAX_REACT_ITERATIONS},
        )
    except Exception as e:
        err = str(e)
        logging.warning("WorkBench SAS error: %s", err[:200])
        latency = time.perf_counter() - t0
        return {
            "function_calls": [],
            "total_tokens": 0,
            "latency_sec": latency,
            "error": err,
            "final_response": "",
        }
    latency = time.perf_counter() - t0
    messages = result.get("messages", [])
    calls = tool_calls_to_strings(messages)
    tokens = 0
    for msg in messages:
        if hasattr(msg, "response_metadata"):
            usage = (msg.response_metadata or {}).get("token_usage") or {}
            tokens += usage.get("total_tokens", 0)
    final = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            final = str(msg.content)
            break
    return {
        "function_calls": calls,
        "total_tokens": tokens,
        "latency_sec": latency,
        "error": err,
        "final_response": final,
    }


# --- CMAS: supervisor + one ReAct worker per domain (workers filtered by CSV domains) ---

_WORKER_SPECS: list[tuple[str, str, list[str]]] = [
    ("email_agent", "You are the Email agent. Use email and directory tools only.", ["email"]),
    ("calendar_agent", "You are the Calendar agent. Use calendar tools only.", ["calendar"]),
    ("analytics_agent", "You are the Analytics agent. Use analytics tools only.", ["analytics"]),
    ("project_management_agent", "You are Project Management. Use project tools only.", ["project_management"]),
    (
        "crm_agent",
        "You are the CRM agent. Use customer relationship tools only.",
        ["customer_relationship_manager"],
    ),
]


def _tag_for_worker(worker_key: str) -> str:
    if worker_key == "crm_agent":
        return "crm"
    return worker_key.replace("_agent", "")


def _active_worker_specs(domains: list[str]) -> list[tuple[str, str, list[str]]]:
    active = [s for s in _WORKER_SPECS if s[2][0] in domains]
    return active if active else list(_WORKER_SPECS)


def build_workbench_cmas_graph(domains: list[str], collected_calls: list[str]):
    """Hub-and-spoke graph; appends each worker's tool calls to collected_calls."""
    from langgraph.prebuilt import create_react_agent

    active = _active_worker_specs(domains)
    prefix = workbench_system_prompt_prefix()
    base_llm = _worker_llm()

    worker_reacts: dict[str, Any] = {}
    for node_name, worker_prompt, dlist in active:
        wt = get_tools_for_domains(dlist)
        llm_w = base_llm.bind_tools(wt, parallel_tool_calls=False)
        worker_reacts[node_name] = create_react_agent(
            llm_w,
            wt,
            prompt=prefix + worker_prompt,
        )

    def _run_worker(node_name: str, task: str, tag: str) -> dict:
        react = worker_reacts[node_name]
        try:
            result = react.invoke(
                {"messages": [("user", task)]},
                config={"recursion_limit": MAX_WORKER_REACT},
            )
        except Exception as e:
            return {
                "aggregated_context": [f"[{tag}] ERROR: {str(e)[:120]}"],
                "total_tokens": 0,
                "failure_taxonomy": str(e)[:200],
            }
        messages = result.get("messages", [])
        collected_calls.extend(tool_calls_to_strings(messages))
        tokens = 0
        for msg in messages:
            if hasattr(msg, "response_metadata"):
                usage = (msg.response_metadata or {}).get("token_usage") or {}
                tokens += usage.get("total_tokens", 0)
        final = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                final = str(msg.content)
                break
        return {
            "aggregated_context": [f"[{tag}] {final}"],
            "total_tokens": tokens,
        }

    descriptions = {
        "email_agent": "Email / inbox operations",
        "calendar_agent": "Calendar events",
        "analytics_agent": "Analytics / plots / traffic",
        "project_management_agent": "Project tasks / boards",
        "crm_agent": "Customer relationship records",
    }

    def _extract_tokens(response) -> int:
        if hasattr(response, "response_metadata"):
            u = (response.response_metadata or {}).get("token_usage") or {}
            return u.get("total_tokens", 0)
        return 0

    def _detect_cycle(history: list[str]) -> str | None:
        if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
            return "Inter-agent Misalignment: Cyclic Repetition"
        if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
            return "Inter-agent Misalignment: Oscillating Dispatch"
        return None

    def _loop_score(history: list[str]) -> float:
        if len(history) < 2:
            return 0.0
        worker_entries = [h.split("|")[0] for h in history if h != "FINISH"]
        if not worker_entries:
            return 0.0
        unique = len(set(worker_entries))
        return 1.0 - (unique / len(worker_entries))

    def _message_content_as_str(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(c) for c in content)
        return str(content) if content is not None else ""

    def supervisor_node(state: CentralizedState) -> dict:
        context = state.get("aggregated_context", [])
        context_str = str(context).lower()
        task = state.get("task", "")
        history = state.get("execution_path", [])

        if len(history) > MAX_CMAS_SUPERVISOR_TURNS:
            return {
                "next_action": "FINISH",
                "final_synthesis": "HALTED: Maximum coordination turns exceeded.",
                "failure_taxonomy": "System Design Issue: Supervisor Turn Limit",
                "total_tokens": 0,
            }

        ctag = _detect_cycle(history)
        if ctag:
            return {
                "next_action": "FINISH",
                "final_synthesis": f"HALTED: {ctag}",
                "failure_taxonomy": ctag,
                "total_tokens": 0,
            }

        loop_sc = _loop_score(history)
        if len(history) >= 4 and loop_sc >= LOOP_SCORE_THRESHOLD:
            return {
                "next_action": "FINISH",
                "final_synthesis": f"HALTED: Loop score {loop_sc:.2f}",
                "failure_taxonomy": f"System Design Issue: Loop Score {loop_sc:.2f}",
                "total_tokens": 0,
            }

        done = set()
        for node_name, _, _ in active:
            tag = _tag_for_worker(node_name)
            if f"[{tag}]" in context_str:
                done.add(node_name)

        avail = [w for w in active if w[0] not in done]

        if not avail:
            prompt = f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize a clear final answer."
            response = base_llm.invoke(prompt)
            return {
                "next_action": "FINISH",
                "final_synthesis": response.content,
                "total_tokens": _extract_tokens(response),
                "execution_path": ["FINISH"],
            }

        avail_desc = "\n".join(f"- {w[0]}: {descriptions.get(w[0], w[0])}" for w in avail)
        dispatch_prompt = (
            "You are a supervisor coordinating specialized workers.\n\n"
            f"Task: {task}\n\nData so far:\n{context if context else 'Nothing yet.'}\n\n"
            f"Available workers:\n{avail_desc}\n\n"
            "If you have enough data to answer fully, respond:\n"
            "REASONING: ...\nACTION: FINISH\n"
            "Otherwise dispatch exactly one worker:\n"
            "REASONING: ...\nACTION: <worker_name>\n"
            "worker_name must be one of the available worker names listed above."
        )
        response = base_llm.invoke(dispatch_prompt)
        tokens = _extract_tokens(response)
        text = _message_content_as_str(response.content).strip()
        action_str = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("ACTION:"):
                action_str = stripped[7:].strip().lower()
        if not action_str:
            action_str = text.lower()

        chosen = None
        for w in avail:
            key = w[0].lower()
            if key in action_str or key.replace("_agent", "") in action_str:
                chosen = w[0]
                break

        if chosen is None and "finish" in action_str:
            synth = base_llm.invoke(
                f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize a clear final answer."
            )
            return {
                "next_action": "FINISH",
                "final_synthesis": synth.content,
                "total_tokens": tokens + _extract_tokens(synth),
                "execution_path": ["FINISH"],
            }

        if chosen:
            return {
                "next_action": chosen,
                "total_tokens": tokens,
                "execution_path": [chosen],
            }

        synth = base_llm.invoke(
            f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize a clear final answer."
        )
        return {
            "next_action": "FINISH",
            "final_synthesis": synth.content,
            "total_tokens": tokens + _extract_tokens(synth),
            "execution_path": ["FINISH"],
        }

    def supervisor_router(state: CentralizedState):
        next_a = state.get("next_action")
        if next_a is None or next_a == "FINISH":
            return END
        return next_a

    builder = StateGraph(CentralizedState)
    builder.add_node("supervisor", supervisor_node)

    for node_name, _, _ in active:
        tag = _tag_for_worker(node_name)

        def make_node(nm: str, tg: str):
            def _node(state: CentralizedState) -> dict:
                return _run_worker(nm, state.get("task", ""), tg)

            return _node

        builder.add_node(node_name, make_node(node_name, tag))
        builder.add_edge(node_name, "supervisor")

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", supervisor_router)
    return builder.compile()


def run_workbench_cmas(task: str, domains: list[str]) -> dict[str, Any]:
    """Hub-and-spoke CMAS on fresh sandbox; function_calls lists all worker tool invocations."""
    collected: list[str] = []
    reset_workbench_state()
    graph = build_workbench_cmas_graph(domains, collected)
    state: CentralizedState = {
        "task": task,
        "aggregated_context": [],
        "next_action": "",
        "final_synthesis": "",
        "total_tokens": 0,
        "execution_path": [],
    }
    t0 = time.perf_counter()
    err = ""
    try:
        result = graph.invoke(state, config={"recursion_limit": 80})
    except Exception as e:
        err = str(e)
        logging.warning("WorkBench CMAS error: %s", err[:200])
        return {
            "function_calls": list(collected),
            "total_tokens": 0,
            "latency_sec": time.perf_counter() - t0,
            "error": err,
            "final_response": "",
            "execution_path": [],
        }
    lat = time.perf_counter() - t0
    tax = result.get("failure_taxonomy")
    return {
        "function_calls": list(collected),
        "total_tokens": result.get("total_tokens") or 0,
        "latency_sec": lat,
        "error": err,
        "final_response": result.get("final_synthesis") or "",
        "execution_path": result.get("execution_path") or [],
        "failure_taxonomy": tax,
    }
