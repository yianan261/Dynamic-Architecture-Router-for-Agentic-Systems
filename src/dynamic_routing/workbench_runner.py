"""
Run Single-Agent and Centralized MAS on WorkBench tools + sandbox.

Uses LangGraph create_react_agent with WorkBench LangChain tools. Grading is
outcome-centric (DataFrame equality after executing predicted vs gold calls).

Design notes (motivated by WorkBench paper + "Science of Scaling" SAS baseline):

* Anti-hallucination: the system prompt explicitly forbids passing a tool call
  (or pseudo-code like ``email.get_most_recent_email_id()``) as a string
  argument; IDs must be discovered first via search/getter tools, then passed
  as literal values.
* Pagination: search_* tools in WorkBench cap at ~5 results. The prompt tells
  the agent to loop until the cap is hit and nothing new is found.
* Matched-compute SAS: recursion limit raised to 40 so SAS is not artificially
  starved of reasoning steps when compared to CMAS.
* Token accounting: langchain-core >=1.0 exposes usage on ``msg.usage_metadata``
  (with ``total_tokens`` / ``input_tokens`` / ``output_tokens``). We read that
  first, then fall back to ``response_metadata.token_usage`` for older paths.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import Field, create_model

from dynamic_routing.chat_models import bind_tools_safely, get_worker_chat_model, llm_backend
from dynamic_routing.state import CentralizedState
from dynamic_routing.workbench_env import (
    get_tools_for_domains,
    reset_workbench_state,
    workbench_system_prompt_prefix,
)

MAX_REACT_ITERATIONS = 40
MAX_CMAS_SUPERVISOR_TURNS = 20
MAX_WORKER_REACT = 30
LOOP_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Rate-limit / quota handling for hosted APIs (Gemini free-tier TPM, etc.)
# ---------------------------------------------------------------------------
#
# LangChain's internal retry only fires while the HTTP call is in flight; once
# the exception bubbles out of the chat model (inside a LangGraph ReAct loop or
# our supervisor dispatch), we need an outer wrapper that sleeps and retries
# for *that entire invocation*. This is the "bulletproof fix" requested: if we
# see a 429 / RESOURCE_EXHAUSTED / quota / rate-limit marker in the error
# string, sleep for 60s (exponentially increasing) and retry up to
# ``RATE_LIMIT_MAX_RETRIES`` times.
RATE_LIMIT_MARKERS = (
    "429",
    "resource_exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
)


def _default_rate_limit_retries() -> int:
    try:
        return max(0, int(os.environ.get("RATE_LIMIT_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def _default_rate_limit_sleep() -> float:
    try:
        return max(1.0, float(os.environ.get("RATE_LIMIT_SLEEP_SEC", "60")))
    except ValueError:
        return 60.0


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in RATE_LIMIT_MARKERS)


def _invoke_with_rate_limit_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)``; on quota errors, back off and retry.

    The first retry waits ``RATE_LIMIT_SLEEP_SEC`` (default 60s), and each
    subsequent retry doubles that interval so we don't hammer the API if the
    quota window is wider than one minute. Only re-raises after exhausting
    ``RATE_LIMIT_MAX_RETRIES`` retries, or immediately for non-quota errors so
    the benchmark loop can record a proper ``error`` / ``failure_taxonomy``.
    """
    retries = _default_rate_limit_retries()
    base_sleep = _default_rate_limit_sleep()
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — we classify below
            if not _is_rate_limit_error(e):
                raise
            last_exc = e
            if attempt >= retries:
                break
            sleep_s = base_sleep * (2**attempt)
            logging.warning(
                "Rate-limit hit (attempt %d/%d): %s — sleeping %.0fs before retry",
                attempt + 1,
                retries,
                str(e)[:140],
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc

_WB_TOOL_SAFE_TO_CANONICAL: dict[str, str] = {}


def _hosted_api_needs_safe_tool_names() -> bool:
    return llm_backend() in ("openai", "google")


def _clear_wb_tool_name_remap() -> None:
    _WB_TOOL_SAFE_TO_CANONICAL.clear()


def _build_string_args_schema(tool_name: str, func: Any) -> type:
    """Build a Pydantic model where every parameter is a plain ``str``.

    WorkBench tools are declared without type annotations (``def send_email(
    recipient=None, subject=None, body=None)``), so LangChain's auto-inferred
    args_schema has properties with no ``type`` field. OpenAI's function-calling
    converter tolerates this, but ``langchain-google-genai`` rejects it with
    ``3 validation errors for Schema: properties.recipient ... Input should be
    a valid dictionary or object``.

    We type every param as ``str`` (not ``Optional[str]``) because Gemini's
    function-calling schema does not accept nullable / anyOf types. The default
    is the empty string rather than ``None`` so that if Gemini does emit an
    unused argument, WorkBench's ``if not x`` guards still treat it as "not
    provided". This also avoids crashes in pandas-backed tools like
    ``calendar.search_events`` which calls ``.str.contains(query)`` — passing
    ``None`` raises ``TypeError: first argument must be string or compiled
    pattern``.
    """
    sig = inspect.signature(func)
    fields: dict[str, tuple[Any, Any]] = {}
    for pname in sig.parameters:
        fields[pname] = (str, Field(default="", description=f"{pname} (optional)"))
    model_name = re.sub(r"\W+", "_", tool_name) + "_Args"
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def _retype_tool_for_google(t: Any, new_name: str | None = None) -> Any:
    """Rebuild a StructuredTool with an explicit string-typed args_schema."""
    func = getattr(t, "func", None)
    if func is None:
        # Fallback: we can't inspect; return best-effort name-renamed copy.
        return t.model_copy(update={"name": new_name}) if new_name else t
    schema = _build_string_args_schema(t.name, func)
    return StructuredTool.from_function(
        func=func,
        name=new_name or t.name,
        description=t.description,
        args_schema=schema,
    )


def _prepare_tools_for_hosted_api(tools: list[Any]) -> list[Any]:
    """Copy tools with dot-free names for hosted LLM APIs; fill remap for grading strings.

    For the Google backend we also rebuild each tool with an explicit Pydantic
    args_schema (all params typed as Optional[str]), because WorkBench's
    un-annotated ``@tool`` functions otherwise produce a malformed Gemini
    Schema and every tool call fails with a validation error.
    """
    if not _hosted_api_needs_safe_tool_names():
        return tools
    backend = llm_backend()
    out: list[Any] = []
    for t in tools:
        name = t.name
        safe: str | None
        if "." in name:
            safe_str = name.replace(".", "_")
            _WB_TOOL_SAFE_TO_CANONICAL[safe_str] = name
            safe = safe_str
        else:
            safe = None
        if backend == "google":
            out.append(_retype_tool_for_google(t, new_name=safe))
        elif safe is not None:
            out.append(t.model_copy(update={"name": safe}))
        else:
            out.append(t)
    return out


def _worker_llm():
    return get_worker_chat_model(temperature=0.1)


# ---------------------------------------------------------------------------
# Prompt engineering (WorkBench-specific)
# ---------------------------------------------------------------------------

WORKBENCH_AGENT_GUIDELINES = """You are an efficient workplace assistant executing tasks against a realistic sandbox (email, calendar, analytics, project management, CRM, company directory).

Tool-use rules (read carefully — violating these is the #1 failure mode):
1. NEVER invent, guess, or compose identifiers (email IDs, event IDs, customer IDs, task IDs, dates, timestamps). If you don't already have an exact value, CALL A SEARCH / GET TOOL FIRST to obtain it, then pass the returned value as a plain string literal.
2. NEVER pass a function call, Python expression, f-string, or pseudo-code as a tool argument. For example, do NOT set `email_id="email.get_most_recent_email_id()"` — call that tool, read the returned ID, then set `email_id="<the actual id string>"`.
3. Tool arguments are literals only. Dates are "YYYY-MM-DD" strings, times are "HH:MM" strings, names use the exact casing returned by the directory/search tool.
4. Pagination: many `search_*` / list tools return at most 5 items per call. If the task requires operating on potentially more matches, keep calling with a refined filter (next page, different sender, narrower date range) until you see fewer than 5 results or the set stops changing.
5. Only emit side-effect calls (delete, update, send, create, move, reassign, plot) that the user explicitly asked for. Redundant or speculative side-effects will fail grading.
6. Do exactly one tool call per step unless the tools are independent. Chain dependent calls — read the result, then act.
7. When the task is complete, stop calling tools and produce a short final message. Do not emit further tool calls "just in case".

If a tool returns an error or an empty result, try a different, more specific query before giving up. Do NOT repeat the same call with the same arguments.
"""


def _sas_system_prompt() -> str:
    return workbench_system_prompt_prefix() + WORKBENCH_AGENT_GUIDELINES


def _worker_system_prompt(specialization: str) -> str:
    return (
        workbench_system_prompt_prefix()
        + WORKBENCH_AGENT_GUIDELINES
        + f"\nYour specialization for this task: {specialization}\n"
    )


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------


def _tokens_from_message(msg: Any) -> int:
    """Read token usage from either msg.usage_metadata (langchain >=1.0) or
    msg.response_metadata.token_usage (older / OpenAI path)."""
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        tt = um.get("total_tokens")
        if isinstance(tt, int) and tt > 0:
            return tt
        it = int(um.get("input_tokens", 0) or 0)
        ot = int(um.get("output_tokens", 0) or 0)
        if it or ot:
            return it + ot
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") or {}
    return int(tu.get("total_tokens", 0) or 0)


def _aggregate_tokens(messages: list[Any]) -> int:
    return sum(_tokens_from_message(m) for m in messages)


def tool_calls_to_strings(messages: list) -> list[str]:
    """Build WorkBench-style call strings from LangGraph message list."""
    out: list[str] = []
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None) or []
        for call in tcs:
            name = call["name"]
            name = _WB_TOOL_SAFE_TO_CANONICAL.get(name, name)
            args = call.get("args") or {}
            if not isinstance(args, dict):
                continue
            parts = [f'{k}="{v}"' for k, v in sorted(args.items())]
            out.append(f"{name}.func(" + ", ".join(parts) + ")")
    return out


def _final_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content:
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# SAS
# ---------------------------------------------------------------------------


def run_workbench_sas(task: str, tools: list) -> dict[str, Any]:
    """Single ReAct agent with full tool set for this task's domains."""
    from langgraph.prebuilt import create_react_agent

    reset_workbench_state()
    _clear_wb_tool_name_remap()
    tools = _prepare_tools_for_hosted_api(tools)

    llm = bind_tools_safely(_worker_llm(), tools)
    agent = create_react_agent(llm, tools, prompt=_sas_system_prompt())

    err = ""
    t0 = time.perf_counter()
    try:
        result = _invoke_with_rate_limit_retry(
            agent.invoke,
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
    return {
        "function_calls": tool_calls_to_strings(messages),
        "total_tokens": _aggregate_tokens(messages),
        "latency_sec": latency,
        "error": err,
        "final_response": _final_text(messages),
    }


# ---------------------------------------------------------------------------
# CMAS: supervisor + one ReAct worker per domain (workers filtered by CSV domains)
# ---------------------------------------------------------------------------

_WORKER_SPECS: list[tuple[str, str, list[str]]] = [
    (
        "email_agent",
        "Email / inbox operations. Use email and company-directory tools only. Search first for IDs, then act.",
        ["email"],
    ),
    (
        "calendar_agent",
        "Calendar operations. Use calendar and company-directory tools only. Search events first, then modify by exact event_id.",
        ["calendar"],
    ),
    (
        "analytics_agent",
        "Analytics / plots / traffic queries. Use analytics tools only. Check values before plotting; only plot when the task's condition holds.",
        ["analytics"],
    ),
    (
        "project_management_agent",
        "Project management. Use project tools only. Search tasks first, act on exact task_id.",
        ["project_management"],
    ),
    (
        "crm_agent",
        "Customer relationship management. Use CRM tools only. Search customers first, modify by exact customer_id.",
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


_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_action_line(text: str, candidates: list[str]) -> str | None:
    """Return the chosen worker name or 'FINISH' from the supervisor text.

    Tolerates fenced/prefixed output from frontier models."""
    stripped = text.strip().strip("`")
    m = _ACTION_RE.search(stripped)
    raw = m.group(1).strip().lower() if m else stripped.lower()

    if "finish" in raw:
        return "FINISH"
    for c in candidates:
        base = c.lower()
        if base in raw or base.replace("_agent", "") in raw:
            return c
    # Last resort: scan entire text.
    low = stripped.lower()
    for c in candidates:
        base = c.lower()
        if base in low or base.replace("_agent", "") in low:
            return c
    if "finish" in low:
        return "FINISH"
    return None


def build_workbench_cmas_graph(domains: list[str], collected_calls: list[str]):
    """Hub-and-spoke graph; appends each worker's tool calls to collected_calls."""
    from langgraph.prebuilt import create_react_agent

    active = _active_worker_specs(domains)
    base_llm = _worker_llm()
    _clear_wb_tool_name_remap()

    worker_reacts: dict[str, Any] = {}
    for node_name, worker_prompt, dlist in active:
        wt = _prepare_tools_for_hosted_api(get_tools_for_domains(dlist))
        llm_w = bind_tools_safely(base_llm, wt)
        worker_reacts[node_name] = create_react_agent(
            llm_w,
            wt,
            prompt=_worker_system_prompt(worker_prompt),
        )

    def _run_worker(node_name: str, task: str, tag: str) -> dict:
        react = worker_reacts[node_name]
        try:
            result = _invoke_with_rate_limit_retry(
                react.invoke,
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
        return {
            "aggregated_context": [f"[{tag}] {_final_text(messages)}"],
            "total_tokens": _aggregate_tokens(messages),
        }

    descriptions = {
        "email_agent": "Email / inbox operations",
        "calendar_agent": "Calendar events",
        "analytics_agent": "Analytics / plots / traffic",
        "project_management_agent": "Project tasks / boards",
        "crm_agent": "Customer relationship records",
    }

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
            response = _invoke_with_rate_limit_retry(base_llm.invoke, prompt)
            return {
                "next_action": "FINISH",
                "final_synthesis": _message_content_as_str(response.content),
                "total_tokens": _tokens_from_message(response),
                "execution_path": ["FINISH"],
            }

        avail_names = [w[0] for w in avail]
        avail_desc = "\n".join(f"- {w[0]}: {descriptions.get(w[0], w[0])}" for w in avail)
        dispatch_prompt = (
            "You are a supervisor coordinating specialized workers.\n\n"
            f"Task: {task}\n\nData collected so far:\n{context if context else 'Nothing yet.'}\n\n"
            f"Available workers:\n{avail_desc}\n\n"
            "Dispatch exactly one worker, or finish. Respond on two lines with no markdown fences:\n"
            "REASONING: <one-sentence rationale>\n"
            "ACTION: <one of: "
            + ", ".join(avail_names + ["FINISH"])
            + ">"
        )
        response = _invoke_with_rate_limit_retry(base_llm.invoke, dispatch_prompt)
        tokens = _tokens_from_message(response)
        text = _message_content_as_str(response.content)

        chosen = _parse_action_line(text, avail_names)

        if chosen == "FINISH" or chosen is None:
            synth = _invoke_with_rate_limit_retry(
                base_llm.invoke,
                f"Task: {task}\n\nAggregated Data:\n{context}\n\nSynthesize a clear final answer.",
            )
            return {
                "next_action": "FINISH",
                "final_synthesis": _message_content_as_str(synth.content),
                "total_tokens": tokens + _tokens_from_message(synth),
                "execution_path": ["FINISH"],
            }

        return {
            "next_action": chosen,
            "total_tokens": tokens,
            "execution_path": [chosen],
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
