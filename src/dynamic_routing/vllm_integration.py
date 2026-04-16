"""
Routing-metadata classifier for the Dynamic Architecture Router.

For a user query, returns three estimates (sequential depth, parallelization
factor, tool count) that the router's thresholds use to pick SAS / CMAS / DMAS.

Backends (resolved by ``dynamic_routing.chat_models.get_router_chat_model``):
  * ``vllm``  — OpenAI-compatible vLLM (default)
  * ``openai`` — API (e.g. gpt-5.4-mini)
  * ``google`` — Gemini

Robustness
----------
Frontier models (gpt-5.4-mini, gemini-3.1-flash-lite-preview) sometimes refuse
``with_structured_output`` or wrap JSON in markdown fences. This module tries,
in order:

1. ``with_structured_output(RoutingMetadata)`` — preferred, strict.
2. Raw prompt → parse first JSON object in the response (fence-tolerant).
3. Keyword heuristics (never raises).

The function always returns a valid metadata dict; it never returns ``None``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field


class RoutingMetadata(BaseModel):
    """Metadata required by the dynamic router's quantitative thresholds."""

    estimated_sequential_depth: int = Field(
        description="Number of highly dependent, step-by-step reasoning blocks. (e.g., 1 to 10)",
        ge=1,
        le=20,
    )
    parallelization_factor: float = Field(
        description="Degree to which the prompt has orthogonal, separable sub-goals. Scale 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
    estimated_tool_count: int = Field(
        description="Estimated number of distinct API tools required to fulfill the objective.",
        ge=1,
        le=50,
    )


# ---------------------------------------------------------------------------
# 1. Keyword fallback (never raises)
# ---------------------------------------------------------------------------


def _keyword_fallback(query: str) -> dict[str, Any]:
    """Rule-based fallback. Tuned so WorkBench-style single-domain tasks get
    a moderate profile (bias toward SAS rather than the no-op DMAS branch)."""
    q = query.lower()

    multi_domain = sum(
        1
        for kw in (
            "email",
            "calendar",
            "analytics",
            "project",
            "task",
            "crm",
            "customer",
            "drive",
            "maps",
            "commute",
        )
        if kw in q
    )

    if "simultaneously" in q or "aggregate" in q or multi_domain >= 3:
        return {
            "estimated_sequential_depth": 2,
            "parallelization_factor": 0.85,
            "estimated_tool_count": 6,
        }
    if "step-by-step" in q or "complex workflow" in q:
        return {
            "estimated_sequential_depth": 8,
            "parallelization_factor": 0.15,
            "estimated_tool_count": 12,
        }
    # Default for typical single-domain WorkBench task: moderate tools, low
    # parallelism — router's threshold logic will route this to SAS.
    return {
        "estimated_sequential_depth": 3,
        "parallelization_factor": 0.3,
        "estimated_tool_count": 4,
    }


# ---------------------------------------------------------------------------
# 2. Structured-output attempt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are an expert architectural routing classifier for an agentic orchestration system.
Your job is to analyze incoming user queries and estimate their structural complexity.

You must estimate three properties:

1. Sequential Depth (1-20): How many steps absolutely depend on the output of the previous step?
   "Find my next meeting, then map the commute" ~= 2.
   "Execute step-by-step using all available tools" ~= 8.

2. Parallelization Factor (0.0-1.0): Can this task be split into independent branches?
   "Summarize my Calendar and summarize my Drive" ~= 0.8.
   "Aggregate Calendar and Maps simultaneously" ~= 0.9.

3. Tool Count (1-50): How many different data sources or APIs will this touch?
   "Cross-reference calendar, maps, drive notes" ~= 5-8.
"""


def _structured_call(query: str) -> dict[str, Any] | None:
    """Preferred path: LangChain structured output."""
    try:
        from langchain_core.prompts import ChatPromptTemplate

        from dynamic_routing.chat_models import get_router_chat_model

        llm = get_router_chat_model(max_tokens=200, temperature=0.1)
        structured_llm = llm.with_structured_output(RoutingMetadata)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "User Query: {query}\n\nReturn JSON with the three fields only."),
        ])
        result = (prompt | structured_llm).invoke({"query": query})
    except Exception as e:
        logging.warning("[router] structured_output failed: %s", str(e)[:160])
        return None

    if isinstance(result, RoutingMetadata):
        return _normalize(result.model_dump())
    if isinstance(result, dict):
        return _normalize(result)
    dumped = getattr(result, "model_dump", lambda: None)()
    return _normalize(dumped) if dumped else None


_JSON_RE = re.compile(r"\{[^{}]*?\}", re.DOTALL)


def _raw_text_call(query: str) -> dict[str, Any] | None:
    """Fallback: plain chat call, extract first JSON object."""
    try:
        from dynamic_routing.chat_models import get_router_chat_model

        llm = get_router_chat_model(max_tokens=200, temperature=0.1)
        user_msg = (
            _SYSTEM_PROMPT
            + f"\n\nUser Query: {query}\n\n"
            + "Return ONLY a single JSON object with keys "
            + "`estimated_sequential_depth` (int), `parallelization_factor` (float 0-1), "
            + "`estimated_tool_count` (int). No code fences, no prose."
        )
        resp = llm.invoke(user_msg)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        logging.warning("[router] raw_text call failed: %s", str(e)[:160])
        return None

    for match in _JSON_RE.findall(text):
        try:
            obj = json.loads(match)
        except Exception:
            continue
        norm = _normalize(obj)
        if norm is not None:
            return norm
    logging.warning("[router] raw_text response had no parseable JSON: %r", text[:200])
    return None


def _normalize(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    try:
        depth = int(obj.get("estimated_sequential_depth", 3))
        parallel = float(obj.get("parallelization_factor", 0.4))
        tools = int(obj.get("estimated_tool_count", 3))
    except (TypeError, ValueError):
        return None
    depth = max(1, min(20, depth))
    parallel = max(0.0, min(1.0, parallel))
    tools = max(1, min(50, tools))
    return {
        "estimated_sequential_depth": depth,
        "parallelization_factor": round(parallel, 2),
        "estimated_tool_count": tools,
    }


# ---------------------------------------------------------------------------
# 3. Public entry point (never raises, never returns None)
# ---------------------------------------------------------------------------


def predict_routing_metadata(query: str) -> dict[str, Any]:
    """Return routing metadata for ``query``. Falls through to keyword heuristics
    on any failure. Set ``USE_LLM_ROUTER=false`` to skip LLM entirely."""
    if os.environ.get("USE_LLM_ROUTER", "true").lower() in ("false", "0", "no"):
        return _keyword_fallback(query)

    result = _structured_call(query)
    if result is not None:
        return result

    result = _raw_text_call(query)
    if result is not None:
        return result

    logging.info("[router] falling back to keyword heuristics for: %r", query[:80])
    return _keyword_fallback(query)


if __name__ == "__main__":
    q1 = (
        "Find a 45-minute meeting slot with my advisor next week, then choose the best "
        "coffee shop nearby based on commute time, then generate a suggested schedule."
    )
    print(f"\nQuery: {q1}")
    print(json.dumps(predict_routing_metadata(q1), indent=2))

    q2 = (
        "Prepare my Monday morning briefing using calendar, tasks, important documents, "
        "commute, and reminders."
    )
    print(f"\nQuery: {q2}")
    print(json.dumps(predict_routing_metadata(q2), indent=2))
