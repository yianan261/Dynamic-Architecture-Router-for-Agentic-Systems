"""
Routing-metadata classifier for the Dynamic Architecture Router.

For a user query, returns routing metadata (core threshold features + richer
coordination-tax signals) used by threshold and learned policies.

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

from pydantic import BaseModel, ConfigDict, Field


class RoutingMetadata(BaseModel):
    """
    Routing metadata for architecture selection.

    Goal:
    Represent a query as a compact structured feature vector for:
      1) threshold-based routing rules, and
      2) learned tabular router models.
    """

    model_config = ConfigDict(extra="ignore")

    # A. Task structure
    estimated_sequential_depth: int = Field(
        ...,
        ge=1,
        le=20,
        description=(
            "Estimated number of strictly dependent reasoning/action stages. "
            "Higher means the task must proceed step by step."
        ),
    )
    parallelization_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Degree to which the task decomposes into independent branches. "
            "0.0 = mostly sequential, 1.0 = highly parallelizable."
        ),
    )
    num_subgoals: int = Field(
        default=1,
        ge=1,
        le=50,
        description=(
            "Number of distinct sub-questions, branches, or requested outputs."
        ),
    )
    entity_count: int = Field(
        default=1,
        ge=1,
        le=100,
        description=(
            "Number of independent target entities (companies, docs, people, "
            "services, products, or time periods)."
        ),
    )
    constraint_tightness: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "Strictness of ordering/correctness constraints. "
            "1=loose/open-ended, 5=highly constrained."
        ),
    )
    open_endedness: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "How exploratory or underspecified the task is. "
            "1=highly specific, 5=broad/open-ended."
        ),
    )

    # B. Execution cost
    estimated_tool_count: int = Field(
        ...,
        ge=1,
        le=50,
        description="Estimated number of distinct tools/APIs/backend systems.",
    )
    expected_retrieval_fanout: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Estimated number of separate retrieval/evidence batches.",
    )
    domain_span: int = Field(
        default=1,
        ge=1,
        le=20,
        description=(
            "Estimated number of distinct domains/tool ecosystems involved."
        ),
    )
    expected_context_expansion: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "How much evidence/intermediate state is expected to accumulate. "
            "1=minimal growth, 5=very large growth."
        ),
    )

    # C. Coordination burden
    aggregation_required: bool = Field(
        default=False,
        description="Whether multiple branches/sources must be synthesized.",
    )
    final_synthesis_complexity: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Difficulty of final merge/synthesis across branches.",
    )
    cross_branch_dependency: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "How much branches depend on one another's outputs. "
            "1=nearly independent, 5=tightly interdependent."
        ),
    )
    communication_load_estimate: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "Likely amount of inter-agent coordination/message passing."
        ),
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
            "num_subgoals": max(3, multi_domain),
            "entity_count": max(2, multi_domain),
            "constraint_tightness": 2,
            "open_endedness": 3,
            "aggregation_required": True,
            "expected_retrieval_fanout": max(4, multi_domain + 1),
            "domain_span": max(2, multi_domain),
            "expected_context_expansion": 3,
            "final_synthesis_complexity": 3,
            "cross_branch_dependency": 2,
            "communication_load_estimate": 3,
        }
    if "step-by-step" in q or "complex workflow" in q:
        return {
            "estimated_sequential_depth": 8,
            "parallelization_factor": 0.15,
            "estimated_tool_count": 12,
            "num_subgoals": 4,
            "entity_count": 3,
            "constraint_tightness": 5,
            "open_endedness": 2,
            "aggregation_required": False,
            "expected_retrieval_fanout": 6,
            "domain_span": max(2, multi_domain),
            "expected_context_expansion": 4,
            "final_synthesis_complexity": 2,
            "cross_branch_dependency": 4,
            "communication_load_estimate": 2,
        }
    # Open-ended / exploratory phrasing → high parallelism (routes to DMAS in tests)
    if "pros and cons" in q or "different approaches" in q:
        return {
            "estimated_sequential_depth": 2,
            "parallelization_factor": 0.9,
            "estimated_tool_count": 4,
            "num_subgoals": 2,
            "entity_count": 2,
            "constraint_tightness": 2,
            "open_endedness": 5,
            "aggregation_required": True,
            "expected_retrieval_fanout": 3,
            "domain_span": max(1, multi_domain),
            "expected_context_expansion": 3,
            "final_synthesis_complexity": 4,
            "cross_branch_dependency": 3,
            "communication_load_estimate": 3,
        }
    # Default for typical single-domain WorkBench task: moderate tools, low
    # parallelism — router's threshold logic will route this to SAS.
    return {
        "estimated_sequential_depth": 3,
        "parallelization_factor": 0.3,
        "estimated_tool_count": 4,
        "num_subgoals": 2,
        "entity_count": max(1, multi_domain),
        "constraint_tightness": 3,
        "open_endedness": 2,
        "aggregation_required": False,
        "expected_retrieval_fanout": 2,
        "domain_span": max(1, multi_domain),
        "expected_context_expansion": 2,
        "final_synthesis_complexity": 2,
        "cross_branch_dependency": 2,
        "communication_load_estimate": 2,
    }


# ---------------------------------------------------------------------------
# 2. Structured-output attempt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are an expert architectural routing classifier for an agentic orchestration system.
Your job is to analyze incoming user queries and estimate their structural complexity.

You must estimate the following fields:

1. Sequential Depth (1-20): How many steps absolutely depend on the output of the previous step?
   "Find my next meeting, then map the commute" ~= 2.
   "Execute step-by-step using all available tools" ~= 8.

2. Parallelization Factor (0.0-1.0): Can this task be split into independent branches?
   "Summarize my Calendar and summarize my Drive" ~= 0.8.
   "Aggregate Calendar and Maps simultaneously" ~= 0.9.

3. Tool Count (1-50): How many different data sources or APIs will this touch?
   "Cross-reference calendar, maps, drive notes" ~= 5-8.

4. num_subgoals (1-50): distinct branches/sub-questions.
5. entity_count (1-100): independent entities/time periods involved.
6. constraint_tightness (1-5): 1 loose, 5 strict constraints.
7. open_endedness (1-5): 1 specific, 5 exploratory.
8. expected_retrieval_fanout (1-100): number of evidence batches.
9. domain_span (1-20): number of tool/domain ecosystems involved.
10. expected_context_expansion (1-5): context growth during execution.
11. aggregation_required (bool): whether branch/source synthesis is required.
12. final_synthesis_complexity (1-5): merge complexity at the end.
13. cross_branch_dependency (1-5): branch interdependence.
14. communication_load_estimate (1-5): coordination/message-passing load.

Rubric for cross_branch_dependency:
1 = branches can be solved independently
3 = some shared assumptions or mild dependency
5 = branches heavily depend on one another's outputs
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
            (
                "human",
                "User Query: {query}\n\nReturn JSON with all routing fields "
                "(estimated_sequential_depth, parallelization_factor, estimated_tool_count, "
                "num_subgoals, entity_count, constraint_tightness, open_endedness, "
                "expected_retrieval_fanout, domain_span, expected_context_expansion, "
                "aggregation_required, final_synthesis_complexity, cross_branch_dependency, "
                "communication_load_estimate).",
            ),
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
            + "`estimated_tool_count` (int), `num_subgoals` (int), `entity_count` (int), "
            + "`constraint_tightness` (int 1-5), `open_endedness` (int 1-5), "
            + "`expected_retrieval_fanout` (int), `domain_span` (int), "
            + "`expected_context_expansion` (int 1-5), `aggregation_required` (bool), "
            + "`final_synthesis_complexity` (int 1-5), `cross_branch_dependency` (int 1-5), "
            + "`communication_load_estimate` (int 1-5). "
            + "No code fences, no prose."
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
    try:
        nsg = int(obj.get("num_subgoals", 1))
    except (TypeError, ValueError):
        nsg = 1
    nsg = max(1, min(50, nsg))
    try:
        entity_count = int(obj.get("entity_count", 1))
    except (TypeError, ValueError):
        entity_count = 1
    entity_count = max(1, min(100, entity_count))
    try:
        constraint_tightness = int(obj.get("constraint_tightness", 2))
    except (TypeError, ValueError):
        constraint_tightness = 2
    constraint_tightness = max(1, min(5, constraint_tightness))
    try:
        open_endedness = int(obj.get("open_endedness", 2))
    except (TypeError, ValueError):
        open_endedness = 2
    open_endedness = max(1, min(5, open_endedness))
    try:
        fan = int(obj.get("expected_retrieval_fanout", 1))
    except (TypeError, ValueError):
        fan = 1
    fan = max(1, min(100, fan))
    try:
        domain_span = int(obj.get("domain_span", 1))
    except (TypeError, ValueError):
        domain_span = 1
    domain_span = max(1, min(20, domain_span))
    try:
        expected_context_expansion = int(obj.get("expected_context_expansion", 2))
    except (TypeError, ValueError):
        expected_context_expansion = 2
    expected_context_expansion = max(1, min(5, expected_context_expansion))
    agg = bool(obj.get("aggregation_required", False))
    try:
        final_synthesis_complexity = int(obj.get("final_synthesis_complexity", 2))
    except (TypeError, ValueError):
        final_synthesis_complexity = 2
    final_synthesis_complexity = max(1, min(5, final_synthesis_complexity))
    try:
        cross_branch_dependency = int(obj.get("cross_branch_dependency", 1))
    except (TypeError, ValueError):
        cross_branch_dependency = 1
    cross_branch_dependency = max(1, min(5, cross_branch_dependency))
    try:
        communication_load_estimate = int(obj.get("communication_load_estimate", 1))
    except (TypeError, ValueError):
        communication_load_estimate = 1
    communication_load_estimate = max(1, min(5, communication_load_estimate))
    return {
        "estimated_sequential_depth": depth,
        "parallelization_factor": round(parallel, 2),
        "estimated_tool_count": tools,
        "num_subgoals": nsg,
        "entity_count": entity_count,
        "constraint_tightness": constraint_tightness,
        "open_endedness": open_endedness,
        "aggregation_required": agg,
        "expected_retrieval_fanout": fan,
        "domain_span": domain_span,
        "expected_context_expansion": expected_context_expansion,
        "final_synthesis_complexity": final_synthesis_complexity,
        "cross_branch_dependency": cross_branch_dependency,
        "communication_load_estimate": communication_load_estimate,
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
