"""
vLLM Integration for the Dynamic Router.

Connects LangGraph to a local Mistral-7B model running via vLLM.
Forces the LLM to output structured JSON matching our quantitative routing thresholds.

Requires: vLLM server running with `python -m vllm.entrypoints.openai.api_server`
          --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

# --- 1. Strict Output Schema ---


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


# --- 2. Keyword Fallback (when vLLM unavailable) ---


def _keyword_fallback(query: str) -> dict[str, Any]:
    """Rule-based fallback when LLM is unavailable. Preserves test behavior."""
    q = query.lower()
    sources = sum(1 for s in ("calendar", "maps", "drive") if s in q)

    if "simultaneously" in q or "aggregate" in q or sources >= 2:
        return {
            "estimated_sequential_depth": 2,
            "parallelization_factor": 0.9,
            "estimated_tool_count": 5,
        }
    if "step-by-step" in q or "complex workflow" in q:
        return {
            "estimated_sequential_depth": 8,
            "parallelization_factor": 0.1,
            "estimated_tool_count": 14,
        }
    return {
        "estimated_sequential_depth": 3,
        "parallelization_factor": 0.4,
        "estimated_tool_count": 3,
    }


# --- 3. LLM-based Classification ---


def _get_structured_llm():
    """Lazy init: only load LangChain + vLLM client when actually needed."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # vLLM serves OpenAI-compatible API on port 8000 by default
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    llm = ChatOpenAI(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url,
        max_tokens=150,
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(RoutingMetadata)

    system_prompt = """
You are an expert architectural routing classifier for an agentic orchestration system.
Your job is to analyze incoming user queries and estimate their structural complexity.

You must estimate three properties:

1. Sequential Depth (1–20): How many steps absolutely depend on the output of the previous step?
   Example: "Find my next meeting, then map the commute" = Depth ~2.
   Example: "Execute step-by-step using all available tools" = Depth ~8.

2. Parallelization Factor (0.0–1.0): Can this task be split into independent branches?
   Example: "Summarize my Calendar and summarize my Drive" = High parallelization, ~0.8.
   Example: "Aggregate Calendar and Maps simultaneously" = ~0.9.

3. Tool Count (1–50): How many different data sources or APIs will this touch?
   Example: "Cross-reference calendar, maps, drive notes" = ~5–8.

Output the structured JSON with these three fields.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Query: {query}"),
    ])
    return prompt_template | structured_llm


def predict_routing_metadata(query: str) -> dict[str, Any]:
    """
    Invokes the local Mistral-7B model via vLLM to classify the query.
    Returns a dict with estimated_sequential_depth, parallelization_factor, estimated_tool_count.

    Falls back to keyword-based heuristics if vLLM is unavailable.
    Set USE_LLM_ROUTER=false to force keyword mode (e.g. for CI without vLLM).
    """
    if os.environ.get("USE_LLM_ROUTER", "true").lower() in ("false", "0", "no"):
        return _keyword_fallback(query)

    try:
        chain = _get_structured_llm()
        result = chain.invoke({"query": query})

        if isinstance(result, RoutingMetadata):
            return {
                "estimated_sequential_depth": result.estimated_sequential_depth,
                "parallelization_factor": round(result.parallelization_factor, 2),
                "estimated_tool_count": result.estimated_tool_count,
            }
        # In case with_structured_output returns a dict
        return {
            "estimated_sequential_depth": int(result.get("estimated_sequential_depth", 3)),
            "parallelization_factor": round(float(result.get("parallelization_factor", 0.5)), 2),
            "estimated_tool_count": int(result.get("estimated_tool_count", 3)),
        }
    except Exception as e:
        print(f"-> [WARNING] LLM classification failed, using keyword fallback. Error: {e}")
        return _keyword_fallback(query)


# --- Example Execution ---

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
