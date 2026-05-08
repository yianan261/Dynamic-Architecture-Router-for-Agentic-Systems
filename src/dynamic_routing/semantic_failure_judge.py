from __future__ import annotations

import json
import os
import re
from typing import Any

SEMANTIC_LABELS = (
    "Specification Failure: Requirement Miss",
    "Inter-agent Misalignment: Wrong Assumption Propagation",
    "Task Verification Failure: Premature Termination",
    "Task Verification Failure: Incorrect Synthesis / Incomplete Aggregation",
    "Inter-agent Misalignment: Consensus Failure",
    "Inter-agent Misalignment: Peer Propagation Drift",
    "Task Verification Failure: Unresolved Conflict at Termination",
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _heuristic_label(
    architecture: str,
    error_type: str,
    runtime_taxonomy: str,
    execution_path: list[str],
    trace_data: dict[str, Any],
    answer: str,
) -> str:
    arch = architecture.lower()
    err = (error_type or "").lower()
    tax = (runtime_taxonomy or "").lower()
    path = " ".join(execution_path).lower()
    trace_s = str(trace_data).lower()
    ans = (answer or "").strip().lower()

    if "cyclic repetition" in tax or "oscillating dispatch" in tax or "loop" in tax:
        return "Inter-agent Misalignment: Consensus Failure" if "dmas" in arch else "Task Verification Failure: Premature Termination"
    if "merge failed" in err or "consensus_merge" in path and ("error" in trace_s or "error" in err):
        return "Task Verification Failure: Incorrect Synthesis / Incomplete Aggregation"
    if "contradict" in trace_s or "conflict" in trace_s:
        return "Task Verification Failure: Unresolved Conflict at Termination"
    if "peer" in trace_s and "drift" in trace_s:
        return "Inter-agent Misalignment: Peer Propagation Drift"
    if not ans or ans in ("no_context", "no_evidence"):
        return "Specification Failure: Requirement Miss"
    if "finish" in path and len(execution_path) <= 1:
        return "Task Verification Failure: Premature Termination"
    return "Specification Failure: Requirement Miss"


def _normalize_label(raw: str) -> str:
    r = (raw or "").strip()
    if r in SEMANTIC_LABELS:
        return r
    low = r.lower()
    for lbl in SEMANTIC_LABELS:
        if lbl.lower() in low or low in lbl.lower():
            return lbl
    return ""


def judge_semantic_failure(
    *,
    architecture: str,
    task: str,
    gold_answer: str,
    final_answer: str,
    verification_score: float,
    verification_backend: str,
    error_type: str = "",
    runtime_taxonomy: str = "",
    execution_path: list[str] | None = None,
    trace_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assign a semantic failure label on *failed* verification cases.

    Returns:
      {
        "label": <one of SEMANTIC_LABELS>,
        "confidence": float,
        "reason": str,
        "evidence": list[str],
        "judge_backend": "llm_structured" | "heuristic_fallback",
      }
    """
    exec_path = execution_path or []
    trace = trace_data or {}
    heur = _heuristic_label(architecture, error_type, runtime_taxonomy, exec_path, trace, final_answer)

    # Prefer a cheap model for failure-only classification.
    try:
        llm = None
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=os.environ.get("SEMANTIC_JUDGE_MODEL", "gpt-4o-mini"),
                    temperature=0.0,
                    timeout=60,
                )
            except Exception:
                llm = None
        if llm is None:
            from dynamic_routing.chat_models import get_worker_chat_model

            llm = get_worker_chat_model(temperature=0.0)
        prompt = (
            "You are labeling multi-agent failure modes for a benchmark run.\n"
            "Pick exactly one label from this closed set:\n"
            f"{json.dumps(list(SEMANTIC_LABELS), ensure_ascii=True)}\n\n"
            "Return strict JSON only, no markdown, using schema:\n"
            '{"label": "<one label>", "confidence": <0..1>, "reason": "<brief>", "evidence": ["...", "..."]}\n\n'
            "Guidance:\n"
            "- Requirement Miss: final answer misses required parts.\n"
            "- Wrong Assumption Propagation: an unsupported assumption spreads across agents.\n"
            "- Premature Termination: system stops before enough work/evidence.\n"
            "- Incorrect Synthesis / Incomplete Aggregation: branch data exists but final merge is wrong/incomplete.\n"
            "- Consensus Failure: decentralized peers fail to converge.\n"
            "- Peer Propagation Drift: incorrect partial result spreads laterally.\n"
            "- Unresolved Conflict at Termination: run ends with conflicting branch outputs.\n\n"
            f"Architecture: {architecture}\n"
            f"Task: {task[:1200]}\n"
            f"Gold answer: {gold_answer[:1200]}\n"
            f"Final answer: {final_answer[:1600]}\n"
            f"Verification score: {verification_score:.4f} (backend={verification_backend})\n"
            f"Error: {error_type[:400]}\n"
            f"Runtime taxonomy: {runtime_taxonomy[:400]}\n"
            f"Execution path: {exec_path[:40]}\n"
            f"Trace data: {str(trace)[:3000]}\n"
            f"If uncertain, use this fallback label: {heur}\n"
        )
        resp = llm.invoke(prompt)
        txt = resp.content if isinstance(resp.content, str) else str(resp.content)
        obj = _extract_json_object(txt) or {}
        lbl = _normalize_label(str(obj.get("label") or ""))
        if not lbl:
            raise ValueError("invalid label")
        conf = float(obj.get("confidence") or 0.5)
        conf = max(0.0, min(1.0, conf))
        reason = str(obj.get("reason") or "")[:500]
        ev = obj.get("evidence")
        if not isinstance(ev, list):
            ev = []
        evidence = [str(x)[:200] for x in ev[:5]]
        return {
            "label": lbl,
            "confidence": conf,
            "reason": reason,
            "evidence": evidence,
            "judge_backend": "llm_structured",
            "model": os.environ.get("SEMANTIC_JUDGE_MODEL", "gpt-4o-mini"),
        }
    except Exception:
        return {
            "label": heur,
            "confidence": 0.35,
            "reason": "Heuristic fallback: LLM structured semantic judge unavailable or malformed output.",
            "evidence": [runtime_taxonomy[:160], error_type[:160]],
            "judge_backend": "heuristic_fallback",
            "model": os.environ.get("SEMANTIC_JUDGE_MODEL", "gpt-4o-mini"),
        }
