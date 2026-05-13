"""
Fin-RATE-style QA over a local JSONL ``corpus`` (SEC-style chunks).

Three architectures (scaffold, offline-friendly):
  * **SAS** — single string context from keyword retrieval over all chunks.
  * **CMAS** — split context by company/year keys when present, then merge text.
  * **DMAS** — two parallel retrieval queries (full + shortened) merged.

For official converted Fin-RATE runs, convert upstream ``qa/*.json`` and
``corpus/corpus.jsonl`` into this runner's JSONL schema before invoking
``run_finrate_sweep.py``. This scaffold does not download official data or emit
the official leaderboard format.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any


FINRATE_ALLOWED_JUDGE_SCORES = (0.0, 0.5, 1.0)


def _tokenize(q: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2}


def load_corpus_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _score_chunk(question: str, chunk_text: str) -> float:
    qt = _tokenize(question)
    low = chunk_text.lower()
    return sum(1 for t in qt if t in low)


def _chunk_doc_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("doc_id") or chunk.get("id") or chunk.get("_id") or "")


def retrieve_context(
    question: str,
    chunks: list[dict[str, Any]],
    top_n: int = 4,
    doc_ids: list[str] | None = None,
) -> str:
    if doc_ids:
        wanted = set(doc_ids)
        parts = [
            str(c.get("chunk") or c.get("text") or "")
            for c in chunks
            if _chunk_doc_id(c) in wanted and str(c.get("chunk") or c.get("text") or "")
        ]
        if parts:
            return "\n\n".join(parts)

    scored: list[tuple[float, str]] = []
    for c in chunks:
        text = str(c.get("chunk") or c.get("text") or "")
        if not text:
            continue
        s = _score_chunk(question, text)
        if s > 0:
            scored.append((float(s), text))
    scored.sort(key=lambda x: -x[0])
    parts = [t for _, t in scored[:top_n]]
    return "\n\n".join(parts) if parts else ""


def run_finrate_sas_offline(
    question: str,
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = retrieve_context(question, chunks, 5, doc_ids=doc_ids)
    return {
        "answer": ctx[:6000] or "NO_CONTEXT",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
        "execution_path": ["sas_retrieve", "finish"],
        "trace": {
            "mode": "sas_offline",
            "query": question[:400],
            "doc_ids": doc_ids or [],
            "context_preview": (ctx[:500] if ctx else ""),
        },
    }


def run_finrate_cmas_offline(
    question: str,
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if doc_ids:
        selected = [c for c in chunks if _chunk_doc_id(c) in set(doc_ids)]
        if selected:
            chunks = selected
    by_company: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        co = str(c.get("company") or "unknown")
        by_company.setdefault(co, []).append(c)
    parts: list[str] = []
    for co, sub in by_company.items():
        ctx = retrieve_context(question, sub, 3)
        if ctx:
            parts.append(f"=== {co} ===\n{ctx}")
    merged = "\n\n".join(parts) if parts else retrieve_context(question, chunks, 5)
    return {
        "answer": merged[:6000] or "NO_CONTEXT",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
        "execution_path": ["cmas_partition", "cmas_merge", "finish"],
        "trace": {
            "mode": "cmas_offline",
            "doc_ids": doc_ids or [],
            "dispatches": [
                {"worker": "company_partition_retriever", "company": co, "context_preview": retrieve_context(question, sub, 1)[:250]}
                for co, sub in list(by_company.items())[:6]
            ],
            "merge_strategy": "company_bucket_merge",
        },
    }


def run_finrate_dmas_offline(
    question: str,
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    q_short = " ".join(question.split()[:8])

    def a(q: str) -> str:
        return retrieve_context(q, chunks, 4, doc_ids=doc_ids)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(a, question)
        f2 = pool.submit(a, q_short)
        c1, c2 = f1.result(), f2.result()
    merged = f"{c1}\n\n---PEER---\n\n{c2}"
    return {
        "answer": merged[:6000] or "NO_CONTEXT",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
        "execution_path": ["dmas_peer_full", "dmas_peer_short", "consensus_merge", "finish"],
        "trace": {
            "mode": "dmas_offline",
            "doc_ids": doc_ids or [],
            "peer_reports": [
                {"peer": "peer_full_query", "query": question[:240], "context_preview": c1[:500]},
                {"peer": "peer_short_query", "query": q_short[:240], "context_preview": c2[:500]},
            ],
            "merge_strategy": "parallel_dual_query_merge",
        },
    }


def build_finrate_local_search_tool(
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
):
    from langchain_core.tools import tool

    @tool
    def local_search(
        query: Annotated[str, "Search query over the fixed Fin-RATE corpus."],
        top_k: Annotated[int, "Number of chunks to return"] = 5,
    ) -> str:
        """Retrieve SEC-style chunks from the fixed local Fin-RATE corpus."""
        k = max(1, min(int(top_k or 5), 12))
        ctx = retrieve_context(query, chunks, top_n=k, doc_ids=doc_ids)
        return ctx[:6000] if ctx else "NO_CONTEXT"

    return local_search


def _invoke_finrate_react_agent(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    mode: str,
    system_prompt: str,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    from dynamic_routing.chat_models import bind_tools_safely, get_worker_chat_model
    from dynamic_routing.workbench_runner import _aggregate_tokens, _final_text

    tool = build_finrate_local_search_tool(chunks, doc_ids=doc_ids)
    llm = bind_tools_safely(get_worker_chat_model(temperature=0.1), [tool])
    agent = create_agent(model=llm, tools=[tool], system_prompt=system_prompt)
    t0 = time.perf_counter()
    err = ""
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 24},
        )
    except Exception as e:
        err = str(e)
        return {
            "answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
            "execution_path": [f"{mode}_error"],
            "trace": {"mode": mode, "query": question[:400], "doc_ids": doc_ids or [], "agent_error": err[:300]},
        }
    messages = result.get("messages", [])
    answer = _final_text(messages)
    return {
        "answer": answer,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": _aggregate_tokens(messages),
        "error": err,
        "execution_path": [f"{mode}_tool_loop", "finish"],
        "trace": {
            "mode": mode,
            "query": question[:400],
            "doc_ids": doc_ids or [],
            "message_count": len(messages),
            "final_preview": answer[:500],
        },
    }


def _finrate_llm_merge(prompt: str) -> tuple[str, int]:
    from dynamic_routing.chat_models import get_worker_chat_model
    from dynamic_routing.workbench_runner import _aggregate_tokens

    resp = get_worker_chat_model(temperature=0.0).invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return text, _aggregate_tokens([resp])


def run_finrate_sas_llm(
    question: str,
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    return _invoke_finrate_react_agent(
        question,
        chunks,
        mode="sas_llm",
        doc_ids=doc_ids,
        system_prompt=(
            "You are a financial QA agent. Use only local_search over the fixed Fin-RATE corpus. "
            "Extract relevant entities, values, dates, and comparison direction. Cite supporting evidence briefly."
        ),
    )


def run_finrate_cmas_llm(
    question: str,
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        evidence = _invoke_finrate_react_agent(
            question,
            chunks,
            mode="cmas_llm_evidence_worker",
            doc_ids=doc_ids,
            system_prompt=(
                "You are the evidence extraction worker in a centralized financial MAS. "
                "Use local_search only. Extract all relevant numeric values, entities, dates, and source clues."
            ),
        )
        reasoning = _invoke_finrate_react_agent(
            question,
            chunks,
            mode="cmas_llm_reasoning_worker",
            doc_ids=doc_ids,
            system_prompt=(
                "You are the financial reasoning worker in a centralized MAS. "
                "Use local_search only. Focus on calculations, comparisons, trends, and missing caveats."
            ),
        )
        merged, merge_tokens = _finrate_llm_merge(
            "You are the CMAS supervisor for a Fin-RATE task. Produce the final answer using only the worker reports. "
            "Preserve key numbers, entities, years, and comparison direction.\n\n"
            f"Question: {question}\n\nEvidence worker:\n{evidence.get('answer', '')[:3000]}\n\n"
            f"Reasoning worker:\n{reasoning.get('answer', '')[:3000]}\n\nFinal answer:"
        )
        total_tokens = int(evidence.get("total_tokens") or 0) + int(reasoning.get("total_tokens") or 0) + merge_tokens
    except Exception as e:
        err = str(e)
        return {
            "answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
            "execution_path": ["cmas_llm_error"],
            "trace": {"mode": "cmas_llm", "query": question[:400], "doc_ids": doc_ids or [], "agent_error": err[:300]},
        }
    return {
        "answer": merged,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": total_tokens,
        "error": "",
        "execution_path": ["cmas_evidence_worker", "cmas_reasoning_worker", "cmas_supervisor_merge"],
        "trace": {
            "mode": "cmas_llm",
            "doc_ids": doc_ids or [],
            "dispatches": [
                {"worker": "evidence_extractor", "final_preview": str(evidence.get("answer") or "")[:500]},
                {"worker": "financial_reasoner", "final_preview": str(reasoning.get("answer") or "")[:500]},
            ],
            "merge_strategy": "llm_supervisor_synthesis",
            "final_preview": merged[:500],
        },
    }


def run_finrate_dmas_llm(
    question: str,
    chunks: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    q_short = " ".join(question.split()[:10])
    peer_specs = [
        ("peer_full_question", question, "Analyze the full question and cite evidence."),
        ("peer_short_query", q_short, "Use a shortened retrieval plan, then reason from the evidence."),
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _invoke_finrate_react_agent,
                    peer_query,
                    chunks,
                    mode=f"dmas_llm_{peer}",
                    doc_ids=doc_ids,
                    system_prompt=(
                        "You are an independent peer in a decentralized financial MAS. "
                        f"{guidance} Use only local_search over the fixed corpus."
                    ),
                )
                for peer, peer_query, guidance in peer_specs
            ]
            reports = [f.result() for f in futures]
        merged, merge_tokens = _finrate_llm_merge(
            "You are the DMAS consensus merger for a Fin-RATE task. Compare the peer reports, resolve conflicts, "
            "and produce one final answer with the key entities, numbers, dates, and comparison direction.\n\n"
            f"Question: {question}\n\nPeer A:\n{reports[0].get('answer', '')[:3000]}\n\n"
            f"Peer B:\n{reports[1].get('answer', '')[:3000]}\n\nFinal answer:"
        )
        total_tokens = sum(int(r.get("total_tokens") or 0) for r in reports) + merge_tokens
    except Exception as e:
        err = str(e)
        return {
            "answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
            "execution_path": ["dmas_llm_error"],
            "trace": {"mode": "dmas_llm", "query": question[:400], "doc_ids": doc_ids or [], "agent_error": err[:300]},
        }
    return {
        "answer": merged,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": total_tokens,
        "error": "",
        "execution_path": ["dmas_peer_full", "dmas_peer_short", "dmas_consensus_merge"],
        "trace": {
            "mode": "dmas_llm",
            "doc_ids": doc_ids or [],
            "peer_reports": [
                {"peer": peer_specs[i][0], "query": peer_specs[i][1][:240], "final_preview": str(reports[i].get("answer") or "")[:500]}
                for i in range(len(reports))
            ],
            "merge_strategy": "llm_consensus_synthesis",
            "final_preview": merged[:500],
        },
    }


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


def _coerce_judge_score(value: Any) -> float | None:
    try:
        raw = float(value)
    except Exception:
        return None
    return min(FINRATE_ALLOWED_JUDGE_SCORES, key=lambda allowed: abs(allowed - raw))


def _finrate_gpt_judge_model() -> str:
    return os.environ.get("FINRATE_GPT_JUDGE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def _legacy_gpt_enabled() -> bool:
    return os.environ.get("FINRATE_USE_GPT_JUDGE", "").strip().lower() in ("1", "true", "yes")


def _finrate_judge_backend() -> str:
    raw = os.environ.get("FINRATE_JUDGE_BACKEND", "").strip().lower()
    if raw in ("local", "gpt", "auto"):
        return raw
    if _legacy_gpt_enabled():
        return "gpt"
    return "local"


def judge_finrate_local(answer: str, gold: str) -> float:
    if not gold.strip():
        return 1.0
    g = set(re.findall(r"[a-z0-9]+", gold.lower()))
    a = set(re.findall(r"[a-z0-9]+", answer.lower()))
    if not g:
        return 0.0
    recall = len(g & a) / len(g)
    bonus = 0.3 if gold.lower()[:80] in answer.lower() else 0.0
    return max(0.0, min(1.0, recall + bonus))


def judge_finrate_local_detail(question: str, answer: str, gold: str) -> dict[str, Any]:
    raw_score = judge_finrate_local(answer, gold)
    score = _coerce_judge_score(raw_score)
    if score is None:
        score = 0.0
    return {
        "score": score,
        "reason": f"Local lexical overlap score {raw_score:.3f} rounded to nearest allowed value.",
        "judge_backend": "local",
        "judge_model": "local_overlap",
    }


def judge_finrate_gpt_detail(question: str, answer: str, gold: str) -> dict[str, Any] | None:
    if not os.environ.get("OPENAI_API_KEY"):
        logging.warning("Fin-RATE GPT judge unavailable: OPENAI_API_KEY is not set")
        return None
    try:
        from langchain_openai import ChatOpenAI

        model = _finrate_gpt_judge_model()
        llm = ChatOpenAI(model=model, temperature=0.0, timeout=60)
        prompt = (
            "You are grading a Fin-RATE-style financial QA answer.\n"
            "Return strict JSON only, no markdown, with schema:\n"
            '{"score": 0.0|0.5|1.0, "reason": "<brief reason>"}\n\n'
            "Scoring rules:\n"
            "- 1.0: candidate fully answers the reference, including key entities, values, dates, and comparison direction.\n"
            "- 0.5: candidate is partially correct but misses or garbles a material part.\n"
            "- 0.0: candidate is wrong, unsupported, or does not answer.\n\n"
            f"Question: {question[:1000]}\n"
            f"Reference answer: {gold[:1000]}\n"
            f"Candidate answer: {answer[:4000]}\n"
        )
        resp = llm.invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        obj = _extract_json_object(text) or {}
        score = _coerce_judge_score(obj.get("score"))
        if score is None:
            raise ValueError("missing or invalid score")
        return {
            "score": score,
            "reason": str(obj.get("reason") or "GPT judge returned a valid score.")[:600],
            "judge_backend": "gpt",
            "judge_model": model,
        }
    except Exception as e:
        logging.warning("Fin-RATE GPT judge failed: %s", str(e)[:120])
    return None


def judge_finrate_gpt(question: str, answer: str, gold: str) -> float | None:
    detail = judge_finrate_gpt_detail(question, answer, gold)
    return None if detail is None else float(detail["score"])


def judge_finrate_answer_detail(question: str, answer: str, gold: str) -> dict[str, Any]:
    backend = _finrate_judge_backend()
    if backend == "local":
        return judge_finrate_local_detail(question, answer, gold)

    local_detail = judge_finrate_local_detail(question, answer, gold)
    if backend == "auto" and float(local_detail["score"]) in (0.0, 1.0):
        return local_detail

    gpt_detail = judge_finrate_gpt_detail(question, answer, gold)
    if gpt_detail is not None:
        return gpt_detail

    fallback = dict(local_detail)
    fallback["judge_backend"] = f"{backend}_fallback_local"
    fallback["reason"] = (
        "GPT judge unavailable or malformed; "
        + str(fallback.get("reason") or "used local fallback")
    )[:600]
    return fallback


def judge_finrate_answer(question: str, answer: str, gold: str) -> tuple[float, str]:
    detail = judge_finrate_answer_detail(question, answer, gold)
    return float(detail["score"]), str(detail["judge_backend"])


def run_finrate_architecture(
    question: str,
    chunks: list[dict[str, Any]],
    architecture: str,
    doc_ids: list[str] | None = None,
    *,
    use_llm: bool = False,
) -> dict[str, Any]:
    a = architecture.strip().lower()
    if use_llm and a in ("sas", "single-agent system"):
        return run_finrate_sas_llm(question, chunks, doc_ids=doc_ids)
    if use_llm and a in ("cmas", "centralized mas"):
        return run_finrate_cmas_llm(question, chunks, doc_ids=doc_ids)
    if use_llm and a in ("dmas", "decentralized mas"):
        return run_finrate_dmas_llm(question, chunks, doc_ids=doc_ids)
    if a in ("sas", "single-agent system"):
        return run_finrate_sas_offline(question, chunks, doc_ids=doc_ids)
    if a in ("cmas", "centralized mas"):
        return run_finrate_cmas_offline(question, chunks, doc_ids=doc_ids)
    if a in ("dmas", "decentralized mas"):
        return run_finrate_dmas_offline(question, chunks, doc_ids=doc_ids)
    raise ValueError(architecture)
