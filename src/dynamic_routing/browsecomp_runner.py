"""
BrowseComp-Plus-style sweep helpers: three architectures over a local corpus.

* **SAS** — single agent (ReAct with ``local_search`` when ``use_llm``; else
  deterministic top-k context string).
* **CMAS** — sequential double-query retrieval + merge (no peer parallelism).
* **DMAS** — two parallel retrieval branches (thread) + merge.

Hybrid judge: ``BROWSECOMP_JUDGE_BACKEND=qwen`` attempts a local OpenAI-compatible
completion (``BROWSECOMP_QWEN_URL``); otherwise ``gpt`` uses the worker LLM;
else ``local`` uses token overlap vs. gold.
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dynamic_routing.browsecomp_env import BrowseCompCorpus
from dynamic_routing.browsecomp_tools import build_local_search_tool


def _retrieve_context(corpus: BrowseCompCorpus, query: str, top_k: int = 5) -> str:
    hits = corpus.search(query, top_k=top_k)
    if not hits:
        return ""
    return "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)


def run_browsecomp_sas_offline(query: str, corpus: BrowseCompCorpus) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = _retrieve_context(corpus, query, 5)
    answer = ctx[:4000] if ctx else "NO_EVIDENCE"
    return {
        "final_answer": answer,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
    }


def run_browsecomp_cmas_offline(query: str, corpus: BrowseCompCorpus) -> dict[str, Any]:
    t0 = time.perf_counter()
    q2 = " ".join(query.split()[: max(3, len(query.split()) // 2)])
    a1 = _retrieve_context(corpus, query, 4)
    a2 = _retrieve_context(corpus, q2, 4)
    merged = f"{a1}\n\n---SECOND_PASS---\n\n{a2}" if a2 != a1 else a1
    return {
        "final_answer": merged[:4000] if merged else "NO_EVIDENCE",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
    }


def run_browsecomp_dmas_offline(query: str, corpus: BrowseCompCorpus) -> dict[str, Any]:
    t0 = time.perf_counter()

    def branch(q: str) -> str:
        return _retrieve_context(corpus, q, 4)

    q_alt = re.sub(r"\s+", " ", " ".join(reversed(query.split()[:12])))
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(branch, query)
        f2 = pool.submit(branch, q_alt if q_alt != query else query + " details")
        c1 = f1.result()
        c2 = f2.result()
    merged = f"{c1}\n\n---PEER---\n\n{c2}"
    return {
        "final_answer": merged[:4000] if merged else "NO_EVIDENCE",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
    }


def _local_judge_score(answer: str, gold: str) -> float:
    if not gold.strip():
        return 1.0
    g = set(re.findall(r"[a-z0-9]+", gold.lower()))
    a = set(re.findall(r"[a-z0-9]+", answer.lower()))
    if not g:
        return 0.0
    inter = len(g & a)
    recall = inter / len(g)
    bonus = 0.25 if gold.lower() in answer.lower() else 0.0
    return max(0.0, min(1.0, recall + bonus))


def _llm_judge_score(query: str, answer: str, gold: str) -> float | None:
    try:
        from dynamic_routing.chat_models import get_worker_chat_model

        llm = get_worker_chat_model(temperature=0.0)
        prompt = (
            "You grade whether the candidate answer is correct w.r.t. the reference.\n"
            f"Question: {query}\n\nReference answer: {gold}\n\nCandidate:\n{answer[:3500]}\n\n"
            "Reply with exactly one token: CORRECT or INCORRECT"
        )
        resp = llm.invoke(prompt)
        text = (resp.content if isinstance(resp.content, str) else str(resp.content)).upper()
        if "INCORRECT" in text:
            return 0.0
        if "CORRECT" in text:
            return 1.0
    except Exception as e:
        logging.warning("GPT judge failed: %s", str(e)[:120])
    return None


def _qwen_judge_score(query: str, answer: str, gold: str) -> float | None:
    url = os.environ.get("BROWSECOMP_QWEN_URL", "").strip()
    if not url:
        return None
    try:
        import json as _json
        import urllib.error
        import urllib.request

        content = (
            "Reply with exactly CORRECT or INCORRECT.\nQ: "
            + query[:500]
            + "\nGold: "
            + gold[:500]
            + "\nAns: "
            + answer[:2000]
        )
        payload = _json.dumps(
            {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 8,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8", errors="replace")
        if "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper()[:120]:
            return 1.0
        if "INCORRECT" in raw.upper():
            return 0.0
    except urllib.error.URLError as e:
        logging.warning("Qwen judge HTTP failed: %s", e)
    except Exception as e:
        logging.warning("Qwen judge failed: %s", str(e)[:120])
    return None


def judge_browsecomp_answer(query: str, answer: str, gold: str) -> tuple[float, str]:
    """Return (score 0..1, backend name)."""
    backend = os.environ.get("BROWSECOMP_JUDGE_BACKEND", "auto").strip().lower()
    if backend == "qwen" or (backend == "auto" and os.environ.get("BROWSECOMP_QWEN_URL")):
        s = _qwen_judge_score(query, answer, gold)
        if s is not None:
            return s, "qwen_http"
    if backend in ("gpt", "openai", "auto"):
        s = _llm_judge_score(query, answer, gold)
        if s is not None:
            return s, "gpt_worker"
    sc = _local_judge_score(answer, gold)
    return sc, "local_overlap"


def run_browsecomp_sas_llm(query: str, corpus: BrowseCompCorpus) -> dict[str, Any]:
    from langchain.agents import create_agent

    from dynamic_routing.chat_models import bind_tools_safely, get_worker_chat_model
    from dynamic_routing.workbench_runner import _aggregate_tokens, _final_text

    tool = build_local_search_tool(corpus)
    llm = bind_tools_safely(get_worker_chat_model(temperature=0.1), [tool])
    agent = create_agent(
        model=llm,
        tools=[tool],
        system_prompt=(
            "You answer using only the local_search tool and its passages. "
            "Cite doc ids briefly then state the final answer in one sentence."
        ),
    )
    t0 = time.perf_counter()
    err = ""
    try:
        result = agent.invoke(
            {"messages": [("user", query)]},
            config={"recursion_limit": 24},
        )
    except Exception as e:
        err = str(e)
        return {
            "final_answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
        }
    messages = result.get("messages", [])
    return {
        "final_answer": _final_text(messages),
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": _aggregate_tokens(messages),
        "error": err,
    }


def run_browsecomp_architecture(
    query: str,
    corpus: BrowseCompCorpus,
    architecture: str,
    *,
    use_llm: bool,
) -> dict[str, Any]:
    arch = architecture.strip().lower()
    if use_llm and arch == "sas":
        return run_browsecomp_sas_llm(query, corpus)
    if arch in ("sas", "single-agent system"):
        return run_browsecomp_sas_offline(query, corpus)
    if arch in ("cmas", "centralized mas"):
        return run_browsecomp_cmas_offline(query, corpus)
    if arch in ("dmas", "decentralized mas"):
        return run_browsecomp_dmas_offline(query, corpus)
    raise ValueError(f"Unknown architecture: {architecture!r}")
