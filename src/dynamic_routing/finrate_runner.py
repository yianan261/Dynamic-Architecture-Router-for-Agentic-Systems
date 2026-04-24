"""
Fin-RATE-style QA over a local JSONL ``corpus`` (SEC-style chunks).

Three architectures (scaffold, offline-friendly):
  * **SAS** — single string context from keyword retrieval over all chunks.
  * **CMAS** — split context by company/year keys when present, then merge text.
  * **DMAS** — two parallel retrieval queries (full + shortened) merged.

For full Fin-RATE, replace fixture paths with ``qa/*.json`` and ``corpus/corpus.jsonl``
from the upstream repo and use ``--use-llm`` for generation + their judge script.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


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


def retrieve_context(question: str, chunks: list[dict[str, Any]], top_n: int = 4) -> str:
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


def run_finrate_sas_offline(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = retrieve_context(question, chunks, 5)
    return {
        "answer": ctx[:6000] or "NO_CONTEXT",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
    }


def run_finrate_cmas_offline(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
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
    }


def run_finrate_dmas_offline(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    q_short = " ".join(question.split()[:8])

    def a(q: str) -> str:
        return retrieve_context(q, chunks, 4)

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
    }


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


def judge_finrate_gpt(question: str, answer: str, gold: str) -> float | None:
    try:
        from dynamic_routing.chat_models import get_worker_chat_model

        llm = get_worker_chat_model(temperature=0.0)
        prompt = (
            "Score 0.0-1.0 how well the candidate answers the question vs reference.\n"
            f"Q: {question[:800]}\nRef: {gold[:800]}\nCand: {answer[:3000]}\n"
            "Reply with only a float like 0.75"
        )
        resp = llm.invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        m = re.search(r"0?\.\d+|1\.0|1|0", text)
        if m:
            return max(0.0, min(1.0, float(m.group(0))))
    except Exception as e:
        logging.warning("Fin-RATE GPT judge failed: %s", str(e)[:120])
    return None


def judge_finrate_answer(question: str, answer: str, gold: str) -> tuple[float, str]:
    if os.environ.get("FINRATE_USE_GPT_JUDGE", "").lower() in ("1", "true", "yes"):
        s = judge_finrate_gpt(question, answer, gold)
        if s is not None:
            return s, "gpt"
    sc = judge_finrate_local(answer, gold)
    return sc, "local"


def run_finrate_architecture(
    question: str,
    chunks: list[dict[str, Any]],
    architecture: str,
) -> dict[str, Any]:
    a = architecture.strip().lower()
    if a in ("sas", "single-agent system"):
        return run_finrate_sas_offline(question, chunks)
    if a in ("cmas", "centralized mas"):
        return run_finrate_cmas_offline(question, chunks)
    if a in ("dmas", "decentralized mas"):
        return run_finrate_dmas_offline(question, chunks)
    raise ValueError(architecture)
