"""
BrowseComp-Plus-style sweep helpers: three architectures over a local corpus.

* **SAS** — single agent (ReAct with ``local_search`` when ``use_llm``; else
  deterministic top-k context string).
* **CMAS** — sequential double-query retrieval + merge (no peer parallelism).
* **DMAS** — two parallel retrieval branches (thread) + merge.

Judge modes:

* ``local`` — normalized exact / contains / overlap heuristic.
* ``llm`` — semantic equivalence judge when an LLM backend is configured.
* ``auto`` — local first, then LLM only for uncertain heuristic scores.
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from langchain_core.messages import HumanMessage

from dynamic_routing.browsecomp_env import BrowseCompCorpus
from dynamic_routing.browsecomp_tools import build_local_search_tool


LOCAL_UNCERTAIN_LOW = 0.35
LOCAL_UNCERTAIN_HIGH = 0.85


def _retrieve_context(
    corpus: BrowseCompCorpus,
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
) -> str:
    hits = corpus.get_by_ids(doc_ids, top_k=top_k) if doc_ids else corpus.search(query, top_k=top_k)
    if not hits:
        return ""
    return "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)


def run_browsecomp_sas_offline(
    query: str,
    corpus: BrowseCompCorpus,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = _retrieve_context(corpus, query, 5, doc_ids=doc_ids)
    answer = ctx[:4000] if ctx else "NO_EVIDENCE"
    return {
        "final_answer": answer,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
        "execution_path": ["sas_retrieve", "finish"],
        "trace": {
            "mode": "sas_offline",
            "query": query,
            "doc_ids": doc_ids or [],
            "context_preview": answer[:500],
        },
    }


def run_browsecomp_cmas_offline(
    query: str,
    corpus: BrowseCompCorpus,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    q2 = " ".join(query.split()[: max(3, len(query.split()) // 2)])
    a1 = _retrieve_context(corpus, query, 4, doc_ids=doc_ids)
    a2 = _retrieve_context(corpus, q2, 4, doc_ids=doc_ids)
    merged = f"{a1}\n\n---SECOND_PASS---\n\n{a2}" if a2 != a1 else a1
    return {
        "final_answer": merged[:4000] if merged else "NO_EVIDENCE",
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": 0,
        "error": "",
        "execution_path": ["cmas_pass_1", "cmas_pass_2", "cmas_merge"],
        "trace": {
            "mode": "cmas_offline",
            "doc_ids": doc_ids or [],
            "dispatches": [
                {"worker": "pass_1_retriever", "query": query, "context_preview": a1[:500]},
                {"worker": "pass_2_retriever", "query": q2, "context_preview": a2[:500]},
            ],
            "merge_strategy": "sequential_double_pass",
        },
    }


def run_browsecomp_dmas_offline(
    query: str,
    corpus: BrowseCompCorpus,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    def branch(q: str) -> str:
        return _retrieve_context(corpus, q, 4, doc_ids=doc_ids)

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
        "execution_path": ["dmas_peer_a", "dmas_peer_b", "consensus_merge"],
        "trace": {
            "mode": "dmas_offline",
            "doc_ids": doc_ids or [],
            "peer_reports": [
                {"peer": "peer_a", "query": query, "context_preview": c1[:500]},
                {"peer": "peer_b", "query": q_alt if q_alt != query else query + " details", "context_preview": c2[:500]},
            ],
            "merge_strategy": "parallel_two_peer_concat",
        },
    }


def _normalize_for_judge(text: str) -> str:
    """Normalize answer strings without assuming official BrowseComp formatting."""
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _judge_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_for_judge(text)))


def _local_judge_score(answer: str, gold: str) -> tuple[float, str, bool]:
    """Return (score, backend_label, uncertain).

    The local mode is intentionally conservative and stable for router/regret
    experiments: exact and contains matches are decisive; token overlap handles
    partial retrieval answers; only mid-band overlap is considered uncertain.
    """
    gold_norm = _normalize_for_judge(gold)
    answer_norm = _normalize_for_judge(answer)
    if not gold_norm:
        return 1.0, "local_no_gold", False
    if not answer_norm:
        return 0.0, "local_empty", False
    if answer_norm == gold_norm:
        return 1.0, "local_exact", False
    if gold_norm in answer_norm:
        return 1.0, "local_contains_gold", False

    gold_tokens = _judge_tokens(gold)
    answer_tokens = _judge_tokens(answer)
    if not gold_tokens:
        return 0.0, "local_no_gold_tokens", False
    if not answer_tokens:
        return 0.0, "local_no_answer_tokens", False

    overlap = len(gold_tokens & answer_tokens)
    recall = overlap / len(gold_tokens)
    precision = overlap / len(answer_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    # Gold recall matters most because fixture answers may be long retrieved
    # evidence snippets. F1 keeps unrelated long contexts from looking perfect.
    score = max(0.0, min(1.0, (0.75 * recall) + (0.25 * f1)))
    uncertain = LOCAL_UNCERTAIN_LOW < score < LOCAL_UNCERTAIN_HIGH
    return score, "local_overlap", uncertain


def _worker_llm_judge_score(query: str, answer: str, gold: str) -> float | None:
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
        logging.warning("BrowseComp worker LLM judge failed: %s", str(e)[:120])
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


def _llm_judge_score(query: str, answer: str, gold: str) -> tuple[float, str] | None:
    s = _qwen_judge_score(query, answer, gold)
    if s is not None:
        return s, "llm_qwen_http"
    s = _worker_llm_judge_score(query, answer, gold)
    if s is not None:
        return s, "llm_worker"
    return None


def judge_browsecomp_answer(query: str, answer: str, gold: str) -> tuple[float, str]:
    """Return (score 0..1, backend name)."""
    backend = os.environ.get("BROWSECOMP_JUDGE_BACKEND", "auto").strip().lower()
    if backend in ("local", "heuristic"):
        score, label, _ = _local_judge_score(answer, gold)
        return score, label
    if backend in ("llm", "gpt", "openai", "qwen"):
        judged = _llm_judge_score(query, answer, gold)
        if judged is not None:
            return judged
        score, label, _ = _local_judge_score(answer, gold)
        return score, f"{label}_fallback"

    score, label, uncertain = _local_judge_score(answer, gold)
    if not uncertain:
        return score, label

    judged = _llm_judge_score(query, answer, gold)
    if judged is not None:
        llm_score, llm_label = judged
        return llm_score, f"auto_{llm_label}"
    return score, f"{label}_uncertain_no_llm"


def _invoke_browsecomp_react_agent(
    query: str,
    corpus: BrowseCompCorpus,
    *,
    mode: str,
    system_prompt: str,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    from langchain.agents import create_agent

    from dynamic_routing.chat_models import bind_tools_safely, get_worker_chat_model
    from dynamic_routing.workbench_runner import _aggregate_tokens, _final_text

    tool = build_local_search_tool(corpus, allowed_doc_ids=doc_ids)
    llm = bind_tools_safely(get_worker_chat_model(temperature=0.1), [tool])
    agent = create_agent(model=llm, tools=[tool], system_prompt=system_prompt)
    t0 = time.perf_counter()
    err = ""
    try:
        result = agent.invoke(
            cast(Any, {"messages": [HumanMessage(content=query)]}),
            config={"recursion_limit": 24},
        )
    except Exception as e:
        err = str(e)
        return {
            "final_answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
            "execution_path": [f"{mode}_error"],
            "trace": {"mode": mode, "query": query, "doc_ids": doc_ids or [], "agent_error": err[:300]},
        }
    messages = result.get("messages", [])
    return {
        "final_answer": _final_text(messages),
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": _aggregate_tokens(messages),
        "error": err,
        "execution_path": [f"{mode}_tool_loop", "finish"],
        "trace": {
            "mode": mode,
            "query": query,
            "doc_ids": doc_ids or [],
            "message_count": len(messages),
            "final_preview": _final_text(messages)[:500],
        },
    }


def _llm_merge(prompt: str) -> tuple[str, int]:
    from dynamic_routing.chat_models import get_worker_chat_model
    from dynamic_routing.workbench_runner import _aggregate_tokens

    resp = get_worker_chat_model(temperature=0.0).invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return text, _aggregate_tokens([resp])


def run_browsecomp_sas_llm(
    query: str,
    corpus: BrowseCompCorpus,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    return _invoke_browsecomp_react_agent(
        query,
        corpus,
        mode="sas_llm",
        doc_ids=doc_ids,
        system_prompt=(
            "You answer using only the local_search tool and its passages. "
            "Search once or twice as needed, cite doc ids briefly, then state the final answer."
        ),
    )


def run_browsecomp_cmas_llm(
    query: str,
    corpus: BrowseCompCorpus,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    err = ""
    try:
        direct = _invoke_browsecomp_react_agent(
            query,
            corpus,
            mode="cmas_llm_direct_worker",
            doc_ids=doc_ids,
            system_prompt=(
                "You are the direct-evidence worker in a centralized MAS. "
                "Use local_search only, identify the strongest supporting passages, and cite doc ids."
            ),
        )
        alt_query = " ".join(query.split()[: max(4, len(query.split()) // 2)])
        alternate = _invoke_browsecomp_react_agent(
            alt_query,
            corpus,
            mode="cmas_llm_alternate_worker",
            doc_ids=doc_ids,
            system_prompt=(
                "You are the alternate-query worker in a centralized MAS. "
                "Use local_search only, look for aliases, shortened phrasing, or missing evidence, and cite doc ids."
            ),
        )
        merged, merge_tokens = _llm_merge(
            "You are the CMAS supervisor. Synthesize a final answer using only the two worker reports. "
            "Resolve disagreement conservatively and cite supporting doc ids.\n\n"
            f"Question: {query}\n\nDirect worker report:\n{direct.get('final_answer', '')[:2500]}\n\n"
            f"Alternate worker report:\n{alternate.get('final_answer', '')[:2500]}\n\nFinal answer:"
        )
        total_tokens = int(direct.get("total_tokens") or 0) + int(alternate.get("total_tokens") or 0) + merge_tokens
    except Exception as e:
        err = str(e)
        return {
            "final_answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
            "execution_path": ["cmas_llm_error"],
            "trace": {"mode": "cmas_llm", "query": query, "doc_ids": doc_ids or [], "agent_error": err[:300]},
        }
    return {
        "final_answer": merged,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": total_tokens,
        "error": err,
        "execution_path": ["cmas_direct_worker", "cmas_alternate_worker", "cmas_supervisor_merge"],
        "trace": {
            "mode": "cmas_llm",
            "doc_ids": doc_ids or [],
            "dispatches": [
                {"worker": "direct_evidence", "query": query, "final_preview": str(direct.get("final_answer") or "")[:500]},
                {"worker": "alternate_query", "query": alt_query, "final_preview": str(alternate.get("final_answer") or "")[:500]},
            ],
            "merge_strategy": "llm_supervisor_synthesis",
            "final_preview": merged[:500],
        },
    }


def run_browsecomp_dmas_llm(
    query: str,
    corpus: BrowseCompCorpus,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    q_alt = re.sub(r"\s+", " ", " ".join(reversed(query.split()[:12])))
    peer_specs = [
        ("peer_literal", query, "Use literal query terms and cite doc ids."),
        ("peer_alternate", q_alt if q_alt != query else query + " background", "Use alternate wording and cite doc ids."),
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _invoke_browsecomp_react_agent,
                    peer_query,
                    corpus,
                    mode=f"dmas_llm_{peer}",
                    doc_ids=doc_ids,
                    system_prompt=(
                        "You are an independent peer in a decentralized MAS. "
                        f"{guidance} Use only local_search and produce a concise evidence-backed answer."
                    ),
                )
                for peer, peer_query, guidance in peer_specs
            ]
            reports = [f.result() for f in futures]
        merged, merge_tokens = _llm_merge(
            "You are the DMAS consensus merger. Compare the independent peer reports and produce one final answer. "
            "Use only claims supported by the reports and cite doc ids.\n\n"
            f"Question: {query}\n\nPeer A:\n{reports[0].get('final_answer', '')[:2500]}\n\n"
            f"Peer B:\n{reports[1].get('final_answer', '')[:2500]}\n\nFinal answer:"
        )
        total_tokens = sum(int(r.get("total_tokens") or 0) for r in reports) + merge_tokens
    except Exception as e:
        err = str(e)
        return {
            "final_answer": "",
            "latency_sec": time.perf_counter() - t0,
            "total_tokens": 0,
            "error": err,
            "execution_path": ["dmas_llm_error"],
            "trace": {"mode": "dmas_llm", "query": query, "doc_ids": doc_ids or [], "agent_error": err[:300]},
        }
    return {
        "final_answer": merged,
        "latency_sec": time.perf_counter() - t0,
        "total_tokens": total_tokens,
        "error": "",
        "execution_path": ["dmas_peer_literal", "dmas_peer_alternate", "dmas_consensus_merge"],
        "trace": {
            "mode": "dmas_llm",
            "doc_ids": doc_ids or [],
            "peer_reports": [
                {"peer": peer_specs[i][0], "query": peer_specs[i][1], "final_preview": str(reports[i].get("final_answer") or "")[:500]}
                for i in range(len(reports))
            ],
            "merge_strategy": "llm_consensus_synthesis",
            "final_preview": merged[:500],
        },
    }


def run_browsecomp_architecture(
    query: str,
    corpus: BrowseCompCorpus,
    architecture: str,
    *,
    use_llm: bool,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    arch = architecture.strip().lower()
    if use_llm and arch == "sas":
        return run_browsecomp_sas_llm(query, corpus, doc_ids=doc_ids)
    if use_llm and arch == "cmas":
        return run_browsecomp_cmas_llm(query, corpus, doc_ids=doc_ids)
    if use_llm and arch == "dmas":
        return run_browsecomp_dmas_llm(query, corpus, doc_ids=doc_ids)
    if arch in ("sas", "single-agent system"):
        return run_browsecomp_sas_offline(query, corpus, doc_ids=doc_ids)
    if arch in ("cmas", "centralized mas"):
        return run_browsecomp_cmas_offline(query, corpus, doc_ids=doc_ids)
    if arch in ("dmas", "decentralized mas"):
        return run_browsecomp_dmas_offline(query, corpus, doc_ids=doc_ids)
    raise ValueError(f"Unknown architecture: {architecture!r}")
