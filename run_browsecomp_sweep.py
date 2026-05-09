#!/usr/bin/env python3
"""
BrowseComp-style sweep over a local JSONL corpus + query list.

Uses ``browsecomp_runner`` (SAS / CMAS / DMAS) and writes benchmark JSON compatible
with ``evaluate_regret.py``.

Defaults use repo fixtures under ``benchmarks/``. Point to real BrowseComp-Plus
decrypted queries + corpus JSONL when available.

Environment (hybrid judge):
  BROWSECOMP_JUDGE_BACKEND=local|llm|auto
  BROWSECOMP_QWEN_URL=http://host:port  — optional OpenAI-compatible Qwen judge for llm/auto

Run:
  python run_browsecomp_sweep.py
  python run_browsecomp_sweep.py --queries path/to/queries.jsonl --corpus path/to/corpus.jsonl
  python run_browsecomp_sweep.py --queries path/to/queries.jsonl --corpus path/to/corpus.jsonl --sample 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()

from dynamic_routing.browsecomp_env import BrowseCompCorpus  # noqa: E402
from dynamic_routing.browsecomp_runner import (  # noqa: E402
    judge_browsecomp_answer,
    run_browsecomp_architecture,
)
from dynamic_routing.results_paths import timestamped_results_path  # noqa: E402
from dynamic_routing.semantic_failure_judge import judge_semantic_failure  # noqa: E402

DEFAULT_QUERIES = project_root / "benchmarks" / "browsecomp_fixture.jsonl"
DEFAULT_CORPUS = project_root / "benchmarks" / "browsecomp_corpus_fixture.jsonl"


def _doc_ids_from_row(row: dict) -> list[str]:
    doc_ids: list[str] = []
    for key in ("gold_doc_ids", "evidence_doc_ids", "doc_ids"):
        value = row.get(key)
        if isinstance(value, list):
            doc_ids.extend(str(doc_id) for doc_id in value if doc_id)
        elif value:
            doc_ids.append(str(value))
    return list(dict.fromkeys(doc_ids))


def _load_queries(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            qid = str(o.get("query_id") or o.get("id") or f"Q-{len(rows)}")
            q = str(o.get("query") or o.get("text") or "")
            gold = str(o.get("gold_answer") or o.get("answer") or "")
            if q:
                rows.append({
                    "query_id": qid,
                    "query": q,
                    "gold_answer": gold,
                    "doc_ids": _doc_ids_from_row(o),
                    "gold_doc_ids": [str(doc_id) for doc_id in o.get("gold_doc_ids", [])] if isinstance(o.get("gold_doc_ids"), list) else [],
                    "evidence_doc_ids": [str(doc_id) for doc_id in o.get("evidence_doc_ids", [])] if isinstance(o.get("evidence_doc_ids"), list) else [],
                })
    return rows


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _dataset_mode(queries_path: Path, corpus_path: Path) -> str:
    if _same_path(queries_path, DEFAULT_QUERIES) and _same_path(corpus_path, DEFAULT_CORPUS):
        return "fixture"
    return "official_or_custom_converted"


def _run_kind(dataset_mode: str) -> str:
    if dataset_mode == "fixture":
        return "fixtures"
    return "sweeps"


def _sample_queries(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    if sample_size <= 0:
        return rows
    import random

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(sample_size, len(shuffled))]


def _semantic_threshold() -> float:
    try:
        return float(os.environ.get("SEMANTIC_FAIL_SCORE_THRESHOLD", "0.999").strip())
    except Exception:
        return 0.999


def main() -> None:
    ap = argparse.ArgumentParser(description="BrowseComp-style SAS/CMAS/DMAS sweep")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="JSONL: query_id, query, gold_answer")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="JSONL corpus: doc_id, text")
    ap.add_argument("--sample", type=int, default=0, help="Random fixed-seed sample size before --limit (0=all)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for --sample")
    ap.add_argument("--limit", type=int, default=0, help="Max queries (0=all)")
    ap.add_argument("--use-llm-sas", action="store_true", help="Use ReAct+local_search for SAS only (needs LLM backend)")
    ap.add_argument("-o", "--output", type=str, default="benchmark_browsecomp_results.json", help="Stem under results/")
    args = ap.parse_args()

    if not args.queries.is_file():
        print(f"ERROR: queries file not found: {args.queries}", file=sys.stderr)
        sys.exit(1)
    if not args.corpus.is_file():
        print(f"ERROR: corpus file not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    dataset_mode = _dataset_mode(args.queries, args.corpus)
    corpus = BrowseCompCorpus.load_jsonl(args.corpus)
    queries = _load_queries(args.queries)
    available_query_count = len(queries)
    if args.sample > 0:
        queries = _sample_queries(queries, args.sample, args.seed)
    if args.limit > 0:
        queries = queries[: args.limit]

    judge_backend_global = "local_overlap"
    threshold = _semantic_threshold()
    tasks_out: list[dict] = []

    for row in queries:
        qid = row["query_id"]
        q = row["query"]
        gold = row["gold_answer"]
        print(f"\n[{qid}] {q[:80]}...")

        results_three: list[dict] = []
        for arch_label, internal in (
            ("Single-Agent System", "sas"),
            ("Centralized MAS", "cmas"),
            ("Decentralized MAS", "dmas"),
        ):
            out = run_browsecomp_architecture(
                q,
                corpus,
                internal,
                use_llm=args.use_llm_sas,
                doc_ids=row.get("doc_ids") or None,
            )
            lat = float(out.get("latency_sec") or 0.0)
            ans = out.get("final_answer") or ""
            score, jbackend = judge_browsecomp_answer(q, ans, gold)
            judge_backend_global = jbackend
            tok = int(out.get("total_tokens") or 0)
            err = out.get("error") or None
            semantic_failure = None
            if arch_label in ("Centralized MAS", "Decentralized MAS") and float(score) < threshold:
                semantic_failure = judge_semantic_failure(
                    architecture=arch_label,
                    task=q,
                    gold_answer=gold,
                    final_answer=ans,
                    verification_score=float(score),
                    verification_backend=jbackend,
                    error_type=str(err or ""),
                    runtime_taxonomy="",
                    execution_path=list(out.get("execution_path") or []),
                    trace_data=dict(out.get("trace") or {}),
                )
            print(f"  {arch_label[:3]} | acc={score:.2f} judge={jbackend} lat={lat:.3f}s tok={tok}")
            if semantic_failure:
                print(
                    f"  {arch_label[:3]} | SEMANTIC: {semantic_failure.get('label')} "
                    f"({semantic_failure.get('judge_backend')})"
                )
            results_three.append({
                "architecture": arch_label,
                "final_answer": ans,
                "accuracy_score": round(float(score), 2),
                "latency_sec": round(float(lat), 4),
                "total_tokens": tok,
                "error_type": err,
                "execution_path": list(out.get("execution_path") or []) + [f"judge={jbackend}"],
                "trace": dict(out.get("trace") or {}),
                "semantic_failure": semantic_failure,
            })

        tasks_out.append({
            "task_id": qid,
            "description": q,
            "category": "browsecomp_plus",
            "gold_answer": gold,
            "doc_ids": list(row.get("doc_ids") or []),
            "gold_doc_ids": list(row.get("gold_doc_ids") or []),
            "evidence_doc_ids": list(row.get("evidence_doc_ids") or []),
            "router_prediction": None,
            "results": results_three,
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = timestamped_results_path(
        project_root,
        "browsecomp",
        _run_kind(dataset_mode),
        Path(args.output),
        ts,
        default_suffix=".json",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": f"browsecomp_{dataset_mode}",
            "dataset_mode": dataset_mode,
            "task_count": len(tasks_out),
            "available_query_count": available_query_count,
            "requested_sample": int(args.sample),
            "seed": int(args.seed),
            "limit": int(args.limit),
            "queries_path": str(args.queries.resolve()),
            "corpus_path": str(args.corpus.resolve()),
            "queries": str(args.queries.resolve()),
            "corpus": str(args.corpus.resolve()),
            "judge_backend_last": judge_backend_global,
            "use_llm_sas": bool(args.use_llm_sas),
            "note": (
                "Integrate official BrowseComp-Plus decrypt + indexes per upstream README when running full eval. "
                "Failure diagnostics are a project-scoped MAST-inspired diagnostic subset: "
                "runtime structural tags plus lightweight post-hoc semantic_failure labels for failed CMAS/DMAS runs. "
                "PCAB-related paths are legacy pilot artifacts; active benchmark focus is WorkBench, BrowseComp-Plus, and Fin-RATE."
            ),
        },
        "tasks": tasks_out,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path.resolve()}")
    print("Oracle / regret: python evaluate_regret.py", out_path)


if __name__ == "__main__":
    main()
