#!/usr/bin/env python3
"""
BrowseComp-style sweep over a local JSONL corpus + query list.

Uses ``browsecomp_runner`` (SAS / CMAS / DMAS) and writes benchmark JSON compatible
with ``evaluate_regret.py``.

Defaults use repo fixtures under ``benchmarks/``. Point to real BrowseComp-Plus
decrypted queries + corpus JSONL when available.

Environment (hybrid judge):
  BROWSECOMP_JUDGE_BACKEND=auto|qwen|gpt|local
  BROWSECOMP_QWEN_URL=http://host:port  — OpenAI-compatible Qwen server (optional)

Run:
  python run_browsecomp_sweep.py
  python run_browsecomp_sweep.py --queries path/to/queries.jsonl --corpus path/to/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_QUERIES = project_root / "benchmarks" / "browsecomp_fixture.jsonl"
DEFAULT_CORPUS = project_root / "benchmarks" / "browsecomp_corpus_fixture.jsonl"
RESULTS_DIR = project_root / "results"


def _load_queries(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
                rows.append({"query_id": qid, "query": q, "gold_answer": gold})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="BrowseComp-style SAS/CMAS/DMAS sweep")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="JSONL: query_id, query, gold_answer")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="JSONL corpus: doc_id, text")
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

    corpus = BrowseCompCorpus.load_jsonl(args.corpus)
    queries = _load_queries(args.queries)
    if args.limit > 0:
        queries = queries[: args.limit]

    judge_backend_global = "local_overlap"
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
            )
            lat = float(out.get("latency_sec") or 0.0)
            ans = out.get("final_answer") or ""
            score, jbackend = judge_browsecomp_answer(q, ans, gold)
            judge_backend_global = jbackend
            tok = int(out.get("total_tokens") or 0)
            err = out.get("error") or None
            print(f"  {arch_label[:3]} | acc={score:.2f} judge={jbackend} lat={lat:.3f}s tok={tok}")
            results_three.append({
                "architecture": arch_label,
                "accuracy_score": round(float(score), 2),
                "latency_sec": round(float(lat), 4),
                "total_tokens": tok,
                "error_type": err,
                "execution_path": [f"judge={jbackend}"],
            })

        tasks_out.append({
            "task_id": qid,
            "description": q,
            "category": "browsecomp_plus",
            "gold_answer": gold,
            "router_prediction": None,
            "results": results_three,
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{Path(args.output).stem}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "browsecomp_fixture",
            "queries": str(args.queries.resolve()),
            "corpus": str(args.corpus.resolve()),
            "judge_backend_last": judge_backend_global,
            "use_llm_sas": bool(args.use_llm_sas),
            "note": "Integrate official BrowseComp-Plus decrypt + indexes per upstream README when running full eval.",
        },
        "tasks": tasks_out,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path.resolve()}")
    print("Oracle / regret: python evaluate_regret.py", out_path)


if __name__ == "__main__":
    main()
