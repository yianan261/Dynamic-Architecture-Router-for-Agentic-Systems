#!/usr/bin/env python3
"""
Fin-RATE-style sweep: load QA JSONL + corpus JSONL, run SAS/CMAS/DMAS, score.

Use upstream ``qa/*.json`` converted to JSONL or this repo's fixture under
``benchmarks/``. Optional GPT judge: ``FINRATE_USE_GPT_JUDGE=true``.

  python run_finrate_sweep.py
  python run_finrate_sweep.py --qa path/to/qa.jsonl --corpus path/to/corpus.jsonl --per-type 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()

from dynamic_routing.finrate_runner import (  # noqa: E402
    judge_finrate_answer,
    load_corpus_chunks,
    run_finrate_architecture,
)

DEFAULT_QA = project_root / "benchmarks" / "finrate_fixture_qa.jsonl"
DEFAULT_CORPUS = project_root / "benchmarks" / "finrate_corpus_fixture.jsonl"
RESULTS_DIR = project_root / "results"


def _load_qa(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append({
                "id": str(o.get("id") or o.get("qid") or f"FR-{len(rows)}"),
                "task_type": str(o.get("task_type") or o.get("type") or "DR"),
                "question": str(o.get("question") or o.get("query") or ""),
                "gold_answer": str(o.get("gold_answer") or o.get("answer") or ""),
            })
    return [r for r in rows if r["question"]]


def _balanced_sample(rows: list[dict[str, str]], per_type: int, seed: int) -> list[dict[str, str]]:
    import random

    rng = random.Random(seed)
    by_t: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_t[r["task_type"]].append(r)
    out: list[dict[str, str]] = []
    for t, lst in by_t.items():
        rng.shuffle(lst)
        out.extend(lst[:per_type])
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fin-RATE-style SAS/CMAS/DMAS sweep")
    ap.add_argument("--qa", type=Path, default=DEFAULT_QA)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--per-type", type=int, default=0, help="If >0, cap each task_type to N after shuffle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--output", type=str, default="benchmark_finrate_results.json")
    args = ap.parse_args()

    if not args.qa.is_file() or not args.corpus.is_file():
        print("ERROR: QA or corpus path missing", file=sys.stderr)
        sys.exit(1)

    chunks = load_corpus_chunks(args.corpus)
    rows = _load_qa(args.qa)
    if args.per_type > 0:
        rows = _balanced_sample(rows, args.per_type, args.seed)
    if args.limit > 0:
        rows = rows[: args.limit]

    tasks_out: list[dict] = []
    last_judge = "local"

    for row in rows:
        qid = row["id"]
        q = row["question"]
        gold = row["gold_answer"]
        tt = row["task_type"]
        print(f"\n[{qid}] ({tt}) {q[:70]}...")
        res_block: list[dict] = []
        for label, key in (
            ("Single-Agent System", "sas"),
            ("Centralized MAS", "cmas"),
            ("Decentralized MAS", "dmas"),
        ):
            out = run_finrate_architecture(q, chunks, key)
            lat = float(out.get("latency_sec") or 0.0)
            sc, jb = judge_finrate_answer(q, out["answer"], gold)
            last_judge = jb
            print(f"  {label[:3]} | acc={sc:.2f} judge={jb} lat={lat:.3f}s")
            res_block.append({
                "architecture": label,
                "accuracy_score": round(float(sc), 2),
                "latency_sec": round(float(lat), 4),
                "total_tokens": int(out.get("total_tokens") or 0),
                "error_type": out.get("error") or None,
                "execution_path": [f"judge={jb}"],
            })
        tasks_out.append({
            "task_id": qid,
            "description": q,
            "category": f"finrate_{tt.lower()}",
            "gold_answer": gold,
            "router_prediction": None,
            "results": res_block,
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{Path(args.output).stem}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "finrate_fixture",
            "qa": str(args.qa.resolve()),
            "corpus": str(args.corpus.resolve()),
            "judge_backend_last": last_judge,
            "note": "For full Fin-RATE, use upstream qa/*.json + corpus/corpus.jsonl and their evaluation/qa_llm_judge.py as reference.",
        },
        "tasks": tasks_out,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path.resolve()}")
    print("Oracle / regret: python evaluate_regret.py", out_path)


if __name__ == "__main__":
    main()
