#!/usr/bin/env python3
"""
Fin-RATE-style sweep: load QA JSONL + corpus JSONL, run SAS/CMAS/DMAS, score.

Defaults use this repo's fixture under ``benchmarks/``.

Official Fin-RATE ``qa/*.json`` files must be converted to this runner's JSONL
schema before use:
  QA JSONL rows: id, task_type, question, gold_answer, optional doc_id/doc_ids/key_points
  corpus JSONL rows: doc_id/id, company, year, chunk/text

Optional GPT judge: ``FINRATE_USE_GPT_JUDGE=true``.

  python run_finrate_sweep.py
  python run_finrate_sweep.py --qa path/to/qa.jsonl --corpus path/to/corpus.jsonl --per-type 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()

from dynamic_routing.finrate_runner import (  # noqa: E402
    judge_finrate_answer_detail,
    load_corpus_chunks,
    run_finrate_architecture,
)
from dynamic_routing.results_paths import finrate_run_kind, timestamped_results_path  # noqa: E402
from dynamic_routing.semantic_failure_judge import judge_semantic_failure  # noqa: E402

DEFAULT_QA = project_root / "benchmarks" / "finrate_fixture_qa.jsonl"
DEFAULT_CORPUS = project_root / "benchmarks" / "finrate_corpus_fixture.jsonl"


def _doc_ids_from_row(row: dict) -> list[str]:
    doc_ids: list[str] = []
    if row.get("doc_id"):
        doc_ids.append(str(row["doc_id"]))
    raw_doc_ids = row.get("doc_ids")
    if isinstance(raw_doc_ids, list):
        doc_ids.extend(str(doc_id) for doc_id in raw_doc_ids if doc_id)
    elif raw_doc_ids:
        doc_ids.append(str(raw_doc_ids))
    return doc_ids


def _load_qa(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            key_points = o.get("key_points") if isinstance(o.get("key_points"), list) else []
            rows.append({
                "id": str(o.get("id") or o.get("qid") or f"FR-{len(rows)}"),
                "task_type": str(o.get("task_type") or o.get("type") or "DR"),
                "question": str(o.get("question") or o.get("query") or ""),
                "gold_answer": str(o.get("gold_answer") or o.get("answer") or ""),
                "doc_ids": _doc_ids_from_row(o),
                "key_points": [str(kp) for kp in key_points],
                "source": str(o.get("source") or ""),
            })
    return [r for r in rows if r["question"]]


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _dataset_mode(qa_path: Path, corpus_path: Path) -> str:
    if _same_path(qa_path, DEFAULT_QA) and _same_path(corpus_path, DEFAULT_CORPUS):
        return "fixture"
    return "official_or_custom_converted"


def _task_type_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["task_type"]] += 1
    return dict(sorted(counts.items()))


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


def _semantic_threshold() -> float:
    try:
        return float(os.environ.get("SEMANTIC_FAIL_SCORE_THRESHOLD", "0.999").strip())
    except Exception:
        return 0.999


def main() -> None:
    ap = argparse.ArgumentParser(description="Fin-RATE-style SAS/CMAS/DMAS sweep")
    ap.add_argument("--qa", type=Path, default=DEFAULT_QA)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--per-type", type=int, default=0, help="If >0, cap each task_type to N after shuffle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-llm-all", action="store_true", help="Use LLM-backed fixed-corpus agents for SAS, CMAS, and DMAS")
    ap.add_argument("-o", "--output", type=str, default="benchmark_finrate_results.json")
    args = ap.parse_args()

    if not args.qa.is_file() or not args.corpus.is_file():
        print("ERROR: QA or corpus path missing", file=sys.stderr)
        sys.exit(1)

    dataset_mode = _dataset_mode(args.qa, args.corpus)
    chunks = load_corpus_chunks(args.corpus)
    rows = _load_qa(args.qa)
    if args.per_type > 0:
        rows = _balanced_sample(rows, args.per_type, args.seed)
    if args.limit > 0:
        rows = rows[: args.limit]
    task_type_counts = _task_type_counts(rows)

    tasks_out: list[dict] = []
    last_judge = "local"
    threshold = _semantic_threshold()

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
            out = run_finrate_architecture(
                q,
                chunks,
                key,
                doc_ids=row.get("doc_ids") or None,
                use_llm=args.use_llm_all,
            )
            lat = float(out.get("latency_sec") or 0.0)
            judgment = judge_finrate_answer_detail(q, out["answer"], gold)
            sc = float(judgment["score"])
            jb = str(judgment["judge_backend"])
            last_judge = jb
            semantic_failure = None
            if label in ("Centralized MAS", "Decentralized MAS") and float(sc) < threshold:
                semantic_failure = judge_semantic_failure(
                    architecture=label,
                    task=q,
                    gold_answer=gold,
                    final_answer=str(out.get("answer") or ""),
                    verification_score=float(sc),
                    verification_backend=jb,
                    error_type=str(out.get("error") or ""),
                    runtime_taxonomy="",
                    execution_path=list(out.get("execution_path") or []),
                    trace_data=dict(out.get("trace") or {}),
                )
            print(f"  {label[:3]} | acc={sc:.2f} judge={jb} lat={lat:.3f}s")
            if semantic_failure:
                print(
                    f"  {label[:3]} | SEMANTIC: {semantic_failure.get('label')} "
                    f"({semantic_failure.get('judge_backend')})"
                )
            res_block.append({
                "architecture": label,
                "final_answer": str(out.get("answer") or ""),
                "accuracy_score": round(float(sc), 2),
                "latency_sec": round(float(lat), 4),
                "total_tokens": int(out.get("total_tokens") or 0),
                "error_type": out.get("error") or None,
                "execution_path": list(out.get("execution_path") or []) + [f"judge={jb}"],
                "trace": dict(out.get("trace") or {}),
                "judgment": judgment,
                "semantic_failure": semantic_failure,
            })
        tasks_out.append({
            "task_id": qid,
            "description": q,
            "category": f"finrate_{tt.lower()}",
            "gold_answer": gold,
            "doc_ids": list(row.get("doc_ids") or []),
            "key_points": list(row.get("key_points") or []),
            "source": row.get("source") or None,
            "router_prediction": None,
            "results": res_block,
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = timestamped_results_path(
        project_root,
        "finrate",
        finrate_run_kind(dataset_mode),
        Path(args.output),
        ts,
        default_suffix=".json",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "sweep_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": f"finrate_{dataset_mode}",
            "dataset_mode": dataset_mode,
            "task_count": len(tasks_out),
            "task_type_counts": task_type_counts,
            "qa_path": str(args.qa.resolve()),
            "corpus_path": str(args.corpus.resolve()),
            "per_type": int(args.per_type),
            "limit": int(args.limit),
            "seed": int(args.seed),
            "qa": str(args.qa.resolve()),
            "corpus": str(args.corpus.resolve()),
            "judge_backend_last": last_judge,
            "use_llm_all": bool(args.use_llm_all),
            "llm_backend": os.environ.get("LLM_BACKEND", "vllm"),
            "openai_worker_model": os.environ.get("OPENAI_WORKER_MODEL", "gpt-5.4-mini"),
            "vllm_worker_model": os.environ.get("VLLM_WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            "google_worker_model": os.environ.get("GOOGLE_WORKER_MODEL", "gemini-3.1-flash-lite-preview"),
            "note": (
                "For official/custom Fin-RATE-style runs, first convert upstream qa/*.json and corpus/corpus.jsonl "
                "to this runner's JSONL schema; official downloading/leaderboard formatting is intentionally not implemented here. "
                "Failure diagnostics are a project-scoped MAST-inspired diagnostic subset: "
                "runtime structural tags plus lightweight post-hoc semantic_failure labels for failed CMAS/DMAS runs. "
                "PCAB code paths remain as pilot legacy; active benchmark focus is WorkBench, BrowseComp-Plus, and Fin-RATE."
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
