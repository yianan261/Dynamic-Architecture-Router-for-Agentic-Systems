#!/usr/bin/env python3
"""
Rejudge saved Fin-RATE benchmark results without rerunning SAS/CMAS/DMAS.

This script updates per-architecture ``accuracy_score`` and ``judgment`` fields
from saved ``final_answer`` values, then refreshes semantic_failure labels for
failed CMAS/DMAS outputs. It preserves the benchmark schema consumed by
``evaluate_regret.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.dotenv_util import load_project_root_dotenv  # noqa: E402

load_project_root_dotenv()

from dynamic_routing.finrate_runner import judge_finrate_answer_detail  # noqa: E402
from dynamic_routing.results_paths import timestamped_results_path  # noqa: E402
from dynamic_routing.semantic_failure_judge import judge_semantic_failure  # noqa: E402


def _semantic_threshold() -> float:
    import os

    try:
        return float(os.environ.get("SEMANTIC_FAIL_SCORE_THRESHOLD", "0.999").strip())
    except Exception:
        return 0.999


def _candidate_answer(result: dict[str, Any]) -> str:
    if result.get("final_answer") is not None:
        return str(result.get("final_answer") or "")
    trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
    for key in ("final_response", "context_preview", "final_preview"):
        if trace.get(key):
            return str(trace.get(key) or "")
    previews: list[str] = []
    for item in trace.get("dispatches") or trace.get("peer_reports") or []:
        if isinstance(item, dict) and item.get("context_preview"):
            previews.append(str(item.get("context_preview") or ""))
    return "\n\n".join(previews)


def _replace_judge_marker(path: list[Any], backend: str) -> list[str]:
    cleaned = [str(item) for item in path if not str(item).startswith("judge=")]
    cleaned.append(f"judge={backend}")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejudge saved Fin-RATE results without rerunning architectures")
    parser.add_argument("input_json", type=Path, help="Saved Fin-RATE benchmark JSON")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path or stem")
    args = parser.parse_args()

    if not args.input_json.is_file():
        print(f"ERROR: input not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    threshold = _semantic_threshold()
    judge_counts: dict[str, int] = {}
    rejudged = 0

    for task in data.get("tasks", []):
        q = str(task.get("description") or "")
        gold = str(task.get("gold_answer") or "")
        for result in task.get("results", []):
            if not isinstance(result, dict):
                continue
            answer = _candidate_answer(result)
            result["final_answer"] = answer
            judgment = judge_finrate_answer_detail(q, answer, gold)
            score = float(judgment["score"])
            backend = str(judgment["judge_backend"])
            result["accuracy_score"] = round(score, 2)
            result["judgment"] = judgment
            result["execution_path"] = _replace_judge_marker(list(result.get("execution_path") or []), backend)
            judge_counts[backend] = judge_counts.get(backend, 0) + 1
            rejudged += 1

            arch = str(result.get("architecture") or "")
            if arch in ("Centralized MAS", "Decentralized MAS") and score < threshold:
                result["semantic_failure"] = judge_semantic_failure(
                    architecture=arch,
                    task=q,
                    gold_answer=gold,
                    final_answer=answer,
                    verification_score=score,
                    verification_backend=backend,
                    error_type=str(result.get("error_type") or ""),
                    runtime_taxonomy="",
                    execution_path=list(result.get("execution_path") or []),
                    trace_data=dict(result.get("trace") or {}),
                )
            elif arch in ("Centralized MAS", "Decentralized MAS"):
                result["semantic_failure"] = None

    metadata = data.setdefault("metadata", {})
    metadata["rejudged_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["rejudged_source"] = str(args.input_json.resolve())
    metadata["rejudged_results"] = rejudged
    metadata["rejudge_backend_counts"] = dict(sorted(judge_counts.items()))
    metadata["judge_backend_last"] = next(reversed(judge_counts), metadata.get("judge_backend_last", ""))

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        out_path = timestamped_results_path(
            project_root,
            "finrate",
            "rejudged",
            Path(f"{args.input_json.stem}_rejudged.json"),
            ts,
            default_suffix=".json",
        )
    else:
        out_path = args.output
        if not out_path.is_absolute():
            out_path = timestamped_results_path(
                project_root,
                "finrate",
                "rejudged",
                out_path,
                ts,
                default_suffix=".json",
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.resolve()} with {rejudged} rejudged architecture results")
    print("Next: python evaluate_regret.py", out_path)


if __name__ == "__main__":
    main()
