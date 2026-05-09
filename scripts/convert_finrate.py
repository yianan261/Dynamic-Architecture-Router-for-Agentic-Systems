#!/usr/bin/env python3
"""Convert Fin-RATE data into this project's runner JSONL schema.

Expected runner output:
  QA JSONL rows:     {"id": str, "task_type": "DR|EC|LT", "question": str, "gold_answer": str, ...}
  corpus JSONL rows: {"doc_id": str, "chunk": str, ...}

Use a local HuggingFace dataset snapshot by default. The script can also fetch a
snapshot with ``huggingface_hub`` when ``--download`` is passed. No answers or
corpus chunks are fabricated; rows missing required question/answer/chunk fields
are skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator


TASK_FILES = (
    ("DR", ("dr_qa.json", "DR-QA.json")),
    ("EC", ("ec_qa.json", "EC-QA.json")),
    ("LT", ("lt_qa.json", "LT-QA.json")),
)
QUESTION_KEYS = ("question", "query", "prompt")
ANSWER_KEYS = ("gold_answer", "answer", "answers", "reference_answer", "ground_truth", "final_answer", "target")
ID_KEYS = ("id", "qid", "q_id", "question_id", "query_id")
CHUNK_KEYS = ("chunk", "text", "contents", "content", "body", "passage")


def _pick(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(_stringify(v) for v in value if _stringify(v)).strip()
    if isinstance(value, dict):
        for key in ANSWER_KEYS + QUESTION_KEYS + CHUNK_KEYS:
            if key in value:
                return _stringify(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [s for item in value if (s := _stringify(item))]
    text = _stringify(value)
    return [text] if text else []


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(_iter_jsonl(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("data", "examples", "questions", "qa", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [obj]
    return []


def _task_type_from_path(path: Path, fallback: str) -> str:
    name = path.name.lower()
    if "dr" in name:
        return "DR"
    if "ec" in name:
        return "EC"
    if "lt" in name:
        return "LT"
    return fallback


def _qa_files(source_dir: Path, qa_dir: Path | None) -> list[tuple[str, Path]]:
    root = qa_dir or source_dir / "qa"
    found: list[tuple[str, Path]] = []
    for task_type, filenames in TASK_FILES:
        for filename in filenames:
            path = root / filename
            if path.is_file():
                found.append((task_type, path))
                break
    if found:
        return found
    return [
        (_task_type_from_path(path, "UNK"), path)
        for path in sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl"))
    ]


def convert_qa(
    source_dir: Path,
    output_path: Path,
    qa_dir: Path | None,
    max_qa_per_type: int,
) -> tuple[int, int, dict[str, int], set[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    counts: dict[str, int] = {}
    selected_doc_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8") as out:
        for fallback_type, path in _qa_files(source_dir, qa_dir):
            type_written = 0
            for index, row in enumerate(_load_json_rows(path)):
                task_type = _stringify(row.get("task_type") or row.get("type") or fallback_type).upper()
                question = _stringify(_pick(row, QUESTION_KEYS))
                gold = _stringify(_pick(row, ANSWER_KEYS))
                qid = _stringify(_pick(row, ID_KEYS)) or f"FR-{task_type}-{index}"
                if not question or not gold:
                    skipped += 1
                    continue
                converted: dict[str, Any] = {
                    "id": qid,
                    "task_type": task_type,
                    "question": question,
                    "gold_answer": gold,
                }
                doc_id = _stringify(row.get("doc_id"))
                doc_ids = _string_list(row.get("doc_ids"))
                if doc_id:
                    converted["doc_id"] = doc_id
                    selected_doc_ids.add(doc_id)
                if doc_ids:
                    converted["doc_ids"] = doc_ids
                    selected_doc_ids.update(doc_ids)
                key_points = _string_list(row.get("key_points"))
                if key_points:
                    converted["key_points"] = key_points
                source = _stringify(row.get("source"))
                if source:
                    converted["source"] = source
                out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1
                type_written += 1
                counts[task_type] = counts.get(task_type, 0) + 1
                if max_qa_per_type > 0 and type_written >= max_qa_per_type:
                    break
    return written, skipped, dict(sorted(counts.items())), selected_doc_ids


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _corpus_candidates(source_dir: Path, corpus_input: Path | None) -> list[Path]:
    if corpus_input:
        return [corpus_input]
    corpus_dir = source_dir / "corpus"
    preferred = corpus_dir / "corpus.jsonl"
    zipped = corpus_dir / "corpus.zip"
    if preferred.is_file():
        return [preferred]
    if zipped.is_file():
        return [zipped]
    if corpus_dir.exists():
        return sorted(corpus_dir.rglob("*.jsonl")) + sorted(corpus_dir.rglob("*.zip"))
    return []


def _iter_corpus_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as td:
            members = [m for m in zf.namelist() if m.endswith(".jsonl")]
            if not members:
                return
            zf.extract(members[0], td)
            yield from _iter_jsonl(Path(td) / members[0])
        return
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(path)
        return
    raise SystemExit(f"Unsupported Fin-RATE corpus input format: {path}")


def _normalize_corpus_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    chunk = _stringify(_pick(row, CHUNK_KEYS))
    if not chunk:
        return None
    out: dict[str, Any] = {"chunk": chunk}
    upstream_id = _stringify(row.get("_id"))
    if upstream_id:
        out["doc_id"] = upstream_id
    for key in ("company", "ticker", "year", "filing_type", "section", "doc_id", "id", "source", "title"):
        value = _stringify(row.get(key))
        if value and key not in out:
            out[key] = value
    if "doc_id" not in out and "id" not in out:
        out["doc_id"] = f"FR-DOC-{index}"
    return out


def convert_corpus(
    source_dir: Path,
    output_path: Path,
    corpus_input: Path | None,
    max_corpus: int,
    include_doc_ids: set[str] | None = None,
) -> tuple[int, int]:
    candidates = _corpus_candidates(source_dir, corpus_input)
    if not candidates:
        return 0, 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in candidates:
            for row in _iter_corpus_rows(path):
                norm = _normalize_corpus_row(row, written + skipped)
                if norm is None:
                    skipped += 1
                    continue
                if include_doc_ids is not None and str(norm.get("doc_id") or norm.get("id") or "") not in include_doc_ids:
                    skipped += 1
                    continue
                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                written += 1
                if max_corpus > 0 and written >= max_corpus:
                    return written, skipped
    return written, skipped


def _download_snapshot(dataset_name: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit("Downloading Fin-RATE requires `pip install huggingface_hub`.") from exc
    snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=str(local_dir))
    return local_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Fin-RATE HF snapshot to runner JSONL.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Local HuggingFace snapshot directory")
    parser.add_argument("--download", action="store_true", help="Download --hf-dataset into --source-dir before converting")
    parser.add_argument("--hf-dataset", default="GGLabYale/Fin-RATE")
    parser.add_argument("--qa-dir", type=Path, default=None, help="Override QA directory containing dr_qa/ec_qa/lt_qa JSON files")
    parser.add_argument("--corpus-input", type=Path, default=None, help="Override corpus JSONL or corpus.zip path")
    parser.add_argument("--output-dir", type=Path, default=Path("data/converted/finrate"))
    parser.add_argument("--qa-output", default="finrate_qa.jsonl")
    parser.add_argument("--corpus-output", default="finrate_corpus.jsonl")
    parser.add_argument("--max-qa-per-type", type=int, default=0, help="Tiny validation limit per type; 0 means all")
    parser.add_argument("--max-corpus", type=int, default=0, help="Tiny validation limit; 0 means all")
    parser.add_argument(
        "--corpus-from-selected-qa",
        action="store_true",
        help="Write only corpus rows referenced by the converted QA rows' doc_id/doc_ids fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir or Path("data/raw/Fin-RATE")
    if args.download:
        source_dir = _download_snapshot(args.hf_dataset, source_dir)
    if not source_dir.exists():
        raise SystemExit(f"Missing source directory: {source_dir}. Download the HuggingFace dataset first or pass --download.")
    if args.qa_dir and not args.qa_dir.exists():
        raise SystemExit(f"Missing QA directory: {args.qa_dir}")
    if args.corpus_input and not args.corpus_input.exists():
        raise SystemExit(f"Missing corpus input: {args.corpus_input}")

    qa_out = args.output_dir / args.qa_output
    corpus_out = args.output_dir / args.corpus_output
    qa_written, qa_skipped, counts, selected_doc_ids = convert_qa(source_dir, qa_out, args.qa_dir, args.max_qa_per_type)
    include_doc_ids = selected_doc_ids if args.corpus_from_selected_qa else None
    corpus_written, corpus_skipped = convert_corpus(
        source_dir,
        corpus_out,
        args.corpus_input,
        args.max_corpus,
        include_doc_ids,
    )

    print(f"Wrote QA:     {qa_out.resolve()} ({qa_written} rows, {qa_skipped} skipped, counts={counts})")
    print(f"Wrote corpus: {corpus_out.resolve()} ({corpus_written} rows, {corpus_skipped} skipped)")
    if qa_written == 0:
        print("WARNING: no QA rows were written; inspect upstream question/answer field names.", file=sys.stderr)
    if corpus_written == 0:
        print("WARNING: no corpus rows were written; unzip corpus.zip or inspect corpus field names.", file=sys.stderr)


if __name__ == "__main__":
    main()
