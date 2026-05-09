#!/usr/bin/env python3
"""Convert BrowseComp-Plus data into this project's runner JSONL schema.

Expected runner output:
  queries JSONL rows: {"query_id": str, "query": str, "gold_answer": str, ...}
  corpus JSONL rows:  {"doc_id": str, "text": str, ...}

The official BrowseComp-Plus decrypt script produces a decrypted JSONL file and
optionally a two-column TSV with ``query_id`` and ``query``. The decrypted JSONL
is the source of truth for gold answers; the TSV is only used to fill/override
query text by id. No answers or corpus documents are fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


QUERY_ID_KEYS = ("query_id", "id", "qid")
QUERY_TEXT_KEYS = ("query", "question", "text")
ANSWER_KEYS = (
    "gold_answer",
    "answer",
    "answers",
    "reference_answer",
    "final_answer",
    "target",
    "gold",
)
DOC_ID_KEYS = ("doc_id", "docid", "_id", "id", "pid")
DOC_TEXT_KEYS = ("text", "contents", "content", "body", "passage", "document")


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
        for key in ANSWER_KEYS + QUERY_TEXT_KEYS + DOC_TEXT_KEYS:
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


def _read_query_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid = row[0].strip()
            query = row[1].strip()
            if qid and query and qid.lower() not in {"query_id", "qid", "id"}:
                out[qid] = query
    return out


def _read_qrels(path: Path | None) -> dict[str, list[str]]:
    if not path:
        return {}
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid, doc_id = parts[0], parts[2]
            if qid and doc_id:
                out.setdefault(qid, []).append(doc_id)
    return out


def _doc_ids_from_docs(value: Any) -> list[str]:
    doc_ids: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                doc_id = _stringify(_pick(item, DOC_ID_KEYS))
                if doc_id:
                    doc_ids.append(doc_id)
    return doc_ids


def convert_queries(
    decrypted_jsonl: Path,
    queries_tsv: Path | None,
    output_path: Path,
    max_queries: int,
    qrel_evidence: Path | None = None,
    qrel_golds: Path | None = None,
) -> tuple[int, int, set[str]]:
    tsv_queries = _read_query_tsv(queries_tsv) if queries_tsv else {}
    evidence_by_qid = _read_qrels(qrel_evidence)
    golds_by_qid = _read_qrels(qrel_golds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    selected_doc_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(decrypted_jsonl):
            qid = _stringify(_pick(row, QUERY_ID_KEYS)) or f"BCP-{written + skipped}"
            query = tsv_queries.get(qid) or _stringify(_pick(row, QUERY_TEXT_KEYS))
            gold = _stringify(_pick(row, ANSWER_KEYS))
            if not query or not gold:
                skipped += 1
                continue
            gold_doc_ids = golds_by_qid.get(qid) or _doc_ids_from_docs(row.get("gold_docs"))
            evidence_doc_ids = evidence_by_qid.get(qid) or gold_doc_ids
            selected_doc_ids.update(gold_doc_ids)
            selected_doc_ids.update(evidence_doc_ids)
            converted: dict[str, Any] = {
                "query_id": qid,
                "query": query,
                "gold_answer": gold,
            }
            if gold_doc_ids:
                converted["gold_doc_ids"] = gold_doc_ids
            if evidence_doc_ids:
                converted["evidence_doc_ids"] = evidence_doc_ids
            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
            if max_queries > 0 and written >= max_queries:
                break
    return written, skipped, selected_doc_ids


def _iter_local_corpus(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".jsonl", ".json", ".tsv", ".parquet"}:
                yield from _iter_local_corpus(child)
        return

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj.get("data", obj.get("documents", obj.get("corpus", obj))) if isinstance(obj, dict) else obj
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row
        return
    if suffix == ".tsv":
        with path.open(encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
            if has_header:
                yield from csv.DictReader(f, delimiter="\t")
            else:
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) >= 2:
                        yield {"doc_id": row[0], "text": row[1]}
        return
    if suffix == ".parquet":
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise SystemExit("Reading parquet corpus exports requires `pip install datasets`.") from exc
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        for row in dataset:
            yield dict(row)
        return
    raise SystemExit(f"Unsupported corpus input format: {path}")


def _iter_hf_corpus(split: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit("Loading the HuggingFace corpus requires `pip install datasets`.") from exc
    dataset = load_dataset("Tevatron/browsecomp-plus-corpus", split=split)
    for row in dataset:
        yield dict(row)


def _normalize_corpus_row(row: dict[str, Any], index: int) -> dict[str, str] | None:
    doc_id = _stringify(_pick(row, DOC_ID_KEYS)) or f"BCP-DOC-{index}"
    text = _stringify(_pick(row, DOC_TEXT_KEYS))
    title = _stringify(row.get("title"))
    if title and text and title not in text[: len(title) + 10]:
        text = f"{title}\n{text}"
    if not text:
        return None
    out = {"doc_id": doc_id, "text": text}
    url = _stringify(row.get("url"))
    if url:
        out["url"] = url
    return out


def _iter_decrypted_docs(decrypted_jsonl: Path, include_doc_ids: set[str] | None) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for row in _iter_jsonl(decrypted_jsonl):
        for key in ("gold_docs", "evidence_docs", "docs"):
            docs = row.get(key)
            if not isinstance(docs, list):
                continue
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                doc_id = _stringify(_pick(doc, DOC_ID_KEYS))
                if include_doc_ids is not None and doc_id not in include_doc_ids:
                    continue
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                yield doc


def convert_corpus(
    output_path: Path,
    corpus_input: Path | None,
    use_hf_corpus: bool,
    hf_split: str,
    max_corpus: int,
    decrypted_jsonl: Path | None = None,
    from_decrypted_docs: bool = False,
    include_doc_ids: set[str] | None = None,
) -> tuple[int, int]:
    if not corpus_input and not use_hf_corpus and not from_decrypted_docs:
        return 0, 0
    if from_decrypted_docs:
        if decrypted_jsonl is None:
            raise SystemExit("--corpus-from-decrypted-docs requires --decrypted-jsonl")
        rows = _iter_decrypted_docs(decrypted_jsonl, include_doc_ids)
    else:
        rows = _iter_hf_corpus(hf_split) if use_hf_corpus else _iter_local_corpus(corpus_input)  # type: ignore[arg-type]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            norm = _normalize_corpus_row(row, index)
            if norm is None:
                skipped += 1
                continue
            if include_doc_ids is not None and norm["doc_id"] not in include_doc_ids:
                skipped += 1
                continue
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            written += 1
            if max_corpus > 0 and written >= max_corpus:
                break
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BrowseComp-Plus decrypted data to runner JSONL.")
    parser.add_argument("--decrypted-jsonl", type=Path, required=True, help="Output from upstream decrypt_dataset.py")
    parser.add_argument("--queries-tsv", type=Path, default=None, help="Optional upstream two-column query_id/query TSV")
    parser.add_argument("--qrel-evidence", type=Path, default=None, help="Optional qrel evidence file with query/document IDs")
    parser.add_argument("--qrel-golds", type=Path, default=None, help="Optional qrel golds file with query/document IDs")
    parser.add_argument("--corpus-input", type=Path, default=None, help="Optional local corpus export: JSONL, JSON, TSV, parquet, or directory")
    parser.add_argument("--hf-corpus", action="store_true", help="Load Tevatron/browsecomp-plus-corpus directly via datasets")
    parser.add_argument(
        "--corpus-from-decrypted-docs",
        action="store_true",
        help="Build corpus from decrypted row gold_docs/evidence_docs/docs, filtered to converted query doc IDs.",
    )
    parser.add_argument("--hf-split", default="train", help="HuggingFace corpus split when --hf-corpus is used")
    parser.add_argument("--output-dir", type=Path, default=Path("data/converted/browsecomp_plus"))
    parser.add_argument("--queries-output", default="browsecomp_plus_queries.jsonl")
    parser.add_argument("--corpus-output", default="browsecomp_plus_corpus.jsonl")
    parser.add_argument("--max-queries", type=int, default=0, help="Tiny validation limit; 0 means all")
    parser.add_argument("--max-corpus", type=int, default=0, help="Tiny validation limit; 0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.decrypted_jsonl.is_file():
        raise SystemExit(f"Missing decrypted JSONL: {args.decrypted_jsonl}")
    if args.queries_tsv and not args.queries_tsv.is_file():
        raise SystemExit(f"Missing queries TSV: {args.queries_tsv}")
    if args.qrel_evidence and not args.qrel_evidence.is_file():
        raise SystemExit(f"Missing qrel evidence file: {args.qrel_evidence}")
    if args.qrel_golds and not args.qrel_golds.is_file():
        raise SystemExit(f"Missing qrel golds file: {args.qrel_golds}")
    if args.corpus_input and not args.corpus_input.exists():
        raise SystemExit(f"Missing corpus input: {args.corpus_input}")

    queries_out = args.output_dir / args.queries_output
    corpus_out = args.output_dir / args.corpus_output
    q_written, q_skipped, selected_doc_ids = convert_queries(
        args.decrypted_jsonl,
        args.queries_tsv,
        queries_out,
        args.max_queries,
        args.qrel_evidence,
        args.qrel_golds,
    )
    include_doc_ids = selected_doc_ids if selected_doc_ids else None
    c_written, c_skipped = convert_corpus(
        corpus_out,
        args.corpus_input,
        args.hf_corpus,
        args.hf_split,
        args.max_corpus,
        args.decrypted_jsonl,
        args.corpus_from_decrypted_docs,
        include_doc_ids,
    )

    print(f"Wrote queries: {queries_out.resolve()} ({q_written} rows, {q_skipped} skipped)")
    if args.corpus_input or args.hf_corpus or args.corpus_from_decrypted_docs:
        print(f"Wrote corpus:  {corpus_out.resolve()} ({c_written} rows, {c_skipped} skipped)")
    else:
        print("Skipped corpus conversion: pass --corpus-input or --hf-corpus to generate browsecomp_plus_corpus.jsonl")
    if q_written == 0:
        print("WARNING: no query rows were written; inspect upstream answer field names.", file=sys.stderr)


if __name__ == "__main__":
    main()
