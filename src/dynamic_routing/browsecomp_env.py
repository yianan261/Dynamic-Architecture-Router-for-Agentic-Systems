"""BrowseComp-Plus-style local corpus + keyword retrieval (fixture or user JSONL)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SearchHit:
    doc_id: str
    text: str
    score: float


class BrowseCompCorpus:
    """Tiny in-memory index: TF-style score over whitespace tokens."""

    def __init__(self, docs: list[dict[str, str]]) -> None:
        self._docs = docs

    @classmethod
    def load_jsonl(cls, path: Path) -> BrowseCompCorpus:
        docs: list[dict[str, str]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = str(row.get("doc_id") or row.get("id") or row.get("docid") or "")
                text = str(row.get("text") or row.get("contents") or row.get("body") or "")
                if doc_id and text:
                    docs.append({"doc_id": doc_id, "text": text})
        return cls(docs)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        q_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 1]
        if not q_terms:
            return []
        scored: list[tuple[float, dict[str, str]]] = []
        for d in self._docs:
            low = d["text"].lower()
            score = sum(low.count(t) for t in q_terms) + 0.01 * len(low)
            if score > 0:
                scored.append((float(score), d))
        scored.sort(key=lambda x: -x[0])
        out: list[SearchHit] = []
        for s, d in scored[:top_k]:
            out.append(SearchHit(doc_id=d["doc_id"], text=d["text"][:1200], score=s))
        return out
