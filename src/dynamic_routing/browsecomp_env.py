"""BrowseComp-Plus-style local corpus + keyword retrieval (fixture or user JSONL)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass
class SearchHit:
    doc_id: str
    text: str
    score: float


class BrowseCompCorpus:
    """Tiny in-memory BM25-style index over local JSONL corpus rows."""

    def __init__(self, docs: list[dict[str, str]]) -> None:
        self._docs = docs
        self._by_id = {doc["doc_id"]: doc for doc in docs}
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_len = 1.0
        self._build_index()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            t
            for t in re.split(r"\W+", text.lower())
            if len(t) > 2 and t not in STOPWORDS
        ]

    def _build_index(self) -> None:
        total_len = 0
        for doc in self._docs:
            tokens = self._tokens(doc["text"])
            self._doc_tokens.append(tokens)
            total_len += len(tokens)
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        if self._doc_tokens:
            self._avg_len = max(1.0, total_len / len(self._doc_tokens))

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
        q_terms = list(dict.fromkeys(self._tokens(query)))
        if not q_terms:
            return []
        scored: list[tuple[float, dict[str, str]]] = []
        corpus_size = max(1, len(self._docs))
        k1 = 1.2
        b = 0.75
        for d, tokens in zip(self._docs, self._doc_tokens):
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for token in tokens:
                if token in q_terms:
                    tf[token] = tf.get(token, 0) + 1
            if not tf:
                continue
            doc_len = len(tokens)
            score = 0.0
            for token, freq in tf.items():
                df = self._doc_freq.get(token, 0)
                idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / self._avg_len)
                score += idf * ((freq * (k1 + 1)) / denom)
            if score > 0:
                scored.append((float(score), d))
        scored.sort(key=lambda x: -x[0])
        out: list[SearchHit] = []
        for s, d in scored[:top_k]:
            out.append(SearchHit(doc_id=d["doc_id"], text=d["text"][:1200], score=s))
        return out

    def get_by_ids(self, doc_ids: list[str], top_k: int = 5) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for doc_id in doc_ids:
            doc = self._by_id.get(str(doc_id))
            if doc:
                hits.append(SearchHit(doc_id=doc["doc_id"], text=doc["text"], score=1.0))
            if len(hits) >= top_k:
                break
        return hits
