"""Tests for BrowseComp-style local corpus search (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_routing.browsecomp_env import BrowseCompCorpus


def test_corpus_search_finds_revenue_doc() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "benchmarks" / "browsecomp_corpus_fixture.jsonl"
    c = BrowseCompCorpus.load_jsonl(path)
    hits = c.search("Acme Corp revenue 2023", top_k=2)
    assert hits
    assert "120" in hits[0].text or any("120" in h.text for h in hits)
