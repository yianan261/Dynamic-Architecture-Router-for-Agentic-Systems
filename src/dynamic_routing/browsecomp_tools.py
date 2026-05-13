"""LangChain tool: local BrowseComp-style corpus search."""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from dynamic_routing.browsecomp_env import BrowseCompCorpus


def build_local_search_tool(corpus: BrowseCompCorpus, allowed_doc_ids: list[str] | None = None):
    @tool
    def local_search(
        query: Annotated[str, "Search query over the fixed benchmark corpus."],
        top_k: Annotated[int, "Number of passages to return"] = 5,
    ) -> str:
        """Retrieve top passages from the curated local corpus (BrowseComp-Plus style)."""
        k = max(1, min(int(top_k or 5), 20))
        hits = corpus.get_by_ids(allowed_doc_ids, top_k=k) if allowed_doc_ids else corpus.search(query, top_k=k)
        if not hits:
            return "NO_RESULTS"
        lines = [f"[{h.doc_id}] score={h.score:.3f}\n{h.text}" for h in hits]
        return "\n\n---\n\n".join(lines)

    return local_search
