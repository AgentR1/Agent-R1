"""Regression test: the arXiv publication-date cutoff must keep old papers.

`PaperSearchRuntimeConfig.max_arxiv_yymm` is a YYMM value (``2410`` == 2024-10).
The cutoff is applied in ``PaperSearchRuntime._is_before_arxiv_cutoff`` and must
drop only papers published *after* it, whatever id scheme the paper uses.
"""

from __future__ import annotations

import asyncio

from recipes.paper_search.env.paper_client import Paper
from recipes.paper_search.runtime import PaperSearchRuntime, PaperSearchRuntimeConfig

CUTOFF_YYMM = 2410  # 2024-10, the value in recipes/paper_search/inference/default.yaml


class _StubPaperClient:
    """Stands in for the paper-search service, returning a fixed candidate list."""

    def __init__(self, papers: list[Paper]) -> None:
        self._papers = papers

    async def search(self, query: str, limit: int = 10, *, year=None, **kwargs) -> list[Paper]:
        return list(self._papers)

    async def close(self) -> None:
        return None


class _StubSelectorClient:
    """Stands in for the selector service and rates every paper as fully relevant."""

    async def classify(self, prompt: str) -> float:
        return 0.0

    async def close(self) -> None:
        return None


def _runtime_with(papers: list[Paper]) -> PaperSearchRuntime:
    runtime = PaperSearchRuntime(
        config=PaperSearchRuntimeConfig(max_arxiv_yymm=CUTOFF_YYMM),
        paper_client=_StubPaperClient(papers),
        selector_client=_StubSelectorClient(),
    )
    runtime.reset("test query")
    return runtime


def _kept_ids(papers: list[Paper]) -> set[str]:
    runtime = _runtime_with(papers)
    asyncio.run(runtime.search("a search query"))
    return set(runtime.paper_pool.papers)


def _paper(paper_id: str, arxiv_id: str, year: int) -> Paper:
    return Paper(paper_id=paper_id, raw_paper_id=paper_id, arxiv_id=arxiv_id, title=paper_id, abstract="a", year=year)


def test_new_style_arxiv_ids_are_cut_at_the_configured_month():
    kept = _kept_ids(
        [
            _paper("2409.08765", "2409.08765", 2024),  # 2024-09, before the cutoff
            _paper("2411.11111", "2411.11111", 2024),  # 2024-11, after the cutoff
        ]
    )
    assert kept == {"2409.08765"}


def test_old_style_arxiv_ids_before_the_cutoff_are_kept():
    """Old-style ids (``hep-th/9901001``) carry no YYMM prefix, so only ``year`` is available."""
    kept = _kept_ids(
        [
            _paper("hep-th/9901001", "hep-th/9901001", 1999),
            _paper("cs/0112017", "cs/0112017", 2001),
        ]
    )
    assert kept == {"hep-th/9901001", "cs/0112017"}


def test_non_arxiv_records_before_the_cutoff_are_kept():
    kept = _kept_ids([_paper("S2HASH0001", "", 1998)])
    assert kept == {"S2HASH0001"}


def test_papers_after_the_cutoff_are_still_dropped():
    kept = _kept_ids(
        [
            _paper("hep-th/9912001", "hep-th/9912001", 1999),
            _paper("CSHASH2025", "", 2025),
        ]
    )
    assert kept == {"hep-th/9912001"}
