import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from recipes.paper_search.env.paper_client import Paper, PaperPool, PaperSearchClient, SelectorClient
from recipes.paper_search.prompts import PAPERSEARCH_SYSTEM_PROMPT, PAPERSEARCH_USER_PROMPT, SELECT_PROMPT
from recipes.paper_search.utils import (
    PAPER_SEARCH_TOOL_NAMES,
    decode_tool_arguments,
    extract_expand_paper_id,
    extract_search_query,
)

logger = logging.getLogger(__file__)


@dataclass
class PaperSearchRuntimeConfig:
    max_steps: int = 5
    max_parallel_calls: int = 5
    reward_top_k: int = 3
    score_threshold: float = 0.4
    search_cost: float = 0.0
    expand_cost: float = 0.0
    use_discrete_reward: bool = False
    search_top_k: int = 10
    citations_limit: int = 30
    references_limit: int = -1
    search_year: Optional[str] = None
    max_arxiv_yymm: Optional[int] = None


class PaperSearchRuntime:
    def __init__(
        self,
        *,
        config: PaperSearchRuntimeConfig,
        paper_client: PaperSearchClient,
        selector_client: SelectorClient,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.client = paper_client
        self.selector_client = selector_client
        self.logger = logger_ or logger
        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.history_search_queries: dict[str, int] = {}
        self.history_actions: list[tuple[str, str]] = []
        self.ordered_paper_ids: list[str] = []
        self.user_query = ""

    def reset(self, user_query: str) -> None:
        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.history_search_queries = {}
        self.history_actions = []
        self.ordered_paper_ids = []
        self.user_query = user_query

    def format_history_actions(self) -> str:
        if not self.history_actions:
            return "None"

        lines: list[str] = []
        for action, value in self.history_actions:
            if action == "search":
                lines.append(f"[Search] {value}")
            elif action == "expand":
                lines.append(f"[Expand] {value}")
            else:
                raise ValueError(f"Invalid action: {action}")
        return "\n".join(lines) if lines else "None"

    def make_user_prompt(self) -> str:
        return PAPERSEARCH_USER_PROMPT.format(
            user_query=self.user_query,
            paper_list=self.paper_pool.paper_list,
            history_actions=self.format_history_actions(),
        )

    def make_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": PAPERSEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": self.make_user_prompt()},
        ]

    def summarize_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            raw_args = getattr(tool_call, "arguments", "")
            try:
                import json

                arguments = json.loads(raw_args) if raw_args else {}
            except Exception:
                arguments = {"raw_arguments": raw_args}
            summaries.append({"name": tool_call.name, "arguments": arguments})
        return summaries

    async def execute_tool_calls(self, tool_calls: list[Any]) -> tuple[float, list[dict[str, Any]], dict[str, int]]:
        tool_calls = tool_calls[: self.config.max_parallel_calls]
        tasks = []
        summaries: list[dict[str, Any]] = []
        counters = {"search": 0, "expand": 0}

        for tool_call in tool_calls:
            if tool_call.name not in PAPER_SEARCH_TOOL_NAMES:
                self.logger.warning("Unknown tool call: %s", tool_call.name)
                continue

            tool_args = decode_tool_arguments(tool_call.name, tool_call.arguments)
            if not tool_args:
                self.logger.warning("Invalid tool arguments for %s: %r", tool_call.name, tool_call.arguments)
                continue

            summaries.append({"name": tool_call.name, "arguments": tool_args})
            if tool_call.name == "search":
                query = extract_search_query(tool_args)
                if query:
                    self.history_actions.append(("search", query))
                    tasks.append(self.search(query))
                    counters["search"] += 1
            elif tool_call.name == "expand":
                paper_id = extract_expand_paper_id(tool_args)
                if paper_id:
                    self.history_actions.append(("expand", paper_id))
                    tasks.append(self.expand(paper_id))
                    counters["expand"] += 1

        reward_total = sum(await asyncio.gather(*tasks)) if tasks else 0.0
        return reward_total, summaries, counters

    async def search(self, query: str) -> float:
        if query in self.history_search_queries:
            return -0.5

        try:
            papers = await self.client.search(query=query, limit=self.config.search_top_k, year=self.config.search_year)
        except Exception as exc:
            self.logger.warning("Error in search %s: %r", query, exc)
            self.history_search_queries[query] = 0
            return 0.0

        new_papers: list[Paper] = []
        tasks = []
        seen_paper_ids: set[str] = set()

        for paper in papers:
            if not paper.paper_id or paper.paper_id in seen_paper_ids:
                continue
            if not self._is_before_arxiv_cutoff(paper):
                continue
            seen_paper_ids.add(paper.paper_id)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores, strict=False):
            if score < 0.01:
                continue

            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

                kept_scores.append(score)
                self.paper_pool.add_paper(paper, "search", query, score)
                self.ordered_paper_ids.append(paper.paper_id)
            self.logger.info("[%.3f] %s", score, paper.title)

        self.history_search_queries[query] = len(kept_scores)
        if not kept_scores:
            return 0.0

        top_k_scores = sorted(kept_scores, reverse=True)[: self.config.reward_top_k]
        if self.config.use_discrete_reward:
            top_k_scores = [1.0 if score >= self.config.score_threshold else 0.0 for score in top_k_scores]
        return sum(top_k_scores) - self.config.search_cost

    async def expand(self, paper_id: str) -> float:
        with self._paper_pool_lock:
            paper_pool_entry = self.paper_pool.get_paper(paper_id)
            if not paper_pool_entry:
                return -0.5
            if paper_pool_entry.expand:
                return -0.5
            paper_pool_entry.expand = True

        try:
            citations, references = await asyncio.gather(
                self.client.get_citations(paper_id, limit=self.config.citations_limit),
                self.client.get_references(paper_id, limit=self.config.references_limit),
            )
        except Exception as exc:
            self.logger.warning("Error in expand %s: %r", paper_id, exc)
            return 0.0

        merged_candidates: list[Paper] = []
        seen_paper_ids: set[str] = set()
        for paper in citations + references:
            if not paper.paper_id or paper.paper_id == paper_id or paper.paper_id in seen_paper_ids:
                continue
            if not self._is_before_arxiv_cutoff(paper):
                continue
            if not paper.abstract:
                continue
            seen_paper_ids.add(paper.paper_id)
            merged_candidates.append(paper)

        new_papers: list[Paper] = []
        tasks = []
        for paper in merged_candidates:
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores, strict=False):
            if score < 0.01:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

                kept_scores.append(score)
                self.paper_pool.add_paper(paper, "expand", paper_pool_entry.paper.title, score)
                self.ordered_paper_ids.append(paper.paper_id)
            self.logger.info("[%.3f] %s", score, paper.title)

        if not kept_scores:
            return 0.0

        top_k_scores = sorted(kept_scores, reverse=True)[: self.config.reward_top_k]
        if self.config.use_discrete_reward:
            top_k_scores = [1.0 if score >= self.config.score_threshold else 0.0 for score in top_k_scores]
        return sum(top_k_scores) - self.config.expand_cost

    async def get_relevance_score(self, query: str, paper: Paper) -> float:
        prompt = SELECT_PROMPT.format(title=paper.title, abstract=paper.abstract, user_query=query)
        try:
            score = await self.selector_client.classify(prompt)
        except Exception as exc:
            self.logger.warning("Selector service failed for paper_id=%s: %r", paper.paper_id, exc)
            return 0.0
        return float(1.0 - score)

    def build_save_items(self) -> dict[str, Any]:
        ranked_entries = list(reversed(self.paper_pool.ranked_papers))
        save_items: dict[str, Any] = {
            "ordered_ids": list(self.ordered_paper_ids),
            "sorted_ids": [entry.paper.paper_id for entry in ranked_entries],
            "details": {},
        }
        for entry in ranked_entries:
            paper = entry.paper
            save_items["details"][paper.paper_id] = {
                "paper_id": paper.paper_id,
                "raw_paper_id": paper.raw_paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "year": paper.year,
                "score": entry.score,
                "source": entry.source,
                "origin": entry.origin,
                "expand": entry.expand,
            }
        return save_items

    def _is_before_arxiv_cutoff(self, paper: Paper) -> bool:
        """Whether *paper* was published at or before ``max_arxiv_yymm`` (YYMM, e.g. ``2410`` == 2024-10)."""
        if self.config.max_arxiv_yymm is None:
            return True
        arxiv_id = paper.arxiv_id or paper.paper_id
        prefix = arxiv_id.split("v", 1)[0].split(".", 1)[0]
        if len(prefix) == 4 and prefix.isdigit():
            return int(prefix) <= self.config.max_arxiv_yymm
        if paper.year is not None:
            # Papers without a new-style arXiv id only expose a 4-digit year, so compare years.
            # YYMM carries a 2-digit year (9108 == Aug 1991), hence the century expansion.
            cutoff_yy = self.config.max_arxiv_yymm // 100
            cutoff_year = 1900 + cutoff_yy if cutoff_yy >= 91 else 2000 + cutoff_yy
            if int(paper.year) > cutoff_year:
                return False
        return True
