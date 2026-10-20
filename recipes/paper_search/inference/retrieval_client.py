"""Paper retrieval client for inference."""

from __future__ import annotations

from typing import Optional

from recipes.paper_search.env.paper_client import Paper, PaperSearchClient
from recipes.paper_search.inference.serper import ApiKeyPool, search_google_via_serper, serper_api_keys_from_env
from recipes.paper_search.inference.you_search import search_arxiv_via_you, you_api_key_from_env


class InferencePaperClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        search_source: str = "local_db",
        paper_from_month: Optional[str] = None,
        paper_to_month: Optional[str] = None,
        timeout: float = 30.0,
        serper_keys: Optional[list[str]] = None,
        serper_search_url: str = "https://google.serper.dev/search",
        you_api_key: Optional[str] = None,
    ) -> None:
        source = (search_source or "local_db").strip().lower()
        if source not in {"local_db", "google", "you"}:
            raise ValueError("search_source must be 'local_db', 'google' or 'you'")

        self.search_source = source
        self.paper_from_month = paper_from_month
        self.paper_to_month = paper_to_month
        self.serper_search_url = serper_search_url
        self._paper = PaperSearchClient(base_url=base_url, timeout=timeout)
        self._serper_pool: Optional[ApiKeyPool] = None
        if self.search_source == "google":
            keys = serper_keys if serper_keys is not None else serper_api_keys_from_env()
            if not keys:
                raise ValueError(
                    "search_source=google requires Serper API keys from PAPER_SEARCH_SERPER_API_KEYS or SERPER_API_KEY"
                )
            self._serper_pool = ApiKeyPool(keys)
        # You.com works without a key (free tier); YDC_API_KEY switches to the keyed endpoint.
        self._you_api_key = (you_api_key or you_api_key_from_env()) if self.search_source == "you" else None

    async def search(self, query: str, limit: int = 10, *, year: Optional[str] = None) -> list[Paper]:
        if self.search_source == "google":
            assert self._serper_pool is not None
            return await search_google_via_serper(
                self._paper,
                key_pool=self._serper_pool,
                query=query,
                limit=min(limit, 10),
                from_month=self.paper_from_month,
                to_month=self.paper_to_month,
                search_url=self.serper_search_url,
            )
        if self.search_source == "you":
            return await search_arxiv_via_you(
                self._paper,
                query=query,
                limit=min(limit, 10),
                api_key=self._you_api_key,
                from_month=self.paper_from_month,
                to_month=self.paper_to_month,
            )
        return await self._paper.search(query=query, limit=limit, year=year)

    async def get_paper(
        self, paper_id: str, fields: str = "title,abstract,year,authors,externalIds"
    ) -> Optional[Paper]:
        return await self._paper.get_paper(paper_id, fields=fields)

    async def get_citations(
        self, paper_id: str, limit: int = 50, fields: str = "title,abstract,year,authors,externalIds"
    ) -> list[Paper]:
        return await self._paper.get_citations(paper_id, limit=limit, fields=fields)

    async def get_references(
        self, paper_id: str, limit: int = 50, fields: str = "title,abstract,year,authors,externalIds"
    ) -> list[Paper]:
        return await self._paper.get_references(paper_id, limit=limit, fields=fields)

    async def close(self) -> None:
        await self._paper.close()
