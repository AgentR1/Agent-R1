"""You.com search support for paper search inference.

Opt-in with ``PAPERSEARCH_INFER_SEARCH_SOURCE=you``. Works without an API key:

* No key: ``GET https://api.you.com/v1/agents/search`` (free tier, 100 searches
  a day, counted per IP).
* ``YDC_API_KEY`` set: ``POST https://ydc-index.io/v1/search`` with ``X-API-Key``.

Queries are restricted with ``site:arxiv.org``. The ``paper_from_month`` /
``paper_to_month`` window is applied to the arXiv ID itself (new-style IDs
encode YYMM), so the date filter does not depend on search-engine date operators.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Optional

import httpx

from recipes.paper_search.env.http_retry import httpx_request_with_retry
from recipes.paper_search.env.paper_client import Paper, PaperSearchClient
from recipes.paper_search.inference.date_utils import parse_year_month_str
from recipes.paper_search.inference.serper import extract_arxiv_id_from_url

YOU_KEYLESS_URL = "https://api.you.com/v1/agents/search"
YOU_KEYED_URL = "https://ydc-index.io/v1/search"
YOU_USER_AGENT = "youdotcom-integration/agentr1-agent-r1"

_NEW_STYLE_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
_OLD_STYLE_ID = re.compile(r"^[a-z.-]+/(\d{2})(\d{2})\d{3}$", re.IGNORECASE)


def you_api_key_from_env() -> Optional[str]:
    key = os.getenv("YDC_API_KEY", "").strip()
    return key or None


def build_you_search_query(query: str) -> str:
    return f"{query.strip()} site:arxiv.org"


def arxiv_id_year_month(paper_id: str) -> Optional[tuple[int, int]]:
    match = _NEW_STYLE_ID.match(paper_id) or _OLD_STYLE_ID.match(paper_id)
    if not match:
        return None
    yy, mm = int(match.group(1)), int(match.group(2))
    # Old-style IDs run 1991-2007; new-style IDs start in 2007.
    year = 1900 + yy if yy >= 91 else 2000 + yy
    return year, mm


def in_month_window(paper_id: str, from_month: Optional[str], to_month: Optional[str]) -> bool:
    if not from_month and not to_month:
        return True
    ym = arxiv_id_year_month(paper_id)
    if ym is None:
        return False
    if from_month and ym < parse_year_month_str(from_month):
        return False
    if to_month and ym > parse_year_month_str(to_month):
        return False
    return True


async def search_arxiv_via_you(
    paper_client: PaperSearchClient,
    *,
    query: str,
    limit: int,
    api_key: Optional[str] = None,
    from_month: Optional[str] = None,
    to_month: Optional[str] = None,
    fields: str = "title,abstract,year,authors,externalIds",
) -> list[Paper]:
    if limit <= 0:
        return []

    q = build_you_search_query(query)
    # Ask for a few extra hits when a month window may drop some of them.
    count = min(limit * 2, 20) if (from_month or to_month) else limit
    headers = {"User-Agent": YOU_USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        method, url, kwargs = "POST", YOU_KEYED_URL, {"json": {"query": q, "count": count}}
    else:
        method, url, kwargs = "GET", YOU_KEYLESS_URL, {"params": {"query": q, "count": count}}

    resp = await httpx_request_with_retry(
        paper_client.client,
        method,
        url,
        headers=headers,
        max_retries=2,
        retry_status_codes={429, 500, 502, 503, 504},
        retry_exceptions=(httpx.RequestError, httpx.TimeoutException),
        **kwargs,
    )
    if resp.status_code in (401, 403):
        raise RuntimeError(f"You.com search rejected the API key (HTTP {resp.status_code}); check YDC_API_KEY")
    if resp.status_code == 402:
        raise RuntimeError("You.com search: out of credits (HTTP 402)")
    if resp.status_code == 429:
        hint = "" if api_key else " The keyless tier allows 100 searches a day; set YDC_API_KEY for more."
        raise RuntimeError(
            f"You.com search rate limited (HTTP 429, Retry-After={resp.headers.get('Retry-After')}).{hint}"
        )
    resp.raise_for_status()

    web = (resp.json().get("results") or {}).get("web")
    if not isinstance(web, list):
        return []

    paper_ids: list[str] = []
    seen: set[str] = set()
    for item in web:
        if not isinstance(item, dict):
            continue
        paper_id = extract_arxiv_id_from_url(str(item.get("url") or ""))
        if not paper_id or paper_id in seen or not in_month_window(paper_id, from_month, to_month):
            continue
        seen.add(paper_id)
        paper_ids.append(paper_id)
        if len(paper_ids) >= limit:
            break

    tasks = [paper_client.get_paper(paper_id, fields=fields) for paper_id in paper_ids]
    papers = await asyncio.gather(*tasks) if tasks else []
    return [paper for paper in papers if paper is not None]
