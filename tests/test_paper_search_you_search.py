"""Tests for the optional You.com source in paper search inference."""

import asyncio

import httpx
import pytest

from recipes.paper_search.inference import you_search
from recipes.paper_search.inference.retrieval_client import InferencePaperClient
from recipes.paper_search.inference.you_search import arxiv_id_year_month, in_month_window, search_arxiv_via_you

WEB = {
    "results": {
        "web": [
            {"url": "https://arxiv.org/abs/2401.12345v2", "title": "A"},
            {"url": "https://arxiv.org/pdf/2312.00001.pdf", "title": "B"},
            {"url": "https://example.com/not-arxiv", "title": "C"},
            {"url": "https://arxiv.org/abs/2401.12345", "title": "A again"},
            {"url": "https://arxiv.org/abs/2410.99999", "title": "D"},
        ]
    }
}


class StubPaperClient:
    def __init__(self, handler):
        self.requests = []

        def record(request):
            self.requests.append(request)
            return handler(request)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(record))
        self.fetched = []

    async def get_paper(self, paper_id, fields=""):
        self.fetched.append(paper_id)
        return {"id": paper_id}


def run(coro):
    return asyncio.run(coro)


def test_keyless_get_with_site_filter():
    stub = StubPaperClient(lambda r: httpx.Response(200, json=WEB))
    papers = run(search_arxiv_via_you(stub, query="tool learning", limit=5))

    req = stub.requests[0]
    assert req.method == "GET"
    assert str(req.url).startswith(you_search.YOU_KEYLESS_URL)
    assert req.url.params["query"] == "tool learning site:arxiv.org"
    assert req.url.params["count"] == "5"
    assert req.headers["User-Agent"] == "youdotcom-integration/agentr1-agent-r1"
    assert "X-API-Key" not in req.headers
    assert stub.fetched == ["2401.12345", "2312.00001", "2410.99999"]
    assert len(papers) == 3


def test_keyed_post():
    stub = StubPaperClient(lambda r: httpx.Response(200, json=WEB))
    run(search_arxiv_via_you(stub, query="q", limit=2, api_key="ydc-test"))

    req = stub.requests[0]
    assert req.method == "POST"
    assert str(req.url) == you_search.YOU_KEYED_URL
    assert req.headers["X-API-Key"] == "ydc-test"
    assert b'"count":2' in req.content.replace(b" ", b"")
    assert stub.fetched == ["2401.12345", "2312.00001"]


def test_month_window_filters_on_arxiv_id():
    stub = StubPaperClient(lambda r: httpx.Response(200, json=WEB))
    run(search_arxiv_via_you(stub, query="q", limit=5, from_month="2024-01", to_month="2024-09"))
    assert stub.fetched == ["2401.12345"]
    assert stub.requests[0].url.params["count"] == "10"


def test_arxiv_id_year_month():
    assert arxiv_id_year_month("2401.12345") == (2024, 1)
    assert arxiv_id_year_month("0704.0001") == (2007, 4)
    assert arxiv_id_year_month("hep-th/9901001") == (1999, 1)
    assert arxiv_id_year_month("bogus") is None
    assert in_month_window("bogus", None, None)
    assert not in_month_window("bogus", "2024-01", None)


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_and_credit_errors(status):
    stub = StubPaperClient(lambda r: httpx.Response(status))
    with pytest.raises(RuntimeError):
        run(search_arxiv_via_you(stub, query="q", limit=3, api_key="bad"))
    assert len(stub.requests) == 1


def test_rate_limit_retries_then_reports():
    stub = StubPaperClient(lambda r: httpx.Response(429, headers={"Retry-After": "0"}))
    with pytest.raises(RuntimeError, match="100 searches a day"):
        run(search_arxiv_via_you(stub, query="q", limit=3))
    assert len(stub.requests) == 3


def test_empty_results():
    stub = StubPaperClient(lambda r: httpx.Response(200, json={"results": {"web": []}}))
    assert run(search_arxiv_via_you(stub, query="q", limit=3)) == []


def test_inference_client_accepts_you_without_key(monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    client = InferencePaperClient(base_url="http://localhost:4000", search_source="you")
    assert client.search_source == "you"
    assert client._you_api_key is None
    monkeypatch.setenv("YDC_API_KEY", "k")
    assert InferencePaperClient(base_url="http://localhost:4000", search_source="YOU")._you_api_key == "k"
    with pytest.raises(ValueError):
        InferencePaperClient(search_source="bing")
