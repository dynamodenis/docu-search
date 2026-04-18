"""Tavily web search wrapper."""
from __future__ import annotations

import logging
from typing import List, Optional

from tavily import TavilyClient

from backend.config import settings
from backend.schemas.search import Source

log = logging.getLogger(__name__)

_client: Optional[TavilyClient] = None


def get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def search_web(query: str, top_k: int = 5) -> List[Source]:
    try:
        res = get_client().search(
            query=query,
            max_results=top_k,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as e:
        log.warning("Tavily search failed: %s", e)
        return []

    sources: List[Source] = []
    for r in res.get("results", []):
        sources.append(
            Source(
                type="web",
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                snippet=(r.get("content") or "")[:500],
                score=float(r.get("score") or 0.0),
            )
        )
    return sources
