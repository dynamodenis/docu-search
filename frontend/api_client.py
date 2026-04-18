"""Thin HTTP client so the Streamlit UI does not touch the backend modules."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class APIClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("BACKEND_URL", "http://localhost:8000")).rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> Dict[str, Any]:
        r = self.client.get("/health")
        r.raise_for_status()
        return r.json()

    def search(
        self,
        query: str,
        top_k: int = 5,
        model: Optional[str] = None,
        force_route: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {"query": query, "top_k": top_k}
        if model:
            payload["model"] = model
        if force_route:
            payload["force_route"] = force_route
        r = self.client.post("/search", json=payload)
        r.raise_for_status()
        return r.json()

    def ingest(
        self,
        urls: List[str],
        sitemap_url: Optional[str],
        source_label: str,
        max_pages: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "urls": urls,
            "source_label": source_label,
        }
        if sitemap_url:
            payload["sitemap_url"] = sitemap_url
        if max_pages:
            payload["max_pages"] = max_pages
        r = self.client.post("/ingest", json=payload)
        r.raise_for_status()
        return r.json()

    def job(self, job_id: str) -> Dict[str, Any]:
        r = self.client.get(f"/jobs/{job_id}")
        r.raise_for_status()
        return r.json()
