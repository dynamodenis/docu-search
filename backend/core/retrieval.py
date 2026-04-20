"""Hybrid search: dense + sparse prefetch, ColBERT rerank."""
from __future__ import annotations

from typing import List, Optional

from qdrant_client import models

from backend.config import settings
from backend.core.embeddings import get_models
from backend.core.qdrant_client import get_client
from backend.schemas.search import Source


def search_docs(query: str, top_k: int = 5, source_label: Optional[str] = None) -> List[Source]:
    m = get_models()
    client = get_client()

    dense_q = list(m.dense.embed([query]))[0]
    sparse_raw = list(m.sparse.embed([query]))[0].as_object()
    sparse_q = models.SparseVector(
        indices=sparse_raw["indices"].tolist(),
        values=sparse_raw["values"].tolist(),
    )
    colbert_q = list(m.colbert.embed([query]))[0]

    query_filter = None
    if source_label:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_label",
                    match=models.MatchValue(value=source_label),
                )
            ]
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=dense_q, using="dense", limit=50, filter=query_filter),
            models.Prefetch(query=sparse_q, using="sparse", limit=50, filter=query_filter),
        ],
        query=colbert_q,
        using="colbert",
        limit=top_k,
        with_payload=True,
    )

    sources: List[Source] = []
    for point in results.points:
        p = point.payload or {}
        sources.append(
            Source(
                type="docs",
                title=f"{p.get('page_title', 'Untitled')} — {p.get('section_title', '')}".strip(" —"),
                url=p.get("section_url", ""),
                snippet=p.get("chunk_text") or "",
                score=float(point.score or 0.0),
            )
        )
    return sources
