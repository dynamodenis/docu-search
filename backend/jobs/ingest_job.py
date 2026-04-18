"""Background ingestion: scrape → chunk → embed → upsert.

Run inside FastAPI BackgroundTasks for now. For multi-worker / durable
retries, move the same function behind Celery or an async queue.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from qdrant_client import models

from backend.config import settings
from backend.core.chunking import Chunk, chunk_markdown
from backend.core.embeddings import get_models
from backend.core.qdrant_client import get_client
from backend.core.scraper import resolve_target_urls, scrape_page
from backend.jobs.store import job_store

log = logging.getLogger(__name__)

UPSERT_BATCH = 16


def _build_points(chunks: List[Chunk], source_label: str):
    m = get_models()
    texts = [c.text for c in chunks]

    dense = list(m.dense.embed(texts))
    sparse = list(m.sparse.embed(texts))
    colbert = list(m.colbert.embed(texts))

    points = []
    for i, chunk in enumerate(chunks):
        sv = sparse[i].as_object()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense[i].tolist(),
                    "sparse": models.SparseVector(**sv),
                    "colbert": colbert[i].tolist(),
                },
                payload={
                    "chunk_text": chunk.text,
                    "page_title": chunk.page_title,
                    "section_title": chunk.section_title,
                    "section_url": chunk.section_url,
                    "chunk_index": chunk.chunk_index,
                    "source_label": source_label,
                },
            )
        )
    return points


def run_ingest_job(
    job_id: str,
    urls: List[str],
    sitemap_url: Optional[str],
    source_label: str,
    max_pages: int,
) -> None:
    job_store.update(job_id, status="running")
    try:
        targets = resolve_target_urls(urls, sitemap_url, limit=max_pages)
        if not targets:
            job_store.update(job_id, status="failed", error="No URLs to ingest.")
            return

        client = get_client()
        total_scraped = 0
        total_chunks = 0

        for url in targets:
            page = scrape_page(url)
            total_scraped += 1
            if not page or not page.markdown.strip():
                job_store.update(job_id, pages_scraped=total_scraped)
                continue

            chunks = chunk_markdown(page.markdown, page_title=page.title, page_url=page.url)
            if not chunks:
                job_store.update(job_id, pages_scraped=total_scraped)
                continue

            for i in range(0, len(chunks), UPSERT_BATCH):
                batch = chunks[i : i + UPSERT_BATCH]
                points = _build_points(batch, source_label=source_label)
                client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points,
                )
                total_chunks += len(points)

            job_store.update(
                job_id,
                pages_scraped=total_scraped,
                chunks_upserted=total_chunks,
            )

        job_store.update(job_id, status="completed")
        log.info(
            "Ingest job %s done: %d pages, %d chunks.",
            job_id,
            total_scraped,
            total_chunks,
        )

    except Exception as e:
        log.exception("Ingest job %s failed", job_id)
        job_store.update(job_id, status="failed", error=str(e))
