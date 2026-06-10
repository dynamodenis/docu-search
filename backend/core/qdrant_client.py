"""Qdrant client singleton + collection lifecycle.

Lifecycle:
    1. ensure_collection()   → creates in bulk-ingest mode (dense m=0,
                               indexing_threshold=0) so uploads are fast
                               disk writes with no graph building.
    2. bulk ingest           → upload everything you plan to load in one go.
    3. finalize_indexing()   → flips dense to m=16 / indexing_threshold=1000
                               and waits for green. Qdrant builds the HNSW
                               graph once across the whole dataset.
    4. subsequent upserts    → handled incrementally in m=16 mode, no
                               re-finalization needed.

HNSW belongs only on the dense vector: ColBERT sets m=0 (rerank-only, no
ANN graph), and sparse uses an inverted index, not HNSW. Putting HNSW at
the collection level would only have applied to dense anyway.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from qdrant_client import QdrantClient, models

from backend.config import settings
from backend.core.embeddings import COLBERT_DIM, DENSE_DIM

log = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            timeout=60,
        )
    return _client


def ensure_collection() -> bool:
    """Create the collection in bulk-ingest mode if missing.

    Returns True if it was created just now, False if it already existed.
    """
    client = get_client()
    name = settings.qdrant_collection

    if client.collection_exists(name):
        log.info("Collection %r already exists.", name)
        return False

    log.info("Creating collection %r (bulk-ingest mode, HNSW disabled)...", name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,               # no graph built during ingest
                    ef_construct=100,  # used later when we flip to m=16
                ),
            ),
            "colbert": models.VectorParams(
                size=COLBERT_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # rerank-only
                on_disk=False,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
        # Hold off indexing until finalize_indexing() is called.
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    # Payload indexes that make filtering cheap.
    for field_name, schema in [
        ("page_title", models.PayloadSchemaType.KEYWORD),
        ("section_title", models.PayloadSchemaType.KEYWORD),
        ("section_url", models.PayloadSchemaType.KEYWORD),
        ("source_label", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
        )

    log.info("Collection %r created. Run finalize_indexing() after bulk ingest.", name)
    return True


def finalize_indexing(wait: bool = True, poll_seconds: float = 3.0) -> None:
    """Flip dense to m=16 and turn indexing on. Idempotent.

    Call this once after your initial bulk ingest. Subsequent per-user
    upserts will be indexed incrementally in m=16 mode — no need to call
    this again.
    """
    client = get_client()
    name = settings.qdrant_collection

    log.info("Finalizing %r: dense m=16, indexing_threshold=1000...", name)
    client.update_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParamsDiff(
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            )
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
    )

    if not wait:
        return

    log.info("Waiting for indexing to complete...")
    while True:
        info = client.get_collection(name)
        status = info.status.value if hasattr(info.status, "value") else str(info.status)
        if status == "green":
            log.info("Collection %r ready (status=green).", name)
            return
        log.info("Status: %s — indexing in progress...", status)
        time.sleep(poll_seconds)


def list_source_labels() -> list[dict]:
    """Return the distinct ingested source labels with chunk counts.

    Uses Qdrant's facet API on the `source_label` keyword index (cheap —
    no full scan). Each entry is {"label": str, "chunks": int}, sorted by
    count descending. Falls back to a payload-only scroll if facet is
    unavailable (e.g. an older Qdrant server), and returns [] if the
    collection does not exist yet.
    """
    client = get_client()
    name = settings.qdrant_collection

    try:
        result = client.facet(collection_name=name, key="source_label", limit=1000)
        labels = [
            {"label": str(hit.value), "chunks": hit.count}
            for hit in result.hits
            if hit.value
        ]
    except Exception as e:  # noqa: BLE001
        log.warning("facet() unavailable (%s); falling back to scroll.", e)
        labels = _scroll_source_labels(client, name)

    labels.sort(key=lambda item: item["chunks"], reverse=True)
    return labels


def _scroll_source_labels(client: QdrantClient, name: str) -> list[dict]:
    """Fallback: scroll payload-only and tally source_label in Python."""
    counts: dict[str, int] = {}
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=name,
                with_payload=["source_label"],
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            for point in points:
                label = (point.payload or {}).get("source_label")
                if label:
                    counts[label] = counts.get(label, 0) + 1
            if offset is None:
                break
    except Exception as e:  # noqa: BLE001
        log.warning("Could not list source labels: %s", e)
        return []
    return [{"label": label, "chunks": count} for label, count in counts.items()]
