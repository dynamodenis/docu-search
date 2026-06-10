from fastapi import APIRouter

from backend.core.qdrant_client import list_source_labels
from backend.schemas.sources import SourceLabel, SourcesResponse

router = APIRouter(tags=["sources"])


@router.get("/sources", response_model=SourcesResponse)
def get_sources() -> SourcesResponse:
    """List the distinct ingested source labels with per-source chunk counts.

    Backed by a Qdrant facet on the `source_label` keyword index, so this
    is cheap to call and safe to refetch after each ingest completes.
    """
    labels = list_source_labels()
    sources = [SourceLabel(label=item["label"], chunks=item["chunks"]) for item in labels]
    return SourcesResponse(
        sources=sources,
        total_chunks=sum(item["chunks"] for item in labels),
    )
