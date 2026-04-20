import logging

from fastapi import APIRouter, HTTPException

from backend.core.rag import answer_query
from backend.schemas.search import SearchRequest, SearchResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        return answer_query(
            query=req.query,
            top_k=req.top_k,
            model=req.model,
            force_route=req.force_route,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Search failed for query=%r", req.query)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
