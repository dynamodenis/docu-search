from fastapi import APIRouter

from backend.config import settings
from backend.core.qdrant_client import get_client

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    """Lightweight probe. Checks Qdrant is reachable and reports index status."""
    client = get_client()
    qdrant_ok = True
    qdrant_err = None
    collection_status = None
    points = None
    try:
        client.get_collections()
        try:
            info = client.get_collection(settings.qdrant_collection)
            collection_status = (
                info.status.value if hasattr(info.status, "value") else str(info.status)
            )
            points = info.points_count
        except Exception:
            collection_status = "missing"
    except Exception as e:  # noqa: BLE001
        qdrant_ok = False
        qdrant_err = str(e)

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "collection": settings.qdrant_collection,
        "collection_status": collection_status,  # green = indexed & ready
        "points": points,
        "model": settings.openrouter_model,
        **({"qdrant_error": qdrant_err} if not qdrant_ok else {}),
    }
