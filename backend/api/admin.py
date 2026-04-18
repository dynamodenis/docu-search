"""Admin endpoints. Protected by a shared ADMIN_TOKEN header.

Not real auth — just enough to keep casual traffic from triggering
collection-level operations. Swap for proper auth when the app grows.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException

from backend.config import settings
from backend.core.qdrant_client import finalize_indexing

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_admin_token: str | None) -> None:
    expected = settings.admin_token
    if not expected:
        raise HTTPException(500, "ADMIN_TOKEN is not configured on the server.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "Invalid or missing X-Admin-Token header.")


@router.post("/finalize")
def finalize(
    bg: BackgroundTasks,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    """Flip the collection from bulk-ingest mode to search mode.

    Runs the graph build in the background so the HTTP call returns
    immediately. Poll /health to see when the collection is green.
    """
    _require_admin(x_admin_token)
    bg.add_task(finalize_indexing, wait=True)
    return {
        "status": "started",
        "message": "finalize_indexing running in background; poll /health for status",
    }
