import re
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


def normalize_source_label(raw: str) -> str:
    """Normalize to a stable keyword-index-safe token.

    "Example Docs!" → "example_docs"
    "Qdrant/Docs"   → "qdrant_docs"

    Keeps filters reliable because KEYWORD index matching is exact
    and case-sensitive.
    """
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "user_submitted"


class IngestRequest(BaseModel):
    # Either a list of individual page URLs, or a sitemap URL to expand.
    urls: List[HttpUrl] = Field(default_factory=list)
    sitemap_url: Optional[HttpUrl] = None
    # Free-form label that will be stored on each chunk's payload,
    # so users can later filter by source. Normalized to lowercase
    # snake_case so that "Example Docs" and "example_docs" become the
    # same bucket.
    source_label: str = Field("user_submitted", max_length=64)
    # None → use server default. Upper bound is generous; the scraper
    # itself dedupes so the true upper bound is sitemap size.
    max_pages: Optional[int] = Field(None, ge=1, le=10000)

    @field_validator("source_label", mode="before")
    @classmethod
    def _normalize_label(cls, v):
        if not isinstance(v, str):
            return v
        return normalize_source_label(v)


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    total_urls: int


JobStatus = Literal["queued", "running", "completed", "failed"]


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    total_urls: int
    pages_scraped: int = 0
    chunks_upserted: int = 0
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
