from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    # Either a list of individual page URLs, or a sitemap URL to expand.
    urls: List[HttpUrl] = Field(default_factory=list)
    sitemap_url: Optional[HttpUrl] = None
    # Free-form label that will be stored on each chunk's payload,
    # so users can later filter by source.
    source_label: str = Field("user_submitted", max_length=64)
    max_pages: Optional[int] = Field(None, ge=1, le=500)


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
