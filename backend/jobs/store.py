"""In-memory job store.

Good enough for a single backend process. Swap for Redis / a database
when we want multiple workers or job persistence across restarts.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from backend.schemas.ingest import JobState, JobStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, total_urls: int) -> JobState:
        now = datetime.now(timezone.utc)
        job = JobState(
            job_id=str(uuid.uuid4()),
            status="queued",
            total_urls=total_urls,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        pages_scraped: Optional[int] = None,
        chunks_upserted: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status is not None:
                job.status = status
            if pages_scraped is not None:
                job.pages_scraped = pages_scraped
            if chunks_upserted is not None:
                job.chunks_upserted = chunks_upserted
            if error is not None:
                job.error = error
            job.updated_at = datetime.now(timezone.utc)


job_store = JobStore()
