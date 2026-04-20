from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.config import settings
from backend.jobs.ingest_job import run_ingest_job
from backend.jobs.store import job_store
from backend.schemas.ingest import IngestRequest, IngestResponse, JobState

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, bg: BackgroundTasks) -> IngestResponse:
    if not req.urls and not req.sitemap_url:
        raise HTTPException(400, "Provide at least one URL or a sitemap_url.")

    # If the caller specified max_pages, respect it. Otherwise fall back
    # to the server-wide default. The default is a convenience ceiling,
    # not a hard cap — callers who know what they're doing can go higher.
    max_pages = req.max_pages if req.max_pages else settings.ingest_max_pages

    # We record an upper bound on URLs now; the actual count after sitemap
    # expansion is updated once the job runs.
    total = len(req.urls) if not req.sitemap_url else max_pages
    job = job_store.create(total_urls=total)

    bg.add_task(
        run_ingest_job,
        job_id=job.job_id,
        urls=[str(u) for u in req.urls],
        sitemap_url=str(req.sitemap_url) if req.sitemap_url else None,
        source_label=req.source_label,
        max_pages=max_pages,
    )
    return IngestResponse(job_id=job.job_id, status="queued", total_urls=total)


@router.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str) -> JobState:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job
