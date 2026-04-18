"""FastAPI entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import admin, health, ingest, search
from backend.config import settings
from backend.core.embeddings import load_models
from backend.core.qdrant_client import ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting docu-search backend...")
    load_models()
    ensure_collection()
    log.info("Ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="docu-search",
    version="0.1.0",
    description="Hybrid RAG over documentation + Tavily web search.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(ingest.router)
app.include_router(admin.router)


@app.get("/", tags=["system"])
def root():
    return {"name": "docu-search", "docs": "/docs", "health": "/health"}
