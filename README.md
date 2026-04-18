# docu-search

Hybrid RAG over documentation with optional live web search.

Retrieval is Qdrant (dense `bge-small` + BM25 sparse + ColBERT rerank). Web
search is Tavily. An LLM picks which tool to call via OpenAI-style tool
calling — routed through **OpenRouter** so you can swap between Claude,
GPT, Gemini, etc. with one env var.

Built on top of the hybrid pipeline from
[QDrant/Documentation engine project](../QDrant/Documentation%20engine%20project)
by Denis Mbugua.

## Architecture

```
┌────────────────────┐    HTTP     ┌────────────────────────────┐
│  Streamlit (UI)    │ ──────────▶ │  FastAPI backend           │
│  frontend/app.py   │             │                            │
└────────────────────┘             │  POST /search              │
                                   │  POST /ingest  (bg task)   │
                                   │  GET  /jobs/{id}           │
                                   │  GET  /health              │
                                   │                            │
                                   │  core/                     │
                                   │   ├─ embeddings (FastEmbed)│
                                   │   ├─ qdrant_client         │
                                   │   ├─ retrieval (hybrid)    │
                                   │   ├─ tavily_search         │
                                   │   ├─ llm (OpenRouter)      │
                                   │   └─ rag (tool-use loop)   │
                                   │                            │
                                   │  jobs/                     │
                                   │   ├─ store (in-memory)     │
                                   │   └─ ingest_job (scrape →  │
                                   │        chunk → embed →     │
                                   │        upsert)             │
                                   └──────┬───────────┬─────────┘
                                          │           │
                               ┌──────────▼──┐   ┌────▼──────┐
                               │ Qdrant Cloud│   │  Tavily   │
                               └─────────────┘   └───────────┘
```

Single shared collection for now. Every chunk carries a `source_label`
payload so we can later filter by submitter or flip to per-user
collections without a data migration.

## Quick start

```bash
# 1. Set up
cp .env.example .env            # then fill in real keys
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create the Qdrant collection (idempotent)
python -m scripts.bootstrap_collection

# 3. Run backend
uvicorn backend.main:app --reload
#    → http://localhost:8000/docs

# 4. Run frontend (new terminal)
streamlit run frontend/app.py
#    → http://localhost:8501

# 5. After your initial bulk ingest (see below), run ONCE:
python -m scripts.finalize_collection
```

Or in one shot with `make dev` (runs both).

## Collection lifecycle

```
ensure_collection()   →  bulk ingest (/ingest)  →  finalize_indexing()
  dense m=0                all HNSW off              dense m=16
  threshold=0              fast disk writes          threshold=1000
                                                     builds graph once
                                                     waits for green
                                                         ↓
                                          subsequent user ingests run
                                          incrementally in m=16 mode
                                          (no re-finalize needed)
```

`m=0` during bulk load is the standard Qdrant pattern: HNSW built once
across the full dataset produces a higher-quality graph than an
incremental build during upload, and the upload itself is much faster.
ColBERT stays at `m=0` forever (it reranks a candidate set, no ANN graph
needed). Sparse uses an inverted index, not HNSW.

## API

### `POST /search`

```json
{
  "query": "how do I configure HNSW for better recall?",
  "top_k": 5,
  "model": "openai/gpt-4o-mini",        // optional override
  "force_route": null                    // null | "docs" | "web" | "both"
}
```

When `force_route` is null the LLM uses tool calling to pick
`search_docs`, `search_web`, or both.

### `POST /ingest`

```json
{
  "urls": ["https://example.com/docs/page"],
  "sitemap_url": "https://example.com/sitemap.xml",
  "source_label": "example_docs",
  "max_pages": 50
}
```

Returns a `job_id`. Poll `GET /jobs/{id}` for progress.

## Ingestion flow

```
URL(s)  ──▶  scrape (requests + BeautifulSoup + markdownify)
        ──▶  chunk_markdown  (heading split + sentence window w/ overlap)
        ──▶  embed (dense + BM25 + ColBERT)
        ──▶  Qdrant upsert (batches of 16)
```

## Deploying

- **Qdrant**: Qdrant Cloud — you already have this.
- **Backend**: Railway, Fly.io, or Render free tier. The first `/search`
  call in a cold process will take ~30s while FastEmbed loads models.
  Consider a `preload` hook or keeping the dyno warm.
- **Frontend**: Streamlit Community Cloud. Point `BACKEND_URL` at the
  deployed API. Swap for Next.js later when you want a proper API product.

## What's next

- Add API keys + usage metering when opening ingestion to the public.
- Move jobs into Redis / Celery so you can scale to multiple workers.
- Add per-user collections once traffic justifies it.
- Bring back the semantic (Layer 3) chunker if eval shows it matters on
  user content.
