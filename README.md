---
title: docu-search
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

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

## Self-hosting the backend

The backend is a standalone HTTP service — FastAPI behind uvicorn, with
plain JSON endpoints (`/search`, `/ingest`, `/jobs/{id}`, `/admin/*`).
The Streamlit app in `frontend/` is just one consumer of that API; it's
a thin client that POSTs to `/search` and `/ingest`.

**This section talks about the backend in isolation on purpose.** If
you'd rather build your own UI — Next.js, SvelteKit, a Slack bot, a CLI,
another agent, a Raycast extension — skip the bundled frontend entirely
and point your client at the API. The backend is the product; the
frontend is interchangeable.

### Option A: Docker Compose (recommended)

```bash
cp .env.example .env            # fill in QDRANT_*, OPENROUTER_API_KEY, TAVILY_API_KEY
docker compose up --build backend
#   → http://localhost:8000/docs  (Swagger UI)
```

`docker compose up backend` starts only the API. Drop the service name
to also bring up the bundled Streamlit frontend.

### Option B: Bare metal

```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.bootstrap_collection      # one-time: create the Qdrant collection
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Minimum env vars to run the backend

| Variable             | Required | Purpose                                     |
| -------------------- | -------- | ------------------------------------------- |
| `QDRANT_URL`         | yes      | Qdrant Cloud cluster URL                    |
| `QDRANT_API_KEY`     | yes      | Qdrant Cloud API key                        |
| `QDRANT_COLLECTION`  | yes      | Collection name (e.g. `docs_search`)        |
| `OPENROUTER_API_KEY` | yes      | LLM gateway key                             |
| `OPENROUTER_MODEL`   | yes      | Default model — must support tool calling   |
| `ADMIN_TOKEN`        | yes      | Shared secret for `/admin/*` endpoints      |
| `TAVILY_API_KEY`     | optional | Only needed if your callers use `search_web`|
| `CORS_ORIGINS`       | optional | Set this if a browser-based client calls in |

### Calling the API

Ingest some docs (returns a `job_id` immediately; the scrape runs in the background):

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "sitemap_url": "https://qdrant.tech/sitemap.xml",
    "source_label": "qdrant_docs",
    "max_pages": 50
  }'
# → {"job_id": "abc123", "status": "queued"}
```

Poll the job:

```bash
curl http://localhost:8000/jobs/abc123
# → {"status": "running", "pages_done": 12, "pages_total": 50, ...}
```

After your first bulk ingest, finalize the index once (builds the HNSW
graph in one pass — see Collection lifecycle below):

```bash
curl -X POST http://localhost:8000/admin/finalize \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Search:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I configure HNSW for better recall?",
    "top_k": 5
  }'
```

Response includes the grounded answer, citations, and which tool the LLM
picked (`search_docs`, `search_web`, or both).

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

## Choosing the LLM

Retrieval (dense + BM25 + ColBERT rerank) does the heavy lifting. By the
time chunks land in the LLM's context they are already highly relevant,
so the LLM's job is narrow: read 3–10 pre-filtered chunks, write a
grounded answer, cite sources, and pick which tool to call (`search_docs`
vs `search_web`). A small model handles that fine.

**Default is `anthropic/claude-haiku-4.5`.** Cheap, fast, supports tool
calling. Don't pay for Sonnet/Opus/GPT-4o unless you have evidence the
small models are failing on your domain.

Good cheap options (all support tool calling):

| Model                          | Strength                                  |
| ------------------------------ | ----------------------------------------- |
| `anthropic/claude-haiku-4.5`   | balanced; solid citation discipline       |
| `openai/gpt-4o-mini`           | cheapest mainstream; good structured out  |
| `google/gemini-2.5-flash`      | fastest; long context                     |

Switch globally with `OPENROUTER_MODEL` in `.env`, or per-request by
passing `"model": "..."` in the `/search` body.

Rule of thumb: **upgrade the retriever, not the generator**, until your
eval shows the LLM itself is the bottleneck.

### Why this is right for this setup

The retrieval pipeline is doing the work that costs money in a naive RAG:

- **Dense retrieval** narrows from the full corpus to 50 semantically relevant candidates.
- **BM25** catches exact keyword matches dense misses.
- **ColBERT rerank** picks the truly relevant ones out of the merged pool.

By the time the LLM sees the context, you have handed it a small, clean,
high-signal set of chunks. Its remaining job is basically "summarize
these 5 paragraphs and cite them" — something Haiku / gpt-4o-mini do
perfectly well.

You would only want a frontier model if:

- The retrieved chunks contradict each other and need complex reasoning to reconcile.
- You're doing multi-hop reasoning (rare for doc Q&A).
- You observe the small model hallucinating or dropping citations.

None of those apply here. The `0.80 MRR` already tells you retrieval is
good — the generator does not need to be smart, it needs to be faithful.
Haiku 4.5 costs roughly 1/10th of Sonnet 4.5 for the same input/output.
Over thousands of LinkedIn demo queries that's the difference between
"free hobby project" and "surprise bill".

## Deploying

- **Qdrant**: Qdrant Cloud — you already have this.
- **Backend**: Hugging Face Spaces (Docker SDK) — free, 16 GB RAM, fits
  the embedding stack. See below.
- **Frontend**: Streamlit Community Cloud — free, points at the Space URL.

### Backend → Hugging Face Spaces

1. Create a new Space → SDK `Docker` → link to your GitHub repo (or push
   this repo directly to the Space remote). HF reads the YAML header at
   the top of this file for title/port/etc.
2. Settings → **Repository secrets** → add: `QDRANT_URL`,
   `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `OPENROUTER_API_KEY`,
   `OPENROUTER_MODEL`, `TAVILY_API_KEY`, `ADMIN_TOKEN`, `CORS_ORIGINS`.
   Set `CORS_ORIGINS` to your Streamlit Cloud URL
   (e.g. `https://<you>-docu-search.streamlit.app`).
3. The first build takes ~5 min (it pre-bakes the three embedding models
   into the image — that's why your cold starts after that are fast).
4. Your backend is live at `https://<user>-<space>.hf.space`.

Notes:
- Free Spaces sleep after ~48h idle. First request after sleep cold-starts
  the container (~30s). BackgroundTasks state (the ingest job store) is
  in-memory, so sleeping mid-ingest loses job status — run bulk ingests
  deliberately.
- The Space disk is ephemeral; that's fine because Qdrant holds all data.

### Frontend → Streamlit Community Cloud

1. Point at `frontend/app.py`.
2. Secrets (`Settings → Secrets`): `BACKEND_URL="https://<user>-<space>.hf.space"`.
3. Deploy. Copy the URL it gives you back into the backend's
   `CORS_ORIGINS` secret on HF, then restart the Space.

## What's next

- Add API keys + usage metering when opening ingestion to the public.
- Move jobs into Redis / Celery so you can scale to multiple workers.
- Add per-user collections once traffic justifies it.
- Bring back the semantic (Layer 3) chunker if eval shows it matters on
  user content.
