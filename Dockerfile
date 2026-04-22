# docu-search backend — FastAPI + fastembed + Qdrant client.
#
# Lives at the repo root because Hugging Face Spaces (Docker SDK)
# requires the Dockerfile at the root of the Space repo. Also used
# locally via docker-compose.
#
#     docker build -t docu-search-backend .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    FASTEMBED_CACHE_PATH=/opt/fastembed-cache

WORKDIR /app

# lxml / onnxruntime need a few native libs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Pre-download the three embedding models at build time so the first
# request after deploy is fast. Adds ~1GB to the image but removes a
# cold-start stall users would otherwise see.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25'); \
LateInteractionTextEmbedding('colbert-ir/colbertv2.0')"

COPY backend ./backend
COPY scripts ./scripts

EXPOSE 8000

# Single worker: embedding models are held in-process and BackgroundTasks
# state lives in memory. Multiple workers would duplicate RAM and split
# job state across processes.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
