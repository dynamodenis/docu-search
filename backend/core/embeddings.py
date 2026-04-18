"""FastEmbed model singletons.

The three models are heavy (few hundred MB combined) so we load them
once at FastAPI startup via the lifespan handler and reuse the same
instances for every request.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)

log = logging.getLogger(__name__)


DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
DENSE_DIM = 384
COLBERT_DIM = 128


@dataclass
class EmbeddingModels:
    dense: TextEmbedding
    sparse: SparseTextEmbedding
    colbert: LateInteractionTextEmbedding


_models: Optional[EmbeddingModels] = None


def load_models() -> EmbeddingModels:
    """Load all three embedding models. Called once at app startup."""
    global _models
    if _models is not None:
        return _models

    log.info("Loading embedding models (this takes a moment on first run)...")
    _models = EmbeddingModels(
        dense=TextEmbedding(DENSE_MODEL_NAME),
        sparse=SparseTextEmbedding(SPARSE_MODEL_NAME),
        colbert=LateInteractionTextEmbedding(COLBERT_MODEL_NAME),
    )
    log.info("Embedding models ready.")
    return _models


def get_models() -> EmbeddingModels:
    if _models is None:
        return load_models()
    return _models
