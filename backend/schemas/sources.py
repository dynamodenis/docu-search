from typing import List

from pydantic import BaseModel


class SourceLabel(BaseModel):
    label: str  # normalized snake_case token, e.g. "qdrant_docs"
    chunks: int  # number of chunks ingested under this label


class SourcesResponse(BaseModel):
    sources: List[SourceLabel]
    total_chunks: int
