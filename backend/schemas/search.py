from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    # Override the default LLM for this request.
    model: Optional[str] = None
    # Force a specific retrieval route. None = let the LLM decide via tool-use.
    force_route: Optional[Literal["docs", "web", "both"]] = None


class Source(BaseModel):
    type: Literal["docs", "web"]
    title: str
    url: str
    snippet: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    route_used: List[Literal["docs", "web"]]
    model: str
