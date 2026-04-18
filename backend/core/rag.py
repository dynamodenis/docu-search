"""Tool-use RAG: the LLM decides when to call search_docs vs search_web.

Flow:
    1. Send user query with two tools exposed.
    2. LLM returns one or more tool_calls.
    3. We execute them and append the results as tool messages.
    4. LLM generates the final answer citing sources.
    5. Loop until the LLM responds without tool calls (bounded by max_iters).
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from backend.config import settings
from backend.core.llm import get_llm
from backend.core.retrieval import search_docs
from backend.core.tavily_search import search_web
from backend.schemas.search import SearchResponse, Source

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a documentation assistant with two retrieval tools:

- `search_docs`  — the indexed documentation collection. Use this for
                   technical how-to, configuration, APIs, concepts.
- `search_web`   — live web search via Tavily. Use this for latest
                   versions, release notes, recent news, comparisons
                   with other products, pricing.

Rules:
- Call the tool(s) that best fit the question. Call both if the question
  has a "how-to" part AND a "latest info" part.
- After you have sources, answer using ONLY those sources.
- Cite sources inline as [Source 1], [Source 2], ... matching the order
  you received them.
- If the sources do not contain the answer, say so — do not invent.
- Keep answers concise and structured."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Hybrid retrieval (dense + BM25 + ColBERT rerank) over the "
                "indexed documentation. Use for how-to, configuration, "
                "concepts, API usage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Live web search via Tavily. Use for recent news, latest "
                "versions, release announcements, comparisons, pricing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _execute_tool(name: str, args: dict) -> List[Source]:
    query = args.get("query", "").strip()
    top_k = int(args.get("top_k", 5))
    if not query:
        return []
    if name == "search_docs":
        return search_docs(query, top_k=top_k)
    if name == "search_web":
        return search_web(query, top_k=top_k)
    log.warning("Unknown tool requested: %s", name)
    return []


def _format_sources_for_llm(sources: List[Source], offset: int) -> str:
    """Render sources as numbered context the LLM can cite."""
    if not sources:
        return "(no results)"
    blocks = []
    for i, s in enumerate(sources, start=offset + 1):
        blocks.append(
            f"[Source {i}] ({s.type}) {s.title}\n"
            f"URL: {s.url}\n"
            f"{s.snippet}"
        )
    return "\n---\n".join(blocks)


def answer_query(
    query: str,
    top_k: int = 5,
    model: Optional[str] = None,
    force_route: Optional[str] = None,
    max_iters: int = 4,
) -> SearchResponse:
    """Run the full tool-use RAG loop and return the final answer."""
    llm = get_llm()
    model_name = model or settings.openrouter_model
    collected: List[Source] = []
    routes_used: set[str] = set()

    # If the caller forced a route we skip tool-use and do it directly.
    if force_route in ("docs", "web", "both"):
        if force_route in ("docs", "both"):
            docs = search_docs(query, top_k=top_k)
            collected.extend(docs)
            if docs:
                routes_used.add("docs")
        if force_route in ("web", "both"):
            web = search_web(query, top_k=top_k)
            collected.extend(web)
            if web:
                routes_used.add("web")

        context = _format_sources_for_llm(collected, offset=0)
        completion = llm.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        answer_text = completion.choices[0].message.content or ""
        return SearchResponse(
            query=query,
            answer=answer_text,
            sources=collected,
            route_used=sorted(routes_used),  # type: ignore[arg-type]
            model=model_name,
        )

    # Tool-use loop.
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for _ in range(max_iters):
        completion = llm.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Append the assistant turn (with tool calls) so the tool results
        # have something to attach to.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ] or None,
            }
        )

        if not tool_calls:
            return SearchResponse(
                query=query,
                answer=msg.content or "",
                sources=collected,
                route_used=sorted(routes_used),  # type: ignore[arg-type]
                model=model_name,
            )

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            new_sources = _execute_tool(tc.function.name, args)
            offset = len(collected)
            collected.extend(new_sources)
            if new_sources:
                routes_used.add("docs" if tc.function.name == "search_docs" else "web")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _format_sources_for_llm(new_sources, offset=offset),
                }
            )

    # Ran out of iterations — ask the LLM to answer with whatever we have.
    messages.append(
        {
            "role": "user",
            "content": "Answer the original question now using the sources above.",
        }
    )
    final = llm.chat.completions.create(model=model_name, messages=messages)
    return SearchResponse(
        query=query,
        answer=final.choices[0].message.content or "",
        sources=collected,
        route_used=sorted(routes_used),  # type: ignore[arg-type]
        model=model_name,
    )
