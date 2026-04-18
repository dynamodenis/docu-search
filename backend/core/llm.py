"""OpenRouter client (uses the OpenAI-compatible SDK).

Centralised here so every module that wants an LLM just imports
`get_llm()` and does not care which provider sits behind it.
"""
from __future__ import annotations

from typing import Optional

from openai import OpenAI

from backend.config import settings

_client: Optional[OpenAI] = None

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm() -> OpenAI:
    global _client
    if _client is None:
        default_headers = {}
        if settings.openrouter_site_url:
            default_headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            default_headers["X-Title"] = settings.openrouter_app_name

        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key,
            default_headers=default_headers or None,
        )
    return _client
