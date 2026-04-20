"""Centralised settings loaded from environment / .env.

Every other module should import `settings` from here rather than
reading os.environ directly. Keeps config changes to one file.
"""
from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "docs_search"

    # OpenRouter
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-haiku-4.5"
    openrouter_site_url: str = ""
    openrouter_app_name: str = "docu-search"

    # Tavily
    tavily_api_key: str

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    # NoDecode prevents pydantic-settings from JSON-parsing the env value
    # so our comma-splitting validator below can handle "a,b,c" format.
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    ingest_max_pages: int = 1000
    scraper_user_agent: str = "docu-search-bot/0.1"
    admin_token: str = ""

    # Frontend
    backend_url: str = "http://localhost:8000"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
