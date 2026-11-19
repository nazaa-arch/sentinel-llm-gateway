"""Configuration settings loaded from environment variables and optional .env files."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ProviderName = Literal["local-demo", "ollama", "gemini", "openai"]
PolicyMode = Literal["block", "redact", "warn", "allow"]


def _parse_dotenv_file(env_path: Path) -> dict[str, str]:
    """Parse key=value lines from a .env file without external dependencies."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


class Settings(BaseModel):
    """Application settings with ``SENTINEL_*`` environment variable binding."""

    app_name: str = "Sentinel LLM Gateway"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    # Provider settings
    provider: ProviderName = "local-demo"
    model: str = "sentinel-demo-v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    provider_timeout_seconds: float = Field(default=30.0, gt=0.1)
    fallback_to_local_demo: bool = True

    # Guardrails & PII policy
    policy_mode: PolicyMode = "block"
    injection_block_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    injection_warn_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    toxicity_block_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    enable_pii_redaction: bool = True
    restore_pii_by_default: bool = False

    # Semantic cache
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = Field(default=0.86, ge=0.5, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    cache_max_entries: int = Field(default=2000, ge=10)

    # Persistence & rate limiting
    db_path: str = "data/sentinel_gateway.db"
    rate_limit_rpm: int = Field(default=120, ge=1)

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        """Load settings from optional ``.env`` file overlaid with ``SENTINEL_*`` env vars."""
        merged_env: dict[str, str] = {}
        if env_file is not None:
            merged_env.update(_parse_dotenv_file(Path(env_file)))
        merged_env.update(os.environ)

        payload: dict[str, Any] = {}
        for field_name in cls.model_fields:
            env_key = f"SENTINEL_{field_name.upper()}"
            if env_key in merged_env:
                raw_value = merged_env[env_key]
                if raw_value == "" and field_name.endswith("_api_key"):
                    payload[field_name] = None
                else:
                    payload[field_name] = raw_value
        return cls.model_validate(payload)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings loaded from environment."""
    return Settings.from_env()
