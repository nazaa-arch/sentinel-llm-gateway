"""Shared pytest fixtures for isolated Sentinel LLM Gateway testing."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel_gateway.api import create_app
from sentinel_gateway.config import Settings


@pytest.fixture()
def gateway_client(tmp_path: Path) -> Iterator[TestClient]:
    """Create a FastAPI TestClient backed by an isolated temporary SQLite database."""
    settings = Settings(
        db_path=str(tmp_path / "test_sentinel.db"),
        provider="local-demo",
        model="sentinel-demo-v1",
        policy_mode="block",
        enable_pii_redaction=True,
        enable_semantic_cache=True,
        cache_similarity_threshold=0.86,
        rate_limit_rpm=100,
    )
    app = create_app(settings=settings, simulated_demo_latency_ms=2.0)
    with TestClient(app) as client:
        yield client
