"""Unit tests for SQLite semantic cache (exact hash, cosine similarity, TTL, scope, eviction)."""

from __future__ import annotations

import time
from pathlib import Path

from sentinel_gateway.cache import SemanticCache


class TestSemanticCache:
    def test_exact_hit_and_semantic_paraphrase_hit(self, tmp_path: Path) -> None:
        cache = SemanticCache(
            db_path=tmp_path / "cache.db",
            default_threshold=0.86,
            ttl_seconds=3600,
        )
        try:
            cache.store(
                prompt="What is semantic caching and how does it reduce LLM latency and costs?",
                response_text=(
                    "Semantic caching stores prompt embeddings to serve similar queries fast."
                ),
                model="sentinel-demo-v1",
                provider="local-demo",
                prompt_tokens_est=18,
                completion_tokens_est=24,
            )

            # Exact hit fast path
            exact = cache.lookup(
                "  what is semantic caching and how does it reduce llm latency and costs? ",
                model="sentinel-demo-v1",
            )
            assert exact is not None
            assert exact.exact_hit is True
            assert exact.similarity == 1.0

            # Semantic paraphrase hit
            semantic = cache.lookup(
                (
                    "Can you explain semantic caching and how it cuts LLM "
                    "response latency and pricing?"
                ),
                model="sentinel-demo-v1",
            )
            assert semantic is not None
            assert semantic.exact_hit is False
            assert semantic.similarity >= 0.86
            assert "Semantic caching" in semantic.response_text

            # Distinct unrelated query must miss
            miss = cache.lookup(
                "How do I configure Kubernetes horizontal pod autoscaling?",
                model="sentinel-demo-v1",
            )
            assert miss is None
        finally:
            cache.close()

    def test_model_and_system_prompt_scope_isolation(self, tmp_path: Path) -> None:
        cache = SemanticCache(db_path=tmp_path / "scope.db")
        try:
            cache.store(
                prompt="Explain Luhn validation",
                response_text="Response A",
                model="model-a",
                provider="local-demo",
                system_prompt="You are a concise assistant.",
            )
            assert cache.lookup("Explain Luhn validation", model="model-b") is None
            assert (
                cache.lookup(
                    "Explain Luhn validation",
                    model="model-a",
                    system_prompt="You are a pirate.",
                )
                is None
            )
            hit = cache.lookup(
                "Explain Luhn validation",
                model="model-a",
                system_prompt="You are a concise assistant.",
            )
            assert hit is not None
            assert hit.response_text == "Response A"
        finally:
            cache.close()

    def test_ttl_expiration_and_clear(self, tmp_path: Path) -> None:
        cache = SemanticCache(db_path=tmp_path / "ttl.db", ttl_seconds=1)
        try:
            cache.store(
                prompt="Short lived entry",
                response_text="Temporary answer",
                model="sentinel-demo-v1",
                provider="local-demo",
                ttl_seconds=1,
            )
            assert cache.count() == 1
            time.sleep(1.1)
            assert cache.lookup("Short lived entry", model="sentinel-demo-v1") is None
            assert cache.count() == 0

            cache.store(
                prompt="Another entry",
                response_text="Answer 2",
                model="sentinel-demo-v1",
                provider="local-demo",
                ttl_seconds=300,
            )
            cleared = cache.clear()
            assert cleared == 1
            assert cache.count() == 0
        finally:
            cache.close()
