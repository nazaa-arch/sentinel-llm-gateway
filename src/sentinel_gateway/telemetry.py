"""SQLite observability store, FinOps metrics aggregation, and sliding-window rate limiter."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sentinel_gateway.providers import estimate_cost_usd
from sentinel_gateway.schemas import AuditRecord, MetricsSummary


class SlidingWindowRateLimiter:
    """In-memory per-client sliding-window RPM rate limiter with idle-bucket eviction."""

    def __init__(
        self,
        max_requests_per_minute: int = 120,
        max_tracked_clients: int = 10_000,
    ) -> None:
        self.max_rpm = max_requests_per_minute
        self.max_tracked_clients = max_tracked_clients
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def _evict_stale_locked(self, cutoff: float) -> None:
        stale_keys = [
            key
            for key, timestamps in self._buckets.items()
            if not timestamps or timestamps[-1] < cutoff
        ]
        for key in stale_keys:
            del self._buckets[key]

    def allow(self, client_key: str) -> bool:
        """Return True if client is within RPM quota, False otherwise."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            if len(self._buckets) >= self.max_tracked_clients:
                self._evict_stale_locked(cutoff)
            bucket = self._buckets.get(client_key)
            if bucket is None:
                bucket = deque()
                self._buckets[client_key] = bucket
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_rpm:
                return False
            bucket.append(now)
            return True

    def retry_after_seconds(self, client_key: str) -> int:
        """Return whole seconds until the oldest request in the client's window expires."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(client_key)
            if not bucket:
                return 1
            remaining = max(1.0, 60.0 - (now - bucket[0]))
            return int(remaining) + (1 if remaining % 1 > 0 else 0)


class TelemetryStore:
    """SQLite audit log and real-time gateway observability aggregator."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """Lazily open the SQLite connection and initialize schema under lock."""
        if self._conn is not None:
            return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn = conn
        self._init_schema(conn)
        return conn

    def initialize(self) -> None:
        """Eagerly initialize the underlying SQLite connection and schema."""
        with self._lock:
            self._ensure_connected()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    prompt_tokens_est INTEGER NOT NULL,
                    completion_tokens_est INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    estimated_cost_saved_usd REAL NOT NULL DEFAULT 0.0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    cache_similarity REAL,
                    pii_count INTEGER NOT NULL DEFAULT 0,
                    pii_types_json TEXT NOT NULL DEFAULT '[]',
                    injection_score REAL NOT NULL DEFAULT 0.0,
                    toxicity_score REAL NOT NULL DEFAULT 0.0,
                    action_taken TEXT NOT NULL,
                    redacted_prompt_preview TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs (timestamp DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs (action_taken);"
            )

    def record(
        self,
        *,
        provider: str,
        model: str,
        latency_ms: float,
        prompt_tokens_est: int,
        completion_tokens_est: int,
        cache_hit: bool,
        cache_similarity: float | None,
        pii_count: int,
        pii_types: list[str],
        injection_score: float,
        toxicity_score: float,
        action_taken: str,
        redacted_prompt_preview: str,
    ) -> int:
        """Insert a single redacted audit record and return its row ID."""
        timestamp = datetime.now(UTC).isoformat()
        inferred_cost = estimate_cost_usd(model, prompt_tokens_est, completion_tokens_est)
        if cache_hit:
            actual_cost = 0.0
            saved_cost = inferred_cost
        elif action_taken == "block":
            actual_cost = 0.0
            saved_cost = 0.0
        else:
            actual_cost = inferred_cost
            saved_cost = 0.0

        preview = redacted_prompt_preview.replace("\n", " ").strip()[:180]
        with self._lock:
            conn = self._ensure_connected()
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO audit_logs (
                        timestamp, provider, model, latency_ms, prompt_tokens_est,
                        completion_tokens_est, estimated_cost_usd, estimated_cost_saved_usd,
                        cache_hit, cache_similarity, pii_count, pii_types_json,
                        injection_score, toxicity_score, action_taken, redacted_prompt_preview
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        provider,
                        model,
                        round(latency_ms, 3),
                        prompt_tokens_est,
                        completion_tokens_est,
                        actual_cost,
                        saved_cost,
                        1 if cache_hit else 0,
                        cache_similarity,
                        pii_count,
                        json.dumps(sorted(set(pii_types))),
                        round(injection_score, 4),
                        round(toxicity_score, 4),
                        action_taken,
                        preview,
                    ),
                )
            return int(cur.lastrowid)

    def get_metrics(self) -> MetricsSummary:
        """Compute live p50/p95 latency, cache hit rate, block rate, and FinOps savings."""
        with self._lock:
            conn = self._ensure_connected()
            rows = conn.execute(
                """
                SELECT provider, latency_ms, estimated_cost_usd, estimated_cost_saved_usd,
                       cache_hit, pii_count, action_taken
                FROM audit_logs
                """
            ).fetchall()

        total = len(rows)
        if total == 0:
            return MetricsSummary(
                total_requests=0,
                blocked_requests=0,
                warned_requests=0,
                redacted_requests=0,
                allowed_requests=0,
                cache_hits=0,
                cache_hit_rate=0.0,
                guardrail_block_rate=0.0,
                total_pii_redacted=0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_latency_ms=0.0,
                total_estimated_cost_usd=0.0,
                estimated_cost_saved_usd=0.0,
                provider_breakdown={},
            )

        latencies = np.array([r[1] for r in rows], dtype=np.float64)
        blocked = sum(1 for r in rows if r[6] == "block")
        warned = sum(1 for r in rows if r[6] == "warn")
        redacted = sum(1 for r in rows if r[6] == "redact")
        allowed = sum(1 for r in rows if r[6] == "allow")
        cache_hits = sum(1 for r in rows if r[4] == 1)
        non_blocked = max(1, total - blocked)
        total_pii = sum(int(r[5]) for r in rows)
        total_cost = sum(float(r[2]) for r in rows)
        saved_cost = sum(float(r[3]) for r in rows)

        provider_breakdown: dict[str, int] = {}
        for r in rows:
            provider_breakdown[r[0]] = provider_breakdown.get(r[0], 0) + 1

        return MetricsSummary(
            total_requests=total,
            blocked_requests=blocked,
            warned_requests=warned,
            redacted_requests=redacted,
            allowed_requests=allowed,
            cache_hits=cache_hits,
            cache_hit_rate=round(cache_hits / non_blocked, 4),
            guardrail_block_rate=round(blocked / total, 4),
            total_pii_redacted=total_pii,
            p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
            avg_latency_ms=round(float(np.mean(latencies)), 2),
            total_estimated_cost_usd=round(total_cost, 6),
            estimated_cost_saved_usd=round(saved_cost, 6),
            provider_breakdown=provider_breakdown,
        )

    def get_recent_audits(
        self,
        limit: int = 50,
        action: str | None = None,
    ) -> list[AuditRecord]:
        """Fetch most recent audit records, optionally filtered by policy action."""
        safe_limit = max(1, min(limit, 500))
        with self._lock:
            conn = self._ensure_connected()
            if action:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, provider, model, latency_ms, prompt_tokens_est,
                           completion_tokens_est, estimated_cost_usd, cache_hit, cache_similarity,
                           pii_count, pii_types_json, injection_score, toxicity_score,
                           action_taken, redacted_prompt_preview
                    FROM audit_logs
                    WHERE action_taken = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (action, safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, provider, model, latency_ms, prompt_tokens_est,
                           completion_tokens_est, estimated_cost_usd, cache_hit, cache_similarity,
                           pii_count, pii_types_json, injection_score, toxicity_score,
                           action_taken, redacted_prompt_preview
                    FROM audit_logs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        return [
            AuditRecord(
                id=r[0],
                timestamp=r[1],
                provider=r[2],
                model=r[3],
                latency_ms=r[4],
                prompt_tokens_est=r[5],
                completion_tokens_est=r[6],
                estimated_cost_usd=r[7],
                cache_hit=bool(r[8]),
                cache_similarity=r[9],
                pii_count=r[10],
                pii_types=json.loads(r[11]) if r[11] else [],
                injection_score=r[12],
                toxicity_score=r[13],
                action_taken=r[14],
                redacted_prompt_preview=r[15],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
