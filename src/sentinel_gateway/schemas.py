"""Pydantic request, response, guardrail, and telemetry schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from sentinel_gateway.config import PolicyMode, ProviderName


class ChatMessage(BaseModel):
    """Single OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(..., max_length=100_000)


class SentinelOptions(BaseModel):
    """Per-request Sentinel gateway policy overrides."""

    redact_pii: bool | None = None
    restore_pii: bool | None = None
    use_cache: bool | None = None
    policy_mode: PolicyMode | None = None
    provider: ProviderName | None = None
    cache_threshold: float | None = Field(default=None, ge=0.5, le=1.0)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible ``/v1/chat/completions`` request payload."""

    model: str | None = Field(default=None, max_length=128)
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=128)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=512, ge=1, le=16384)
    stream: bool = False
    user: str | None = Field(default=None, max_length=128)
    sentinel_options: SentinelOptions | None = None


class UsageStats(BaseModel):
    """Estimated or provider-reported token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """Single completion choice following OpenAI API structure."""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class PIIEntityOut(BaseModel):
    """Serialized PII detection metadata (never exposes full raw secret)."""

    entity_type: str
    placeholder: str
    start: int
    end: int
    masked_preview: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardSignalOut(BaseModel):
    """Single triggered guardrail rule signal."""

    category: Literal["prompt_injection", "toxicity", "obfuscation"]
    rule_id: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    matched_snippet: str


class SentinelMetadata(BaseModel):
    """Security, caching, and latency telemetry attached to completions."""

    action_taken: Literal["allow", "redact", "warn", "block"]
    pii_detected: int = 0
    pii_entities: list[PIIEntityOut] = Field(default_factory=list)
    pii_restored: bool = False
    injection_score: float = 0.0
    toxicity_score: float = 0.0
    cache_hit: bool = False
    cache_similarity: float | None = None
    provider: str
    latency_ms: float
    estimated_cost_usd: float = 0.0
    redacted_prompt_preview: str = ""
    warnings: list[str] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response augmented with Sentinel metadata."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats
    sentinel: SentinelMetadata


class GuardInspectRequest(BaseModel):
    """Dry-run inspection request for ``/v1/guard/inspect``."""

    text: str = Field(..., min_length=1, max_length=100_000)
    policy_mode: PolicyMode | None = None
    redact_pii: bool = True


class GuardInspectResponse(BaseModel):
    """Detailed PII + prompt injection + toxicity inspection report."""

    action: Literal["allow", "redact", "warn", "block"]
    allowed: bool
    redacted_text: str
    pii_count: int
    pii_entities: list[PIIEntityOut]
    injection_score: float
    toxicity_score: float
    risk_score: float
    signals: list[GuardSignalOut]
    latency_ms: float


class MetricsSummary(BaseModel):
    """Aggregated gateway observability metrics."""

    total_requests: int
    blocked_requests: int
    warned_requests: int
    redacted_requests: int
    allowed_requests: int
    cache_hits: int
    cache_hit_rate: float
    guardrail_block_rate: float
    total_pii_redacted: int
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
    total_estimated_cost_usd: float
    estimated_cost_saved_usd: float
    provider_breakdown: dict[str, int]


class AuditRecord(BaseModel):
    """Single persisted audit log entry."""

    id: int
    timestamp: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    estimated_cost_usd: float
    cache_hit: bool
    cache_similarity: float | None = None
    pii_count: int
    pii_types: list[str] = Field(default_factory=list)
    injection_score: float
    toxicity_score: float
    action_taken: str
    redacted_prompt_preview: str


class HealthResponse(BaseModel):
    """Gateway health status payload."""

    status: Literal["ok"] = "ok"
    version: str
    provider: str
    model: str
    policy_mode: str
    cache_entries: int


class CacheClearResponse(BaseModel):
    """Response after clearing semantic cache entries."""

    cleared_entries: int
    status: Literal["cleared"] = "cleared"
