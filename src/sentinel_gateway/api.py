"""FastAPI application exposing OpenAI-compatible gateway endpoints and interactive Web UI."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from sentinel_gateway import __version__
from sentinel_gateway.cache import SemanticCache
from sentinel_gateway.config import PolicyMode, Settings, get_settings
from sentinel_gateway.guardrails import GuardrailsEngine
from sentinel_gateway.pii import PIIRedactor, PIIVault
from sentinel_gateway.providers import ProviderRouter, estimate_cost_usd, estimate_messages_tokens
from sentinel_gateway.schemas import (
    AuditRecord,
    CacheClearResponse,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    GuardInspectRequest,
    GuardInspectResponse,
    GuardSignalOut,
    HealthResponse,
    MetricsSummary,
    PIIEntityOut,
    SentinelMetadata,
    UsageStats,
)
from sentinel_gateway.telemetry import SlidingWindowRateLimiter, TelemetryStore

_STATIC_DIR = Path(__file__).parent / "static"


def _serialize_pii_entities(entities: list) -> list[PIIEntityOut]:
    return [
        PIIEntityOut(
            entity_type=e.entity_type,
            placeholder=e.placeholder,
            start=e.start,
            end=e.end,
            masked_preview=e.masked_preview,
            confidence=e.confidence,
            metadata=e.metadata,
        )
        for e in entities
    ]


def _serialize_signals(signals: list) -> list[GuardSignalOut]:
    return [
        GuardSignalOut(
            category=s.category,
            rule_id=s.rule_id,
            description=s.description,
            weight=s.weight,
            matched_snippet=s.matched_snippet,
        )
        for s in signals
    ]


def create_app(
    settings: Settings | None = None,
    simulated_demo_latency_ms: float = 12.0,
) -> FastAPI:
    """Create and wire a configured Sentinel LLM Gateway FastAPI instance."""
    cfg = settings if settings is not None else get_settings()

    redactor = PIIRedactor()
    guardrails = GuardrailsEngine(
        injection_block_threshold=cfg.injection_block_threshold,
        injection_warn_threshold=cfg.injection_warn_threshold,
        toxicity_block_threshold=cfg.toxicity_block_threshold,
    )
    cache = SemanticCache(
        db_path=cfg.db_path,
        default_threshold=cfg.cache_similarity_threshold,
        ttl_seconds=cfg.cache_ttl_seconds,
        max_entries=cfg.cache_max_entries,
    )
    telemetry = TelemetryStore(db_path=cfg.db_path)
    router = ProviderRouter(cfg, simulated_demo_latency_ms=simulated_demo_latency_ms)
    rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute=cfg.rate_limit_rpm)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        cache.close()
        telemetry.close()

    app = FastAPI(
        title=cfg.app_name,
        version=__version__,
        description=(
            "OpenAI-compatible AI Guardrails, Indonesian & Global PII Redaction, "
            "Semantic Cache, and Observability Gateway."
        ),
        lifespan=lifespan,
    )

    # Expose components on app.state for programmatic inspection & testing
    app.state.settings = cfg
    app.state.redactor = redactor
    app.state.guardrails = guardrails
    app.state.cache = cache
    app.state.telemetry = telemetry
    app.state.router = router
    app.state.rate_limiter = rate_limiter

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui() -> HTMLResponse:
        index_file = _STATIC_DIR / "index.html"
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        provider, model = router.resolve_provider_and_model()
        return HealthResponse(
            status="ok",
            version=__version__,
            provider=provider,
            model=model,
            policy_mode=cfg.policy_mode,
            cache_entries=cache.count(),
        )

    @app.post("/v1/guard/inspect", response_model=GuardInspectResponse, tags=["Guardrails"])
    async def inspect_guard(payload: GuardInspectRequest) -> GuardInspectResponse:
        started = time.perf_counter()
        policy: PolicyMode = payload.policy_mode or cfg.policy_mode

        pii_result = redactor.redact(payload.text)
        guard_eval = guardrails.evaluate(payload.text, policy_mode=policy)

        if payload.redact_pii:
            output_text = (
                redactor.redact(guard_eval.sanitized_text).redacted_text
                if guard_eval.action == "redact"
                else pii_result.redacted_text
            )
        else:
            output_text = guard_eval.sanitized_text

        action = guard_eval.action
        if action == "allow" and payload.redact_pii and pii_result.entities:
            action = "redact"

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return GuardInspectResponse(
            action=action,
            allowed=guard_eval.allowed,
            redacted_text=output_text,
            pii_count=len(pii_result.entities),
            pii_entities=_serialize_pii_entities(pii_result.entities),
            injection_score=guard_eval.injection_score,
            toxicity_score=guard_eval.toxicity_score,
            risk_score=guard_eval.risk_score,
            signals=_serialize_signals(guard_eval.signals),
            latency_ms=elapsed_ms,
        )

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        tags=["OpenAI-Compatible Gateway"],
    )
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        response: Response,
    ) -> ChatCompletionResponse | JSONResponse:
        started = time.perf_counter()

        # 1. Per-client sliding-window rate limiting
        client_id = payload.user or (request.client.host if request.client else "anonymous")
        if not rate_limiter.allow(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Rate limit exceeded ({cfg.rate_limit_rpm} requests/minute).",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        opts = payload.sentinel_options
        policy_mode: PolicyMode = (
            opts.policy_mode if (opts and opts.policy_mode) else cfg.policy_mode
        )
        redact_pii = (
            opts.redact_pii
            if (opts and opts.redact_pii is not None)
            else cfg.enable_pii_redaction
        )
        restore_pii = (
            opts.restore_pii
            if (opts and opts.restore_pii is not None)
            else cfg.restore_pii_by_default
        )
        use_cache = (
            opts.use_cache
            if (opts and opts.use_cache is not None)
            else cfg.enable_semantic_cache
        )
        cache_threshold = (
            opts.cache_threshold
            if (opts and opts.cache_threshold is not None)
            else cfg.cache_similarity_threshold
        )
        req_provider = opts.provider if opts else None
        resolved_provider, resolved_model = router.resolve_provider_and_model(
            req_provider, payload.model
        )

        # 2. Extract user prompt text for guardrail & PII analysis
        user_contents = [m.content for m in payload.messages if m.role == "user"]
        combined_user_text = (
            "\n".join(user_contents) if user_contents else payload.messages[-1].content
        )
        system_prompt = "\n".join(m.content for m in payload.messages if m.role == "system")

        guard_eval = guardrails.evaluate(combined_user_text, policy_mode=policy_mode)

        # 3. Always scan PII so audit logs never store raw sensitive values
        vault = PIIVault()
        all_pii_entities = []
        redacted_messages: list[ChatMessage] = []
        redacted_user_parts: list[str] = []

        for msg in payload.messages:
            if msg.role == "user":
                candidate_text = (
                    guardrails.evaluate(msg.content, policy_mode=policy_mode).sanitized_text
                    if guard_eval.action == "redact"
                    else msg.content
                )
                pii_res = redactor.redact(candidate_text, vault=vault)
                all_pii_entities.extend(pii_res.entities)
                redacted_user_parts.append(pii_res.redacted_text)
                outbound_content = pii_res.redacted_text if redact_pii else candidate_text
                redacted_messages.append(ChatMessage(role=msg.role, content=outbound_content))
            else:
                redacted_messages.append(msg)

        redacted_prompt_preview = " | ".join(redacted_user_parts) or payload.messages[-1].content
        pii_types = [e.entity_type for e in all_pii_entities]

        # 4. Enforce block policy before touching cache or upstream provider
        if not guard_eval.allowed:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            prompt_tokens_est = estimate_messages_tokens(redacted_messages)
            telemetry.record(
                provider=resolved_provider,
                model=resolved_model,
                latency_ms=elapsed_ms,
                prompt_tokens_est=prompt_tokens_est,
                completion_tokens_est=0,
                cache_hit=False,
                cache_similarity=None,
                pii_count=len(all_pii_entities),
                pii_types=pii_types,
                injection_score=guard_eval.injection_score,
                toxicity_score=guard_eval.toxicity_score,
                action_taken="block",
                redacted_prompt_preview=redacted_prompt_preview,
            )
            return JSONResponse(
                status_code=403,
                headers={
                    "X-Sentinel-Action": "block",
                    "X-Sentinel-Cache": "BYPASS",
                    "X-Sentinel-PII-Count": str(len(all_pii_entities)),
                    "X-Sentinel-Latency-Ms": str(elapsed_ms),
                },
                content={
                    "error": {
                        "message": "Request blocked by Sentinel LLM Gateway security guardrails.",
                        "type": "sentinel_policy_violation",
                        "code": "guardrail_blocked",
                    },
                    "sentinel": {
                        "action_taken": "block",
                        "injection_score": guard_eval.injection_score,
                        "toxicity_score": guard_eval.toxicity_score,
                        "risk_score": guard_eval.risk_score,
                        "pii_detected": len(all_pii_entities),
                        "signals": [s.model_dump() for s in _serialize_signals(guard_eval.signals)],
                        "latency_ms": elapsed_ms,
                    },
                },
            )

        # 5. Semantic cache lookup (keyed ONLY on redacted prompt so raw PII is never cached)
        cache_key_prompt = (
            "\n".join(redacted_user_parts) if redacted_user_parts else combined_user_text
        )
        cache_hit = False
        cache_similarity: float | None = None
        warnings = list(guard_eval.warnings)

        if use_cache:
            cached = cache.lookup(
                prompt=cache_key_prompt,
                model=resolved_model,
                system_prompt=system_prompt,
                threshold=cache_threshold,
            )
        else:
            cached = None

        if cached is not None:
            cache_hit = True
            cache_similarity = cached.similarity
            assistant_text = cached.response_text
            actual_provider = cached.provider
            actual_model = cached.model
            prompt_tokens = cached.prompt_tokens_est
            completion_tokens = cached.completion_tokens_est
        else:
            provider_result = await router.generate(
                messages=redacted_messages,
                requested_provider=req_provider,
                requested_model=payload.model,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
            assistant_text = provider_result.content
            actual_provider = provider_result.provider
            actual_model = provider_result.model
            prompt_tokens = provider_result.prompt_tokens
            completion_tokens = provider_result.completion_tokens
            if provider_result.warning:
                warnings.append(provider_result.warning)

            if use_cache:
                cache.store(
                    prompt=cache_key_prompt,
                    response_text=assistant_text,
                    model=resolved_model,
                    provider=actual_provider,
                    system_prompt=system_prompt,
                    prompt_tokens_est=prompt_tokens,
                    completion_tokens_est=completion_tokens,
                )

        # 6. Optional reversible PII restoration on outgoing response
        pii_restored = False
        if restore_pii and redact_pii and len(vault) > 0:
            restored_text, restored_count = vault.restore(assistant_text)
            assistant_text = restored_text
            pii_restored = restored_count > 0

        # 7. Finalize action & telemetry
        action_taken = guard_eval.action
        if action_taken == "allow" and redact_pii and all_pii_entities:
            action_taken = "redact"

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        est_cost = 0.0 if cache_hit else estimate_cost_usd(
            actual_model, prompt_tokens, completion_tokens
        )

        telemetry.record(
            provider=actual_provider,
            model=actual_model,
            latency_ms=elapsed_ms,
            prompt_tokens_est=prompt_tokens,
            completion_tokens_est=completion_tokens,
            cache_hit=cache_hit,
            cache_similarity=cache_similarity,
            pii_count=len(all_pii_entities),
            pii_types=pii_types,
            injection_score=guard_eval.injection_score,
            toxicity_score=guard_eval.toxicity_score,
            action_taken=action_taken,
            redacted_prompt_preview=redacted_prompt_preview,
        )

        response.headers["X-Sentinel-Action"] = action_taken
        response.headers["X-Sentinel-Cache"] = "HIT" if cache_hit else "MISS"
        response.headers["X-Sentinel-PII-Count"] = str(len(all_pii_entities))
        response.headers["X-Sentinel-Latency-Ms"] = str(elapsed_ms)

        return ChatCompletionResponse(
            id=f"chatcmpl-sentinel-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=actual_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=assistant_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            sentinel=SentinelMetadata(
                action_taken=action_taken,
                pii_detected=len(all_pii_entities),
                pii_entities=_serialize_pii_entities(all_pii_entities),
                pii_restored=pii_restored,
                injection_score=guard_eval.injection_score,
                toxicity_score=guard_eval.toxicity_score,
                cache_hit=cache_hit,
                cache_similarity=cache_similarity,
                provider=actual_provider,
                latency_ms=elapsed_ms,
                estimated_cost_usd=est_cost,
                redacted_prompt_preview=redacted_prompt_preview[:240],
                warnings=warnings,
            ),
        )

    @app.get("/v1/metrics", response_model=MetricsSummary, tags=["Observability"])
    async def get_metrics() -> MetricsSummary:
        return telemetry.get_metrics()

    @app.get("/v1/audit", response_model=list[AuditRecord], tags=["Observability"])
    async def get_audit_logs(
        limit: int = Query(default=50, ge=1, le=500),
        action: str | None = Query(default=None),
    ) -> list[AuditRecord]:
        return telemetry.get_recent_audits(limit=limit, action=action)

    @app.delete("/v1/cache", response_model=CacheClearResponse, tags=["Semantic Cache"])
    async def clear_cache() -> CacheClearResponse:
        cleared = cache.clear()
        return CacheClearResponse(cleared_entries=cleared)

    return app


app = create_app()
