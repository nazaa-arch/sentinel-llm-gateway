"""Multi-provider async LLM router (local-demo, Ollama, Google Gemini, OpenAI)."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

from sentinel_gateway.config import ProviderName, Settings
from sentinel_gateway.schemas import ChatMessage

# Reference pricing (USD per 1,000 tokens: prompt_rate, completion_rate)
MODEL_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.00015, 0.00060),
    "gemini-2.0-flash": (0.00010, 0.00040),
    "gemini-1.5-pro": (0.00125, 0.00500),
    "sentinel-demo-v1": (0.00015, 0.00060),
    "qwen2.5:7b": (0.00005, 0.00015),
    "llama3.1:8b": (0.00005, 0.00015),
}

_PLACEHOLDER_RE = re.compile(r"<(?:NIK|PHONE|EMAIL|CARD|NPWP|BANK_ACCOUNT|MENTION)_\d+>")
_GEMINI_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_INDONESIAN_HINTS = re.compile(
    r"(?i)\b(?:apa|bagaimana|jelaskan|tolong|saya|anda|yang|untuk|dengan|tidak|keamanan|"
    r"pembayaran|rekening|pelanggan|data\s+pribadi|nik|npwp|ktp|terima\s+kasih|halo|selamat)\b"
)


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token, minimum 1 for non-empty text)."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """Estimate prompt tokens across chat messages including framing overhead."""
    return sum(estimate_tokens(m.content) + 4 for m in messages)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated USD inference cost for a model and token usage."""
    p_rate, c_rate = MODEL_PRICING_PER_1K.get(model, (0.00015, 0.00060))
    cost = (prompt_tokens / 1000.0) * p_rate + (completion_tokens / 1000.0) * c_rate
    return round(cost, 6)


@dataclass(frozen=True)
class ProviderResult:
    """Normalized response returned by any upstream LLM adapter."""

    content: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    fallback_used: bool = False
    warning: str | None = None


class LocalDemoProvider:
    """Deterministic offline LLM responder for zero-key demos, tests, and fallback."""

    def __init__(self, simulated_latency_ms: float = 12.0) -> None:
        self.simulated_latency_ms = simulated_latency_ms

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ProviderResult:
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        user_messages = [m.content for m in messages if m.role == "user"]
        latest = user_messages[-1] if user_messages else messages[-1].content
        lower = latest.lower()
        is_id = bool(_INDONESIAN_HINTS.search(latest))
        placeholders = _PLACEHOLDER_RE.findall(latest)

        if placeholders:
            unique_tokens = ", ".join(dict.fromkeys(placeholders))
            if is_id:
                reply = (
                    f"Permintaan Anda telah diterima secara aman melalui Sentinel LLM Gateway. "
                    f"Referensi data sensitif yang terlindungi ({unique_tokens}) telah dicatat "
                    f"tanpa pernah terekspos dalam bentuk teks asli ke model upstream. "
                    f"Silakan konfirmasi jika Anda ingin melanjutkan verifikasi akun atau layanan."
                )
            else:
                reply = (
                    f"Your request was safely processed through Sentinel LLM Gateway. "
                    f"The protected sensitive references ({unique_tokens}) were handled as "
                    f"redacted placeholders without exposing raw PII to the upstream LLM. "
                    f"Let me know how you would like to proceed with this account inquiry."
                )
        elif any(k in lower for k in ("semantic cach", "tembolok semantik", "cache")):
            if is_id:
                reply = (
                    "Semantic caching menyimpan embedding vektor dari prompt yang dinormalisasi "
                    "dan membandingkan kueri baru menggunakan cosine similarity. Jika kemiripan "
                    "melampaui ambang batas (default 0.86), respons disajikan langsung dari SQLite "
                    "sehingga memangkas latensi dan biaya token LLM secara signifikan."
                )
            else:
                reply = (
                    "Semantic caching vectorizes normalized prompts and compares incoming queries "
                    "using cosine similarity against cached embeddings. When similarity exceeds "
                    "the configured threshold (default 0.86), Sentinel returns the cached answer "
                    "in sub-millisecond time—cutting both inference latency and LLM token spend."
                )
        elif any(k in lower for k in ("prompt injection", "jailbreak", "guardrail", "injeksi")):
            if is_id:
                reply = (
                    "Pertahanan prompt injection memerlukan strategi berlapis (defense-in-depth): "
                    "normalisasi Unicode/homoglyph, de-obfuskasi leetspeak & Base64, deteksi "
                    "instruction override/role hijacking/delimiter spoofing, dan isolasi tool."
                )
            else:
                reply = (
                    "Defending against prompt injection requires defense-in-depth: Unicode and "
                    "homoglyph normalization, leetspeak & Base64 decoding, weighted detection of "
                    "instruction overrides, role hijacking, delimiter spoofing, and tool isolation."
                )
        elif any(k in lower for k in ("pii", "uu pdp", "nik", "npwp", "data pribadi", "luhn")):
            if is_id:
                reply = (
                    "Sesuai prinsip perlindungan data pribadi (UU PDP No. 27 Tahun 2022), data "
                    "sensitif seperti NIK 16-digit, NPWP, nomor HP (+62/08), email, dan kartu "
                    "kredit wajib dideteksi dan disamarkan sebelum dikirim ke LLM eksternal "
                    "menggunakan token vault reversibel."
                )
            else:
                reply = (
                    "To comply with data privacy frameworks (including Indonesia's UU PDP Law "
                    "No. 27/2022 and GDPR), PII such as 16-digit Indonesian NIK, NPWP, mobile "
                    "numbers, emails, and Luhn-validated card numbers should be redacted into "
                    "reversible placeholders before leaving your trust boundary."
                )
        else:
            snippet = latest.strip().replace("\n", " ")[:140]
            if is_id:
                reply = (
                    f'[Sentinel Local Demo] Ringkasan respons untuk permintaan Anda: "{snippet}". '
                    "Gateway telah memverifikasi keamanan prompt, mencegah kebocoran PII, "
                    "dan mencatat metrik observabilitas secara real-time."
                )
            else:
                reply = (
                    f'[Sentinel Local Demo] Processed your request safely: "{snippet}". '
                    "Sentinel validated prompt guardrails, enforced PII privacy controls, and "
                    "recorded real-time latency and token telemetry."
                )

        if max_tokens is not None and len(reply) > max_tokens * 5:
            reply = reply[: max_tokens * 5].rstrip() + "..."

        prompt_tokens = estimate_messages_tokens(messages)
        completion_tokens = estimate_tokens(reply)
        return ProviderResult(
            content=reply,
            provider="local-demo",
            model=model or "sentinel-demo-v1",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OllamaProvider:
    """Async adapter for local Ollama ``/api/chat`` REST API."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ProviderResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = (data.get("message") or {}).get("content") or ""
        prompt_tokens = int(data.get("prompt_eval_count") or estimate_messages_tokens(messages))
        completion_tokens = int(data.get("eval_count") or estimate_tokens(content))
        return ProviderResult(
            content=content,
            provider="ollama",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class GeminiProvider:
    """Async adapter for Google Gemini ``generateContent`` REST API."""

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ProviderResult:
        if not _GEMINI_MODEL_RE.fullmatch(model):
            raise ValueError("Invalid Gemini model identifier")

        system_parts = [m.content for m in messages if m.role == "system"]
        contents = []
        for msg in messages:
            if msg.role == "system":
                continue
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        if not contents:
            fallback_text = "\n".join(system_parts) if system_parts else ""
            contents.append({"role": "user", "parts": [{"text": fallback_text}]})
            system_parts = []

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates") or []
        parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
        content = "".join(p.get("text") or "" for p in parts if isinstance(p, dict))
        usage = data.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount") or estimate_messages_tokens(messages))
        completion_tokens = int(usage.get("candidatesTokenCount") or estimate_tokens(content))
        return ProviderResult(
            content=content,
            provider="gemini",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OpenAIProvider:
    """Async adapter for OpenAI and OpenAI-compatible ``/v1/chat/completions`` endpoints."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ProviderResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        return ProviderResult(
            content=content,
            provider="openai",
            model=data.get("model") or model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _format_safe_provider_error(exc: Exception) -> str:
    """Return a sanitized error summary that never leaks API keys, URLs, or stack traces."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"upstream HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "upstream request timed out"
    if isinstance(exc, httpx.RequestError):
        return "upstream connection failed"
    if isinstance(exc, (RuntimeError, ValueError)):
        return str(exc)
    return type(exc).__name__


class ProviderRouter:
    """Routes completion calls to the selected LLM provider with optional local-demo fallback."""

    def __init__(self, settings: Settings, simulated_demo_latency_ms: float = 12.0) -> None:
        self.settings = settings
        self.local_demo = LocalDemoProvider(simulated_latency_ms=simulated_demo_latency_ms)

    def resolve_provider_and_model(
        self,
        requested_provider: ProviderName | None = None,
        requested_model: str | None = None,
    ) -> tuple[ProviderName, str]:
        """Resolve effective provider and default model name."""
        provider: ProviderName = requested_provider or self.settings.provider
        if requested_model:
            return provider, requested_model
        if provider == "ollama":
            return provider, self.settings.ollama_model
        if provider == "gemini":
            return provider, self.settings.gemini_model
        if provider == "openai":
            return provider, self.settings.openai_model
        return provider, self.settings.model

    async def generate(
        self,
        messages: list[ChatMessage],
        requested_provider: ProviderName | None = None,
        requested_model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ProviderResult:
        """Dispatch request to upstream provider with graceful fallback if enabled."""
        provider, model = self.resolve_provider_and_model(requested_provider, requested_model)
        if provider == "local-demo":
            return await self.local_demo.complete(messages, model, temperature, max_tokens)

        try:
            if provider == "ollama":
                adapter = OllamaProvider(
                    self.settings.ollama_base_url, self.settings.provider_timeout_seconds
                )
                return await adapter.complete(messages, model, temperature, max_tokens)
            if provider == "gemini":
                if not self.settings.gemini_api_key:
                    raise RuntimeError("SENTINEL_GEMINI_API_KEY is not configured")
                adapter_g = GeminiProvider(
                    self.settings.gemini_api_key, self.settings.provider_timeout_seconds
                )
                return await adapter_g.complete(messages, model, temperature, max_tokens)
            if provider == "openai":
                if not self.settings.openai_api_key:
                    raise RuntimeError("SENTINEL_OPENAI_API_KEY is not configured")
                adapter_o = OpenAIProvider(
                    self.settings.openai_api_key,
                    self.settings.openai_base_url,
                    self.settings.provider_timeout_seconds,
                )
                return await adapter_o.complete(messages, model, temperature, max_tokens)
        except Exception as exc:
            if not self.settings.fallback_to_local_demo:
                raise
            fallback = await self.local_demo.complete(
                messages, "sentinel-demo-v1", temperature, max_tokens
            )
            return ProviderResult(
                content=fallback.content,
                provider=fallback.provider,
                model=fallback.model,
                prompt_tokens=fallback.prompt_tokens,
                completion_tokens=fallback.completion_tokens,
                fallback_used=True,
                warning=(
                    f"Upstream provider '{provider}' unavailable "
                    f"({_format_safe_provider_error(exc)}); fell back to local-demo."
                ),
            )

        return await self.local_demo.complete(messages, model, temperature, max_tokens)
