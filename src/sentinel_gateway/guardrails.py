"""Multi-layer bilingual prompt injection, jailbreak, obfuscation, and toxicity guardrails."""

from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from sentinel_gateway.config import PolicyMode

GuardAction = Literal["allow", "redact", "warn", "block"]
SignalCategory = Literal["prompt_injection", "toxicity", "obfuscation"]

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_BASE64_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])"
)

# Common Cyrillic / Greek homoglyphs mapped to ASCII Latin equivalents
_HOMOGLYPH_MAP = str.maketrans(
    {
        "\u0430": "a",  # Cyrillic a
        "\u0435": "e",  # Cyrillic e
        "\u043e": "o",  # Cyrillic o
        "\u0440": "p",  # Cyrillic r
        "\u0441": "c",  # Cyrillic s
        "\u0445": "x",  # Cyrillic x
        "\u0443": "y",  # Cyrillic y
        "\u0456": "i",  # Cyrillic i
        "\u0455": "s",  # Cyrillic dze
        "\u0458": "j",  # Cyrillic je
        "\u0410": "A",
        "\u0415": "E",
        "\u041e": "O",
        "\u0420": "P",
        "\u0421": "C",
        "\u0425": "X",
        "\u03bf": "o",  # Greek omicron
        "\u03b1": "a",  # Greek alpha
        "\u03b5": "e",  # Greek epsilon
        "\u03c1": "p",  # Greek rho
    }
)

_LEET_MAP = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "!": "i",
    }
)

_DEFENSIVE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:how\s+to\s+(?:prevent|defend|mitigate|detect|protect|stop)|"
    r"explain\s+(?:what|how)\s+.*?\s+(?:works|is\s+mitigated)|"
    r"what\s+is\s+(?:prompt\s+injection|sql\s+injection|phishing|malware|jailbreak)|"
    r"apa\s+itu\s+(?:prompt\s+injection|sql\s+injection|phishing|malware|ransomware)|"
    r"cara\s+(?:mencegah|melindungi|mendeteksi|mitigasi)|"
    r"best\s+practices?\s+to\s+prevent|security\s+awareness|for\s+defensive)\b"
)


@dataclass(frozen=True)
class DetectionRule:
    """Weighted regex detection rule for prompt security or toxicity."""

    rule_id: str
    category: SignalCategory
    description: str
    pattern: re.Pattern[str]
    weight: float
    sanitize_span: bool = False


@dataclass(frozen=True)
class GuardSignal:
    """Matched security signal produced during inspection."""

    category: SignalCategory
    rule_id: str
    description: str
    weight: float
    matched_snippet: str


@dataclass
class GuardEvaluation:
    """Aggregated outcome from the security policy engine."""

    action: GuardAction
    allowed: bool
    injection_score: float
    toxicity_score: float
    risk_score: float
    signals: list[GuardSignal] = field(default_factory=list)
    sanitized_text: str = ""
    warnings: list[str] = field(default_factory=list)


INJECTION_RULES: list[DetectionRule] = [
    # 1. Direct Instruction Override (EN & ID)
    DetectionRule(
        rule_id="INJ_OVERRIDE_EN",
        category="prompt_injection",
        description="Direct attempt to ignore or override prior system instructions",
        pattern=re.compile(
            r"(?i)\b(?:ignore|disregard|forget|bypass|override|void)\s+"
            r"(?:(?:all|any|every|your|the|my|these)\s+){0,2}"
            r"(?:(?:previous|prior|above|earlier|system|initial|safety|original)\s+){1,2}"
            r"(?:instructions?|prompts?|rules?|guidelines?|directives?|constraints?)\b"
        ),
        weight=0.82,
        sanitize_span=True,
    ),
    DetectionRule(
        rule_id="INJ_OVERRIDE_ID",
        category="prompt_injection",
        description="Indonesian instruction override attempt (abaikan/lupakan instruksi)",
        pattern=re.compile(
            r"(?i)\b(?:abaikan|lupakan|hiraukan|lewati|batalkan|timpa)\s+"
            r"(?:semua\s+|seluruh\s+)?(?:instruksi|perintah|aturan|pedoman|prompt)\s*"
            r"(?:sebelumnya|sistem|awal|di\s*atas|keamanan)?\b"
        ),
        weight=0.82,
        sanitize_span=True,
    ),
    DetectionRule(
        rule_id="INJ_NEW_TASK_OVERRIDE",
        category="prompt_injection",
        description="Task reset / priority hijack directive",
        pattern=re.compile(
            r"(?i)\b(?:from\s+now\s+on|starting\s+now|mulai\s+sekarang)\b.{0,40}"
            r"\b(?:only\s+follow|obey\s+only|your\s+new\s+(?:role|task|instruction)|"
            r"hanya\s+ikuti\s+perintah\s+(?:saya|ku)|tugas\s+barumu)\b"
        ),
        weight=0.72,
        sanitize_span=True,
    ),
    # 2. Role Hijacking / Jailbreak / DAN (EN & ID)
    DetectionRule(
        rule_id="INJ_JAILBREAK_DAN",
        category="prompt_injection",
        description="Known jailbreak persona (DAN / Developer Mode / unrestricted AI)",
        pattern=re.compile(
            r"(?i)\b(?:you\s+are\s+now\s+dan|do\s+anything\s+now|developer\s+mode\s+(?:enabled|on)|"
            r"jailbreak(?:\s+mode)?|unrestricted\s+ai|god\s*mode|evil\s+assistant|"
            r"no\s+(?:ethical|moral|safety)\s+(?:boundaries|limits|rules|restrictions)|"
            r"without\s+any\s+(?:restrictions|censorship|filters))\b"
        ),
        weight=0.85,
        sanitize_span=True,
    ),
    DetectionRule(
        rule_id="INJ_JAILBREAK_ID",
        category="prompt_injection",
        description="Indonesian jailbreak / unrestricted persona prompt",
        pattern=re.compile(
            r"(?i)\b(?:berperanlah\s+sebagai\s+ai\s+tanpa\s+(?:batasan|aturan|sensor)|"
            r"mode\s+pengembang\s+tanpa\s+sensor|tanpa\s+(?:aturan\s+)?etika|"
            r"bebas\s+dari\s+semua\s+(?:aturan|batasan|filter)|"
            r"jangan\s+patuhi\s+(?:aturan|kebijakan)\s+(?:keamanan|openai|sistem))\b"
        ),
        weight=0.84,
        sanitize_span=True,
    ),
    # 3. System Prompt & Secret Exfiltration (EN & ID)
    DetectionRule(
        rule_id="INJ_EXFIL_PROMPT_EN",
        category="prompt_injection",
        description="Attempt to reveal hidden system prompt or internal instructions",
        pattern=re.compile(
            r"(?i)(?:\b(?:reveal|show|print|output|repeat|dump|leak|display|expose)\s+"
            r"(?:me\s+)?(?:(?:all|your|the)\s+){0,2}"
            r"(?:(?:exact|full|verbatim|hidden|secret|internal|system|initial|developer)\s+){1,2}"
            r"(?:prompt|message|instructions?|rules?|directives?|api\s+keys?|"
            r"environment\s+variables?|tokens?)\b|"
            r"\b(?:reveal|show|print|output|dump|leak|expose)\s+"
            r"(?:me\s+)?(?:(?:all|your|the)\s+){0,2}"
            r"(?:api\s+keys?|environment\s+variables?|secret\s+tokens?)\b)"
        ),
        weight=0.78,
        sanitize_span=True,
    ),
    DetectionRule(
        rule_id="INJ_EXFIL_PROMPT_ID",
        category="prompt_injection",
        description="Indonesian system prompt or secret exfiltration attempt",
        pattern=re.compile(
            r"(?i)\b(?:tampilkan|bocorkan|sebutkan|tunjukkan|cetak|salin)\s+"
            r"(?:semua\s+)?(?:system\s+prompt|prompt\s+sistem|prompt\s+awal|"
            r"instruksi\s+(?:rahasia|tersembunyi|sistem)|api\s+key|kunci\s+rahasia|"
            r"konfigurasi\s+rahasia)\b"
        ),
        weight=0.78,
        sanitize_span=True,
    ),
    # 4. Delimiter / Markup / Role Spoofing
    DetectionRule(
        rule_id="INJ_DELIMITER_SPOOF",
        category="prompt_injection",
        description="Chat template delimiter or system role tag spoofing",
        pattern=re.compile(
            r"(?i)(?:<\|im_start\|>\s*system|<\|system\|>|<<SYS>>|<</SYS>>|"
            r"\[/?INST\]|</?(?:system|developer|instructions?)>|"
            r"###\s*(?:SYSTEM\s+OVERRIDE|NEW\s+SYSTEM\s+PROMPT|END\s+OF\s+USER\s+PROMPT)|"
            r"---+\s*(?:BEGIN|END)\s+SYSTEM\s*(?:PROMPT)?\s*---+)"
        ),
        weight=0.80,
        sanitize_span=True,
    ),
    # 5. Tool / Command / Data Exfiltration Injection
    DetectionRule(
        rule_id="INJ_TOOL_EXFIL",
        category="prompt_injection",
        description="Indirect command execution or unauthorized data exfiltration directive",
        pattern=re.compile(
            r"(?i)\b(?:execute\s+(?:shell|bash|system)\s+command|run\s+`?rm\s+-rf|"
            r"curl\s+https?://\S+\s*\|\s*(?:ba)?sh|"
            r"exfiltrate\s+(?:all\s+)?(?:user|customer|database)\s+data|"
            r"send\s+all\s+(?:secrets|passwords|pii|tokens)\s+to\s+https?://|"
            r"kirim\s+semua\s+(?:data\s+pengguna|rahasia|token)\s+ke\s+https?://)\b"
        ),
        weight=0.86,
        sanitize_span=True,
    ),
    # 6. Social Engineering / Hypothetical Safety Bypass (moderate weight)
    DetectionRule(
        rule_id="INJ_BYPASS_FRAMING",
        category="prompt_injection",
        description="Hypothetical or fictional framing requesting policy bypass",
        pattern=re.compile(
            r"(?i)\b(?:pretend\s+(?:you\s+have\s+no|there\s+are\s+no)\s+(?:rules|filters|limits)|"
            r"hypothetically\s+(?:ignore|bypass)\s+safety|"
            r"for\s+educational\s+purposes\s+only,?\s+(?:bypass|ignore|give\s+me\s+the\s+exploit)|"
            r"pura-pura\s+(?:kamu\s+)?tidak\s+punya\s+(?:aturan|batasan|filter))\b"
        ),
        weight=0.68,
        sanitize_span=False,
    ),
]

TOXICITY_RULES: list[DetectionRule] = [
    DetectionRule(
        rule_id="TOX_WEAPONS_VIOLENCE",
        category="toxicity",
        description="Request for explosives, weapons manufacturing, or lethal harm",
        pattern=re.compile(
            r"(?i)\b(?:how\s+to\s+(?:make|build|construct)\s+(?:a\s+)?(?:bomb|explosive|molotov|poison)|"
            r"cara\s+(?:membuat|merakit)\s+(?:bom|bahan\s+peledak|racun\s+mematikan)|"
            r"step-by-step\s+guide\s+to\s+(?:poison|assassinate|kill))\b"
        ),
        weight=0.92,
    ),
    DetectionRule(
        rule_id="TOX_SELF_HARM",
        category="toxicity",
        description="Self-harm encouragement or suicide instruction",
        pattern=re.compile(
            r"(?i)\b(?:go\s+kill\s+yourself|how\s+to\s+commit\s+suicide|"
            r"cara\s+bunuh\s+diri|suruh\s+dia\s+bunuh\s+diri)\b"
        ),
        weight=0.95,
    ),
    DetectionRule(
        rule_id="TOX_MALWARE_EXPLOIT",
        category="toxicity",
        description="Actionable malware, ransomware, or credential theft creation request",
        pattern=re.compile(
            r"(?i)\b(?:write\s+(?:a\s+)?(?:working\s+)?(?:ransomware|keylogger|trojan|phishing\s+kit)|"
            r"generate\s+(?:a\s+)?(?:stealer|ransomware|exploit)\s+payload|"
            r"buatkan\s+(?:kode\s+)?(?:ransomware|keylogger|phishing\s+untuk\s+mencuri\s+password)|"
            r"cara\s+membobol\s+(?:rekening|akun\s+bank|database)\s+orang)\b"
        ),
        weight=0.84,
    ),
    DetectionRule(
        rule_id="TOX_ABUSE_EN_ID",
        category="toxicity",
        description="Severe targeted harassment, hate speech, or abusive profanity",
        pattern=re.compile(
            r"(?i)\b(?:you\s+are\s+(?:worthless\s+trash|subhuman)|"
            r"all\s+\w+\s+should\s+(?:die|be\s+eliminated)|"
            r"(?:dasar|kamu)\s+(?:anjing|bangsat|bajingan|tolol\s+banget|sampah\s+masyarakat)|"
            r"bunuh\s+saja\s+semua)\b"
        ),
        weight=0.72,
    ),
]


def normalize_obfuscated_text(text: str) -> tuple[str, bool]:
    """Strip zero-width characters and fold NFKC/homoglyphs; return (normalized, modified)."""
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    normalized = unicodedata.normalize("NFKC", cleaned).translate(_HOMOGLYPH_MAP)
    modified = normalized != text
    return normalized, modified


def decode_leetspeak(text: str) -> str:
    """Translate common leetspeak substitutions into Latin characters."""
    return text.translate(_LEET_MAP)


def extract_base64_payloads(
    text: str,
    max_depth: int = 2,
    max_candidates: int = 20,
    max_token_len: int = 4096,
) -> list[tuple[str, str]]:
    """Detect and recursively decode printable UTF-8 base64 payloads embedded in text."""
    decoded_items: list[tuple[str, str]] = []
    seen_tokens: set[str] = set()

    def _decode_layer(current_text: str, depth: int, root_token: str | None = None) -> None:
        if depth <= 0 or len(decoded_items) >= max_candidates:
            return
        for match in _BASE64_CANDIDATE_RE.finditer(current_text):
            if len(decoded_items) >= max_candidates:
                break
            token = match.group(1)
            if len(token) > max_token_len or len(token) % 4 == 1 or token in seen_tokens:
                continue
            seen_tokens.add(token)
            padded = token + ("=" * ((4 - len(token) % 4) % 4))
            try:
                raw = base64.b64decode(padded, validate=True)
                decoded = raw.decode("utf-8")
            except Exception:
                continue
            # Only inspect human-readable text payloads
            printable_ratio = sum(1 for ch in decoded if ch.isprintable() or ch.isspace()) / max(
                1, len(decoded)
            )
            if len(decoded.strip()) >= 12 and printable_ratio >= 0.90:
                outer = root_token or token
                decoded_items.append((outer, decoded))
                if depth > 1:
                    _decode_layer(decoded, depth - 1, root_token=outer)

    _decode_layer(text, max_depth)
    return decoded_items


def noisy_or(weights: list[float]) -> float:
    """Combine independent rule probabilities via Noisy-OR: ``1 - prod(1 - w_i)``."""
    if not weights:
        return 0.0
    complement = 1.0
    for w in weights:
        clipped = min(max(w, 0.0), 0.999)
        complement *= 1.0 - clipped
    return round(1.0 - complement, 4)


class GuardrailsEngine:
    """Evaluates prompts against bilingual injection, obfuscation, and toxicity policies."""

    def __init__(
        self,
        injection_block_threshold: float = 0.65,
        injection_warn_threshold: float = 0.35,
        toxicity_block_threshold: float = 0.60,
    ) -> None:
        self.injection_block_threshold = injection_block_threshold
        self.injection_warn_threshold = injection_warn_threshold
        self.toxicity_block_threshold = toxicity_block_threshold

    def _scan_rules(
        self,
        text: str,
        rules: list[DetectionRule],
        seen_rule_ids: set[str],
        weight_multiplier: float = 1.0,
        suffix: str = "",
    ) -> list[GuardSignal]:
        signals: list[GuardSignal] = []
        for rule in rules:
            if rule.rule_id in seen_rule_ids:
                continue
            match = rule.pattern.search(text)
            if match:
                seen_rule_ids.add(rule.rule_id)
                rule_key = f"{rule.rule_id}{suffix}"
                snippet = match.group(0)[:120]
                signals.append(
                    GuardSignal(
                        category=rule.category,
                        rule_id=rule_key,
                        description=rule.description,
                        weight=round(min(0.98, rule.weight * weight_multiplier), 4),
                        matched_snippet=snippet,
                    )
                )
        return signals

    def _sanitize_prompt(self, text: str, malicious_b64_tokens: set[str] | None = None) -> str:
        """Remove matched injection directives/delimiters for ``redact`` policy mode."""
        sanitized = text
        for rule in INJECTION_RULES:
            if rule.sanitize_span:
                sanitized = rule.pattern.sub("[REDACTED_DIRECTIVE]", sanitized)
        if malicious_b64_tokens:
            for token in malicious_b64_tokens:
                sanitized = sanitized.replace(token, "[REDACTED_ENCODED_PAYLOAD]")
        return sanitized

    def evaluate(
        self,
        text: str,
        policy_mode: PolicyMode = "block",
    ) -> GuardEvaluation:
        """Run multi-layer inspection and return risk scores and policy action."""
        signals: list[GuardSignal] = []
        seen: set[str] = set()
        malicious_b64_tokens: set[str] = set()

        # Layer 1: Unicode NFKC + zero-width + homoglyph normalization
        normalized, was_unicode_obfuscated = normalize_obfuscated_text(text)

        # Layer 2: Direct scan on normalized text
        direct_inj = self._scan_rules(normalized, INJECTION_RULES, seen)
        direct_tox = self._scan_rules(normalized, TOXICITY_RULES, seen)
        signals.extend(direct_inj)
        signals.extend(direct_tox)

        # Layer 3: Leetspeak de-obfuscation scan (only adds newly discovered rules)
        leet_text = decode_leetspeak(normalized)
        if leet_text != normalized:
            leet_inj = self._scan_rules(
                leet_text, INJECTION_RULES, seen, weight_multiplier=0.95, suffix="_LEET"
            )
            leet_tox = self._scan_rules(
                leet_text, TOXICITY_RULES, seen, weight_multiplier=0.95, suffix="_LEET"
            )
            if leet_inj or leet_tox:
                signals.extend(leet_inj)
                signals.extend(leet_tox)
                first_leet = leet_inj[0] if leet_inj else leet_tox[0]
                signals.append(
                    GuardSignal(
                        category="obfuscation",
                        rule_id="OBF_LEETSPEAK",
                        description="Leetspeak obfuscation used to conceal adversarial directive",
                        weight=0.30,
                        matched_snippet=first_leet.matched_snippet,
                    )
                )

        # Layer 4: Recursive Base64 payload decoding scan
        b64_flagged = False
        for raw_b64, decoded_payload in extract_base64_payloads(normalized):
            decoded_norm, _ = normalize_obfuscated_text(decoded_payload)
            b64_inj = self._scan_rules(
                decoded_norm, INJECTION_RULES, seen, weight_multiplier=1.0, suffix="_B64"
            )
            b64_tox = self._scan_rules(
                decoded_norm, TOXICITY_RULES, seen, weight_multiplier=1.0, suffix="_B64"
            )
            if b64_inj or b64_tox:
                malicious_b64_tokens.add(raw_b64)
                signals.extend(b64_inj)
                signals.extend(b64_tox)
                if not b64_flagged:
                    b64_flagged = True
                    signals.append(
                        GuardSignal(
                            category="obfuscation",
                            rule_id="OBF_BASE64_PAYLOAD",
                            description="Base64-encoded adversarial payload detected",
                            weight=0.40,
                            matched_snippet=f"{raw_b64[:36]}... -> {decoded_payload[:60]}",
                        )
                    )

        if was_unicode_obfuscated and any(s.category == "prompt_injection" for s in signals):
            signals.append(
                GuardSignal(
                    category="obfuscation",
                    rule_id="OBF_UNICODE_HOMOGLYPH",
                    description="Zero-width or homoglyph characters obscured an injection attempt",
                    weight=0.25,
                    matched_snippet="[unicode-normalized]",
                )
            )

        # Compute scores via Noisy-OR
        inj_weights = [
            s.weight for s in signals if s.category in ("prompt_injection", "obfuscation")
        ]
        tox_weights = [s.weight for s in signals if s.category == "toxicity"]

        injection_score = noisy_or(inj_weights)
        toxicity_score = noisy_or(tox_weights)

        # Defensive / educational framing dampener (only if no direct override/jailbreak)
        has_strong_override = any(
            s.rule_id.startswith(("INJ_OVERRIDE", "INJ_JAILBREAK", "INJ_DELIMITER", "INJ_TOOL"))
            for s in signals
        )
        if _DEFENSIVE_CONTEXT_RE.search(normalized) and not has_strong_override:
            injection_score = round(injection_score * 0.35, 4)
            toxicity_score = round(toxicity_score * 0.35, 4)

        risk_score = round(max(injection_score, toxicity_score), 4)

        exceeds_block = (
            injection_score >= self.injection_block_threshold
            or toxicity_score >= self.toxicity_block_threshold
        )
        exceeds_warn = (
            injection_score >= self.injection_warn_threshold
            or toxicity_score >= self.injection_warn_threshold
        )

        warnings: list[str] = []
        sanitized_text = text

        if exceeds_block:
            if policy_mode == "block":
                action: GuardAction = "block"
                allowed = False
                warnings.append(
                    f"Blocked by Sentinel policy (injection={injection_score:.2f}, "
                    f"toxicity={toxicity_score:.2f})"
                )
            elif policy_mode == "redact":
                action = "redact"
                allowed = True
                sanitized_text = self._sanitize_prompt(normalized, malicious_b64_tokens)
                warnings.append("Adversarial directives sanitized under redact policy mode")
            elif policy_mode == "warn":
                action = "warn"
                allowed = True
                warnings.append(
                    f"High-risk prompt allowed under warn mode (risk={risk_score:.2f})"
                )
            else:
                action = "allow"
                allowed = True
                warnings.append(
                    f"Observed high-risk prompt under allow mode (risk={risk_score:.2f})"
                )
        elif exceeds_warn:
            action = "warn" if policy_mode in ("block", "warn", "redact") else "allow"
            allowed = True
            warnings.append(f"Elevated guardrail risk score detected (risk={risk_score:.2f})")
        else:
            action = "allow"
            allowed = True

        return GuardEvaluation(
            action=action,
            allowed=allowed,
            injection_score=injection_score,
            toxicity_score=toxicity_score,
            risk_score=risk_score,
            signals=signals,
            sanitized_text=sanitized_text,
            warnings=warnings,
        )
