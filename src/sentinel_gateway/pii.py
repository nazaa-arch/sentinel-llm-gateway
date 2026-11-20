"""Indonesian and global PII detection, validation, and reversible token vault."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Official BPS / Kemendagri 2-digit Indonesian province codes (11..96)
INDONESIAN_PROVINCE_CODES: dict[str, str] = {
    "11": "Aceh",
    "12": "Sumatera Utara",
    "13": "Sumatera Barat",
    "14": "Riau",
    "15": "Jambi",
    "16": "Sumatera Selatan",
    "17": "Bengkulu",
    "18": "Lampung",
    "19": "Kepulauan Bangka Belitung",
    "21": "Kepulauan Riau",
    "31": "DKI Jakarta",
    "32": "Jawa Barat",
    "33": "Jawa Tengah",
    "34": "DI Yogyakarta",
    "35": "Jawa Timur",
    "36": "Banten",
    "51": "Bali",
    "52": "Nusa Tenggara Barat",
    "53": "Nusa Tenggara Timur",
    "61": "Kalimantan Barat",
    "62": "Kalimantan Tengah",
    "63": "Kalimantan Selatan",
    "64": "Kalimantan Timur",
    "65": "Kalimantan Utara",
    "71": "Sulawesi Utara",
    "72": "Sulawesi Tengah",
    "73": "Sulawesi Selatan",
    "74": "Sulawesi Tenggara",
    "75": "Gorontalo",
    "76": "Sulawesi Barat",
    "81": "Maluku",
    "82": "Maluku Utara",
    "91": "Papua",
    "92": "Papua Barat",
    "93": "Papua Selatan",
    "94": "Papua Tengah",
    "95": "Papua Pegunungan",
    "96": "Papua Barat Daya",
}

_DAYS_IN_MONTH: dict[int, int] = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

# Compiled detection patterns
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_NPWP_FORMATTED_RE = re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}(?!\d)")
_NPWP_CONTEXT_16_RE = re.compile(
    r"(?i)\bnpwp(?:\s*(?:no|nomor|:|#|-))*\s*(\d{15,16})\b"
)
_CARD_CANDIDATE_RE = re.compile(r"(?<![\dA-Za-z\-])(?:\d[ -]?){13,19}(?![\dA-Za-z])")
_NIK_CANDIDATE_RE = re.compile(r"(?<![\dA-Za-z\-/])(\d{16})(?![\dA-Za-z])")
_ID_PHONE_RE = re.compile(
    r"(?<![\dA-Za-z])(?:\+62|62|0)[\s\-]?8[1-9]\d(?:[\s\-]?\d){6,10}(?!\d)"
)
_BANK_CONTEXT_RE = re.compile(
    r"(?i)(?:\b(?:no\.?\s*rek(?:ening)?|norek|rekening|bank\s+account|account\s*(?:no|number|#)?|"
    r"acct\.?\s*(?:no|#)?|transfer\s+ke|bca|bri|bni|mandiri|bsi|cimb|permata|danamon|btn|"
    r"seabank|jenius|ocbc|maybank)\b)"
    r"(?:[^\d\n]{0,28}?)(\d(?:[\s\-]?\d){9,15})(?!\d)"
)
_MENTION_RE = re.compile(r"(?:(?<=^)|(?<=[\s(\[\{\"',;:!]))(@[A-Za-z][A-Za-z0-9_.]{2,29})\b")

_CARD_CONTEXT_RE = re.compile(
    r"(?i)\b(?:credit\s*card|debit\s*card|kartu\s*kredit|kartu\s*debit|visa|mastercard|"
    r"amex|american\s*express|jcb|discover|card\s*(?:no|number|#)|cc\b|cvv|exp)\b"
)
_NIK_CONTEXT_RE = re.compile(
    r"(?i)\b(?:nik|ktp|e-?ktp|kartu\s+tanda\s+penduduk|nomor\s+induk|identitas)\b"
)
_NON_PII_PREFIX_RE = re.compile(
    r"(?i)(?:ord|order|inv|invoice|trx|txn|ref|sku|po|ticket|id|rp|idr|usd)\s*[:#\-]?\s*$"
)


def luhn_check(number: str) -> bool:
    """Validate a numeric string using the Luhn checksum algorithm."""
    digits_only = re.sub(r"\D", "", number)
    if not (13 <= len(digits_only) <= 19) or len(set(digits_only)) == 1:
        return False

    total = 0
    reverse_digits = [int(ch) for ch in reversed(digits_only)]
    for idx, digit in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit
    return total % 10 == 0


def identify_card_network(digits: str) -> str | None:
    """Return card brand if the digits match major IIN/BIN prefix ranges."""
    n = len(digits)
    if digits.startswith("4") and n in (13, 16, 19):
        return "Visa"
    if n == 16:
        prefix2 = int(digits[:2])
        prefix4 = int(digits[:4])
        if 51 <= prefix2 <= 55 or 2221 <= prefix4 <= 2720:
            return "Mastercard"
        if digits.startswith("6011") or digits.startswith("65") or 644 <= int(digits[:3]) <= 649:
            return "Discover"
        if 3528 <= prefix4 <= 3589:
            return "JCB"
    if n == 15 and digits[:2] in ("34", "37"):
        return "American Express"
    return None


def validate_indonesian_nik(nik: str) -> tuple[bool, dict[str, Any]]:
    """Validate a 16-digit Indonesian NIK (province, regency, district, DDMMYY +40 female rule)."""
    if not re.fullmatch(r"\d{16}", nik):
        return False, {}

    province_code = nik[0:2]
    regency_code = nik[2:4]
    district_code = nik[4:6]
    day_raw = int(nik[6:8])
    month = int(nik[8:10])
    year_short = nik[10:12]
    serial = nik[12:16]

    if province_code not in INDONESIAN_PROVINCE_CODES:
        return False, {}
    if regency_code == "00" or district_code == "00":
        return False, {}
    if serial == "0000":
        return False, {}
    if not (1 <= month <= 12):
        return False, {}

    if 41 <= day_raw <= 71:
        actual_day = day_raw - 40
        gender = "female"
    elif 1 <= day_raw <= 31:
        actual_day = day_raw
        gender = "male"
    else:
        return False, {}

    if actual_day > _DAYS_IN_MONTH[month]:
        return False, {}
    if month == 2 and actual_day == 29 and int(year_short) % 4 != 0:
        return False, {}

    return True, {
        "province_code": province_code,
        "province": INDONESIAN_PROVINCE_CODES[province_code],
        "gender": gender,
        "birth_date_dmy": f"{actual_day:02d}-{month:02d}-{year_short}",
    }


def mask_preview(value: str, entity_type: str) -> str:
    """Create a safe partially masked preview for telemetry and UI display."""
    if entity_type == "EMAIL" and "@" in value:
        local, domain = value.split("@", 1)
        shown = local[:2] if len(local) > 2 else local[:1]
        return f"{shown}***@{domain}"
    if entity_type == "MENTION":
        return f"@{value[1:3]}***" if len(value) > 3 else "@***"
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}***{value[-3:]}"


@dataclass(frozen=True)
class PIIEntity:
    """Detected PII entity with span and assigned reversible placeholder."""

    entity_type: str
    value: str
    placeholder: str
    start: int
    end: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def masked_preview(self) -> str:
        return mask_preview(self.value, self.entity_type)


@dataclass
class PIIVault:
    """Ephemeral per-request bidirectional map between placeholders and original PII."""

    _placeholder_to_value: dict[str, str] = field(default_factory=dict)
    _value_to_placeholder: dict[tuple[str, str], str] = field(default_factory=dict)
    _type_counters: dict[str, int] = field(default_factory=dict)

    def register(self, entity_type: str, raw_value: str, avoid_text: str = "") -> str:
        """Assign or reuse a deterministic placeholder such as ``<NIK_1>``."""
        key = (entity_type, raw_value)
        if key in self._value_to_placeholder:
            return self._value_to_placeholder[key]
        count = self._type_counters.get(entity_type, 0) + 1
        placeholder = f"<{entity_type}_{count}>"
        while placeholder in self._placeholder_to_value or (
            avoid_text and placeholder in avoid_text
        ):
            count += 1
            placeholder = f"<{entity_type}_{count}>"
        self._type_counters[entity_type] = count
        self._value_to_placeholder[key] = placeholder
        self._placeholder_to_value[placeholder] = raw_value
        return placeholder

    def restore(self, text: str) -> tuple[str, int]:
        """Atomically replace placeholders in ``text`` with original values from the vault."""
        if not self._placeholder_to_value or not text:
            return text, 0
        pattern = re.compile(
            "|".join(re.escape(ph) for ph in self._placeholder_to_value)
        )
        replacements = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal replacements
            replacements += 1
            return self._placeholder_to_value[match.group(0)]

        restored = pattern.sub(_replace, text)
        return restored, replacements

    def __len__(self) -> int:
        return len(self._placeholder_to_value)


@dataclass
class RedactionResult:
    """Result of scanning and redacting a text payload."""

    original_text: str
    redacted_text: str
    entities: list[PIIEntity]
    vault: PIIVault


@dataclass(frozen=True)
class _CandidateSpan:
    entity_type: str
    start: int
    end: int
    value: str
    priority: int
    confidence: float
    metadata: dict[str, Any]


class PIIRedactor:
    """Indonesian and global PII detector and reversible redactor."""

    def detect_candidates(self, text: str) -> list[_CandidateSpan]:
        """Collect validated PII candidate spans before overlap resolution."""
        candidates: list[_CandidateSpan] = []

        # 1. Email addresses (highest priority to prevent @mention collisions)
        for match in _EMAIL_RE.finditer(text):
            candidates.append(
                _CandidateSpan(
                    entity_type="EMAIL",
                    start=match.start(),
                    end=match.end(),
                    value=match.group(0),
                    priority=100,
                    confidence=0.98,
                    metadata={"domain": match.group(0).split("@", 1)[1].lower()},
                )
            )

        # 2. Indonesian NPWP (formatted 15-digit & keyword-anchored 15/16-digit)
        for match in _NPWP_FORMATTED_RE.finditer(text):
            candidates.append(
                _CandidateSpan(
                    entity_type="NPWP",
                    start=match.start(),
                    end=match.end(),
                    value=match.group(0),
                    priority=95,
                    confidence=0.99,
                    metadata={"format": "15-digit-formatted"},
                )
            )
        for match in _NPWP_CONTEXT_16_RE.finditer(text):
            digits = match.group(1)
            candidates.append(
                _CandidateSpan(
                    entity_type="NPWP",
                    start=match.start(1),
                    end=match.end(1),
                    value=digits,
                    priority=94,
                    confidence=0.95,
                    metadata={"format": f"{len(digits)}-digit-contextual"},
                )
            )

        # 3. Credit cards (Luhn-validated + valid BIN prefix)
        for match in _CARD_CANDIDATE_RE.finditer(text):
            raw = match.group(0).strip(" -")
            start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip(" -")))
            end = start + len(raw)
            digits = re.sub(r"\D", "", raw)
            network = identify_card_network(digits)
            if network is None or not luhn_check(digits):
                continue

            prefix_window = text[max(0, start - 32) : start]
            is_formatted = (" " in raw) or ("-" in raw)
            has_card_context = bool(_CARD_CONTEXT_RE.search(prefix_window))
            if (
                not is_formatted
                and not has_card_context
                and _NON_PII_PREFIX_RE.search(prefix_window)
            ):
                continue
            has_nik_context = bool(_NIK_CONTEXT_RE.search(prefix_window))
            nik_valid, _ = validate_indonesian_nik(digits) if len(digits) == 16 else (False, {})

            # Disambiguate rare 16-digit unformatted NIK vs Mastercard/JCB overlap
            if nik_valid and has_nik_context and not is_formatted and not has_card_context:
                continue

            candidates.append(
                _CandidateSpan(
                    entity_type="CARD",
                    start=start,
                    end=end,
                    value=raw,
                    priority=92 if (is_formatted or has_card_context) else 88,
                    confidence=0.99 if (is_formatted or has_card_context) else 0.94,
                    metadata={"network": network, "luhn_valid": True, "digits": len(digits)},
                )
            )

        # 4. Indonesian 16-digit NIK (province + regency/district + DDMMYY +40 female rule)
        for match in _NIK_CANDIDATE_RE.finditer(text):
            digits = match.group(1)
            start, end = match.start(1), match.end(1)
            prefix_window = text[max(0, start - 24) : start]
            if _NON_PII_PREFIX_RE.search(prefix_window):
                continue
            is_valid, meta = validate_indonesian_nik(digits)
            if not is_valid:
                continue

            has_nik_context = bool(_NIK_CONTEXT_RE.search(text[max(0, start - 32) : start]))
            candidates.append(
                _CandidateSpan(
                    entity_type="NIK",
                    start=start,
                    end=end,
                    value=digits,
                    priority=91 if has_nik_context else 89,
                    confidence=0.98 if has_nik_context else 0.93,
                    metadata=meta,
                )
            )

        # 5. Indonesian mobile phone numbers (+628..., 628..., 08...)
        for match in _ID_PHONE_RE.finditer(text):
            raw = match.group(0).strip()
            digits = re.sub(r"\D", "", raw)
            normalized_len = len(digits) - 1 if digits.startswith("62") else len(digits)
            if not (10 <= normalized_len <= 14):
                continue
            prefix_window = text[max(0, match.start() - 16) : match.start()]
            if _NON_PII_PREFIX_RE.search(prefix_window):
                continue

            candidates.append(
                _CandidateSpan(
                    entity_type="PHONE",
                    start=match.start(),
                    end=match.end(),
                    value=raw,
                    priority=85,
                    confidence=0.95,
                    metadata={"country": "ID", "digits": len(digits)},
                )
            )

        # 6. Indonesian / international bank account numbers (context-anchored)
        for match in _BANK_CONTEXT_RE.finditer(text):
            raw = match.group(1).strip()
            digits = re.sub(r"\D", "", raw)
            if not (10 <= len(digits) <= 16):
                continue
            bank_prefix = text[max(0, match.start(1) - 16) : match.start(1)]
            if _NON_PII_PREFIX_RE.search(bank_prefix):
                continue
            candidates.append(
                _CandidateSpan(
                    entity_type="BANK_ACCOUNT",
                    start=match.start(1),
                    end=match.end(1),
                    value=raw,
                    priority=80,
                    confidence=0.90,
                    metadata={"digits": len(digits)},
                )
            )

        # 7. @mentions (lower priority than email)
        for match in _MENTION_RE.finditer(text):
            candidates.append(
                _CandidateSpan(
                    entity_type="MENTION",
                    start=match.start(1),
                    end=match.end(1),
                    value=match.group(1),
                    priority=70,
                    confidence=0.88,
                    metadata={},
                )
            )

        return candidates

    def _resolve_overlaps(self, candidates: list[_CandidateSpan]) -> list[_CandidateSpan]:
        """Select non-overlapping spans prioritizing higher confidence/specificity and length."""
        ordered = sorted(
            candidates,
            key=lambda c: (-c.priority, -(c.end - c.start), c.start),
        )
        selected: list[_CandidateSpan] = []
        for cand in ordered:
            overlaps = any(not (cand.end <= s.start or cand.start >= s.end) for s in selected)
            if not overlaps:
                selected.append(cand)
        return sorted(selected, key=lambda c: c.start)

    def redact(self, text: str, vault: PIIVault | None = None) -> RedactionResult:
        """Detect all supported PII in ``text`` and replace with reversible placeholders."""
        active_vault = vault if vault is not None else PIIVault()
        resolved = self._resolve_overlaps(self.detect_candidates(text))

        if not resolved:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                entities=[],
                vault=active_vault,
            )

        parts: list[str] = []
        entities: list[PIIEntity] = []
        cursor = 0
        for span in resolved:
            placeholder = active_vault.register(span.entity_type, span.value, avoid_text=text)
            parts.append(text[cursor : span.start])
            parts.append(placeholder)
            cursor = span.end
            entities.append(
                PIIEntity(
                    entity_type=span.entity_type,
                    value=span.value,
                    placeholder=placeholder,
                    start=span.start,
                    end=span.end,
                    confidence=span.confidence,
                    metadata=span.metadata,
                )
            )
        parts.append(text[cursor:])

        return RedactionResult(
            original_text=text,
            redacted_text="".join(parts),
            entities=entities,
            vault=active_vault,
        )
