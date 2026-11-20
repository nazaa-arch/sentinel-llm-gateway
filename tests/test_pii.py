"""Unit tests for Indonesian & global PII detection, validation, and reversible token vault."""

from __future__ import annotations

import pytest

from sentinel_gateway.pii import (
    PIIRedactor,
    PIIVault,
    luhn_check,
    validate_indonesian_nik,
)


class TestNIKValidation:
    def test_valid_male_nik_jakarta(self) -> None:
        valid, meta = validate_indonesian_nik("3174051508950002")
        assert valid is True
        assert meta["province"] == "DKI Jakarta"
        assert meta["gender"] == "male"
        assert meta["birth_date_dmy"] == "15-08-95"

    def test_valid_female_nik_plus_40_rule(self) -> None:
        # Day 55 => 55 - 40 = 15 (female)
        valid, meta = validate_indonesian_nik("3273015506920004")
        assert valid is True
        assert meta["province"] == "Jawa Barat"
        assert meta["gender"] == "female"
        assert meta["birth_date_dmy"] == "15-06-92"

    @pytest.mark.parametrize(
        "bad_nik",
        [
            "9974051508950002",  # invalid province code 99
            "4074051508950002",  # invalid province code 40
            "3100051508950002",  # zero regency code
            "3174001508950002",  # zero district code
            "3174053508950002",  # day 35 invalid (between 32..40)
            "3174057508950002",  # day 75 invalid (> 71)
            "3174051513950002",  # month 13 invalid
            "3174053102950002",  # Feb 31 invalid
            "3174052902950002",  # Feb 29 on non-leap year 1995 invalid
            "3174051508950000",  # serial 0000 invalid
            "31740515089500",  # wrong length
        ],
    )
    def test_invalid_nik_rejected(self, bad_nik: str) -> None:
        valid, _ = validate_indonesian_nik(bad_nik)
        assert valid is False


class TestLuhnValidation:
    @pytest.mark.parametrize(
        "card",
        [
            "4532-0151-1283-0366",
            "5425 2334 3010 9903",
            "378282246310005",
            "3530111333300000",
        ],
    )
    def test_valid_cards(self, card: str) -> None:
        assert luhn_check(card) is True

    @pytest.mark.parametrize(
        "card",
        [
            "4532-0151-1283-0367",
            "5425 2334 3010 9904",
            "4444444444444444",
            "123456789012",
        ],
    )
    def test_invalid_cards(self, card: str) -> None:
        assert luhn_check(card) is False


class TestPIIRedactorAndVault:
    def test_multi_entity_redaction_and_reversible_restoration(self) -> None:
        redactor = PIIRedactor()
        text = (
            "Halo @budi_dev, NIK saya 3174051508950002, NPWP 09.254.294.3-407.000, "
            "HP +62 812-3456-7890, email budi@example.co.id, kartu kredit 4532-0151-1283-0366, "
            "dan rekening BCA 1234567890."
        )
        result = redactor.redact(text)
        detected_types = {e.entity_type for e in result.entities}

        assert detected_types == {
            "MENTION",
            "NIK",
            "NPWP",
            "PHONE",
            "EMAIL",
            "CARD",
            "BANK_ACCOUNT",
        }
        assert "3174051508950002" not in result.redacted_text
        assert "09.254.294.3-407.000" not in result.redacted_text
        assert "+62 812-3456-7890" not in result.redacted_text
        assert "budi@example.co.id" not in result.redacted_text
        assert "4532-0151-1283-0366" not in result.redacted_text
        assert "1234567890" not in result.redacted_text
        assert "<NIK_1>" in result.redacted_text
        assert "<EMAIL_1>" in result.redacted_text

        restored, count = result.vault.restore(result.redacted_text)
        assert count == 7
        assert restored == text

    def test_vault_deduplicates_repeated_values(self) -> None:
        redactor = PIIRedactor()
        vault = PIIVault()
        text = "Call 081234567890 now or message 081234567890 later."
        result = redactor.redact(text, vault=vault)
        assert result.redacted_text == "Call <PHONE_1> now or message <PHONE_1> later."
        assert len(vault) == 1

    def test_false_positive_traps_not_redacted(self) -> None:
        redactor = PIIRedactor()
        text = (
            "Order ORD-3174051508950002 in year 2026 cost Rp 150.000.000 with "
            "SKU 9988776655, ORD 4111111111111111, and invalid card 4532-0151-1283-0367."
        )
        result = redactor.redact(text)
        assert result.entities == []
        assert result.redacted_text == text

    def test_vault_avoids_literal_placeholder_collision_and_restores_atomically(self) -> None:
        redactor = PIIRedactor()
        text = "Literal token <NIK_1> alongside real NIK 3174051508950002."
        result = redactor.redact(text)
        assert len(result.entities) == 1
        assert result.entities[0].placeholder == "<NIK_2>"
        assert result.redacted_text == "Literal token <NIK_1> alongside real NIK <NIK_2>."

        restored, count = result.vault.restore(result.redacted_text)
        assert count == 1
        assert restored == text
