"""Tests de la inyeccion de fallas generica (reemplaza los hooks de Gemini)."""

from __future__ import annotations

import asyncio

import pytest

from app.services.vision.errors import ConfigurationError, FailureReason, LimitKind, ProviderError
from app.services.vision.fault_injection import FaultInjectingProvider, FaultRule, parse_fault_spec
from app.services.vision.models import ExtractionRequest
from tests.vision_helpers import FakeProvider

REQUEST = ExtractionRequest(images=[b"img"])


def test_selector_with_model_targets_that_model_only() -> None:
    rules = parse_fault_spec("gemini:gemini-3.6-flash=503")

    assert rules == [FaultRule("gemini", "gemini-3.6-flash", "503")]
    assert rules[0].matches("gemini", "gemini-3.6-flash")
    assert not rules[0].matches("gemini", "gemini-3.8-flash")


def test_selector_without_model_targets_the_whole_provider() -> None:
    (rule,) = parse_fault_spec("OpenAI=Timeout")

    assert rule == FaultRule("openai", None, "timeout")
    assert rule.matches("openai", "cualquier-modelo")


def test_several_rules_can_be_combined() -> None:
    rules = parse_fault_spec("gemini:gemini-3.6-flash=503, openai=credit")

    assert [r.provider_id for r in rules] == ["gemini", "openai"]


def test_bare_token_is_the_deprecated_alias_for_the_first_gemini_model() -> None:
    (rule,) = parse_fault_spec("503", legacy_gemini_model="gemini-3.6-flash")

    assert rule == FaultRule("gemini", "gemini-3.6-flash", "503")


def test_bare_token_without_a_gemini_tier_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="selector"):
        parse_fault_spec("503")


def test_unknown_fault_is_rejected_listing_the_valid_ones() -> None:
    with pytest.raises(ConfigurationError, match="validas"):
        parse_fault_spec("gemini=999")


def test_a_matching_rule_raises_the_canonical_error_without_calling_the_provider() -> None:
    inner = FakeProvider("gemini")
    faulty = FaultInjectingProvider(inner, parse_fault_spec("gemini:gemini-3.6-flash=503"))

    with pytest.raises(ProviderError) as info:
        asyncio.run(faulty.extract(REQUEST, model="gemini-3.6-flash"))

    error = info.value
    assert error.reason is FailureReason.ALTA_DEMANDA
    assert error.http_status == 503
    assert "[PRUEBA MANUAL]" in error.message
    assert inner.calls == []
    assert faulty.real_call_count == 0


def test_other_models_pass_through_and_are_counted_as_real_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inner = FakeProvider("gemini")
    faulty = FaultInjectingProvider(inner, parse_fault_spec("gemini:gemini-3.6-flash=503"))

    with caplog.at_level("INFO", logger="app.services.vision.fault_injection"):
        result = asyncio.run(faulty.extract(REQUEST, model="gemini-3.8-flash"))

    assert len(result.entries) == 1
    assert inner.calls == ["gemini-3.8-flash"]
    assert faulty.real_call_count == 1
    assert any("REAL_CALL #1 provider=gemini model=gemini-3.8-flash" in r.getMessage()
               for r in caplog.records)


def test_rules_for_other_providers_do_not_apply() -> None:
    inner = FakeProvider("gemini")
    faulty = FaultInjectingProvider(inner, parse_fault_spec("openai=503"))

    asyncio.run(faulty.extract(REQUEST, model="gemini-3.6-flash"))

    assert faulty.real_call_count == 1


@pytest.mark.parametrize(
    ("token", "reason", "kind"),
    [
        ("429", FailureReason.LIMITE_ALCANZADO, LimitKind.DESCONOCIDO),
        ("rate", FailureReason.LIMITE_ALCANZADO, LimitKind.RATE_MINUTO),
        ("daily", FailureReason.LIMITE_ALCANZADO, LimitKind.CUOTA_DIARIA),
        ("credit", FailureReason.LIMITE_ALCANZADO, LimitKind.CREDITO_O_GASTO),
        ("404", FailureReason.MODELO_NO_ENCONTRADO, None),
        ("401", FailureReason.AUTENTICACION, None),
        ("400", FailureReason.SOLICITUD_INVALIDA, None),
        ("timeout", FailureReason.TIMEOUT, None),
        ("connection", FailureReason.CONEXION, None),
        ("invalid", FailureReason.RESPUESTA_INVALIDA, None),
        ("529", FailureReason.ALTA_DEMANDA, None),
    ],
)
def test_every_fault_token_maps_to_its_canonical_reason(
    token: str, reason: FailureReason, kind: LimitKind | None
) -> None:
    faulty = FaultInjectingProvider(FakeProvider("gemini"), parse_fault_spec(f"gemini={token}"))

    with pytest.raises(ProviderError) as info:
        asyncio.run(faulty.extract(REQUEST, model="m"))

    assert info.value.reason is reason
    assert info.value.limit_kind is kind


def test_the_wrapper_delegates_identity_close_and_configuration() -> None:
    inner = FakeProvider("openai", display_name="OpenAI")
    faulty = FaultInjectingProvider(inner, [])

    assert (faulty.provider_id, faulty.display_name) == ("openai", "OpenAI")
    assert faulty.is_configured() is True
    asyncio.run(faulty.aclose())
    assert inner.closed == 1
