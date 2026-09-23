"""Tests de la fabrica de la cadena: tiers inactivos, validacion al arrancar,
avisos de deprecacion e inyeccion de fallas (Escenarios A/B con los hooks
portados). Sin red: el `_send` de Gemini se reemplaza.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.adapters.vision.factory import build_default_registry, build_fallback_chain
from app.config import Settings
from app.services.vision.errors import ConfigurationError
from app.services.vision.fault_injection import FaultInjectingProvider
from app.services.vision.registry import ProviderRegistry
from app.services.vision.schema import DeliveryEntryDTO
from tests.vision_helpers import FakeProvider, ListSink

FACTORY_LOGGER = "app.adapters.vision.factory"


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings()


def _ok_response() -> SimpleNamespace:
    entry = DeliveryEntryDTO(screenshot_index=1, address="Calle Falsa 123")
    return SimpleNamespace(parsed=[entry], candidates=None, usage_metadata=None)


def test_with_only_gemini_configured_tiers_three_and_four_are_inactive(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(monkeypatch, GEMINI_API_KEY="k")

    with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
        chain = build_fallback_chain(settings, recorder=ListSink())

    assert [t.active for t in chain.tiers] == [True, True, False, False]
    assert [t.spec.model for t in chain.tiers] == [
        "gemini-3.6-flash", "gemini-3.8-flash", "gpt-5.6-terra", "claude-sonnet-5",
    ]
    assert chain.tiers[2].inactive_reason == "sin OPENAI_API_KEY"
    assert chain.tiers[3].inactive_reason == "sin ANTHROPIC_API_KEY"
    summary = next(r.getMessage() for r in caplog.records if "Cadena de vision" in r.getMessage())
    assert "1) Gemini gemini-3.6-flash [activo]" in summary
    assert "3) OpenAI gpt-5.6-terra [inactivo: sin OPENAI_API_KEY]" in summary
    assert "4) Anthropic claude-sonnet-5 [inactivo: sin ANTHROPIC_API_KEY]" in summary


def test_both_gemini_tiers_share_one_provider_instance(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = build_fallback_chain(_settings(monkeypatch, GEMINI_API_KEY="k"), recorder=ListSink())

    assert chain.tiers[0].provider is chain.tiers[1].provider


def test_without_any_key_startup_fails_fast(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigurationError, match="Ningun tier"):
        build_fallback_chain(_settings(monkeypatch), recorder=ListSink())


def test_an_unknown_provider_in_the_chain_is_a_configuration_error(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        monkeypatch, GEMINI_API_KEY="k", VISION_CHAIN="gemini:gemini-x,opnai:gpt-y"
    )

    with pytest.raises(ConfigurationError, match="opnai"):
        build_fallback_chain(settings, recorder=ListSink())


def test_a_known_provider_without_a_registered_adapter_is_inactive_as_not_implemented(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un proveedor conocido (`KNOWN_PROVIDERS`) cuyo adaptador no esta en el
    registro no es un error de configuracion: su tier queda inactivo."""
    settings = _settings(monkeypatch, GEMINI_API_KEY="k", OPENAI_API_KEY="k", ANTHROPIC_API_KEY="k")
    registry = ProviderRegistry()
    registry.register("gemini", lambda: FakeProvider("gemini"), key_env_var="GEMINI_API_KEY")

    chain = build_fallback_chain(settings, registry=registry, recorder=ListSink())

    assert [t.active for t in chain.tiers] == [True, True, False, False]
    assert chain.tiers[2].inactive_reason == "adaptador no implementado todavia"
    assert chain.tiers[3].inactive_reason == "adaptador no implementado todavia"


def test_the_default_registry_registers_the_three_real_adapters(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = build_default_registry(_settings(monkeypatch, GEMINI_API_KEY="k"))

    assert [registry.get(p) is not None for p in ("gemini", "openai", "anthropic")] == [True] * 3


def test_a_registered_adapter_with_key_becomes_an_active_tier(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(monkeypatch, GEMINI_API_KEY="k")
    registry = build_default_registry(settings)
    registry.register("openai", lambda: FakeProvider("openai"), key_env_var="OPENAI_API_KEY")

    chain = build_fallback_chain(settings, registry=registry, recorder=ListSink())

    assert [t.active for t in chain.tiers] == [True, True, True, False]


def test_deprecated_aliases_are_applied_and_warned_about_at_startup(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(
        monkeypatch,
        GEMINI_API_KEY="k",
        GEMINI_MODEL="gemini-custom-1",
        GEMINI_FALLBACK_MODEL="gemini-custom-2",
    )

    with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
        chain = build_fallback_chain(settings, recorder=ListSink())

    assert [t.spec.model for t in chain.tiers[:2]] == ["gemini-custom-1", "gemini-custom-2"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("DEPRECADO" in w for w in warnings) == 2
    assert any("GEMINI_MODEL" in w for w in warnings)
    assert any("GEMINI_FALLBACK_MODEL" in w for w in warnings)


def test_retry_policy_comes_from_each_provider_block(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        monkeypatch,
        GEMINI_API_KEY="k",
        GEMINI_MAX_ATTEMPTS="3",
        GEMINI_RETRY_MAX_DELAY_SECONDS="5",
        OPENAI_MAX_ATTEMPTS="4",
    )

    chain = build_fallback_chain(settings, recorder=ListSink())

    policies = chain._policy.retry_policies
    assert (policies["gemini"].max_attempts, policies["gemini"].max_delay_seconds) == (3, 5.0)
    assert policies["openai"].max_attempts == 4


def test_default_retry_policies_one_attempt_for_gemini_two_for_the_rest(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(monkeypatch, GEMINI_API_KEY="k")

    chain = build_fallback_chain(settings, recorder=ListSink())

    policies = chain._policy.retry_policies
    assert policies["gemini"].max_attempts == 1
    assert policies["openai"].max_attempts == 2
    assert policies["anthropic"].max_attempts == 2


# ------------------------------------------------------------------ inyeccion de fallas


def _run(chain: Any) -> Any:
    return asyncio.run(chain.extract_entries(images=[b"img"], screenshot_ids=["fid-1"]))


def test_force_vision_error_makes_the_primary_fail_and_the_fallback_answer(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Equivale al Escenario C/B de las pruebas en vivo: el modelo principal
    falla sin gastar cuota y el respaldo (real) responde."""
    settings = _settings(
        monkeypatch,
        GEMINI_API_KEY="k",
        GEMINI_MAX_ATTEMPTS="1",
        FORCE_VISION_ERROR="gemini:gemini-3.6-flash=503",
    )
    chain = build_fallback_chain(settings, recorder=ListSink())
    faulty = chain.tiers[0].provider
    assert isinstance(faulty, FaultInjectingProvider)
    faulty._inner._send = AsyncMock(return_value=_ok_response())  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO):
        entries = _run(chain)

    assert len(entries) == 1
    assert faulty.real_call_count == 1  # solo el respaldo llego a la API
    assert faulty._inner._send.await_args.args[1] == "gemini-3.8-flash"  # type: ignore[attr-defined]
    assert any("MODO DE PRUEBA" in r.getMessage() for r in caplog.records)


def test_the_deprecated_force_gemini_model_error_still_works_and_is_warned_about(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(
        monkeypatch,
        GEMINI_API_KEY="k",
        GEMINI_TEST_RETRY_ATTEMPTS="1",
        FORCE_GEMINI_MODEL_ERROR="503",
    )

    with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
        chain = build_fallback_chain(settings, recorder=ListSink())

    faulty = chain.tiers[0].provider
    assert isinstance(faulty, FaultInjectingProvider)
    faulty._inner._send = AsyncMock(return_value=_ok_response())  # type: ignore[attr-defined]
    assert len(_run(chain)) == 1
    assert chain._policy.retry_policies["gemini"].max_attempts == 1  # alias de MAX_ATTEMPTS
    warnings = " ".join(r.getMessage() for r in caplog.records)
    assert "FORCE_GEMINI_MODEL_ERROR es un alias" in warnings
    assert "GEMINI_TEST_RETRY_ATTEMPTS es un alias" in warnings


def test_without_the_hooks_the_providers_are_not_wrapped(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = build_fallback_chain(_settings(monkeypatch, GEMINI_API_KEY="k"), recorder=ListSink())

    assert not isinstance(chain.tiers[0].provider, FaultInjectingProvider)


# ------------------------------------------------------------------ guardia de cuota


def test_the_quota_guard_is_wired_with_the_pacific_reset_for_gemini_only(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from app.services.vision.errors import FailureReason, LimitKind, ProviderError

    settings = _settings(monkeypatch, GEMINI_API_KEY="k")
    chain = build_fallback_chain(settings, recorder=ListSink())
    guard = chain._quota_guard
    assert guard is not None

    def daily(provider: str) -> ProviderError:
        return ProviderError(
            FailureReason.LIMITE_ALCANZADO,
            provider=provider,
            model="m",
            limit_kind=LimitKind.CUOTA_DIARIA,
        )

    gemini_block = guard.record_failure(daily("gemini"))
    openai_block = guard.record_failure(daily("openai"))

    assert gemini_block is not None and openai_block is not None
    # Gemini: medianoche del Pacifico (siempre a las 07:00 o 08:00 UTC, en punto).
    assert gemini_block.until is not None and gemini_block.until.minute == 0
    assert gemini_block.until.hour in (7, 8)
    # Sin reset documentado: duracion fija (6 h por defecto) desde ahora.
    assert openai_block.until is not None
    assert 5.9 * 3600 < (openai_block.until - datetime.now(UTC)).total_seconds() <= 6 * 3600


def test_the_quota_guard_can_be_disabled_from_the_environment(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(monkeypatch, GEMINI_API_KEY="k", VISION_QUOTA_GUARD_ENABLED="false")

    chain = build_fallback_chain(settings, recorder=ListSink())

    assert chain._quota_guard is None


def test_a_forced_daily_quota_failure_blocks_the_tier_on_the_next_request(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """De punta a punta con los hooks de prueba (cero llamadas reales): la
    primera solicitud descubre la cuota diaria agotada; la segunda ni intenta."""
    settings = _settings(
        monkeypatch,
        GEMINI_API_KEY="k",
        GEMINI_MAX_ATTEMPTS="1",
        FORCE_VISION_ERROR="gemini:gemini-3.6-flash=daily",
    )
    sink = ListSink()
    chain = build_fallback_chain(settings, recorder=sink)
    faulty = chain.tiers[0].provider
    assert isinstance(faulty, FaultInjectingProvider)
    faulty._inner._send = AsyncMock(return_value=_ok_response())  # type: ignore[attr-defined]

    _run(chain)
    _run(chain)

    assert faulty.real_call_count == 2  # solo el tier 2, una vez por solicitud
    assert sink.outcomes == ["FALLO", "EXITO", "SALTADO_CUOTA", "EXITO"]
