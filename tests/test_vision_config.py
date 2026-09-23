"""Tests de la configuracion anidada por proveedor, la cadena y los alias B'.

Todos usan `isolated_env`: sin `.env` ni variables reales del desarrollador.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_ADVANCE_ON,
    DEFAULT_CHAIN,
    GeminiSettings,
    Settings,
    parse_chain,
)
from app.services.vision.errors import FailureReason
from app.services.vision.models import TierSpec


def test_the_default_chain_is_the_four_decided_tiers(isolated_env: Path) -> None:
    resolution = Settings().resolve_chain()

    assert resolution.tiers == (
        TierSpec("gemini", "gemini-3.6-flash"),
        TierSpec("gemini", "gemini-3.8-flash"),
        TierSpec("openai", "gpt-5.6-terra"),
        TierSpec("anthropic", "claude-sonnet-5"),
    )
    assert resolution.tiers == DEFAULT_CHAIN
    assert resolution.notices == ()


def test_vision_chain_env_var_replaces_the_default(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_CHAIN", "gemini:gemini-x, openai:gpt-y")

    resolution = Settings().resolve_chain()

    assert resolution.tiers == (TierSpec("gemini", "gemini-x"), TierSpec("openai", "gpt-y"))


def test_a_blank_vision_chain_falls_back_to_the_default(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_CHAIN", "")

    assert Settings().resolve_chain().tiers == DEFAULT_CHAIN


@pytest.mark.parametrize("value", ["gemini", "gemini:", ":modelo", "gemini:a,openai"])
def test_a_malformed_chain_fails_fast(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("VISION_CHAIN", value)

    with pytest.raises(ValidationError, match="proveedor:modelo"):
        Settings()


def test_parse_chain_lowercases_the_provider_and_keeps_the_model() -> None:
    assert parse_chain("Gemini:Gemini-X") == (TierSpec("gemini", "Gemini-X"),)


# ------------------------------------------------------------------ alias B'


def test_deprecated_model_vars_map_to_tiers_one_and_two(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-1")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-custom-2")

    resolution = Settings().resolve_chain()

    assert resolution.tiers[:2] == (
        TierSpec("gemini", "gemini-custom-1"),
        TierSpec("gemini", "gemini-custom-2"),
    )
    assert resolution.tiers[2:] == DEFAULT_CHAIN[2:]  # los tiers 3 y 4 no cambian
    assert len(resolution.notices) == 2
    assert all("deprecada" in notice for notice in resolution.notices)


def test_only_the_alias_that_is_set_is_applied(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-custom-2")

    resolution = Settings().resolve_chain()

    assert resolution.tiers[0] == DEFAULT_CHAIN[0]
    assert resolution.tiers[1] == TierSpec("gemini", "gemini-custom-2")
    assert len(resolution.notices) == 1


def test_blank_deprecated_aliases_are_ignored(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "  ")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")

    resolution = Settings().resolve_chain()

    assert resolution.tiers == DEFAULT_CHAIN
    assert resolution.notices == ()


def test_the_aliases_are_ignored_with_a_notice_when_vision_chain_is_defined(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_CHAIN", "gemini:gemini-x")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-1")

    resolution = Settings().resolve_chain()

    assert resolution.tiers == (TierSpec("gemini", "gemini-x"),)
    assert len(resolution.notices) == 1
    assert "VISION_CHAIN" in resolution.notices[0]


def test_the_test_retry_hook_is_an_alias_of_max_attempts(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_MAX_ATTEMPTS", "3")
    assert GeminiSettings().effective_max_attempts == 3

    monkeypatch.setenv("GEMINI_TEST_RETRY_ATTEMPTS", "1")
    assert GeminiSettings().effective_max_attempts == 1


# ------------------------------------------------------------------ bloques por proveedor


def test_each_provider_block_is_independent(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    settings = Settings()

    assert settings.gemini.timeout_seconds == 15.0
    assert settings.openai.timeout_seconds == 60.0  # no se comparte con Gemini
    assert settings.openai.has_api_key() is True
    assert settings.gemini.has_api_key() is False
    assert settings.anthropic.has_api_key() is False


def test_gemini_defaults_to_one_attempt_and_the_others_keep_two(isolated_env: Path) -> None:
    """Gemini baja a 1 intento por tier; OpenAI y Anthropic no cambian."""
    settings = Settings()

    assert settings.gemini.max_attempts == 1
    assert settings.gemini.effective_max_attempts == 1
    assert settings.openai.max_attempts == 2
    assert settings.anthropic.max_attempts == 2


def test_gemini_max_attempts_is_still_overridable(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_MAX_ATTEMPTS", "2")

    assert Settings().gemini.effective_max_attempts == 2


def test_the_provider_timeout_is_a_setting_with_the_temporary_local_default(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """60 s es TEMPORAL (etapa local): debe ser configuracion por proveedor,
    no una constante enterrada en el adaptador."""
    assert Settings().gemini.timeout_seconds == 60.0

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("ANTHROPIC_TIMEOUT_SECONDS", "45")
    settings = Settings()

    assert settings.gemini.timeout_seconds == 20.0
    assert settings.anthropic.timeout_seconds == 45.0


def test_api_keys_are_secret_and_blank_means_unconfigured(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    settings = GeminiSettings()

    assert settings.has_api_key() is True
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings.model_dump())

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert GeminiSettings().has_api_key() is False


def test_settings_read_every_block_from_the_same_dotenv_file(isolated_env: Path) -> None:
    (isolated_env / ".env").write_text(
        "GEMINI_API_KEY=k1\nOPENAI_API_KEY=k2\nVISION_CHAIN=gemini:a\nUNRELATED_VAR=x\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.gemini.has_api_key() and settings.openai.has_api_key()
    assert settings.resolve_chain().tiers == (TierSpec("gemini", "a"),)


def test_the_telegram_token_is_still_required(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")

    with pytest.raises(ValidationError):
        Settings()


# ------------------------------------------------------------------ bloque de la cadena


def test_advance_on_default_only_excludes_invalid_requests(isolated_env: Path) -> None:
    advance_on = Settings().vision.advance_on

    assert advance_on == DEFAULT_ADVANCE_ON
    assert FailureReason.SOLICITUD_INVALIDA not in advance_on
    assert FailureReason.AUTENTICACION in advance_on
    assert FailureReason.RESPUESTA_INVALIDA in advance_on


def test_advance_on_can_be_overridden_from_the_environment(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_ADVANCE_ON", "alta_demanda, TIMEOUT")

    assert Settings().vision.advance_on == frozenset(
        {FailureReason.ALTA_DEMANDA, FailureReason.TIMEOUT}
    )


def test_an_unknown_advance_on_reason_fails_fast(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_ADVANCE_ON", "INVENTADA")

    with pytest.raises(ValidationError):
        Settings()


def test_the_attempts_log_defaults_on_and_a_blank_value_turns_it_off(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Settings().vision.attempts_log_path == Path("logs/vision_attempts.jsonl")

    monkeypatch.setenv("VISION_ATTEMPTS_LOG_PATH", "")
    assert Settings().vision.attempts_log_path is None


def test_circuit_breaker_and_deadline_defaults_are_configurable(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vision = Settings().vision
    assert vision.circuit_breaker_enabled is True
    assert (vision.circuit_breaker_failure_threshold, vision.circuit_breaker_cooldown_seconds) == (
        3,
        120.0,
    )
    assert vision.chain_deadline_seconds == 300.0

    monkeypatch.setenv("VISION_CIRCUIT_BREAKER_ENABLED", "false")
    monkeypatch.setenv("VISION_CHAIN_DEADLINE_SECONDS", "90")
    vision = Settings().vision
    assert vision.circuit_breaker_enabled is False
    assert vision.chain_deadline_seconds == 90.0


def test_fault_injection_reads_the_new_and_the_deprecated_variable(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Settings().fault_injection.force_error is None

    monkeypatch.setenv("FORCE_VISION_ERROR", "openai=503")
    monkeypatch.setenv("FORCE_GEMINI_MODEL_ERROR", "503")
    fault = Settings().fault_injection

    assert fault.force_error == "openai=503"
    assert fault.legacy_force_gemini_model_error == "503"


def test_quota_guard_defaults_and_overrides(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vision = Settings().vision
    assert vision.quota_guard_enabled is True
    assert vision.daily_quota_fallback_block_seconds == 6 * 3600

    monkeypatch.setenv("VISION_QUOTA_GUARD_ENABLED", "false")
    monkeypatch.setenv("VISION_DAILY_QUOTA_FALLBACK_BLOCK_SECONDS", "7200")
    vision = Settings().vision
    assert vision.quota_guard_enabled is False
    assert vision.daily_quota_fallback_block_seconds == 7200.0
