"""Conteo de llamadas reales de los scripts que gastan cuota real.

Incidente (2026-09-23): el script contaba por lineas de log de `httpx` e
informo 2 llamadas cuando fueron 4: los SDK de OpenAI y Anthropic loguean por
`httpx2`, y una llamada que vence por timeout no deja ninguna linea. Ahora
ambos scripts cuentan por los intentos que registra la cadena
(`telemetry.real_provider_calls`).
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from app.services.vision.chain import RetryPolicy, Tier
from app.services.vision.errors import AllProvidersFailedError, FailureReason
from app.services.vision.fault_injection import FaultInjectingProvider, FaultRule
from app.services.vision.models import TierSpec
from app.services.vision.telemetry import real_provider_calls
from tests.vision_helpers import ChainHarness, FakeProvider, failure, ok_result, tier

R = FailureReason
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_NAMES = ("validate_multi_image", "test_gemini_connection")


def _load_script(name: str = "validate_multi_image") -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counts_every_attempt_that_reached_a_provider_including_timeouts() -> None:
    """La secuencia real de chain_request_id=557bca1b2a4c: 503, 503, TIMEOUT,
    EXITO = 4 llamadas (el script viejo informo 2)."""
    g1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)])
    g2 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)])
    oa = FakeProvider("openai", [failure(R.TIMEOUT), ok_result(51)])
    harness = ChainHarness(
        [
            tier(g1, "gemini-3.6-flash"),
            tier(g2, "gemini-3.8-flash"),
            tier(oa, "gpt-5.6-terra"),
            tier(None, "claude-sonnet-5", provider_id="anthropic", inactive_reason="sin key"),
        ],
        retry={
            "gemini": RetryPolicy(max_attempts=1),
            "openai": RetryPolicy(2, initial_delay_seconds=0.0, max_delay_seconds=0.0),
        },
    )

    asyncio.run(harness.run())

    assert real_provider_calls(harness.sink.attempts) == 4
    assert len(g1.calls) + len(g2.calls) + len(oa.calls) == 4


def test_forced_faults_and_skipped_tiers_are_not_real_calls() -> None:
    """Una falla forzada (FORCE_VISION_ERROR) nunca sale del proceso, y un
    tier inactivo no se intenta: ninguno de los dos cuenta."""
    real = FakeProvider("gemini")
    forced = FaultInjectingProvider(real, [FaultRule("gemini", "gemini-3.6-flash", "503")])
    backup = FakeProvider("openai", [failure(R.ALTA_DEMANDA)])
    harness = ChainHarness(
        [
            tier(None, "gemini-x", inactive_reason="sin key"),
            Tier(
                spec=TierSpec("gemini", "gemini-3.6-flash"),
                display_name="Gemini",
                provider=forced,
                inactive_reason=None,
            ),
            tier(backup, "gpt-5.6-terra"),
        ],
        retry={"gemini": RetryPolicy(max_attempts=1), "openai": RetryPolicy(max_attempts=1)},
    )

    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())

    assert real.calls == []  # la falla forzada no llego al proveedor
    assert backup.calls == ["gpt-5.6-terra"]
    assert "SALTADO_INACTIVO" in harness.sink.outcomes
    assert real_provider_calls(harness.sink.attempts) == 1


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_both_scripts_count_with_the_shared_function(name: str) -> None:
    """Ninguno de los dos scripts puede volver a un conteo propio."""
    script = _load_script(name)

    assert script.real_provider_calls is real_provider_calls


def test_the_http_cross_check_listens_to_httpx_and_httpx2() -> None:
    assert set(_load_script()._HttpCallCounter.LOGGERS) == {"httpx", "httpx2"}
    assert set(_load_script("test_gemini_connection")._HttpCallCounter._LOGGERS) == {
        "httpx",
        "httpx2",
    }
