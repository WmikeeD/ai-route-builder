"""Regresion de los Escenarios A y B (validados con Telegram real) sobre la
arquitectura nueva: `GeminiProvider` real + cadena de 2 tiers de Gemini.

Es el port de los antiguos tests de `test_gemini_client.py`. Los errores del
SDK se lanzan de verdad desde `_send`, asi que se ejercita tambien la
traduccion al vocabulario canonico. Un intento por tier (equivale a
`GEMINI_TEST_RETRY_ATTEMPTS=1` de las pruebas en vivo).

Cambios de comportamiento DECIDIDOS respecto del codigo anterior, cubiertos
al final: una respuesta invalida y una key mala ahora continuan al siguiente
tier en vez de abortar.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai import errors

from app.adapters.vision.gemini import GeminiProvider
from app.services.vision.chain import RetryPolicy, Tier
from app.services.vision.errors import AllProvidersFailedError, FailureReason
from app.services.vision.models import TierSpec
from app.services.vision.schema import DeliveryEntryDTO
from app.services.vision.telemetry import AttemptRecorder
from tests.vision_helpers import ChainHarness

PRIMARY = "gemini-3.6-flash"
FALLBACK = "gemini-3.8-flash"


def _server_error() -> errors.ServerError:
    return errors.ServerError(503, {"message": "high demand", "status": "UNAVAILABLE"})


def _rate_limit_error() -> errors.ClientError:
    return errors.ClientError(429, {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"})


def _model_not_found_error() -> errors.ClientError:
    return errors.ClientError(404, {"message": "model not found", "status": "NOT_FOUND"})


def _bad_request_error() -> errors.ClientError:
    return errors.ClientError(400, {"message": "invalid argument", "status": "INVALID_ARGUMENT"})


def _auth_error() -> errors.ClientError:
    return errors.ClientError(403, {"message": "key revoked", "status": "PERMISSION_DENIED"})


def _fake_response() -> SimpleNamespace:
    entry = DeliveryEntryDTO(screenshot_index=1, address="Calle Falsa 123")
    return SimpleNamespace(parsed=[entry], candidates=None, usage_metadata=None)


def _harness(effects: list[Any], *, collect: bool = False) -> tuple[ChainHarness, AsyncMock]:
    """`collect=True` guarda los intentos en `harness.sink`; si no, se usa el
    registro real (`AttemptRecorder`) para poder comprobar el log."""
    provider = GeminiProvider(api_key="test-key", timeout_seconds=60.0)
    send = AsyncMock(side_effect=effects)
    provider._send = send  # type: ignore[method-assign]
    tiers = [
        Tier(TierSpec("gemini", PRIMARY), "Gemini", provider),
        Tier(TierSpec("gemini", FALLBACK), "Gemini", provider),
    ]
    harness = ChainHarness(
        tiers,
        retry={"gemini": RetryPolicy(1)},
        recorder=None if collect else AttemptRecorder(None),
    )
    return harness, send


def _models_called(send: AsyncMock) -> list[str]:
    return [call.args[1] for call in send.call_args_list]


def _run(harness: ChainHarness) -> Any:
    return asyncio.run(
        harness.chain.extract_entries(images=[b"img"], screenshot_ids=["fid-1"])
    )


def test_uses_the_primary_model_when_it_succeeds() -> None:
    harness, send = _harness([_fake_response()])

    entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY]


def test_falls_back_to_the_secondary_model_on_server_error() -> None:
    harness, send = _harness([_server_error(), _fake_response()])

    entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY, FALLBACK]


def test_raises_if_both_models_are_saturated() -> None:
    harness, send = _harness([_server_error(), _server_error()])

    with pytest.raises(AllProvidersFailedError) as info:
        _run(harness)

    assert info.value.dominant_reason is FailureReason.ALTA_DEMANDA
    assert _models_called(send) == [PRIMARY, FALLBACK]


def test_scenario_b_rate_limit_falls_back_and_logs_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness, send = _harness([_rate_limit_error(), _fake_response()])

    with caplog.at_level(logging.INFO, logger="vision.attempts"):
        entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY, FALLBACK]
    (record,) = [r for r in caplog.records if r.getMessage().startswith("LIMITE_ALCANZADO")]
    assert record.levelno == logging.WARNING
    assert "LIMITE_ALCANZADO en Gemini (gemini-3.6-flash)" in record.getMessage()


def test_scenario_a_model_not_found_falls_back_and_logs_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Un 404 en el modelo principal (retirado/renombrado por Google, el
    incidente que origino la auditoria) conmuta al de respaldo y se ve en
    ERROR para que se corrija la config."""
    harness, send = _harness([_model_not_found_error(), _fake_response()])

    with caplog.at_level(logging.INFO, logger="vision.attempts"):
        entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY, FALLBACK]
    (record,) = [r for r in caplog.records if r.getMessage().startswith("MODELO_NO_ENCONTRADO")]
    assert record.levelno == logging.ERROR
    assert "MODELO_NO_ENCONTRADO en Gemini (gemini-3.6-flash)" in record.getMessage()


def test_does_not_fall_back_on_a_bad_request() -> None:
    """Un 400 es un error real (nuestro payload), no saturacion: no tiene
    sentido probar otro modelo y la cadena se detiene de inmediato."""
    harness, send = _harness([_bad_request_error()])

    with pytest.raises(AllProvidersFailedError) as info:
        _run(harness)

    assert _models_called(send) == [PRIMARY]
    assert info.value.dominant_reason is FailureReason.SOLICITUD_INVALIDA
    assert info.value.aborted_by is not None


def test_a_transport_timeout_on_the_primary_falls_back() -> None:
    harness, send = _harness([httpx.ReadTimeout("timed out"), _fake_response()])

    entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY, FALLBACK]


def test_real_sdk_error_details_reach_the_attempt_record() -> None:
    harness, _send = _harness([_server_error(), _fake_response()], collect=True)

    _run(harness)

    failed, succeeded = harness.sink.attempts
    assert failed.reason is FailureReason.ALTA_DEMANDA
    assert (failed.http_status, failed.provider_code) == (503, "UNAVAILABLE")
    assert (failed.provider, failed.model, failed.tier) == ("gemini", PRIMARY, 1)
    assert (succeeded.model, succeeded.tier, str(succeeded.outcome)) == (FALLBACK, 2, "EXITO")


# ------------------------------------------------------------------ cambios decididos


def test_an_invalid_response_now_continues_to_the_next_tier() -> None:
    """Antes: `GeminiExtractionError` con UNA sola llamada. Ahora (decision #3)
    la cadena prueba el siguiente tier: una llamada extra vale mas que perder el
    lote."""
    invalid = SimpleNamespace(parsed=None)
    harness, send = _harness([invalid, _fake_response()])

    entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY, FALLBACK]


def test_two_invalid_responses_end_as_an_invalid_response_failure() -> None:
    invalid = SimpleNamespace(parsed=None)
    harness, send = _harness([invalid, invalid])

    with pytest.raises(AllProvidersFailedError) as info:
        _run(harness)

    assert info.value.dominant_reason is FailureReason.RESPUESTA_INVALIDA
    assert _models_called(send) == [PRIMARY, FALLBACK]


def test_a_bad_key_now_continues_to_the_next_tier_instead_of_aborting() -> None:
    """Decision #2: una key mala de un proveedor no debe bloquear la cadena."""
    harness, send = _harness([_auth_error(), _fake_response()])

    entries = _run(harness)

    assert len(entries) == 1
    assert _models_called(send) == [PRIMARY, FALLBACK]
