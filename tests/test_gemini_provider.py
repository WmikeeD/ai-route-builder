"""Tests de `GeminiProvider`: traduccion de errores del SDK al vocabulario
canonico, lectura de la respuesta, registro de uso y configuracion del cliente.

Se reemplaza `_send` (el seam que hace la llamada real al SDK) en vez de
tocar `genai.Client`, para no depender de la forma interna del cliente de
terceros. Nada de esto toca la red.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai import errors, types

from app.adapters.vision.gemini import (
    GeminiProvider,
    classify_rate_limit,
    parse_retry_after,
)
from app.config import GeminiSettings
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest
from app.services.vision.schema import DeliveryEntryDTO

R = FailureReason
REQUEST = ExtractionRequest(images=[b"img"], screenshot_ids=["fid-1"])


def _provider() -> GeminiProvider:
    return GeminiProvider(api_key="test-key", timeout_seconds=60.0)


def _extract_raising(exc: BaseException) -> ProviderError:
    provider = _provider()
    provider._send = AsyncMock(side_effect=exc)  # type: ignore[method-assign]
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(REQUEST, model="gemini-x"))
    return info.value


def _api_error(cls: type[errors.APIError], code: int, status: str, message: str = "m") -> Any:
    return cls(code, {"message": message, "status": status})


# ------------------------------------------------------------------ errores del SDK

DAILY_BODY = (
    "You exceeded your current quota. Quota exceeded for metric: "
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20"
)
MINUTE_BODY = (
    "Quota exceeded for metric: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, "
    "limit: 5. Please retry in 45.8s."
)


SERVER, CLIENT = errors.ServerError, errors.ClientError
LIMIT = R.LIMITE_ALCANZADO
UNKNOWN, DAILY, PER_MIN = (
    LimitKind.DESCONOCIDO,
    LimitKind.CUOTA_DIARIA,
    LimitKind.RATE_MINUTO,
)


def _e(cls: type[errors.APIError], code: int, status: str, message: str = "m") -> Any:
    return _api_error(cls, code, status, message)


@pytest.mark.parametrize(
    ("exc", "reason", "kind", "http", "code"),
    [
        (_e(SERVER, 503, "UNAVAILABLE"), R.ALTA_DEMANDA, None, 503, "UNAVAILABLE"),
        (_e(SERVER, 500, "INTERNAL"), R.ALTA_DEMANDA, None, 500, "INTERNAL"),
        (_e(SERVER, 504, "DEADLINE"), R.ALTA_DEMANDA, None, 504, "DEADLINE"),
        (_e(CLIENT, 429, "EXHAUSTED", "rate limited"), LIMIT, UNKNOWN, 429, "EXHAUSTED"),
        (_e(CLIENT, 429, "EXHAUSTED", DAILY_BODY), LIMIT, DAILY, 429, "EXHAUSTED"),
        (_e(CLIENT, 429, "EXHAUSTED", MINUTE_BODY), LIMIT, PER_MIN, 429, "EXHAUSTED"),
        (_e(CLIENT, 402, "PAYMENT"), LIMIT, LimitKind.CREDITO_O_GASTO, 402, "PAYMENT"),
        (_e(CLIENT, 404, "NOT_FOUND"), R.MODELO_NO_ENCONTRADO, None, 404, "NOT_FOUND"),
        (_e(CLIENT, 401, "UNAUTH"), R.AUTENTICACION, None, 401, "UNAUTH"),
        (_e(CLIENT, 403, "DENIED"), R.AUTENTICACION, None, 403, "DENIED"),
        (_e(CLIENT, 400, "INVALID"), R.SOLICITUD_INVALIDA, None, 400, "INVALID"),
        (_e(CLIENT, 413, "TOO_LARGE"), R.SOLICITUD_INVALIDA, None, 413, "TOO_LARGE"),
        (_e(CLIENT, 408, "TIMEOUT"), R.TIMEOUT, None, 408, "TIMEOUT"),
        (_e(errors.APIError, 302, "REDIRECT"), R.DESCONOCIDO, None, 302, "REDIRECT"),
    ],
)
def test_sdk_errors_are_translated_to_the_canonical_vocabulary(
    exc: Exception, reason: FailureReason, kind: LimitKind | None, http: int, code: str
) -> None:
    error = _extract_raising(exc)

    assert error.reason is reason
    assert error.limit_kind is kind
    assert error.http_status == http
    assert error.provider_code == code
    assert error.provider == "gemini"
    assert error.model == "gemini-x"


def test_the_retry_delay_of_a_429_is_carried_for_the_chain() -> None:
    error = _extract_raising(_api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", MINUTE_BODY))

    assert error.retry_after_seconds == pytest.approx(45.8)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Please retry in 45.8s.", 45.8),
        ("{'retryDelay': '30s'}", 30.0),
        ("nada que ver", None),
    ],
)
def test_parse_retry_after(text: str, expected: float | None) -> None:
    assert parse_retry_after(text) == expected


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("... PerDay ...", LimitKind.CUOTA_DIARIA),
        ("daily quota reached", LimitKind.CUOTA_DIARIA),
        ("rate_limit_exceeded", LimitKind.RATE_MINUTO),
        ("... PerMinute ...", LimitKind.RATE_MINUTO),
        ("Resource exhausted", LimitKind.DESCONOCIDO),
    ],
)
def test_classify_rate_limit_heuristic(text: str, kind: LimitKind) -> None:
    assert classify_rate_limit(text) is kind


def test_transport_timeout_is_translated_to_timeout() -> None:
    """El SDK NO envuelve los errores de transporte de httpx (hallazgo 1.5):
    en la prueba real llego un `ReadTimeout` crudo con code=None."""
    error = _extract_raising(httpx.ReadTimeout("The read operation timed out"))

    assert error.reason is R.TIMEOUT
    assert error.http_status is None


def test_transport_failure_is_translated_to_connection() -> None:
    assert _extract_raising(httpx.ConnectError("boom")).reason is R.CONEXION


def test_programming_errors_are_not_disguised_as_provider_failures() -> None:
    provider = _provider()
    provider._send = AsyncMock(side_effect=ValueError("bug nuestro"))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="bug nuestro"):
        asyncio.run(provider.extract(REQUEST, model="gemini-x"))


def test_a_hung_call_is_cut_by_the_hard_timeout() -> None:
    provider = _provider()
    provider._hard_timeout_seconds = 0.05

    async def hang(request: ExtractionRequest, model: str) -> Any:
        await asyncio.sleep(10)

    provider._send = hang  # type: ignore[method-assign,assignment]

    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(REQUEST, model="gemini-x"))

    assert info.value.reason is R.TIMEOUT
    assert "tope duro" in info.value.message


# ------------------------------------------------------------------ respuesta y uso


def _response_with(
    finish_reason: types.FinishReason, *, valid_json: bool = False
) -> types.GenerateContentResponse:
    """Respuesta real del SDK. Sin `valid_json`, `parsed` queda en None (como
    cuando el JSON llega cortado)."""
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=finish_reason)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=4200,
            candidates_token_count=6100,
            thoughts_token_count=900,
            total_token_count=11200,
        ),
    )
    if valid_json:
        response.parsed = [DeliveryEntryDTO(screenshot_index=1, address="Calle Falsa 123")]
    return response


def _extract_response(response: Any) -> Any:
    provider = _provider()
    provider._send = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return asyncio.run(provider.extract(REQUEST, model="gemini-x"))


def test_a_valid_response_becomes_domain_entries_with_the_source_screenshot() -> None:
    entry = DeliveryEntryDTO(screenshot_index=1, address="Calle Falsa 123")
    response = SimpleNamespace(parsed=[entry], candidates=None, usage_metadata=None)

    result = _extract_response(response)

    assert [e.address for e in result.entries] == ["Calle Falsa 123"]
    assert result.entries[0].source_screenshot_id == "fid-1"
    assert result.usage.finish_reason is None


def test_an_unparseable_response_is_an_invalid_response_error_not_a_type_error() -> None:
    """El SDK deja `response.parsed` en None (sin excepcion) cuando el JSON llega
    cortado; debe fallar de forma explicita como RESPUESTA_INVALIDA."""
    provider = _provider()
    provider._send = AsyncMock(return_value=SimpleNamespace(parsed=None))  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(REQUEST, model="gemini-x"))

    assert info.value.reason is R.RESPUESTA_INVALIDA


def test_usage_is_logged_at_info_when_generation_stops_normally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.services.vision.base"):
        _extract_response(_response_with(types.FinishReason.STOP, valid_json=True))

    (record,) = [r for r in caplog.records if "VISION_RESPONSE" in r.getMessage()]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    for fragment in (
        "provider=gemini", "model=gemini-x", "finish_reason=STOP", "prompt_tokens=4200",
        "output_tokens=6100", "reasoning_tokens=900", "total_tokens=11200",
    ):
        assert fragment in message


def test_truncated_output_is_logged_even_though_validation_then_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regresion: `GEMINI_RESPONSE` se emitia ANTES de validar el JSON, y eso
    permite ver `finish_reason=MAX_TOKENS` en una respuesta cortada. El log de
    uso debe salir aunque la respuesta luego se rechace."""
    provider = _provider()
    provider._send = AsyncMock(return_value=_response_with(types.FinishReason.MAX_TOKENS))  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO, logger="app.services.vision.base"), pytest.raises(
        ProviderError
    ) as info:
        asyncio.run(provider.extract(REQUEST, model="gemini-x"))

    (record,) = [r for r in caplog.records if "VISION_RESPONSE" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "finish_reason=MAX_TOKENS" in record.getMessage()
    assert info.value.reason is R.RESPUESTA_INVALIDA
    assert "finish_reason=MAX_TOKENS" in info.value.message


def test_usage_tolerates_missing_candidates_and_usage_metadata() -> None:
    result = _extract_response(
        SimpleNamespace(parsed=[DeliveryEntryDTO(screenshot_index=1, address="x")])
    )

    assert result.usage.prompt_tokens is None
    assert result.usage.finish_ok is True


# ------------------------------------------------------------------ configuracion


def test_the_sdk_client_makes_one_attempt_and_has_an_explicit_timeout() -> None:
    """R2: el reintento esta en la cadena, no en el SDK (que reintentaba 429 y
    5xx por igual). Y ya no hay `timeout=None`: una llamada llego a colgarse
    ~4.5 min."""
    options = _provider()._client._api_client._http_options

    assert options.retry_options is not None
    assert options.retry_options.attempts == 1
    assert options.timeout == 60_000  # el SDK espera milisegundos


def test_from_settings_returns_none_without_an_explicit_key(isolated_env: Any) -> None:
    assert GeminiProvider.from_settings(GeminiSettings()) is None
    assert GeminiProvider.from_settings(GeminiSettings(api_key="   ")) is None


def test_from_settings_builds_the_provider_and_never_leaks_the_key(isolated_env: Any) -> None:
    settings = GeminiSettings(api_key="super-secret-key", timeout_seconds=15.0)

    provider = GeminiProvider.from_settings(settings)

    assert provider is not None
    assert provider._client._api_client._http_options.timeout == 15_000
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings)
