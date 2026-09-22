"""Tests de `OpenAIProvider`: solicitud, respuesta, uso y traduccion de errores.

Las excepciones son las REALES del SDK (`openai.RateLimitError`, ...),
construidas con una respuesta `httpx2` como lo hace el propio SDK, asi el
mapeo se prueba contra la forma verdadera (`exc.code`, `exc.type`,
`exc.status_code`, cabeceras). El cliente se reemplaza en `_send`/`responses`:
nada de esto toca la red ni usa una API key real.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import openai
import pytest

from app.adapters.vision.openai import (
    SCHEMA_NAME,
    OpenAIProvider,
    build_strict_schema,
    classify_rate_limit,
)
from app.config import OpenAISettings
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest
from app.services.vision.prompt import SYSTEM_INSTRUCTION

R = FailureReason
REQUEST = ExtractionRequest(images=[b"img-1", b"img-2"], screenshot_ids=["fid-1", "fid-2"])
URL = "https://api.openai.com/v1/responses"


def _provider(**kwargs: Any) -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key", timeout_seconds=60.0, **kwargs)


def _status_error(
    cls: type[openai.APIStatusError],
    status: int,
    *,
    code: str | None = None,
    type_: str | None = None,
    message: str = "m",
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    body = {"message": message, "type": type_, "code": code, "param": None}
    response = httpx2.Response(
        status,
        headers=headers or {},
        json={"error": body},
        request=httpx2.Request("POST", URL),
    )
    return cls(message, response=response, body=body)


def _extract_raising(exc: BaseException) -> ProviderError:
    provider = _provider()
    provider._send = AsyncMock(side_effect=exc)  # type: ignore[method-assign]
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(REQUEST, model="gpt-x"))
    return info.value


# ------------------------------------------------------------------ esquema estricto


def test_the_schema_is_a_strict_object_root_without_unsupported_constraints() -> None:
    schema = build_strict_schema()
    entry = schema["$defs"]["_OpenAIEntry"]

    assert schema["type"] == "object"  # la lista va envuelta: raiz = objeto
    assert schema["required"] == ["entries"]
    assert schema["additionalProperties"] is False
    assert entry["additionalProperties"] is False
    assert sorted(entry["required"]) == sorted(entry["properties"])  # todos requeridos
    dumped = json.dumps(schema)
    assert "minimum" not in dumped  # Field(ge=1) del DTO NO viaja al proveedor
    assert '"default"' not in dumped


# ------------------------------------------------------------------ solicitud


def test_the_request_carries_images_labels_schema_and_privacy_flags() -> None:
    provider = _provider(image_detail="high")
    create = AsyncMock(return_value=_response(entries=[_entry()]))
    provider._client.responses.create = create  # type: ignore[method-assign]

    asyncio.run(provider.extract(REQUEST, model="gpt-5.6-terra"))

    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-5.6-terra"
    assert kwargs["instructions"] == SYSTEM_INSTRUCTION
    assert kwargs["store"] is False  # no dejar capturas de clientes en OpenAI
    assert "reasoning" not in kwargs  # sin esfuerzo configurado: default del modelo
    fmt = kwargs["text"]["format"]
    assert (fmt["type"], fmt["name"], fmt["strict"]) == ("json_schema", SCHEMA_NAME, True)
    assert fmt["schema"] == build_strict_schema()

    (message,) = kwargs["input"]
    parts = message["content"]
    images = [p for p in parts if p["type"] == "input_image"]
    labels = [p["text"] for p in parts if p["type"] == "input_text"]
    assert labels[:2] == ["Imagen 1:", "Imagen 2:"]  # ayudan a mapear screenshot_index
    assert [p["detail"] for p in images] == ["high", "high"]
    expected = "data:image/jpeg;base64," + base64.b64encode(b"img-1").decode()
    assert images[0]["image_url"] == expected


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    provider = _provider(reasoning_effort="low", store=True)
    create = AsyncMock(return_value=_response(entries=[_entry()]))
    provider._client.responses.create = create  # type: ignore[method-assign]

    asyncio.run(provider.extract(REQUEST, model="gpt-x"))

    assert create.await_args.kwargs["reasoning"] == {"effort": "low"}
    assert create.await_args.kwargs["store"] is True


# ------------------------------------------------------------------ respuesta


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "screenshot_index": 1,
        "order": 54,
        "package_id": "696733800652",
        "customer_name": None,
        "address": "PANAMERICANA NORTE 8000",
        "locality": "QUILICURA",
        "delivery_type": "Entrega Estandar",
        "eta": "12:34",
        "time_window": "07:00 - 21:00",
        "status": "pending",
    }
    return base | overrides


def _response(
    entries: list[dict[str, Any]] | None = None,
    *,
    text: str | None = None,
    status: str = "completed",
    incomplete_reason: str | None = None,
    output: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_123",
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        output_text=text if text is not None else json.dumps({"entries": entries or []}),
        output=output or [],
        error=None,
        usage=SimpleNamespace(
            input_tokens=1500,
            output_tokens=600,
            total_tokens=2100,
            output_tokens_details=SimpleNamespace(reasoning_tokens=400),
        ),
    )


def _extract(response: Any) -> Any:
    provider = _provider()
    provider._send = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return asyncio.run(provider.extract(REQUEST, model="gpt-x"))


def test_a_valid_response_becomes_domain_entries_mapped_to_their_screenshot() -> None:
    response = _response(
        [_entry(), _entry(screenshot_index=2, order=55, address="PANAMERICANA NORTE 1500")]
    )

    result = _extract(response)

    assert [e.order for e in result.entries] == [54, 55]
    assert [e.source_screenshot_id for e in result.entries] == ["fid-1", "fid-2"]
    assert result.entries[0].locality == "QUILICURA"
    assert result.provider_request_id == "resp_123"


def test_usage_and_finish_reason_are_normalised(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.services.vision.base"):
        result = _extract(_response([_entry()]))

    usage = result.usage
    assert (usage.prompt_tokens, usage.output_tokens, usage.reasoning_tokens) == (1500, 600, 400)
    assert (usage.total_tokens, usage.finish_reason, usage.finish_ok) == (2100, "completed", True)
    (record,) = [r for r in caplog.records if "VISION_RESPONSE" in r.getMessage()]
    assert "provider=openai" in record.getMessage()
    assert record.levelno == logging.INFO


def test_the_request_id_prefers_the_sdk_header_value() -> None:
    response = _response([_entry()])
    response._request_id = "req_from_header"

    assert _extract(response).provider_request_id == "req_from_header"


def test_an_incomplete_response_is_invalid_but_its_usage_is_still_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Equivalente al MAX_TOKENS de Gemini: la respuesta truncada se rechaza pero
    debe verse por que termino."""
    response = _response(
        text='{"entries": [', status="incomplete", incomplete_reason="max_output_tokens"
    )
    provider = _provider()
    provider._send = AsyncMock(return_value=response)  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO, logger="app.services.vision.base"), pytest.raises(
        ProviderError
    ) as info:
        asyncio.run(provider.extract(REQUEST, model="gpt-x"))

    assert info.value.reason is R.RESPUESTA_INVALIDA
    assert "max_output_tokens" in info.value.message
    (record,) = [r for r in caplog.records if "VISION_RESPONSE" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "finish_reason=max_output_tokens" in record.getMessage()


def test_a_refusal_is_an_invalid_response() -> None:
    refusal = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])
    error = _extract_expecting_error(_response(text="", output=[refusal]))

    assert error.reason is R.RESPUESTA_INVALIDA
    assert "refusal" in error.message


def test_an_empty_response_is_an_invalid_response() -> None:
    assert _extract_expecting_error(_response(text="  ")).reason is R.RESPUESTA_INVALIDA


def test_a_failed_response_is_an_invalid_response() -> None:
    response = _response(text="")
    response.status = "failed"
    response.error = SimpleNamespace(code="server_error")

    error = _extract_expecting_error(response)

    assert error.reason is R.RESPUESTA_INVALIDA
    assert "server_error" in error.message


def test_malformed_json_never_leaks_capture_content_into_the_error() -> None:
    secret = "CALLE SECRETA 999"
    error = _extract_expecting_error(_response(text='{"entries": [{"address": "' + secret + '"'))

    assert error.reason is R.RESPUESTA_INVALIDA
    assert secret not in error.message
    assert secret not in str(error)
    assert error.__cause__ is not None and secret not in repr(error.__cause__)


def test_an_entry_with_an_invalid_screenshot_index_is_rejected_locally() -> None:
    """`ge=1` no viaja al proveedor (no lo soporta el modo estricto): se valida
    aqui al convertir al DTO neutral."""
    error = _extract_expecting_error(_response([_entry(screenshot_index=0)]))

    assert error.reason is R.RESPUESTA_INVALIDA
    assert "entradas invalidas" in error.message


def _extract_expecting_error(response: Any) -> ProviderError:
    provider = _provider()
    provider._send = AsyncMock(return_value=response)  # type: ignore[method-assign]
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(REQUEST, model="gpt-x"))
    return info.value


# ------------------------------------------------------------------ errores del SDK

RATE = LimitKind.RATE_MINUTO
CREDIT = LimitKind.CREDITO_O_GASTO
DAILY = LimitKind.CUOTA_DIARIA


@pytest.mark.parametrize(
    ("exc", "reason", "kind", "http", "code"),
    [
        # Sobrecarga / 5xx
        (_status_error(openai.InternalServerError, 503, code="server_is_overloaded",
                       type_="service_unavailable_error"),
         R.ALTA_DEMANDA, None, 503, "server_is_overloaded"),
        (_status_error(openai.InternalServerError, 500), R.ALTA_DEMANDA, None, 500, None),
        (_status_error(openai.InternalServerError, 502), R.ALTA_DEMANDA, None, 502, None),
        # 429: credito y gasto (no transitorio)
        (_status_error(openai.RateLimitError, 429, code="credit_balance_exhausted"),
         R.LIMITE_ALCANZADO, CREDIT, 429, "credit_balance_exhausted"),
        (_status_error(openai.RateLimitError, 429, code="organization_spend_limit_exceeded"),
         R.LIMITE_ALCANZADO, CREDIT, 429, "organization_spend_limit_exceeded"),
        (_status_error(openai.RateLimitError, 429, code="project_spend_limit_exceeded"),
         R.LIMITE_ALCANZADO, CREDIT, 429, "project_spend_limit_exceeded"),
        (_status_error(openai.RateLimitError, 429, code="organization_usage_limit_exceeded"),
         R.LIMITE_ALCANZADO, CREDIT, 429, "organization_usage_limit_exceeded"),
        (_status_error(
            openai.RateLimitError, 429, code="insufficient_quota", type_="insufficient_quota"),
         R.LIMITE_ALCANZADO, CREDIT, 429, "insufficient_quota"),
        # 429: transitorio
        (_status_error(openai.RateLimitError, 429, code="slow_down", type_="rate_limit_error"),
         R.LIMITE_ALCANZADO, RATE, 429, "slow_down"),
        (_status_error(
            openai.RateLimitError, 429, code="rate_limit_exceeded", type_="rate_limit_error"),
         R.LIMITE_ALCANZADO, RATE, 429, "rate_limit_exceeded"),
        (_status_error(openai.RateLimitError, 429, type_="rate_limit_error"),
         R.LIMITE_ALCANZADO, RATE, 429, None),
        # 429: diario (heuristica por texto)
        (_status_error(openai.RateLimitError, 429, code="rate_limit_exceeded",
                       message="Rate limit reached for requests per day (RPD): Limit 200"),
         R.LIMITE_ALCANZADO, DAILY, 429, "rate_limit_exceeded"),
        # 429 sin ninguna pista
        (_status_error(openai.RateLimitError, 429),
         R.LIMITE_ALCANZADO, LimitKind.DESCONOCIDO, 429, None),
        # Resto de estados
        (_status_error(openai.NotFoundError, 404, code="model_not_found"),
         R.MODELO_NO_ENCONTRADO, None, 404, "model_not_found"),
        (_status_error(openai.BadRequestError, 400, code="model_not_found"),
         R.MODELO_NO_ENCONTRADO, None, 400, "model_not_found"),
        (_status_error(openai.AuthenticationError, 401, code="invalid_api_key"),
         R.AUTENTICACION, None, 401, "invalid_api_key"),
        (_status_error(openai.PermissionDeniedError, 403), R.AUTENTICACION, None, 403, None),
        (_status_error(openai.BadRequestError, 400, code="invalid_value"),
         R.SOLICITUD_INVALIDA, None, 400, "invalid_value"),
        (_status_error(openai.UnprocessableEntityError, 422),
         R.SOLICITUD_INVALIDA, None, 422, None),
        (_status_error(openai.APIStatusError, 413), R.SOLICITUD_INVALIDA, None, 413, None),
        (_status_error(openai.APIStatusError, 408), R.TIMEOUT, None, 408, None),
        (_status_error(openai.APIStatusError, 402), R.LIMITE_ALCANZADO, CREDIT, 402, None),
        (_status_error(openai.ConflictError, 409), R.DESCONOCIDO, None, 409, None),
    ],
)
def test_sdk_errors_are_translated_to_the_canonical_vocabulary(
    exc: Exception, reason: FailureReason, kind: LimitKind | None, http: int, code: str | None
) -> None:
    error = _extract_raising(exc)

    assert error.reason is reason
    assert error.limit_kind is kind
    assert error.http_status == http
    assert error.provider_code == code
    assert error.provider == "openai"
    assert error.model == "gpt-x"


def test_retry_after_headers_are_carried_for_the_chain() -> None:
    seconds = _extract_raising(
        _status_error(openai.RateLimitError, 429, code="slow_down", headers={"retry-after": "7"})
    )
    millis = _extract_raising(
        _status_error(
            openai.RateLimitError, 429, code="slow_down", headers={"retry-after-ms": "2500"}
        )
    )
    absent = _extract_raising(_status_error(openai.RateLimitError, 429, code="slow_down"))

    assert seconds.retry_after_seconds == pytest.approx(7.0)
    assert millis.retry_after_seconds == pytest.approx(2.5)
    assert absent.retry_after_seconds is None


def test_connection_errors_and_timeouts_are_told_apart() -> None:
    request = httpx2.Request("POST", URL)

    timeout = _extract_raising(openai.APITimeoutError(request=request))
    connection = _extract_raising(openai.APIConnectionError(request=request))

    assert timeout.reason is R.TIMEOUT  # APITimeoutError es subclase de APIConnectionError
    assert connection.reason is R.CONEXION


def test_an_api_error_without_status_is_unknown() -> None:
    error = _extract_raising(
        openai.APIError("stream roto", httpx2.Request("POST", URL), body={"code": "boom"})
    )

    assert error.reason is R.DESCONOCIDO
    assert error.provider_code == "boom"


def test_the_request_id_of_a_failed_call_is_kept_for_support_tickets() -> None:
    exc = _status_error(
        openai.InternalServerError, 503, headers={"x-request-id": "req_abc"}
    )

    assert _extract_raising(exc).provider_request_id == "req_abc"


def test_programming_errors_are_not_disguised_as_provider_failures() -> None:
    provider = _provider()
    provider._send = AsyncMock(side_effect=ValueError("bug nuestro"))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="bug nuestro"):
        asyncio.run(provider.extract(REQUEST, model="gpt-x"))


def test_a_hung_call_is_cut_by_the_hard_timeout() -> None:
    provider = _provider()
    provider._hard_timeout_seconds = 0.05

    async def hang(request: ExtractionRequest, model: str) -> Any:
        await asyncio.sleep(10)

    provider._send = hang  # type: ignore[method-assign,assignment]

    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(REQUEST, model="gpt-x"))

    assert info.value.reason is R.TIMEOUT


@pytest.mark.parametrize(
    ("code", "type_", "message", "kind"),
    [
        ("credit_balance_exhausted", None, "", CREDIT),
        (None, "insufficient_quota", "", CREDIT),
        ("slow_down", None, "", RATE),
        ("rate_limit_exceeded", None, "Limit reached for tokens per min (TPM)", RATE),
        ("rate_limit_exceeded", None, "Limit reached for tokens per day (TPD)", DAILY),
        (None, None, "requests per day", DAILY),
        (None, None, "", LimitKind.DESCONOCIDO),
    ],
)
def test_classify_rate_limit(
    code: str | None, type_: str | None, message: str, kind: LimitKind
) -> None:
    assert classify_rate_limit(code, type_, message) is kind


# ------------------------------------------------------------------ configuracion


def test_the_sdk_client_has_no_retries_of_its_own_and_an_explicit_timeout() -> None:
    """R2: el reintento esta en la cadena. Los SDK reintentan 429 y 5xx por igual,
    y un 429 por credito agotado nunca debe reintentarse."""
    client = _provider()._client

    assert client.max_retries == 0
    assert client.timeout == 60.0


def test_from_settings_returns_none_without_an_explicit_key(isolated_env: Any) -> None:
    assert OpenAIProvider.from_settings(OpenAISettings()) is None
    assert OpenAIProvider.from_settings(OpenAISettings(api_key="  ")) is None


def test_from_settings_builds_the_provider_and_never_leaks_the_key(isolated_env: Any) -> None:
    settings = OpenAISettings(
        api_key="sk-super-secret", timeout_seconds=15.0, reasoning_effort="low", image_detail="low"
    )

    provider = OpenAIProvider.from_settings(settings)

    assert provider is not None
    assert provider._client.timeout == 15.0
    assert provider._reasoning_effort == "low"
    assert provider._store is False  # por defecto no se guarda nada en OpenAI
    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in str(settings.model_dump())


def test_a_blank_reasoning_effort_means_the_model_default(
    isolated_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    assert OpenAISettings().reasoning_effort is None
