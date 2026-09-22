"""Tests de `AnthropicProvider`: solicitud, respuesta, uso y traduccion de errores.

A diferencia de los tests de OpenAI, aqui NO se fabrican excepciones a mano: el
cliente real del SDK se conecta a un `httpx2.MockTransport`, asi que la
solicitud se construye y los errores se parsean con el codigo real del SDK
(clase de la excepcion, `body`, `request_id`, cabeceras). Ningun test abre un
socket ni usa una API key real (la key es un texto de relleno).
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic
from anthropic.resources.messages import AsyncMessages

import app.adapters.vision.anthropic as anthropic_adapter
from app.adapters.vision.anthropic import (
    API_BASE_URL,
    AnthropicProvider,
    build_strict_schema,
)
from app.config import AnthropicSettings
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest
from app.services.vision.prompt import SYSTEM_INSTRUCTION

R = FailureReason
K = LimitKind
REQUEST = ExtractionRequest(images=[b"img-1", b"img-2"], screenshot_ids=["fid-1", "fid-2"])
Handler = Callable[[httpx2.Request], httpx2.Response]


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "screenshot_index": 1, "order": 54, "package_id": "696733800652",
        "customer_name": None, "address": "PANAMERICANA NORTE 8000", "locality": "QUILICURA",
        "delivery_type": None, "eta": "12:34", "time_window": "07:00 - 21:00",
        "status": "pending",
    }
    return entry | overrides


def _message(
    *,
    entries: list[dict[str, Any]] | None = None,
    text: str | None = None,
    stop_reason: str = "end_turn",
    content: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> httpx2.Response:
    body_text = text if text is not None else json.dumps({"entries": entries or [_entry()]})
    return httpx2.Response(
        200,
        headers={"request-id": "req_ok"},
        json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": content if content is not None else [{"type": "text", "text": body_text}],
            "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        },
    )


def _error(
    status: int,
    error_type: str,
    message: str = "boom",
    *,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx2.Response:
    error: dict[str, Any] = {"type": error_type, "message": message}
    if details is not None:
        error["details"] = details
    return httpx2.Response(
        status,
        headers={"request-id": "req_err", **(headers or {})},
        json={"type": "error", "error": error, "request_id": "req_err"},
    )


class _Wire:
    """Transporte simulado que cuenta las solicitudes. El proveedor construye su
    cliente EXACTAMENTE como en produccion (key, base_url, timeout, max_retries);
    solo se le inyecta este `http_client` para que no toque la red."""

    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx2.Request] = []

        def counting(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return handler(request)

        self.http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(counting))

    @property
    def body(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.requests[-1].content)
        return loaded


def _provider(handler: Handler, **kwargs: Any) -> tuple[AnthropicProvider, _Wire]:
    wire = _Wire(handler)
    real_client = AsyncAnthropic

    def with_mock_transport(**client_kwargs: Any) -> AsyncAnthropic:
        return real_client(**client_kwargs, http_client=wire.http_client)

    with patch.object(anthropic_adapter, "AsyncAnthropic", with_mock_transport):
        provider = AnthropicProvider(api_key="test-key", timeout_seconds=60.0, **kwargs)
    return provider, wire


def _run(provider: AnthropicProvider, model: str = "claude-sonnet-5") -> Any:
    return asyncio.run(provider.extract(REQUEST, model=model))


def _fail(handler: Handler, **kwargs: Any) -> ProviderError:
    provider, _ = _provider(handler, **kwargs)
    with pytest.raises(ProviderError) as info:
        _run(provider)
    return info.value


# ------------------------------------------------------------------ esquema estricto


def test_the_schema_is_a_strict_object_root_without_unsupported_constraints() -> None:
    schema = build_strict_schema()
    entry = schema["$defs"]["_AnthropicEntry"]

    assert schema["type"] == "object"  # la lista va envuelta: raiz = objeto
    assert schema["required"] == ["entries"]
    assert schema["additionalProperties"] is False
    assert entry["additionalProperties"] is False
    assert sorted(entry["required"]) == sorted(entry["properties"])  # todos requeridos
    dumped = json.dumps(schema)
    assert "minimum" not in dumped  # Field(ge=1) del DTO NO viaja al proveedor
    assert '"default"' not in dumped


# ------------------------------------------------------------------ solicitud (SDK real)


def test_the_request_on_the_wire_carries_images_labels_schema_and_no_temperature() -> None:
    provider, wire = _provider(lambda request: _message())

    _run(provider)

    (request,) = wire.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert "authorization" not in request.headers  # ni token ni credencial ambiental
    body = wire.body
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] == 16000
    assert body["system"] == SYSTEM_INSTRUCTION
    assert "temperature" not in body  # los modelos recientes de Anthropic lo rechazan (400)
    assert "thinking" not in body
    assert body["output_config"] == {
        "format": {"type": "json_schema", "schema": build_strict_schema()}
    }

    (message,) = body["messages"]
    assert message["role"] == "user"
    parts = message["content"]
    assert [p["type"] for p in parts] == ["text", "image", "text", "image", "text"]
    assert [p["text"] for p in parts if p["type"] == "text"][:2] == ["Imagen 1:", "Imagen 2:"]
    assert parts[1]["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64.b64encode(b"img-1").decode(),
    }


def test_effort_and_max_tokens_are_sent_when_configured() -> None:
    provider, wire = _provider(lambda request: _message(), effort="low", max_tokens=4000)

    _run(provider)

    assert wire.body["max_tokens"] == 4000
    assert wire.body["output_config"]["effort"] == "low"
    assert wire.body["output_config"]["format"]["type"] == "json_schema"


def test_the_sdk_has_no_per_request_retention_switch_like_openai_store() -> None:
    """OpenAI necesita `store=False`; en Anthropic NO existe un equivalente por
    solicitud (la retencion se gestiona a nivel de organizacion). Si una version
    nueva del SDK lo agrega, este test falla y hay que aplicarlo."""
    params = set(inspect.signature(AsyncMessages.create).parameters)

    assert not {"store", "retention", "zero_retention", "data_retention"} & params


# ------------------------------------------------------------------ respuesta


def test_a_valid_response_yields_entries_usage_and_the_request_id() -> None:
    provider, _ = _provider(
        lambda request: _message(
            entries=[_entry(), _entry(order=55, screenshot_index=2)],
            usage={
                "input_tokens": 1000, "output_tokens": 300,
                "output_tokens_details": {"thinking_tokens": 120},
            },
        )
    )

    result = _run(provider)

    assert [(e.order, e.source_screenshot_id) for e in result.entries] == [
        (54, "fid-1"), (55, "fid-2")
    ]
    assert result.provider_request_id == "req_ok"  # `req_...` (cabecera), no `msg_...`
    usage = result.usage
    assert (usage.prompt_tokens, usage.output_tokens, usage.total_tokens) == (1000, 300, 1300)
    assert usage.reasoning_tokens == 120
    assert (usage.finish_reason, usage.finish_ok) == ("end_turn", True)


def test_cached_input_tokens_are_added_to_the_prompt_count() -> None:
    provider, _ = _provider(
        lambda request: _message(
            usage={
                "input_tokens": 10, "output_tokens": 5,
                "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
            }
        )
    )

    assert _run(provider).usage.prompt_tokens == 1110


def test_thinking_blocks_are_ignored_and_only_text_is_parsed() -> None:
    provider, _ = _provider(
        lambda request: _message(
            content=[
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "text", "text": json.dumps({"entries": [_entry()]})},
            ]
        )
    )

    assert [e.address for e in _run(provider).entries] == ["PANAMERICANA NORTE 8000"]


SECRET = "CLIENTE-SECRETO-JUAN-PEREZ"


@pytest.mark.parametrize(
    ("response", "expected_text"),
    [
        pytest.param(_message(stop_reason="max_tokens"), "incompleta", id="truncated"),
        pytest.param(
            _message(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}),
            "refusal",
            id="refusal",
        ),
        pytest.param(_message(content=[]), "sin texto", id="no-content"),
        pytest.param(_message(text="   "), "sin texto", id="blank-text"),
        pytest.param(_message(text=f"{{esto no es json {SECRET}"), "JSON no valido", id="bad-json"),
        pytest.param(
            _message(text=json.dumps({"entries": [_entry(address=SECRET, screenshot_index=0)]})),
            "entradas invalidas",
            id="index-below-one",  # el `ge=1` se valida localmente: no viaja al proveedor
        ),
        pytest.param(
            _message(text=json.dumps({"entries": [{"address": SECRET}]})),
            "JSON no valido",
            id="missing-required-fields",
        ),
    ],
)
def test_an_unusable_response_is_invalid_without_leaking_its_content(
    response: httpx2.Response, expected_text: str
) -> None:
    error = _fail(lambda request: response)

    assert error.reason is R.RESPUESTA_INVALIDA
    assert expected_text in error.message
    assert SECRET not in error.message  # los mensajes van a logs y al JSONL
    assert error.provider_request_id == "req_ok"


def test_usage_is_logged_before_validating_so_truncated_responses_are_visible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        _fail(lambda request: _message(stop_reason="max_tokens"))

    line = next(r for r in caplog.records if "VISION_RESPONSE" in r.getMessage())
    assert line.levelno == logging.WARNING
    assert "finish_reason=max_tokens" in line.getMessage()


# ------------------------------------------------------------------ errores (SDK real)

BILLING = "Your credit balance is too low to access the Anthropic API."
TIER_CAP = (
    "You have reached your API usage limits: your organization has crossed its monthly "
    "API usage threshold, set based on your organization's API tier. "
    "You will regain access on 2026-10-01 at 00:00 UTC."
)
USER_CAP = "You have reached your specified API usage limits. You will regain access on 2026-10-01."
WORKSPACE_CAP = "You have reached your specified workspace API usage limits."


@pytest.mark.parametrize(
    ("response", "reason", "limit_kind", "retry_after"),
    [
        # 5xx: siempre ALTA_DEMANDA. Cada codigo por separado, porque en el SDK 529 y
        # 500 son clases distintas y 503/504 caen en InternalServerError.
        pytest.param(_error(529, "overloaded_error", "Overloaded"), R.ALTA_DEMANDA, None, None,
                     id="529-overloaded"),
        pytest.param(_error(500, "api_error"), R.ALTA_DEMANDA, None, None, id="500"),
        pytest.param(_error(502, "api_error"), R.ALTA_DEMANDA, None, None, id="502"),
        pytest.param(_error(503, "api_error"), R.ALTA_DEMANDA, None, None, id="503"),
        pytest.param(_error(504, "timeout_error"), R.ALTA_DEMANDA, None, None, id="504"),
        # 429 con retry-after = limite por minuto (RPM/ITPM/OTPM o aceleracion)
        pytest.param(_error(429, "rate_limit_error", headers={"retry-after": "7"}),
                     R.LIMITE_ALCANZADO, K.RATE_MINUTO, 7.0, id="429-retry-after"),
        pytest.param(_error(429, "rate_limit_error", headers={"retry-after-ms": "1500"}),
                     R.LIMITE_ALCANZADO, K.RATE_MINUTO, 1.5, id="429-retry-after-ms"),
        # Tope de gasto del tier: codigo documentado, sin retry-after
        pytest.param(_error(429, "rate_limit_error", TIER_CAP,
                            details={"error_code": "enforced_spend_limit_reached"}),
                     R.LIMITE_ALCANZADO, K.CREDITO_O_GASTO, None, id="429-tier-spend-cap"),
        # 402 billing_error
        pytest.param(_error(402, "billing_error", BILLING),
                     R.LIMITE_ALCANZADO, K.CREDITO_O_GASTO, None, id="402-billing"),
        # Limite de gasto fijado por el usuario: llega como 400 (no 402 ni 429)
        pytest.param(_error(400, "invalid_request_error", USER_CAP),
                     R.LIMITE_ALCANZADO, K.CREDITO_O_GASTO, None, id="400-user-spend-limit"),
        pytest.param(_error(400, "invalid_request_error", WORKSPACE_CAP),
                     R.LIMITE_ALCANZADO, K.CREDITO_O_GASTO, None, id="400-workspace-spend-limit"),
        # Resto de la tabla
        pytest.param(_error(400, "invalid_request_error", "messages: bad"),
                     R.SOLICITUD_INVALIDA, None, None, id="400-invalid-request"),
        pytest.param(_error(413, "request_too_large"), R.SOLICITUD_INVALIDA, None, None, id="413"),
        pytest.param(_error(422, "invalid_request_error"),
                     R.SOLICITUD_INVALIDA, None, None, id="422"),
        pytest.param(_error(404, "not_found_error", "model: claude-x"),
                     R.MODELO_NO_ENCONTRADO, None, None, id="404"),
        pytest.param(_error(401, "authentication_error", "invalid x-api-key"),
                     R.AUTENTICACION, None, None, id="401"),
        pytest.param(_error(403, "permission_error"), R.AUTENTICACION, None, None, id="403"),
        pytest.param(_error(408, "timeout_error"), R.TIMEOUT, None, None, id="408"),
        pytest.param(_error(409, "conflict_error"), R.DESCONOCIDO, None, None, id="409"),
        pytest.param(_error(418, "api_error"), R.DESCONOCIDO, None, None, id="unlisted-4xx"),
    ],
)
def test_every_cell_of_the_error_table(
    response: httpx2.Response,
    reason: FailureReason,
    limit_kind: LimitKind | None,
    retry_after: float | None,
) -> None:
    error = _fail(lambda request: response)

    assert error.reason is reason
    assert error.limit_kind is limit_kind
    assert error.retry_after_seconds == retry_after
    assert error.http_status == response.status_code
    assert error.provider_request_id == "req_err"


def test_a_529_is_overloaded_error_and_not_an_internal_server_error() -> None:
    """Contradice la tabla 3.3 del documento de arquitectura ("`InternalServerError`
    ... y 529"): en el SDK real 529 es `OverloadedError`, hermana (no hija) de
    `InternalServerError`. Un mapeo por clase lo dejaria como DESCONOCIDO."""
    provider, _ = _provider(lambda request: _error(529, "overloaded_error", "Overloaded"))

    with pytest.raises(anthropic.APIStatusError) as info:
        asyncio.run(provider._client.messages.create(model="m", max_tokens=1, messages=[]))

    assert type(info.value) is anthropic.OverloadedError
    assert not isinstance(info.value, anthropic.InternalServerError)
    assert _fail(lambda request: _error(529, "overloaded_error")).reason is R.ALTA_DEMANDA


def test_a_402_has_no_dedicated_sdk_class_and_is_still_credit() -> None:
    provider, _ = _provider(lambda request: _error(402, "billing_error", BILLING))

    with pytest.raises(anthropic.APIStatusError) as info:
        asyncio.run(provider._client.messages.create(model="m", max_tokens=1, messages=[]))

    assert type(info.value) is anthropic.APIStatusError  # sin clase propia para 402
    error = _fail(lambda request: _error(402, "billing_error", BILLING))
    assert error.limit_kind is K.CREDITO_O_GASTO


def test_a_429_without_retry_after_or_documented_code_is_credit_by_default() -> None:
    """HEURISTICA NO VERIFICADA con una respuesta real (por eso es configurable):
    la documentacion dice que el 429 del tope de gasto no trae `retry-after`,
    pero no que todo 429 sin el sea un tope de gasto."""
    error = _fail(lambda request: _error(429, "rate_limit_error", "?"))

    assert (error.reason, error.limit_kind) == (R.LIMITE_ALCANZADO, K.CREDITO_O_GASTO)


def test_the_429_heuristic_can_be_turned_off() -> None:
    error = _fail(
        lambda request: _error(429, "rate_limit_error", "?"),
        treat_429_without_retry_after_as_credit=False,
    )

    assert (error.reason, error.limit_kind) == (R.LIMITE_ALCANZADO, K.DESCONOCIDO)


def test_the_documented_spend_cap_code_wins_even_with_the_heuristic_off_or_a_retry_after() -> None:
    def cap(headers: dict[str, str]) -> Handler:
        return lambda request: _error(
            429, "rate_limit_error", TIER_CAP,
            details={"error_code": "enforced_spend_limit_reached"}, headers=headers,
        )

    off = _fail(cap({}), treat_429_without_retry_after_as_credit=False)
    with_header = _fail(cap({"retry-after": "30"}))

    assert off.limit_kind is K.CREDITO_O_GASTO
    assert with_header.limit_kind is K.CREDITO_O_GASTO


def test_the_provider_code_keeps_the_type_and_the_documented_error_code() -> None:
    spend_cap = {"error_code": "enforced_spend_limit_reached"}
    capped = _fail(lambda request: _error(429, "rate_limit_error", TIER_CAP, details=spend_cap))
    overloaded = _fail(lambda request: _error(529, "overloaded_error"))

    assert capped.provider_code == "rate_limit_error/enforced_spend_limit_reached"
    assert overloaded.provider_code == "overloaded_error"


def test_the_message_is_the_error_message_not_the_whole_body_dump() -> None:
    error = _fail(lambda request: _error(402, "billing_error", BILLING))

    assert error.message == BILLING  # no "Error code: 402 - {'type': 'error', ...}"


def test_a_non_json_error_body_still_maps_by_status_and_is_truncated() -> None:
    html = "<html>" + "x" * 2000 + "</html>"
    error = _fail(lambda request: httpx2.Response(502, text=html))

    assert error.reason is R.ALTA_DEMANDA
    assert error.provider_code is None
    assert len(error.message) <= 300


def test_a_failed_call_is_exactly_one_http_request_the_sdk_never_retries() -> None:
    """Con el default del SDK (`max_retries=2`) un 529 se enviaria 3 veces."""
    provider, wire = _provider(lambda request: _error(529, "overloaded_error"))

    with pytest.raises(ProviderError):
        _run(provider)

    assert len(wire.requests) == 1
    assert provider._client.max_retries == 0


@pytest.mark.parametrize(
    ("raised", "reason"),
    [
        pytest.param(httpx2.ReadTimeout("t"), R.TIMEOUT, id="read-timeout"),
        pytest.param(httpx2.ConnectTimeout("t"), R.TIMEOUT, id="connect-timeout"),
        pytest.param(httpx2.ConnectError("c"), R.CONEXION, id="connect-error"),
    ],
)
def test_transport_failures_are_timeout_or_connection(
    raised: Exception, reason: FailureReason
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise raised

    provider, wire = _provider(handler)
    with pytest.raises(ProviderError) as info:
        _run(provider)

    assert info.value.reason is reason
    assert len(wire.requests) == 1  # timeout/conexion tampoco se reintentan en el SDK


def test_a_programming_error_is_not_swallowed_as_a_provider_failure() -> None:
    provider, _ = _provider(lambda request: _message())

    async def broken_send(request: ExtractionRequest, model: str) -> Any:
        raise ZeroDivisionError

    provider._send = broken_send  # type: ignore[method-assign]
    with pytest.raises(ZeroDivisionError):
        _run(provider)


# ------------------------------------------------------------ cliente: credenciales ambientales

HOSTILE_ENV = {
    "ANTHROPIC_AUTH_TOKEN": "ambient-token",
    "ANTHROPIC_PROFILE": "ambient-profile",
    "ANTHROPIC_BASE_URL": "https://ambient-proxy.example.invalid",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Injected: from-ambient-env",
    "ANTHROPIC_WEBHOOK_SIGNING_KEY": "ambient-webhook-key",
}


class _ClientMustNotBeBuiltError(AssertionError):
    pass


def _forbid_client_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise _ClientMustNotBeBuiltError("se instancio AsyncAnthropic")

    monkeypatch.setattr(anthropic_adapter, "AsyncAnthropic", forbidden)


def test_without_an_explicit_key_the_client_is_never_built_even_with_ambient_credentials(
    isolated_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Inactivo" significa inactivo: token, perfil y demas credenciales del
    entorno NO activan el tier (el SDK las resolveria si se construyera)."""
    for name, value in HOSTILE_ENV.items():
        monkeypatch.setenv(name, value)
    _forbid_client_construction(monkeypatch)

    assert AnthropicProvider.from_settings(AnthropicSettings()) is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")  # una key en blanco tampoco cuenta
    assert AnthropicProvider.from_settings(AnthropicSettings()) is None


def test_with_an_explicit_key_ambient_variables_cannot_redirect_or_inject(
    isolated_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in HOSTILE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "explicit-key")

    provider = AnthropicProvider.from_settings(AnthropicSettings())

    assert provider is not None
    client = provider._client
    assert client.api_key == "explicit-key"
    assert client.auth_token is None  # el token ambiental no se usa
    assert str(client.base_url).rstrip("/") == API_BASE_URL  # sin proxy ambiental
    assert client.auth_headers == {"X-Api-Key": "explicit-key"}
    assert "X-Injected" not in client.default_headers  # ANTHROPIC_CUSTOM_HEADERS
    assert client.max_retries == 0  # el reintento es de la cadena (R2)
    assert client.timeout == 60.0


def test_the_explicit_base_url_holds_even_if_the_env_scrub_were_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensa en profundidad: dos capas independientes (variables ocultas y
    `base_url` explicito) para que el trafico jamas salga hacia un proxy ambiental."""
    monkeypatch.setattr(anthropic_adapter, "_without_ambient_anthropic_env", nullcontext)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", HOSTILE_ENV["ANTHROPIC_BASE_URL"])

    provider = AnthropicProvider(api_key="k", timeout_seconds=60.0)

    assert str(provider._client.base_url).rstrip("/") == API_BASE_URL


def test_the_ambient_environment_is_restored_after_building_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in HOSTILE_ENV.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)

    AnthropicProvider(api_key="k", timeout_seconds=60.0)

    assert dict(os.environ) == before


def test_the_ambient_environment_is_restored_even_if_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in HOSTILE_ENV.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    _forbid_client_construction(monkeypatch)

    with pytest.raises(_ClientMustNotBeBuiltError):
        AnthropicProvider(api_key="k", timeout_seconds=60.0)

    assert dict(os.environ) == before


def test_settings_are_passed_through_to_the_provider(
    isolated_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "5000")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
    monkeypatch.setenv("ANTHROPIC_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ANTHROPIC_TREAT_429_WITHOUT_RETRY_AFTER_AS_CREDIT", "false")

    provider = AnthropicProvider.from_settings(AnthropicSettings())

    assert provider is not None
    assert provider._max_tokens == 5000
    assert provider._effort == "medium"
    assert provider._client.timeout == 45.0
    assert provider._hard_timeout_seconds == 47.0
    assert provider._429_without_retry_after_is_credit is False


def test_settings_defaults_and_secret_handling(
    isolated_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "  ")  # en blanco = default del modelo

    settings = AnthropicSettings()

    assert (settings.max_tokens, settings.effort) == (16000, None)
    assert settings.treat_429_without_retry_after_as_credit is True
    assert "sk-ant-secret-value" not in repr(settings)  # SecretStr


def test_an_unknown_effort_value_is_rejected_at_startup(
    isolated_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_EFFORT", "turbo")

    with pytest.raises(ValueError, match="effort"):
        AnthropicSettings()
