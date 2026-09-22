"""Anthropic como tier 4 de la cadena real (fabrica + proveedores reales, sin red).

Con key de relleno el tier esta activo y se prueba completo con el `_send`
reemplazado; sin key debe quedar INACTIVO aunque el entorno traiga
credenciales de Anthropic. Ninguna llamada real: jamas hay una key valida.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import anthropic
import httpx2
import pytest

import app.adapters.vision.anthropic as anthropic_adapter
from app.adapters.vision.anthropic import AnthropicProvider
from app.adapters.vision.factory import build_fallback_chain
from app.config import Settings
from app.services.vision.errors import AllProvidersFailedError, FailureReason, LimitKind
from app.services.vision.fault_injection import FaultInjectingProvider
from app.services.vision.telemetry import format_attempt
from tests.vision_helpers import ListSink

URL = "https://api.anthropic.com/v1/messages"


def _settings(
    monkeypatch: pytest.MonkeyPatch, *, drop: tuple[str, ...] = (), **env: str
) -> Settings:
    base = {
        "GEMINI_API_KEY": "k-gemini", "OPENAI_API_KEY": "k-openai",
        "ANTHROPIC_API_KEY": "k-anthropic",
        "GEMINI_MAX_ATTEMPTS": "1", "OPENAI_MAX_ATTEMPTS": "1", "ANTHROPIC_MAX_ATTEMPTS": "1",
        # Los tres primeros tiers fallan por sobrecarga sin gastar nada:
        "FORCE_VISION_ERROR": "gemini=503,openai=503",
    }
    for name, value in (base | env).items():
        if name not in drop:
            monkeypatch.setenv(name, value)
    return Settings()


def _anthropic_ok() -> SimpleNamespace:
    entry = {
        "screenshot_index": 1, "order": 54, "package_id": "696733800652",
        "customer_name": None, "address": "PANAMERICANA NORTE 8000", "locality": "QUILICURA",
        "delivery_type": None, "eta": "12:34", "time_window": "07:00 - 21:00",
        "status": "pending",
    }
    return SimpleNamespace(
        id="msg_1", _request_id="req_1", stop_reason="end_turn", stop_details=None,
        content=[SimpleNamespace(type="text", text=json.dumps({"entries": [entry]}))],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
            cache_read_input_tokens=0, output_tokens_details=None,
        ),
    )


def _anthropic_error(
    cls: type[anthropic.APIStatusError], status: int, error_type: str
) -> Exception:
    body = {"type": "error", "error": {"type": error_type, "message": "m"}, "request_id": "req_e"}
    response = httpx2.Response(
        status, headers={"request-id": "req_e"}, json=body, request=httpx2.Request("POST", URL)
    )
    return cls("m", response=response, body=body)


def _anthropic_inner(chain: Any) -> AnthropicProvider:
    """Con FORCE_VISION_ERROR la fabrica envuelve a TODOS los proveedores en el
    inyector; el real esta en `_inner`."""
    wrapper = chain.tiers[3].provider
    assert isinstance(wrapper, FaultInjectingProvider)
    inner = wrapper._inner
    assert isinstance(inner, AnthropicProvider)
    return inner


def _run(chain: Any) -> Any:
    return asyncio.run(chain.extract_entries(images=[b"img"], screenshot_ids=["fid-1"]))


def test_with_an_anthropic_key_tier_four_is_a_real_active_provider(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = build_fallback_chain(_settings(monkeypatch, FORCE_VISION_ERROR=""), recorder=ListSink())

    assert [t.active for t in chain.tiers] == [True, True, True, True]
    assert isinstance(chain.tiers[3].provider, AnthropicProvider)
    assert chain.tiers[3].spec.model == "claude-sonnet-5"


def test_when_gemini_and_openai_are_overloaded_anthropic_answers(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = ListSink()
    chain = build_fallback_chain(_settings(monkeypatch), recorder=sink)
    _anthropic_inner(chain)._send = AsyncMock(return_value=_anthropic_ok())  # type: ignore[method-assign]

    entries = _run(chain)

    assert [e.address for e in entries] == ["PANAMERICANA NORTE 8000"]
    assert entries[0].source_screenshot_id == "fid-1"
    assert sink.outcomes == ["FALLO", "FALLO", "FALLO", "EXITO"]
    assert sink.attempts[3].provider == "anthropic"
    assert sink.attempts[3].provider_request_id == "req_1"


def test_an_overloaded_529_from_anthropic_is_high_demand_and_exhausts_the_chain(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = ListSink()
    chain = build_fallback_chain(_settings(monkeypatch), recorder=sink)
    _anthropic_inner(chain)._send = AsyncMock(  # type: ignore[method-assign]
        side_effect=_anthropic_error(anthropic.OverloadedError, 529, "overloaded_error")
    )

    with pytest.raises(AllProvidersFailedError) as info:
        _run(chain)

    assert info.value.dominant_reason is FailureReason.ALTA_DEMANDA
    assert sink.attempts[3].reason is FailureReason.ALTA_DEMANDA
    assert sink.attempts[3].http_status == 529


def test_a_billing_error_blocks_anthropic_and_the_next_request_skips_it(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sink = ListSink()
    chain = build_fallback_chain(_settings(monkeypatch), recorder=sink)
    send = AsyncMock(side_effect=_anthropic_error(anthropic.APIStatusError, 402, "billing_error"))
    _anthropic_inner(chain)._send = send  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO), pytest.raises(AllProvidersFailedError) as first:
        _run(chain)
    with pytest.raises(AllProvidersFailedError):
        _run(chain)

    assert send.await_count == 1  # la 2a solicitud NO llama a Anthropic
    assert "SALTADO_CUOTA" in sink.outcomes
    assert first.value.dominant_limit_kind is LimitKind.CREDITO_O_GASTO
    logged = [r for r in caplog.records if "BLOQUEO_CREDITO_O_GASTO" in r.getMessage()]
    assert len(logged) == 1 and "anthropic" in logged[0].getMessage()


def test_without_a_key_tier_four_is_skipped_and_ambient_credentials_do_not_activate_it(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La cadena llega al tier 4 (los tres primeros fallan) y lo salta limpio:
    ni se instancia el SDK ni se hace una llamada, aunque el entorno traiga
    token, perfil y proxy de Anthropic."""
    for name, value in {
        "ANTHROPIC_AUTH_TOKEN": "ambient-token",
        "ANTHROPIC_PROFILE": "ambient-profile",
        "ANTHROPIC_BASE_URL": "https://ambient-proxy.example.invalid",
    }.items():
        monkeypatch.setenv(name, value)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("se instancio AsyncAnthropic sin key explicita")

    monkeypatch.setattr(anthropic_adapter, "AsyncAnthropic", forbidden)
    sink = ListSink()
    settings = _settings(monkeypatch, drop=("ANTHROPIC_API_KEY",))

    chain = build_fallback_chain(settings, recorder=sink)
    with pytest.raises(AllProvidersFailedError):
        _run(chain)

    assert chain.tiers[3].active is False
    assert chain.tiers[3].inactive_reason == "sin ANTHROPIC_API_KEY"
    assert sink.outcomes == ["FALLO", "FALLO", "FALLO", "SALTADO_INACTIVO"]
    # Es el texto que emite el registrador real (`ListSink` solo guarda los intentos):
    assert format_attempt(sink.attempts[3]) == "tier 4 (Anthropic) no configurado, se omite"
