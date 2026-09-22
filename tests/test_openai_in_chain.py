"""OpenAI como tier 3 de la cadena real (fabrica + proveedores reales, sin red).

Los `_send` de Gemini y de OpenAI se reemplazan; todo lo demas (fabrica,
cadena, traduccion de errores, cortacircuitos, guardia de cuota) es el codigo
de produccion. La API key es un texto de relleno: jamas se hace una llamada.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import openai
import pytest

from app.adapters.vision.factory import build_fallback_chain
from app.adapters.vision.openai import OpenAIProvider
from app.config import Settings
from app.services.vision.errors import AllProvidersFailedError, FailureReason, LimitKind
from app.services.vision.fault_injection import FaultInjectingProvider
from tests.vision_helpers import ListSink


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    base = {"GEMINI_API_KEY": "k-gemini", "OPENAI_API_KEY": "k-openai", "GEMINI_MAX_ATTEMPTS": "1",
            "OPENAI_MAX_ATTEMPTS": "1"}
    for name, value in (base | env).items():
        monkeypatch.setenv(name, value)
    return Settings()


def _openai_ok() -> SimpleNamespace:
    entry = {
        "screenshot_index": 1, "order": 54, "package_id": "696733800652",
        "customer_name": None, "address": "PANAMERICANA NORTE 8000", "locality": "QUILICURA",
        "delivery_type": None, "eta": "12:34", "time_window": "07:00 - 21:00",
        "status": "pending",
    }
    return SimpleNamespace(
        id="resp_1", status="completed", incomplete_details=None, error=None, output=[],
        output_text=json.dumps({"entries": [entry]}),
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, total_tokens=15,
            output_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
    )


def _openai_error(cls: type[openai.APIStatusError], status: int, **body: str) -> Exception:
    payload = {"message": "m", "type": None, "code": None, "param": None} | body
    response = httpx2.Response(
        status, json={"error": payload}, request=httpx2.Request("POST", "https://x/v1/responses")
    )
    return cls("m", response=response, body=payload)


def _openai_inner(chain: Any) -> OpenAIProvider:
    """Con FORCE_VISION_ERROR la fabrica envuelve a TODOS los proveedores en el
    inyector (para contar sus llamadas reales); el real esta en `_inner`."""
    wrapper = chain.tiers[2].provider
    assert isinstance(wrapper, FaultInjectingProvider)
    inner = wrapper._inner
    assert isinstance(inner, OpenAIProvider)
    return inner


def _run(chain: Any) -> Any:
    return asyncio.run(chain.extract_entries(images=[b"img"], screenshot_ids=["fid-1"]))


def test_with_an_openai_key_tier_three_is_a_real_active_provider(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = build_fallback_chain(_settings(monkeypatch), recorder=ListSink())

    assert [t.active for t in chain.tiers] == [True, True, True, False]
    assert isinstance(chain.tiers[2].provider, OpenAIProvider)
    assert chain.tiers[3].inactive_reason == "sin ANTHROPIC_API_KEY"


def test_when_both_gemini_models_are_overloaded_openai_answers(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """El caso que motivo todo: Gemini saturado (los dos modelos) y la cadena
    llega a otra plataforma. Los dos 503 de Gemini estan forzados (cero llamadas
    reales); OpenAI responde con un JSON estricto valido."""
    sink = ListSink()
    chain = build_fallback_chain(
        _settings(monkeypatch, FORCE_VISION_ERROR="gemini=503"), recorder=sink
    )
    openai_provider = _openai_inner(chain)
    openai_provider._send = AsyncMock(return_value=_openai_ok())  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO):
        entries = _run(chain)

    assert [e.address for e in entries] == ["PANAMERICANA NORTE 8000"]
    assert entries[0].source_screenshot_id == "fid-1"
    assert sink.outcomes == ["FALLO", "FALLO", "EXITO"]  # el tier 4 ni se alcanza
    assert [(a.provider, a.reason) for a in sink.attempts[:2]] == [
        ("gemini", FailureReason.ALTA_DEMANDA)
    ] * 2
    assert sink.attempts[2].provider == "openai"
    assert sink.attempts[2].usage is not None and sink.attempts[2].usage.prompt_tokens == 10


def test_a_credit_exhausted_429_blocks_openai_and_the_next_request_skips_it(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sink = ListSink()
    chain = build_fallback_chain(
        _settings(monkeypatch, FORCE_VISION_ERROR="gemini=503"), recorder=sink
    )
    openai_provider = _openai_inner(chain)
    send = AsyncMock(
        side_effect=_openai_error(
            openai.RateLimitError, 429, code="credit_balance_exhausted", type="insufficient_quota"
        )
    )
    openai_provider._send = send  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO), pytest.raises(AllProvidersFailedError) as first:
        _run(chain)
    with pytest.raises(AllProvidersFailedError) as second:
        _run(chain)

    assert send.await_count == 1  # la 2a solicitud NO llama a OpenAI: requiere intervencion
    credit_failure = next(a for a in sink.attempts if a.provider == "openai")
    assert credit_failure.reason is FailureReason.LIMITE_ALCANZADO
    assert credit_failure.limit_kind is LimitKind.CREDITO_O_GASTO
    assert "SALTADO_CUOTA" in sink.outcomes
    # La UI recibe el subtipo correcto para decir "reintentar no ayuda":
    for error in (first.value, second.value):
        assert error.dominant_reason is FailureReason.LIMITE_ALCANZADO
        assert error.dominant_limit_kind is LimitKind.CREDITO_O_GASTO
    logged = [r for r in caplog.records if "BLOQUEO_CREDITO_O_GASTO" in r.getMessage()]
    assert len(logged) == 1 and logged[0].levelno == logging.ERROR
    assert "openai" in logged[0].getMessage()


def test_an_invalid_openai_key_does_not_block_the_chain_or_the_provider_forever(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUTENTICACION continua al siguiente tier (decision #2) y NO bloquea: una
    key corregida en caliente no aplica, pero tampoco se trata como credito."""
    sink = ListSink()
    chain = build_fallback_chain(
        _settings(monkeypatch, FORCE_VISION_ERROR="gemini=503"), recorder=sink
    )
    openai_provider = _openai_inner(chain)
    send = AsyncMock(
        side_effect=_openai_error(openai.AuthenticationError, 401, code="invalid_api_key")
    )
    openai_provider._send = send  # type: ignore[method-assign]

    for _ in range(2):
        with pytest.raises(AllProvidersFailedError):
            _run(chain)

    assert send.await_count == 2  # se reintenta en cada solicitud (no es un bloqueo de cuota)
    assert "SALTADO_CUOTA" not in sink.outcomes
