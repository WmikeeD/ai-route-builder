"""Arranque del bot: timeouts de la API de Telegram y reintentos de `initialize()`.

Incidente: en Render, un unico `TimedOut` en `getMe` (con el timeout por
defecto de 5 s de la libreria) tumbo el arranque completo, sin reintentos.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from telegram.error import BadRequest, InvalidToken, NetworkError, TelegramError, TimedOut
from telegram.ext import Application
from telegram.request import HTTPXRequest

import app.main as main_module
from app.adapters import telegram_bot
from app.config import get_settings


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Reemplaza la espera entre reintentos para no dormir de verdad."""
    sleep = AsyncMock()
    monkeypatch.setattr(telegram_bot.asyncio, "sleep", sleep)
    return sleep


def _fake_application(*outcomes: TelegramError | None) -> Any:
    return SimpleNamespace(initialize=AsyncMock(side_effect=list(outcomes)))


def _slept(sleep: AsyncMock) -> list[float]:
    return [call.args[0] for call in sleep.await_args_list]


def test_create_application_sets_explicit_telegram_timeouts() -> None:
    application = telegram_bot.create_application(
        token="123456:test", route_extractor=Mock(), session_manager=None
    )

    request = application.bot.request
    assert isinstance(request, HTTPXRequest)
    timeout = request._client.timeout
    assert timeout.connect == telegram_bot.TELEGRAM_CONNECT_TIMEOUT_SECONDS == 10
    assert timeout.read == telegram_bot.TELEGRAM_READ_TIMEOUT_SECONDS == 10
    # Sin tocar: defaults de la libreria.
    assert timeout.write == 5
    assert timeout.pool == 1
    assert request._media_write_timeout == 20


@pytest.mark.parametrize("error", [TimedOut(), NetworkError("Bad Gateway")])
def test_initialize_retries_transient_network_errors(
    sleeps: AsyncMock, caplog: pytest.LogCaptureFixture, error: TelegramError
) -> None:
    application = _fake_application(error, error, None)

    with caplog.at_level(logging.WARNING, logger=telegram_bot.__name__):
        asyncio.run(telegram_bot.initialize_with_retry(application))

    assert application.initialize.await_count == 3
    assert _slept(sleeps) == [2.0, 4.0]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "intento 1/3" in warnings[0].getMessage()
    assert "intento 2/3" in warnings[1].getMessage()


@pytest.mark.parametrize("error", [InvalidToken("Unauthorized"), BadRequest("Bad request")])
def test_initialize_does_not_retry_non_transient_errors(
    sleeps: AsyncMock, error: TelegramError
) -> None:
    application = _fake_application(error, None)

    with pytest.raises(type(error)):
        asyncio.run(telegram_bot.initialize_with_retry(application))

    assert application.initialize.await_count == 1
    sleeps.assert_not_awaited()


def test_initialize_gives_up_after_third_attempt(sleeps: AsyncMock) -> None:
    last = TimedOut("tercero")
    application = _fake_application(TimedOut(), TimedOut(), last, None)

    with pytest.raises(TimedOut) as exc_info:
        asyncio.run(telegram_bot.initialize_with_retry(application))

    assert exc_info.value is last
    assert application.initialize.await_count == 3
    assert _slept(sleeps) == [2.0, 4.0]


def test_lifespan_retries_initialize_and_sets_bootstrap_retries(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, sleeps: AsyncMock
) -> None:
    """El arranque real de la app usa el reintento y `bootstrap_retries`."""
    updater = SimpleNamespace(start_polling=AsyncMock(), stop=AsyncMock())
    telegram_app = SimpleNamespace(
        initialize=AsyncMock(side_effect=[TimedOut(), None]),
        start=AsyncMock(),
        stop=AsyncMock(),
        shutdown=AsyncMock(),
        updater=updater,
    )
    chain = SimpleNamespace(aclose=AsyncMock())

    def _create_application(**_: Any) -> Application:
        return telegram_app  # type: ignore[return-value]

    def _build_fallback_chain(*_: Any, **__: Any) -> SimpleNamespace:
        return chain

    monkeypatch.setattr(main_module, "create_application", _create_application)
    monkeypatch.setattr(main_module, "build_fallback_chain", _build_fallback_chain)
    # El logging ya tiene sus propios tests; aqui no se altera el global.
    monkeypatch.setattr(main_module, "setup_logging", lambda _level: None)
    get_settings.cache_clear()

    async def _run() -> None:
        async with main_module.lifespan(FastAPI()):
            pass

    try:
        asyncio.run(_run())
    finally:
        get_settings.cache_clear()

    assert telegram_app.initialize.await_count == 2
    updater.start_polling.assert_awaited_once_with(
        bootstrap_retries=telegram_bot.POLLING_BOOTSTRAP_RETRIES
    )
    assert telegram_bot.POLLING_BOOTSTRAP_RETRIES == 3
