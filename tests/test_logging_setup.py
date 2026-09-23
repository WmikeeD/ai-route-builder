"""Regresion: el token del bot de Telegram no debe llegar a los logs.

Incidente: `httpx` loguea en INFO la URL completa de cada request, y la API de
Telegram lleva el token en la URL. Estos tests fallan si alguien quita el
ajuste de nivel de `httpx`/`httpcore` o el filtro de redaccion.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

import app.main as main_module
from app.config import get_settings
from app.logging_setup import (
    NOISY_HTTP_LOGGERS,
    NOISY_SDK_LOGGERS,
    REDACTED,
    REDACTED_API_KEY,
    SecretRedactingFilter,
    redact_secrets,
    setup_logging,
)

# Armado en runtime para no dejar un literal con forma de token en el repo.
FAKE_TOKEN = "123456789" + ":" + "AAbb_-" + "x" * 29
FAKE_TOKEN_URLENCODED = FAKE_TOKEN.replace(":", "%3A")
FAKE_URL = f"https://api.telegram.org/bot{FAKE_TOKEN}/getUpdates"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    names = ("", *NOISY_HTTP_LOGGERS, *NOISY_SDK_LOGGERS)
    levels = {name: logging.getLogger(name).level for name in names}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    handlers = list(logging.getLogger().handlers)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            handlers.extend(logger.handlers)
    for handler in handlers:
        for f in [f for f in handler.filters if isinstance(f, SecretRedactingFilter)]:
            handler.removeFilter(f)


def _assert_http_loggers_quiet() -> None:
    for name in (*NOISY_HTTP_LOGGERS, *NOISY_SDK_LOGGERS):
        logger = logging.getLogger(name)
        assert logger.getEffectiveLevel() >= logging.WARNING, name
        assert not logger.isEnabledFor(logging.INFO), name


def test_every_http_library_and_provider_sdk_logger_is_covered() -> None:
    """Lista explicita: si alguien quita un logger de las tuplas, el resto de
    los tests (que iteran las tuplas) seguiria pasando sin cubrirlo.
    `httpx2`/`httpcore2` son paquetes aparte, usados por los SDK de OpenAI y
    Anthropic; `openai`/`anthropic` vuelcan el cuerpo del request en DEBUG."""
    covered = {*NOISY_HTTP_LOGGERS, *NOISY_SDK_LOGGERS}

    assert {"httpx", "httpcore", "httpx2", "httpcore2", "openai", "anthropic"} <= covered


@pytest.mark.parametrize("app_level", ["DEBUG", "INFO", "WARNING"])
def test_setup_logging_forces_http_loggers_to_warning(app_level: str) -> None:
    for name in (*NOISY_HTTP_LOGGERS, *NOISY_SDK_LOGGERS):
        logging.getLogger(name).setLevel(logging.DEBUG)

    setup_logging(app_level)

    _assert_http_loggers_quiet()


def test_setup_logging_attaches_filter_to_root_handlers() -> None:
    setup_logging("INFO")

    root_handlers = logging.getLogger().handlers
    assert root_handlers
    for handler in root_handlers:
        assert any(isinstance(f, SecretRedactingFilter) for f in handler.filters)


def test_setup_logging_is_idempotent() -> None:
    setup_logging("INFO")
    setup_logging("INFO")

    for handler in logging.getLogger().handlers:
        count = sum(isinstance(f, SecretRedactingFilter) for f in handler.filters)
        assert count == 1


@pytest.mark.parametrize(
    "logger_name,level",
    [("telegram.ext.Updater", logging.ERROR), ("app.cualquiera", logging.INFO)],
)
def test_token_is_redacted_end_to_end(
    caplog: pytest.LogCaptureFixture, logger_name: str, level: int
) -> None:
    setup_logging("DEBUG")
    caplog.set_level(logging.DEBUG)

    logging.getLogger(logger_name).log(level, "HTTP Request: POST %s", FAKE_URL)
    logging.getLogger(logger_name).log(level, "descarga %s", FAKE_TOKEN_URLENCODED.lower())

    assert FAKE_TOKEN not in caplog.text
    assert FAKE_TOKEN.split(":")[1] not in caplog.text
    assert caplog.text.count(REDACTED) == 2


def test_token_in_traceback_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    setup_logging("INFO")

    try:
        raise RuntimeError(f"fallo contra {FAKE_URL}")
    except RuntimeError:
        logging.getLogger("telegram.ext.Updater").exception("error de red")

    record = caplog.records[-1]
    assert record.exc_text is not None
    assert "RuntimeError" in record.exc_text
    assert FAKE_TOKEN not in record.exc_text
    assert FAKE_TOKEN not in caplog.text
    assert REDACTED in caplog.text


def test_filter_on_raw_record() -> None:
    record = logging.LogRecord(
        name="httpx",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="HTTP Request: GET %s %s",
        args=(FAKE_URL, "200 OK"),
        exc_info=None,
    )

    assert SecretRedactingFilter().filter(record) is True
    assert record.getMessage() == (
        f"HTTP Request: GET https://api.telegram.org/bot{REDACTED}/getUpdates 200 OK"
    )


# Armadas en runtime por la misma razon que FAKE_TOKEN.
FAKE_OPENAI_KEYS = [
    "sk-" + "proj-" + "Ab1_" * 12,
    "sk-" + "svcacct-" + "x" * 30,
    "sk-" + "Z9" * 24,
]
FAKE_ANTHROPIC_KEY = "sk-" + "ant-api03-" + "Qw_-" * 20


@pytest.mark.parametrize("key", [*FAKE_OPENAI_KEYS, FAKE_ANTHROPIC_KEY])
def test_provider_api_keys_are_redacted(caplog: pytest.LogCaptureFixture, key: str) -> None:
    setup_logging("INFO")

    logging.getLogger("app.cualquiera").info("Authorization: Bearer %s", key)
    try:
        raise RuntimeError(f"fallo con x-api-key={key}")
    except RuntimeError:
        logging.getLogger("app.cualquiera").exception("error del proveedor")

    assert key not in caplog.text
    assert caplog.text.count(REDACTED_API_KEY) == 2


@pytest.mark.parametrize(
    "text",
    [
        "sin secretos",
        "orden=123456789 bulto=ABC",
        "latency=52500 ms",
        "12345678:corto",
        "sk-corto",
        "task-1234567890123456789012345",
    ],
)
def test_redaction_leaves_ordinary_text_untouched(text: str) -> None:
    assert redact_secrets(text) == text


def test_app_lifespan_applies_logging_setup(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    """La configuracion llega por el arranque real de la app, no solo por
    llamar a `setup_logging` a mano."""

    async def _noop() -> None:
        return None

    telegram_app = SimpleNamespace(
        initialize=_noop, start=_noop, stop=_noop, shutdown=_noop, updater=None
    )
    chain = SimpleNamespace(aclose=_noop)

    def _create_application(**_: Any) -> SimpleNamespace:
        return telegram_app

    def _build_fallback_chain(*_: Any, **__: Any) -> SimpleNamespace:
        return chain

    monkeypatch.setattr(main_module, "create_application", _create_application)
    monkeypatch.setattr(main_module, "build_fallback_chain", _build_fallback_chain)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    for name in (*NOISY_HTTP_LOGGERS, *NOISY_SDK_LOGGERS):
        logging.getLogger(name).setLevel(logging.DEBUG)
    get_settings.cache_clear()

    async def _run() -> None:
        async with main_module.lifespan(FastAPI()):
            _assert_http_loggers_quiet()
            for handler in logging.getLogger().handlers:
                assert any(isinstance(f, SecretRedactingFilter) for f in handler.filters)

    try:
        asyncio.run(_run())
    finally:
        get_settings.cache_clear()
