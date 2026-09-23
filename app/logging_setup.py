"""Configuracion de logging de la app, con defensa contra fugas de secretos.

La API de Telegram lleva el token del bot DENTRO de la URL
(`https://api.telegram.org/bot<id>:<token>/getUpdates`), y `httpx` loguea la
URL completa de cada request en INFO. Por eso, en capas:

1. `httpx` y `httpcore` quedan SIEMPRE en WARNING, sin importar LOG_LEVEL ni
   ninguna variable de entorno.
2. Un filtro redacta cualquier token con forma de token de bot de Telegram
   (tambien la variante URL-encodeada `%3A` de las descargas de archivos) de
   toda linea que pase por los handlers configurados, incluidos tracebacks.
"""

from __future__ import annotations

import logging
import re

# Loggers de librerias HTTP que loguean URLs completas en INFO/DEBUG.
NOISY_HTTP_LOGGERS = ("httpx", "httpcore")

# `<id>:<secreto>` o `<id>%3A<secreto>` (el `%3a` en minuscula tambien es valido).
TELEGRAM_TOKEN_PATTERN = re.compile(r"\d{8,10}(?::|%3[Aa])[A-Za-z0-9_-]{35}")
REDACTED = "[TOKEN REDACTADO]"


def redact_secrets(text: str) -> str:
    return TELEGRAM_TOKEN_PATTERN.sub(REDACTED, text)


class SecretRedactingFilter(logging.Filter):
    """Redacta tokens de bot de Telegram del mensaje, del traceback y del
    stack de un record. Nunca descarta records: solo los reescribe."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # args mal formados: que el handler reporte el error
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = None

        # El Formatter cachea el traceback en `exc_text` y reutiliza ese texto;
        # se genera aqui para poder redactarlo antes de que llegue al handler.
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_secrets(record.stack_info)
        return True


def _all_handlers() -> list[logging.Handler]:
    handlers = list(logging.getLogger().handlers)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            handlers.extend(logger.handlers)
    return handlers


def install_redaction_filter() -> None:
    """Agrega el filtro a todos los handlers existentes (root y loggers con
    handlers propios, p. ej. los de uvicorn). Idempotente."""
    for handler in _all_handlers():
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())


def setup_logging(level: str | int) -> None:
    """Configura el logging de la app. Llamar al arrancar, antes de crear
    cualquier cliente HTTP."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
    else:
        logging.basicConfig(level=level)

    for name in NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    install_redaction_filter()
