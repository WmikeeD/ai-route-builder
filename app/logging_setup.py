"""Configuracion de logging de la app, con defensa contra fugas de secretos.

La API de Telegram lleva el token del bot DENTRO de la URL
(`https://api.telegram.org/bot<id>:<token>/getUpdates`), y `httpx` loguea la
URL completa de cada request en INFO. Por eso, en capas:

1. Las librerias HTTP quedan SIEMPRE en WARNING, sin importar LOG_LEVEL ni
   ninguna variable de entorno: `httpx`/`httpcore` (Telegram, Gemini) y
   `httpx2`/`httpcore2` (paquetes aparte que usan los SDK de OpenAI y
   Anthropic). Tambien los loggers de esos dos SDK: en DEBUG vuelcan el cuerpo
   completo del request (prompt + capturas en base64 = datos de clientes).
2. Un filtro redacta de toda linea que pase por los handlers configurados,
   incluidos tracebacks: tokens de bot de Telegram (tambien la variante
   URL-encodeada `%3A` de las descargas de archivos) y, como defensa en
   profundidad, API keys de OpenAI (`sk-...`) y Anthropic (`sk-ant-...`),
   que hoy viajan en headers y no aparecen en ningun log.
"""

from __future__ import annotations

import logging
import re

# Loggers de librerias HTTP que loguean URLs completas en INFO/DEBUG.
NOISY_HTTP_LOGGERS = ("httpx", "httpcore", "httpx2", "httpcore2")
# Loggers de SDK de proveedores que en DEBUG loguean el cuerpo de cada request.
NOISY_SDK_LOGGERS = ("openai", "anthropic")

# `<id>:<secreto>` o `<id>%3A<secreto>` (el `%3a` en minuscula tambien es valido).
TELEGRAM_TOKEN_PATTERN = re.compile(r"\d{8,10}(?::|%3[Aa])[A-Za-z0-9_-]{35}")
REDACTED = "[TOKEN REDACTADO]"

# `sk-...` (OpenAI, incluidos `sk-proj-`, `sk-svcacct-`, `sk-admin-`) y
# `sk-ant-...` (Anthropic). El minimo de 20 caracteres evita falsos positivos
# con texto comun que empiece con "sk-".
API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")
REDACTED_API_KEY = "[API KEY REDACTADA]"


def redact_secrets(text: str) -> str:
    text = TELEGRAM_TOKEN_PATTERN.sub(REDACTED, text)
    return API_KEY_PATTERN.sub(REDACTED_API_KEY, text)


class SecretRedactingFilter(logging.Filter):
    """Redacta secretos (tokens de Telegram, API keys) del mensaje, del
    traceback y del stack de un record. Nunca descarta records: solo los
    reescribe."""

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

    for name in (*NOISY_HTTP_LOGGERS, *NOISY_SDK_LOGGERS):
        logging.getLogger(name).setLevel(logging.WARNING)

    install_redaction_filter()
