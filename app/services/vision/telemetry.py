"""Registro de cada intento de la cadena (T1 + T2).

* T1: dos lineas por intento en el log estandar (stdout): una legible con el
  vocabulario canonico ("ALTA_DEMANDA en Gemini (gemini-3.8-flash) ...") y una
  JSON consultable con `jq`.
* T2: el mismo JSON, una linea por intento, en un archivo JSONL rotativo.

Regla de privacidad: solo conteos, codigos y tiempos; nunca contenido de las
capturas ni de las respuestas. Un fallo del archivo nunca rompe la
solicitud: se avisa una vez y se sigue con T1.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol

from app.services.vision.errors import FailureReason, LimitKind
from app.services.vision.models import AttemptOutcome, ProviderAttempt

human_logger = logging.getLogger("vision.attempts")
json_logger = logging.getLogger("vision.attempts.json")
_own_logger = logging.getLogger(__name__)

_MAX_JSONL_BYTES = 5 * 1024 * 1024
_JSONL_BACKUPS = 3

# Nivel del log por causa. Lo que exige accion nuestra (config rota, limite
# de credito) sube a ERROR; lo transitorio del proveedor queda en WARNING.
_LEVEL_BY_REASON: dict[FailureReason, int] = {
    FailureReason.ALTA_DEMANDA: logging.WARNING,
    FailureReason.LIMITE_ALCANZADO: logging.WARNING,
    FailureReason.MODELO_NO_ENCONTRADO: logging.ERROR,
    FailureReason.AUTENTICACION: logging.ERROR,
    FailureReason.SOLICITUD_INVALIDA: logging.ERROR,
    FailureReason.RESPUESTA_INVALIDA: logging.WARNING,
    FailureReason.TIMEOUT: logging.WARNING,
    FailureReason.CONEXION: logging.WARNING,
    FailureReason.DESCONOCIDO: logging.ERROR,
}


class AttemptSink(Protocol):
    """Lo que la cadena necesita para registrar un intento."""

    def record(self, attempt: ProviderAttempt) -> None: ...


def format_attempt(attempt: ProviderAttempt) -> str:
    """Linea legible con el vocabulario canonico: causa + proveedor + modelo."""
    where = f"{attempt.display_name} ({attempt.model}) [tier {attempt.tier}/{attempt.tiers_total}]"
    if attempt.outcome is AttemptOutcome.SALTADO_INACTIVO:
        return f"tier {attempt.tier} ({attempt.display_name}) no configurado, se omite"
    if attempt.outcome is AttemptOutcome.SALTADO_CIRCUITO:
        return (
            f"tier {attempt.tier} ({attempt.display_name}, {attempt.model}) omitido: "
            f"cortacircuitos abierto ({attempt.detail})"
        )
    if attempt.outcome is AttemptOutcome.SALTADO_CUOTA:
        kind = f" kind={attempt.limit_kind}" if attempt.limit_kind else ""
        return (
            f"tier {attempt.tier} ({attempt.display_name}, {attempt.model}) omitido por "
            f"LIMITE_ALCANZADO{kind}: {attempt.detail}"
        )

    parts: list[str] = []
    if attempt.outcome is AttemptOutcome.EXITO:
        head = f"EXITO en {where}"
        usage = attempt.usage
        if usage is not None:
            parts.append(f"finish_reason={usage.finish_reason}")
            parts.append(f"prompt_tokens={usage.prompt_tokens}")
            parts.append(f"output_tokens={usage.output_tokens}")
    else:
        reason = attempt.reason or FailureReason.DESCONOCIDO
        head = f"{reason} en {where}"
        if attempt.limit_kind is not None:
            parts.append(f"kind={attempt.limit_kind}")
        if attempt.http_status is not None:
            parts.append(f"http={attempt.http_status}")
        if attempt.provider_code:
            parts.append(f"code={attempt.provider_code}")
    if attempt.latency_ms is not None:
        parts.append(f"latency={attempt.latency_ms / 1000:.1f}s")
    if attempt.max_attempts > 1:
        parts.append(f"attempt={attempt.attempt}/{attempt.max_attempts}")
    if attempt.detail and attempt.outcome is AttemptOutcome.FALLO:
        parts.append(f"detail={attempt.detail}")
    return " ".join([head, *parts])


def level_for(attempt: ProviderAttempt) -> int:
    if attempt.outcome is not AttemptOutcome.FALLO:
        return logging.INFO
    reason = attempt.reason or FailureReason.DESCONOCIDO
    if reason is FailureReason.LIMITE_ALCANZADO and attempt.limit_kind is LimitKind.CREDITO_O_GASTO:
        return logging.ERROR
    return _LEVEL_BY_REASON[reason]


class AttemptRecorder:
    """Emite cada `ProviderAttempt` por T1 (log) y T2 (JSONL, opcional)."""

    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._jsonl_path = jsonl_path
        self._file_logger: logging.Logger | None = None
        self._file_handler: RotatingFileHandler | None = None
        self._file_broken = False
        if jsonl_path is not None:
            self._open_file(jsonl_path)

    def _open_file(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                path, maxBytes=_MAX_JSONL_BYTES, backupCount=_JSONL_BACKUPS, encoding="utf-8"
            )
        except OSError:
            _own_logger.exception("No se pudo abrir el archivo de intentos %s; solo T1.", path)
            self._file_broken = True
            return
        handler.setFormatter(logging.Formatter("%(message)s"))
        # Logger propio por archivo, sin propagar: el JSONL no debe mezclarse
        # con el log de la aplicacion.
        file_logger = logging.getLogger(f"vision.attempts.file.{id(self)}")
        file_logger.setLevel(logging.INFO)
        file_logger.propagate = False
        file_logger.addHandler(handler)
        self._file_logger = file_logger
        self._file_handler = handler

    def record(self, attempt: ProviderAttempt) -> None:
        human_logger.log(level_for(attempt), format_attempt(attempt))
        try:
            line = json.dumps(attempt.to_record(), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            _own_logger.exception("No se pudo serializar el intento a JSON")
            return
        json_logger.info(line)
        if self._file_logger is not None and not self._file_broken:
            try:
                self._file_logger.info(line)
            except Exception:
                self._file_broken = True
                _own_logger.exception("Fallo el archivo de intentos; se continua solo con T1.")

    def close(self) -> None:
        if self._file_logger is not None and self._file_handler is not None:
            self._file_logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_logger = None
            self._file_handler = None
