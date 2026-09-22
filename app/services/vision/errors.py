"""Tipos canonicos de falla de la cadena de vision multi-proveedor.

Vocabulario unico para logs, metricas y decisiones de la cadena, sin importar
de que proveedor (Gemini, OpenAI, Anthropic, ...) venga el error. Cada
adaptador traduce las excepciones de su SDK a `ProviderError`; nada por
encima de la capa de adaptadores debe importar un SDK.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.vision.models import ProviderAttempt

_MAX_MESSAGE_CHARS = 300


class FailureReason(StrEnum):
    """Causa canonica de una falla, independiente del proveedor."""

    # El proveedor esta sobrecargado o no disponible (5xx: 500, 502, 503, 504,
    # 529). Transitorio y ajeno a nuestra cuota.
    ALTA_DEMANDA = "ALTA_DEMANDA"
    # Limite propio de cuota o credito (429, o 402 de credito). Accionable de
    # nuestro lado; el subtipo esta en `LimitKind`.
    LIMITE_ALCANZADO = "LIMITE_ALCANZADO"
    # 404: modelo retirado o renombrado; la config quedo desactualizada.
    MODELO_NO_ENCONTRADO = "MODELO_NO_ENCONTRADO"
    # 401/403: key invalida, revocada o sin permiso.
    AUTENTICACION = "AUTENTICACION"
    # 400/413/422: payload o schema rechazado.
    SOLICITUD_INVALIDA = "SOLICITUD_INVALIDA"
    # Respuesta 200 sin JSON utilizable: truncada, rechazada o malformada.
    RESPUESTA_INVALIDA = "RESPUESTA_INVALIDA"
    # No llego respuesta dentro del timeout del cliente.
    TIMEOUT = "TIMEOUT"
    # Error de red/transporte.
    CONEXION = "CONEXION"
    # Cualquier otra falla de API no clasificable.
    DESCONOCIDO = "DESCONOCIDO"


class LimitKind(StrEnum):
    """Subtipo de `FailureReason.LIMITE_ALCANZADO`: que accion corresponde."""

    RATE_MINUTO = "RATE_MINUTO"  # limite por minuto/rampa: esperar segundos
    CUOTA_DIARIA = "CUOTA_DIARIA"  # cupo diario agotado: esperar el reinicio
    CREDITO_O_GASTO = "CREDITO_O_GASTO"  # credito/tope de gasto: pagar o subir limite
    DESCONOCIDO = "DESCONOCIDO"  # no distinguible con el cuerpo disponible


class ConfigurationError(RuntimeError):
    """Configuracion invalida de la cadena; se detecta al arrancar (fail-fast)."""


class ProviderError(Exception):
    """Falla de un proveedor ya traducida al vocabulario canonico.

    Es la unica excepcion que un adaptador deja salir por problemas de API,
    transporte o respuesta. Nunca lleva contenido de las capturas ni
    secretos: `message` viene truncado del error del SDK.
    """

    def __init__(
        self,
        reason: FailureReason,
        *,
        provider: str,
        model: str,
        message: str = "",
        http_status: int | None = None,
        provider_code: str | None = None,
        limit_kind: LimitKind | None = None,
        retry_after_seconds: float | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        if reason is FailureReason.LIMITE_ALCANZADO and limit_kind is None:
            limit_kind = LimitKind.DESCONOCIDO
        self.reason = reason
        self.provider = provider
        self.model = model
        self.message = message[:_MAX_MESSAGE_CHARS]
        self.http_status = http_status
        self.provider_code = provider_code
        self.limit_kind = limit_kind if reason is FailureReason.LIMITE_ALCANZADO else None
        self.retry_after_seconds = retry_after_seconds
        self.provider_request_id = provider_request_id
        detail = f": {self.message}" if self.message else ""
        super().__init__(f"{reason} en {provider} ({model}){detail}")


# Prioridad para elegir la causa dominante de una cadena agotada (la que
# decide el mensaje al usuario): primero lo que exige accion del operador.
_REASON_PRIORITY: tuple[FailureReason, ...] = (
    FailureReason.LIMITE_ALCANZADO,
    FailureReason.ALTA_DEMANDA,
    FailureReason.TIMEOUT,
    FailureReason.CONEXION,
    FailureReason.AUTENTICACION,
    FailureReason.MODELO_NO_ENCONTRADO,
    FailureReason.RESPUESTA_INVALIDA,
    FailureReason.SOLICITUD_INVALIDA,
    FailureReason.DESCONOCIDO,
)
_LIMIT_KIND_PRIORITY: tuple[LimitKind, ...] = (
    LimitKind.CREDITO_O_GASTO,
    LimitKind.CUOTA_DIARIA,
    LimitKind.RATE_MINUTO,
    LimitKind.DESCONOCIDO,
)


def dominant_failure(
    failures: Sequence[ProviderError],
) -> tuple[FailureReason, LimitKind | None]:
    """Causa dominante de un conjunto de fallas y, si es un limite, su subtipo.

    Orden: `LIMITE_ALCANZADO` (con `CREDITO_O_GASTO` por encima de
    `CUOTA_DIARIA`, `RATE_MINUTO` y `DESCONOCIDO`), luego `ALTA_DEMANDA` y el
    resto segun `_REASON_PRIORITY`.
    """
    present = {failure.reason for failure in failures}
    reason = next((r for r in _REASON_PRIORITY if r in present), FailureReason.DESCONOCIDO)
    if reason is not FailureReason.LIMITE_ALCANZADO:
        return reason, None
    kinds = {
        failure.limit_kind or LimitKind.DESCONOCIDO
        for failure in failures
        if failure.reason is FailureReason.LIMITE_ALCANZADO
    }
    kind = next((k for k in _LIMIT_KIND_PRIORITY if k in kinds), LimitKind.DESCONOCIDO)
    return reason, kind


class AllProvidersFailedError(Exception):
    """La cadena termino sin resultado: se agotaron los tiers o una falla no
    continuable la detuvo (`aborted_by`).

    La capa de UI decide el mensaje al usuario con `dominant_reason` y
    `dominant_limit_kind`, sin importar ningun SDK.
    """

    def __init__(
        self,
        attempts: Sequence[ProviderAttempt],
        *,
        dominant_reason: FailureReason,
        dominant_limit_kind: LimitKind | None = None,
        aborted_by: ProviderError | None = None,
    ) -> None:
        self.attempts = tuple(attempts)
        self.dominant_reason = dominant_reason
        self.dominant_limit_kind = dominant_limit_kind
        self.aborted_by = aborted_by
        kind = f"/{dominant_limit_kind}" if dominant_limit_kind else ""
        stop = " (cadena detenida por falla no continuable)" if aborted_by else ""
        super().__init__(f"Cadena de vision sin resultado: {dominant_reason}{kind}{stop}")
