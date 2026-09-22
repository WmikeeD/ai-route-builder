"""Modelos de datos de la cadena de vision: pedido, resultado, tiers e intentos.

Todo es neutral respecto de proveedores. `ProviderAttempt` es el registro
canonico que alimenta el log estructurado y el archivo JSONL de intentos:
guarda conteos, codigos y tiempos, NUNCA contenido de las capturas
(direcciones, nombres de clientes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.domain.models import RawDeliveryEntry
from app.services.vision.errors import FailureReason, LimitKind


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """Lote de capturas a leer. `screenshot_ids` (si viene) tiene el mismo
    largo y orden que `images` y puebla `source_screenshot_id`."""

    images: list[bytes]
    mime_type: str = "image/jpeg"
    screenshot_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class UsageInfo:
    """Consumo y motivo de fin de una respuesta, en forma comun a todos los
    proveedores (reemplaza al log especifico `GEMINI_RESPONSE`)."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    # False si el proveedor corto la generacion (p. ej. MAX_TOKENS): sube el
    # nivel del log a WARNING porque explica un JSON truncado.
    finish_ok: bool = True


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    entries: list[RawDeliveryEntry]
    usage: UsageInfo = field(default_factory=UsageInfo)
    provider_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class TierSpec:
    """Un eslabon de la cadena: proveedor + modelo (`provider_id:model`)."""

    provider_id: str
    model: str


class AttemptOutcome(StrEnum):
    EXITO = "EXITO"
    FALLO = "FALLO"
    SALTADO_INACTIVO = "SALTADO_INACTIVO"
    SALTADO_CIRCUITO = "SALTADO_CIRCUITO"
    # Tier omitido por un bloqueo de cuota diaria o de credito (quota_guard.py):
    # distinto del cortacircuitos, que solo cubre fallas transitorias.
    SALTADO_CUOTA = "SALTADO_CUOTA"


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    """Registro canonico de un intento (o de un tier omitido)."""

    ts: datetime
    chain_request_id: str
    tier: int
    tiers_total: int
    provider: str
    display_name: str
    model: str
    outcome: AttemptOutcome
    attempt: int = 1
    max_attempts: int = 1
    reason: FailureReason | None = None
    limit_kind: LimitKind | None = None
    http_status: int | None = None
    provider_code: str | None = None
    latency_ms: int | None = None
    images: int = 0
    payload_bytes: int = 0
    usage: UsageInfo | None = None
    provider_request_id: str | None = None
    detail: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Version serializable a JSON (una linea del JSONL de intentos)."""
        usage = self.usage
        return {
            "ts": self.ts.isoformat(),
            "chain_request_id": self.chain_request_id,
            "tier": self.tier,
            "tiers_total": self.tiers_total,
            "provider": self.provider,
            "model": self.model,
            "outcome": str(self.outcome),
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "reason": str(self.reason) if self.reason else None,
            "limit_kind": str(self.limit_kind) if self.limit_kind else None,
            "http_status": self.http_status,
            "provider_code": self.provider_code,
            "latency_ms": self.latency_ms,
            "images": self.images,
            "payload_bytes": self.payload_bytes,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "reasoning_tokens": usage.reasoning_tokens if usage else None,
            "finish_reason": usage.finish_reason if usage else None,
            "provider_request_id": self.provider_request_id,
            "detail": self.detail,
        }
