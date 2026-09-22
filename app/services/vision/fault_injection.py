"""Inyeccion de fallas para pruebas manuales, independiente del proveedor.

Reemplaza a los hooks que vivian dentro de `gemini_client.py`
(`FORCE_GEMINI_MODEL_ERROR` y el contador `_real_call_count`). El decorador
`FaultInjectingProvider` envuelve CUALQUIER proveedor de la cadena y, para
los modelos elegidos, lanza un `ProviderError` canonico sin llamar a la API
real (no gasta cuota); el resto pasa al proveedor real y se cuenta.

Formato de `FORCE_VISION_ERROR` (lista separada por comas):

    proveedor[:modelo]=falla      p. ej.  gemini:gemini-3.6-flash=503,openai=timeout

Fallas: 503 / 500 / 529 (ALTA_DEMANDA), 429 (LIMITE_ALCANZADO sin subtipo),
rate / daily / credit (LIMITE_ALCANZADO con subtipo), 404, 401, 400,
timeout, connection, invalid.

Alias deprecado `FORCE_GEMINI_MODEL_ERROR=503`: una falla sin selector aplica
al modelo del primer tier de Gemini, como hacia el hook original.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.vision.errors import (
    ConfigurationError,
    FailureReason,
    LimitKind,
    ProviderError,
)
from app.services.vision.models import ExtractionRequest, ExtractionResult
from app.services.vision.ports import VisionExtractorProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Fault:
    reason: FailureReason
    http_status: int | None = None
    limit_kind: LimitKind | None = None
    provider_code: str | None = None


_FAULTS: dict[str, _Fault] = {
    "503": _Fault(FailureReason.ALTA_DEMANDA, 503, provider_code="UNAVAILABLE"),
    "500": _Fault(FailureReason.ALTA_DEMANDA, 500, provider_code="INTERNAL"),
    "529": _Fault(FailureReason.ALTA_DEMANDA, 529, provider_code="overloaded_error"),
    "429": _Fault(FailureReason.LIMITE_ALCANZADO, 429, provider_code="RESOURCE_EXHAUSTED"),
    "rate": _Fault(
        FailureReason.LIMITE_ALCANZADO, 429, LimitKind.RATE_MINUTO, "rate_limit_exceeded"
    ),
    "daily": _Fault(
        FailureReason.LIMITE_ALCANZADO, 429, LimitKind.CUOTA_DIARIA, "quota_exceeded"
    ),
    "credit": _Fault(
        FailureReason.LIMITE_ALCANZADO, 429, LimitKind.CREDITO_O_GASTO, "credit_balance_exhausted"
    ),
    "404": _Fault(FailureReason.MODELO_NO_ENCONTRADO, 404, provider_code="NOT_FOUND"),
    "401": _Fault(FailureReason.AUTENTICACION, 401, provider_code="UNAUTHENTICATED"),
    "400": _Fault(FailureReason.SOLICITUD_INVALIDA, 400, provider_code="INVALID_ARGUMENT"),
    "timeout": _Fault(FailureReason.TIMEOUT),
    "connection": _Fault(FailureReason.CONEXION),
    "invalid": _Fault(FailureReason.RESPUESTA_INVALIDA),
}


@dataclass(frozen=True, slots=True)
class FaultRule:
    provider_id: str
    model: str | None  # None = todos los modelos del proveedor
    fault_token: str

    def matches(self, provider_id: str, model: str) -> bool:
        return self.provider_id == provider_id and self.model in (None, model)


def parse_fault_spec(spec: str, *, legacy_gemini_model: str | None = None) -> list[FaultRule]:
    """Convierte el texto de `FORCE_VISION_ERROR` en reglas.

    `legacy_gemini_model` es el modelo del primer tier de Gemini, al que se
    aplican las fallas sin selector (alias deprecado del hook original).
    """
    rules: list[FaultRule] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        selector, sep, token = item.partition("=")
        if not sep:
            token, selector = selector, ""  # falla sin selector (alias legacy)
        token = token.strip().lower()
        if token not in _FAULTS:
            raise ConfigurationError(
                f"Falla forzada invalida {token!r}; validas: {', '.join(sorted(_FAULTS))}"
            )
        if selector:
            provider_id, _, model = selector.strip().partition(":")
            rules.append(FaultRule(provider_id.lower(), model or None, token))
        else:
            if legacy_gemini_model is None:
                raise ConfigurationError(
                    f"La falla {token!r} no lleva selector proveedor[:modelo] y no hay un "
                    "tier de Gemini al que aplicar el alias deprecado"
                )
            rules.append(FaultRule("gemini", legacy_gemini_model, token))
    return rules


class FaultInjectingProvider:
    """Decorador de prueba: envuelve un proveedor y simula fallas canonicas."""

    def __init__(self, inner: VisionExtractorProvider, rules: Sequence[FaultRule]) -> None:
        self._inner = inner
        self._rules = tuple(rule for rule in rules if rule.provider_id == inner.provider_id)
        self.provider_id = inner.provider_id
        self.display_name = inner.display_name
        # Llamadas que SI llegaron al proveedor real (fuente de verdad del
        # consumo de cuota durante una prueba, sin depender del dashboard).
        self.real_call_count = 0

    def is_configured(self) -> bool:
        return self._inner.is_configured()

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def extract(self, request: ExtractionRequest, *, model: str) -> ExtractionResult:
        rule = next((r for r in self._rules if r.matches(self.provider_id, model)), None)
        if rule is not None:
            fault = _FAULTS[rule.fault_token]
            raise ProviderError(
                fault.reason,
                provider=self.provider_id,
                model=model,
                message=f"[PRUEBA MANUAL] falla {rule.fault_token!r} forzada para '{model}'",
                http_status=fault.http_status,
                provider_code=fault.provider_code,
                limit_kind=fault.limit_kind,
            )
        self.real_call_count += 1
        logger.info(
            "REAL_CALL #%d provider=%s model=%s", self.real_call_count, self.provider_id, model
        )
        return await self._inner.extract(request, model=model)
