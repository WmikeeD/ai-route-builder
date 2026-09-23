"""Adaptador de Google Gemini para extraccion multimodal de rutas de entrega.

Traduce capturas de la aplicacion de origen (pantalla "Visitas") a entidades de
dominio usando Structured Outputs (JSON) de Gemini. Junto con los demas
modulos de `app.adapters.vision`, es la unica capa que conoce la libreria
`google-genai`: el dominio y los servicios no la importan.

Decisiones de esta capa (ver docs/ARQUITECTURA_FALLBACK_MULTIPROVEEDOR.md):

* El SDK va en UN solo intento (`attempts=1`): el reintento esta
  centralizado en la cadena (R2), que es la unica que sabe no reintentar un
  429 por cuota. Antes el SDK reintentaba 429 y 5xx por igual.
* Timeout de cliente explicito (configurable en `GeminiSettings`). Antes el
  SDK corria con `timeout=None` y una llamada llego a colgarse ~4.5 min.
* Los errores de transporte de `httpx` (timeout, red) no vienen envueltos
  por el SDK: se mapean aqui a `TIMEOUT` / `CONEXION`.
* No se envian `temperature`, `top_p` ni `top_k`: estan deprecados para los
  modelos Gemini 3.x de la cadena (changelog del 21-jul-2026) y Google
  recomienda dejar `temperature` en su default (1.0) en toda la familia 3.
  La consistencia de la extraccion la da el schema estructurado.
"""

from __future__ import annotations

import re
from typing import Any, cast

import httpx
from google import genai
from google.genai import errors, types

from app.config import GeminiSettings
from app.services.vision.base import BaseVisionProvider, InvalidResponseError
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest, UsageInfo
from app.services.vision.prompt import SYSTEM_INSTRUCTION
from app.services.vision.schema import DeliveryEntryDTO

# Heuristica para distinguir el tipo de limite de un 429 a partir del cuerpo.
# La documentacion de Gemini nombra `rate_limit_exceeded` (por minuto) y
# `quota_exceeded` (diario), y los cuerpos clasicos traen ids como
# `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. NO esta verificada
# contra cuerpos reales de este proyecto: ante la duda queda DESCONOCIDO y el
# cuerpo se conserva (truncado) en el registro para ajustarla con datos.
_DAILY_MARKERS = ("perday", "per_day", "per day", "daily", "quota_exceeded")
_MINUTE_MARKERS = (
    "perminute",
    "per_minute",
    "per minute",
    "persecond",
    "per second",
    "rate_limit_exceeded",
)
_RETRY_DELAY_RE = re.compile(r"retry(?:delay)?\W{0,6}(?:in\s+)?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def classify_rate_limit(text: str) -> LimitKind:
    lowered = text.lower()
    if any(marker in lowered for marker in _DAILY_MARKERS):
        return LimitKind.CUOTA_DIARIA
    if any(marker in lowered for marker in _MINUTE_MARKERS):
        return LimitKind.RATE_MINUTO
    return LimitKind.DESCONOCIDO


def parse_retry_after(text: str) -> float | None:
    match = _RETRY_DELAY_RE.search(text)
    return float(match.group(1)) if match else None


class GeminiProvider(BaseVisionProvider):
    provider_id = "gemini"
    display_name = "Gemini"

    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),  # el SDK espera milisegundos
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    @classmethod
    def from_settings(cls, settings: GeminiSettings) -> GeminiProvider | None:
        """None si no hay key explicita: el tier queda inactivo."""
        if settings.api_key is None or not settings.has_api_key():
            return None
        return cls(
            api_key=settings.api_key.get_secret_value(),
            timeout_seconds=settings.timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def _send(self, request: ExtractionRequest, model: str) -> Any:
        parts = [
            types.Part.from_bytes(data=image, mime_type=request.mime_type)
            for image in request.images
        ]
        return await self._client.aio.models.generate_content(
            model=model,
            contents=parts,  # type: ignore[arg-type]  # list[Part] es valido en runtime; el stub de contents no lo modela bien
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=list[DeliveryEntryDTO],
            ),
        )

    def _usage(self, response: Any) -> UsageInfo:
        candidates = getattr(response, "candidates", None)
        finish_reason = candidates[0].finish_reason if candidates else None
        usage = getattr(response, "usage_metadata", None)
        return UsageInfo(
            prompt_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            reasoning_tokens=usage.thoughts_token_count if usage else None,
            total_tokens=usage.total_token_count if usage else None,
            finish_reason=finish_reason.name if finish_reason else None,
            finish_ok=finish_reason in (None, types.FinishReason.STOP),
        )

    def _entries(
        self, response: Any, *, request: ExtractionRequest, model: str
    ) -> list[DeliveryEntryDTO]:
        # El SDK deja `response.parsed` en None (sin excepcion) cuando el JSON
        # llega cortado o invalido, p. ej. por MAX_TOKENS.
        if response.parsed is None:
            raise InvalidResponseError("Gemini no devolvio un JSON valido para las capturas")
        # `response.parsed` esta tipado como BaseModel | dict | Enum en el stub
        # de la SDK, pero con response_schema=list[Model] la SDK arma un
        # wrapper interno y aqui siempre entrega una list[DeliveryEntryDTO].
        return list(cast(list[DeliveryEntryDTO], response.parsed))

    def _request_id(self, response: Any) -> str | None:
        value = getattr(response, "response_id", None)
        return value if isinstance(value, str) else None

    def _translate_error(self, exc: Exception, model: str) -> ProviderError | None:
        if isinstance(exc, errors.APIError):
            return self._translate_api_error(exc, model)
        if isinstance(exc, httpx.TimeoutException):
            return self._error(
                FailureReason.TIMEOUT, model, message=f"{type(exc).__name__}: {exc}"
            )
        if isinstance(exc, httpx.TransportError):
            return self._error(
                FailureReason.CONEXION, model, message=f"{type(exc).__name__}: {exc}"
            )
        return None  # no es una falla de API: se propaga tal cual

    def _translate_api_error(self, exc: errors.APIError, model: str) -> ProviderError:
        code = exc.code
        message = str(exc.message or exc)

        def build(
            reason: FailureReason,
            limit_kind: LimitKind | None = None,
            retry_after: float | None = None,
        ) -> ProviderError:
            return self._error(
                reason,
                model,
                message=message,
                http_status=code,
                provider_code=exc.status,
                limit_kind=limit_kind,
                retry_after_seconds=retry_after,
            )

        if isinstance(exc, errors.ServerError):
            return build(FailureReason.ALTA_DEMANDA)
        body = f"{exc.message or ''} {exc.details or ''}"
        if code == 429:
            return build(
                FailureReason.LIMITE_ALCANZADO,
                classify_rate_limit(body),
                parse_retry_after(body),
            )
        if code == 402:
            return build(FailureReason.LIMITE_ALCANZADO, LimitKind.CREDITO_O_GASTO)
        if code == 404:
            return build(FailureReason.MODELO_NO_ENCONTRADO)
        if code in (401, 403):
            return build(FailureReason.AUTENTICACION)
        if code == 408:
            return build(FailureReason.TIMEOUT)
        if isinstance(exc, errors.ClientError):
            return build(FailureReason.SOLICITUD_INVALIDA)
        return build(FailureReason.DESCONOCIDO)
