"""Adaptador de OpenAI (API de Responses) para extraccion multimodal de rutas.

Traduce capturas de la aplicacion de origen a entidades de dominio usando Structured
Outputs (`text.format` con `json_schema` estricto). Junto con los demas
modulos de `app.adapters.vision`, es la unica capa que conoce el SDK de
OpenAI.

Diferencias con el adaptador de Gemini (por eso cada proveedor es dueno de su
schema y de su mapeo de errores):

* El schema raiz DEBE ser un objeto: la lista de tarjetas va envuelta en
  `{"entries": [...]}`. Gemini acepta una lista en la raiz.
* Modo estricto: todos los campos son obligatorios (los opcionales son
  `nullable`) y sin restricciones como `minimum`; el `screenshot_index >= 1` se
  valida localmente al convertir al DTO neutral.
* El SDK va con `max_retries=0`: el reintento esta centralizado en la cadena
  (R2), la unica que sabe no reintentar un 429 por credito agotado.
* `store=False`: la API de Responses guarda por defecto una copia de la
  solicitud; las capturas traen datos de clientes.
* Los errores vienen tipados por el SDK (`RateLimitError`,
  `InternalServerError`, ...) y el detalle machine-readable esta en `exc.code`
  y `exc.type`; el mapeo usa (codigo HTTP, code/type), no solo la clase.
"""

from __future__ import annotations

import base64
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import OpenAISettings
from app.domain.models import DeliveryStatus
from app.services.vision.base import BaseVisionProvider, InvalidResponseError
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest, UsageInfo
from app.services.vision.prompt import SYSTEM_INSTRUCTION
from app.services.vision.schema import DeliveryEntryDTO

SCHEMA_NAME = "route_entries"

# `error.code` de un 429 que NO se arregla esperando: credito o tope de gasto.
# Los cuatro primeros estan documentados por OpenAI; `insufficient_quota` es el
# codigo clasico de cuota de facturacion agotada (no listado en la pagina leida).
_CREDIT_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "insufficient_quota",
    }
)
# Heuristica por texto para un limite DIARIO (RPD/TPD): OpenAI no expone un
# codigo propio. No esta verificada con cuerpos reales de este proyecto.
_DAILY_MARKERS = ("per day", "(rpd)", "(tpd)")


class _OpenAIEntry(BaseModel):
    """Lo que se le pide a OpenAI por tarjeta: como `DeliveryEntryDTO` pero sin
    restricciones (`ge=1`) y con todos los campos requeridos (modo estricto)."""

    screenshot_index: int
    order: int | None
    package_id: str | None
    customer_name: str | None
    address: str
    locality: str | None
    delivery_type: str | None
    eta: str | None
    time_window: str | None
    status: DeliveryStatus


class _OpenAIEnvelope(BaseModel):
    entries: list[_OpenAIEntry]


def build_strict_schema() -> dict[str, Any]:
    """Schema JSON estricto del sobre `{"entries": [...]}`, generado con el
    helper publico del SDK (aplica `additionalProperties: false` y `required`
    en todos los objetos)."""
    tool: Any = openai.pydantic_function_tool(_OpenAIEnvelope, name=SCHEMA_NAME)
    schema: dict[str, Any] = tool["function"]["parameters"]
    return schema


def classify_rate_limit(code: str | None, error_type: str | None, message: str) -> LimitKind:
    """Subtipo de un 429 a partir de `error.code`, `error.type` y el texto."""
    if code in _CREDIT_CODES or error_type == "insufficient_quota":
        return LimitKind.CREDITO_O_GASTO
    if code == "slow_down":
        return LimitKind.RATE_MINUTO
    lowered = message.lower()
    if any(marker in lowered for marker in _DAILY_MARKERS):
        return LimitKind.CUOTA_DIARIA
    if code == "rate_limit_exceeded" or error_type == "rate_limit_error" or "per min" in lowered:
        return LimitKind.RATE_MINUTO
    return LimitKind.DESCONOCIDO


def _retry_after_seconds(exc: openai.APIStatusError) -> float | None:
    """`Retry-After` (segundos) o `retry-after-ms`, si el proveedor los entrega."""
    headers = exc.response.headers
    for name, scale in (("retry-after", 1.0), ("retry-after-ms", 0.001)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            return float(raw) * scale
        except ValueError:
            continue
    return None


def _error_message(exc: openai.APIError) -> str:
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return str(body["message"])
    return exc.message


class OpenAIProvider(BaseVisionProvider):
    provider_id = "openai"
    display_name = "OpenAI"

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        reasoning_effort: str | None = None,
        image_detail: str = "auto",
        store: bool = False,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._reasoning_effort = reasoning_effort
        self._image_detail = image_detail
        self._store = store
        self._schema = build_strict_schema()
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)

    @classmethod
    def from_settings(cls, settings: OpenAISettings) -> OpenAIProvider | None:
        """None si no hay key explicita: el tier queda inactivo."""
        if settings.api_key is None or not settings.has_api_key():
            return None
        return cls(
            api_key=settings.api_key.get_secret_value(),
            timeout_seconds=settings.timeout_seconds,
            reasoning_effort=settings.reasoning_effort,
            image_detail=settings.image_detail,
            store=settings.store_responses,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def _send(self, request: ExtractionRequest, model: str) -> Any:
        content: list[dict[str, Any]] = []
        for index, image in enumerate(request.images, start=1):
            encoded = base64.b64encode(image).decode("ascii")
            content.append({"type": "input_text", "text": f"Imagen {index}:"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{request.mime_type};base64,{encoded}",
                    "detail": self._image_detail,
                }
            )
        content.append(
            {
                "type": "input_text",
                "text": "Extrae las tarjetas de todas las imagenes segun las reglas indicadas.",
            }
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": SYSTEM_INSTRUCTION,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": SCHEMA_NAME,
                    "strict": True,
                    "schema": self._schema,
                }
            },
            "store": self._store,
        }
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        return await self._client.responses.create(**kwargs)

    def _usage(self, response: Any) -> UsageInfo:
        usage = getattr(response, "usage", None)
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        finish = getattr(incomplete, "reason", None) or status
        details = getattr(usage, "output_tokens_details", None)
        return UsageInfo(
            prompt_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            finish_reason=finish,
            finish_ok=status in (None, "completed"),
        )

    def _entries(
        self, response: Any, *, request: ExtractionRequest, model: str
    ) -> list[DeliveryEntryDTO]:
        # Los mensajes NUNCA incluyen contenido del JSON: van a logs y al JSONL.
        status = getattr(response, "status", None)
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            raise InvalidResponseError(f"respuesta incompleta ({reason})")
        if status == "failed":
            code = getattr(getattr(response, "error", None), "code", None)
            raise InvalidResponseError(f"la respuesta fallo en el proveedor (code={code})")
        if _has_refusal(response):
            raise InvalidResponseError("el modelo rechazo la solicitud (refusal)")
        text = getattr(response, "output_text", None)
        if not text or not text.strip():
            raise InvalidResponseError("respuesta sin texto")
        try:
            envelope = _OpenAIEnvelope.model_validate_json(text)
        except ValidationError as exc:
            raise InvalidResponseError(
                f"JSON no valido contra el schema ({exc.error_count()} errores)"
            ) from None
        try:
            return [DeliveryEntryDTO.model_validate(e.model_dump()) for e in envelope.entries]
        except ValidationError as exc:
            raise InvalidResponseError(
                f"entradas invalidas ({exc.error_count()} errores)"
            ) from None

    def _request_id(self, response: Any) -> str | None:
        value = getattr(response, "_request_id", None) or getattr(response, "id", None)
        return value if isinstance(value, str) else None

    def _translate_error(self, exc: Exception, model: str) -> ProviderError | None:
        if isinstance(exc, openai.APITimeoutError):
            return self._error(FailureReason.TIMEOUT, model, message=_error_message(exc))
        if isinstance(exc, openai.APIConnectionError):
            return self._error(FailureReason.CONEXION, model, message=_error_message(exc))
        if isinstance(exc, openai.APIStatusError):
            return self._translate_status_error(exc, model)
        if isinstance(exc, openai.APIError):
            return self._error(
                FailureReason.DESCONOCIDO,
                model,
                message=_error_message(exc),
                provider_code=exc.code,
            )
        return None  # no es una falla de API: se propaga tal cual

    def _translate_status_error(self, exc: openai.APIStatusError, model: str) -> ProviderError:
        status = exc.status_code
        message = _error_message(exc)

        def build(
            reason: FailureReason,
            limit_kind: LimitKind | None = None,
            retry_after: float | None = None,
        ) -> ProviderError:
            return self._error(
                reason,
                model,
                message=message,
                http_status=status,
                provider_code=exc.code,
                limit_kind=limit_kind,
                retry_after_seconds=retry_after,
                provider_request_id=exc.request_id,
            )

        if exc.code == "model_not_found":
            return build(FailureReason.MODELO_NO_ENCONTRADO)
        if status >= 500:  # 503 con code `server_is_overloaded` = sobrecarga
            return build(FailureReason.ALTA_DEMANDA)
        if status == 429:
            return build(
                FailureReason.LIMITE_ALCANZADO,
                classify_rate_limit(exc.code, exc.type, message),
                _retry_after_seconds(exc),
            )
        if status == 402:
            return build(FailureReason.LIMITE_ALCANZADO, LimitKind.CREDITO_O_GASTO)
        if status == 404:
            return build(FailureReason.MODELO_NO_ENCONTRADO)
        if status in (401, 403):
            return build(FailureReason.AUTENTICACION)
        if status == 408:
            return build(FailureReason.TIMEOUT)
        if status in (400, 413, 422):
            return build(FailureReason.SOLICITUD_INVALIDA)
        return build(FailureReason.DESCONOCIDO)


def _has_refusal(response: Any) -> bool:
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return True
    return False
