"""Adaptador de Anthropic (API de Messages) para extraccion multimodal de rutas.

Traduce capturas de la aplicacion de origen a entidades de dominio usando salida
estructurada (`output_config.format` con `json_schema`). Junto con los demas
modulos de `app.adapters.vision`, es la unica capa que conoce el SDK de
Anthropic.

Diferencias con los otros adaptadores (por eso cada proveedor es dueno de su
schema y de su mapeo de errores):

* El schema raiz DEBE ser un objeto: la lista de tarjetas va envuelta en
  `{"entries": [...]}`. Sin `minimum`; el `screenshot_index >= 1` se valida
  localmente al convertir al DTO neutral.
* Modelos recientes de Anthropic rechazan `temperature`: no se envia.
* No existe un equivalente al `store=False` de OpenAI: la API no tiene un
  parametro por solicitud para desactivar la retencion. La retencion
  (eliminacion en 30 dias por defecto, cero retencion por acuerdo) se
  gestiona a nivel de organizacion, no desde este adaptador.
* El SDK va con `max_retries=0`: el reintento esta centralizado en la cadena.
* El SDK de Anthropic resuelve credenciales y destino del ENTORNO
  (ANTHROPIC_AUTH_TOKEN, perfiles, ANTHROPIC_BASE_URL, ANTHROPIC_CUSTOM_HEADERS
  ...). Para que "inactivo" signifique inactivo y para que ninguna variable
  ambiental redirija el trafico o inyecte cabeceras, el cliente solo se
  construye con la key explicita del bloque de config, con `base_url`
  explicito y con esas variables ocultas durante la construccion.
* Los errores 5xx NO comparten clase: 529 es `OverloadedError` (no hereda de
  `InternalServerError`). Por eso el mapeo usa el codigo HTTP y el cuerpo,
  nunca la clase de la excepcion.
* `exc.body` es el sobre completo `{"type": "error", "error": {...}}` (en
  OpenAI es el objeto interno); el tipo y el mensaje estan en `body["error"]`.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from app.config import AnthropicSettings
from app.domain.models import DeliveryStatus
from app.services.vision.base import BaseVisionProvider, InvalidResponseError
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest, UsageInfo
from app.services.vision.prompt import SYSTEM_INSTRUCTION
from app.services.vision.schema import DeliveryEntryDTO

API_BASE_URL = "https://api.anthropic.com"

# Codigo documentado (`error.details.error_code`) del 429 por tope de gasto
# mensual del tier: sin `retry-after`, no se arregla esperando.
SPEND_CAP_ERROR_CODE = "enforced_spend_limit_reached"
# Un limite de gasto fijado por el usuario responde 400 `invalid_request_error`
# (NO 402 ni 429) con un mensaje que empieza asi (o "...your specified
# workspace API usage limits"). Documentado; no verificado con una respuesta real.
_USER_SPEND_LIMIT_PREFIX = "you have reached your specified"

# `stop_reason` con los que la generacion termino de forma normal.
_OK_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})


class _AnthropicEntry(BaseModel):
    """Lo que se le pide a Anthropic por tarjeta: como `DeliveryEntryDTO` pero
    sin restricciones (`ge=1`) y con todos los campos requeridos."""

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


class _AnthropicEnvelope(BaseModel):
    entries: list[_AnthropicEntry]


def build_strict_schema() -> dict[str, Any]:
    """Schema JSON del sobre `{"entries": [...]}`, transformado con el helper
    publico del SDK (`additionalProperties: false`, `required` en todos los
    objetos, restricciones no soportadas movidas a la descripcion)."""
    return anthropic.transform_schema(_AnthropicEnvelope)


@contextmanager
def _without_ambient_anthropic_env() -> Iterator[None]:
    """Oculta las variables `ANTHROPIC_*` mientras se construye el cliente y las
    restaura al salir. El constructor del SDK las lee (token, base_url,
    cabeceras, perfil) aunque se le pase una key explicita. Solo se usa en el
    arranque, de forma sincrona y sin `await` dentro."""
    hidden = {name: value for name, value in os.environ.items() if name.startswith("ANTHROPIC_")}
    for name in hidden:
        del os.environ[name]
    try:
        yield
    finally:
        os.environ.update(hidden)


def _error_body(exc: anthropic.APIError) -> dict[str, Any]:
    """El objeto `error` del sobre (`{"type", "message", "details"}`), o {}."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error
    return {}


def _error_message(exc: anthropic.APIError) -> str:
    message = _error_body(exc).get("message")
    return message if isinstance(message, str) else exc.message


def _provider_code(exc: anthropic.APIStatusError) -> str | None:
    """`error.type`, con `/error_code` si trae `details.error_code`."""
    error = _error_body(exc)
    error_type = error.get("type")
    if not isinstance(error_type, str):
        return None
    details = error.get("details")
    code = details.get("error_code") if isinstance(details, dict) else None
    return f"{error_type}/{code}" if isinstance(code, str) else error_type


def _retry_after_seconds(exc: anthropic.APIStatusError) -> float | None:
    """`retry-after` (segundos) o `retry-after-ms`, si el proveedor los entrega."""
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


def _has_retry_after(exc: anthropic.APIStatusError) -> bool:
    return "retry-after" in exc.response.headers or "retry-after-ms" in exc.response.headers


class AnthropicProvider(BaseVisionProvider):
    provider_id = "anthropic"
    display_name = "Anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        max_tokens: int = 16000,
        effort: str | None = None,
        treat_429_without_retry_after_as_credit: bool = True,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._max_tokens = max_tokens
        self._effort = effort
        self._429_without_retry_after_is_credit = treat_429_without_retry_after_as_credit
        self._schema = build_strict_schema()
        with _without_ambient_anthropic_env():
            self._client = AsyncAnthropic(
                api_key=api_key,
                base_url=API_BASE_URL,
                timeout=timeout_seconds,
                max_retries=0,
            )

    @classmethod
    def from_settings(cls, settings: AnthropicSettings) -> AnthropicProvider | None:
        """None si no hay key explicita en NUESTRA config: el tier queda
        inactivo y el SDK ni se instancia (no puede resolver credenciales
        ambientales)."""
        if settings.api_key is None or not settings.has_api_key():
            return None
        return cls(
            api_key=settings.api_key.get_secret_value(),
            timeout_seconds=settings.timeout_seconds,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
            treat_429_without_retry_after_as_credit=settings.treat_429_without_retry_after_as_credit,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def _send(self, request: ExtractionRequest, model: str) -> Any:
        # Imagenes antes del texto final, cada una etiquetada (recomendacion de
        # Anthropic para varias imagenes).
        content: list[dict[str, Any]] = []
        for index, image in enumerate(request.images, start=1):
            content.append({"type": "text", "text": f"Imagen {index}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": request.mime_type,
                        "data": base64.b64encode(image).decode("ascii"),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": "Extrae las tarjetas de todas las imagenes segun las reglas indicadas.",
            }
        )
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": self._schema}
        }
        if self._effort:
            output_config["effort"] = self._effort
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": content}],
            "output_config": output_config,
        }
        return await self._client.messages.create(**kwargs)

    def _usage(self, response: Any) -> UsageInfo:
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        # `input_tokens` excluye lo leido/escrito en cache: se suma para que sea
        # comparable con el prompt de los otros proveedores.
        parts = [
            getattr(usage, name, None)
            for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        ]
        prompt = sum(p for p in parts if isinstance(p, int)) if usage is not None else None
        output = getattr(usage, "output_tokens", None)
        details = getattr(usage, "output_tokens_details", None)
        total = None
        if prompt is not None and isinstance(output, int):
            total = prompt + output
        return UsageInfo(
            prompt_tokens=prompt,
            output_tokens=output,
            reasoning_tokens=getattr(details, "thinking_tokens", None),
            total_tokens=total,
            finish_reason=stop_reason,
            finish_ok=stop_reason in _OK_STOP_REASONS,
        )

    def _entries(
        self, response: Any, *, request: ExtractionRequest, model: str
    ) -> list[DeliveryEntryDTO]:
        # Los mensajes NUNCA incluyen contenido del JSON: van a logs y al JSONL.
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise InvalidResponseError(f"el modelo rechazo la solicitud (refusal, {category})")
        if stop_reason not in _OK_STOP_REASONS:
            raise InvalidResponseError(f"respuesta incompleta o inesperada ({stop_reason})")
        text = "".join(
            block.text
            for block in getattr(response, "content", None) or []
            if getattr(block, "type", None) == "text" and isinstance(block.text, str)
        )
        if not text.strip():
            raise InvalidResponseError("respuesta sin texto")
        try:
            envelope = _AnthropicEnvelope.model_validate_json(text)
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
        # APITimeoutError hereda de APIConnectionError: el orden importa.
        if isinstance(exc, anthropic.APITimeoutError):
            return self._error(FailureReason.TIMEOUT, model, message=_error_message(exc))
        if isinstance(exc, anthropic.APIConnectionError):
            return self._error(FailureReason.CONEXION, model, message=_error_message(exc))
        if isinstance(exc, anthropic.APIStatusError):
            return self._translate_status_error(exc, model)
        if isinstance(exc, anthropic.APIError):
            return self._error(FailureReason.DESCONOCIDO, model, message=_error_message(exc))
        return None  # no es una falla de API: se propaga tal cual

    def _translate_status_error(self, exc: anthropic.APIStatusError, model: str) -> ProviderError:
        status = exc.status_code
        message = _error_message(exc)
        code = _provider_code(exc)

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
                provider_code=code,
                limit_kind=limit_kind,
                retry_after_seconds=retry_after,
                provider_request_id=exc.request_id,
            )

        if status >= 500:  # 500, 502, 503, 504 y 529 `overloaded_error`
            return build(FailureReason.ALTA_DEMANDA)
        if status == 429:
            limit_kind, retry_after = self._classify_rate_limit(exc)
            return build(FailureReason.LIMITE_ALCANZADO, limit_kind, retry_after)
        if status == 402:  # `billing_error`
            return build(FailureReason.LIMITE_ALCANZADO, LimitKind.CREDITO_O_GASTO)
        if status == 400 and message.lower().startswith(_USER_SPEND_LIMIT_PREFIX):
            # Tope de gasto fijado por el usuario: llega como 400, que de otro
            # modo seria SOLICITUD_INVALIDA (la unica razon que NO avanza).
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

    def _classify_rate_limit(
        self, exc: anthropic.APIStatusError
    ) -> tuple[LimitKind, float | None]:
        """Subtipo de un 429 (siempre `rate_limit_error`) y su `retry-after`."""
        code = _provider_code(exc) or ""
        if code.endswith(f"/{SPEND_CAP_ERROR_CODE}"):
            return LimitKind.CREDITO_O_GASTO, None
        if _has_retry_after(exc):
            return LimitKind.RATE_MINUTO, _retry_after_seconds(exc)
        # Sin `retry-after` y sin el codigo documentado. HEURISTICA NO
        # verificada con una respuesta real: la documentacion dice que el 429 del
        # tope de gasto no trae `retry-after`, pero no que todo 429 sin el sea un
        # tope de gasto. Un falso positivo bloquearia a Anthropic hasta reiniciar.
        if self._429_without_retry_after_is_credit:
            return LimitKind.CREDITO_O_GASTO, None
        return LimitKind.DESCONOCIDO, None
