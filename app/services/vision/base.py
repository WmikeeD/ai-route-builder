"""Clase base para adaptadores de proveedores de vision (plantilla).

El `Protocol` de `ports.py` sirve para tipado e inyeccion; esta clase evita
copiar y pegar el flujo comun: aplicar el timeout duro, traducir las
excepciones del SDK a `ProviderError`, registrar el uso (tokens y motivo de
fin) y convertir el DTO a dominio. Cada proveedor implementa solo estos
ganchos: enviar, leer el uso, leer las entradas, traducir errores y cerrar.

Para agregar un proveedor nuevo (p. ej. Kimi o DeepSeek): heredar de esta
clase, agregar su bloque de settings, registrarlo en la fabrica y sumar una
linea a `VISION_CHAIN`. Nada mas cambia.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import ExtractionRequest, ExtractionResult, UsageInfo
from app.services.vision.schema import DeliveryEntryDTO, to_domain_entry

logger = logging.getLogger(__name__)

# Margen del tope duro sobre el timeout del cliente HTTP: el timeout del SDK
# es el que normalmente corta; este tope solo evita colgarse si el SDK falla
# en aplicarlo (p. ej. un servidor que gotea bytes).
_HARD_TIMEOUT_GRACE_SECONDS = 2.0


class InvalidResponseError(Exception):
    """Lo lanza `_entries` cuando la respuesta no trae un JSON utilizable
    (vacio, truncado o rechazado). La base la convierte en
    `ProviderError(RESPUESTA_INVALIDA)`."""


class BaseVisionProvider(ABC):
    provider_id: str
    display_name: str

    def __init__(self, *, timeout_seconds: float) -> None:
        self._hard_timeout_seconds = timeout_seconds + _HARD_TIMEOUT_GRACE_SECONDS

    def is_configured(self) -> bool:
        """Una instancia solo existe si hay credencial explicita: la fabrica
        no construye adaptadores sin key."""
        return True

    @abstractmethod
    async def _send(self, request: ExtractionRequest, model: str) -> Any:
        """Llamada al SDK. Puede lanzar cualquier excepcion del SDK."""

    @abstractmethod
    def _usage(self, response: Any) -> UsageInfo:
        """Tokens y motivo de fin de la respuesta. No debe lanzar: se llama
        antes de validar el JSON para poder loguear tambien las respuestas
        truncadas."""

    @abstractmethod
    def _entries(
        self, response: Any, *, request: ExtractionRequest, model: str
    ) -> list[DeliveryEntryDTO]:
        """Las tarjetas de la respuesta. Lanza `InvalidResponseError` si no hay
        un JSON utilizable."""

    @abstractmethod
    def _translate_error(self, exc: Exception, model: str) -> ProviderError | None:
        """Traduce una excepcion del SDK/transporte a `ProviderError`. Devuelve
        `None` para lo que NO es una falla de API (errores de programacion),
        que entonces se propaga sin tocar."""

    @abstractmethod
    async def aclose(self) -> None: ...

    def _request_id(self, response: Any) -> str | None:
        """Id de la solicitud en el proveedor (para abrir tickets), si existe."""
        return None

    def _error(
        self,
        reason: FailureReason,
        model: str,
        *,
        message: str = "",
        http_status: int | None = None,
        provider_code: str | None = None,
        limit_kind: LimitKind | None = None,
        retry_after_seconds: float | None = None,
        provider_request_id: str | None = None,
    ) -> ProviderError:
        return ProviderError(
            reason,
            provider=self.provider_id,
            model=model,
            message=message,
            http_status=http_status,
            provider_code=provider_code,
            limit_kind=limit_kind,
            retry_after_seconds=retry_after_seconds,
            provider_request_id=provider_request_id,
        )

    async def extract(self, request: ExtractionRequest, *, model: str) -> ExtractionResult:
        try:
            async with asyncio.timeout(self._hard_timeout_seconds):
                response = await self._send(request, model)
        except ProviderError:
            raise
        except TimeoutError as exc:
            raise self._error(
                FailureReason.TIMEOUT,
                model,
                message=f"sin respuesta tras {self._hard_timeout_seconds:.0f}s (tope duro)",
            ) from exc
        except Exception as exc:
            translated = self._translate_error(exc, model)
            if translated is None:
                raise
            raise translated from exc

        # Uso primero: un JSON truncado (p. ej. MAX_TOKENS) debe verse en el
        # log aunque despues falle la validacion.
        usage = self._usage(response)
        self._log_usage(model, usage)
        request_id = self._request_id(response)
        try:
            dtos = self._entries(response, request=request, model=model)
        except InvalidResponseError as exc:
            raise self._error(
                FailureReason.RESPUESTA_INVALIDA,
                model,
                message=f"{exc} (finish_reason={usage.finish_reason})",
                provider_request_id=request_id,
            ) from exc
        return ExtractionResult(
            entries=[to_domain_entry(item, request.screenshot_ids) for item in dtos],
            usage=usage,
            provider_request_id=request_id,
        )

    def _log_usage(self, model: str, usage: UsageInfo) -> None:
        """Por que termino la generacion y cuantos tokens consumio (reemplaza
        al log especifico de Gemini). Un motivo de fin anormal (p. ej.
        MAX_TOKENS) explica un JSON truncado: sube a WARNING. Solo conteos,
        nunca el contenido de la respuesta."""
        level = logging.INFO if usage.finish_ok else logging.WARNING
        logger.log(
            level,
            "VISION_RESPONSE provider=%s model=%s finish_reason=%s prompt_tokens=%s "
            "output_tokens=%s reasoning_tokens=%s total_tokens=%s",
            self.provider_id,
            model,
            usage.finish_reason,
            usage.prompt_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
        )
