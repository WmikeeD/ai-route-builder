"""Puertos de la extraccion por vision.

Dos contratos, uno por lado de la cadena:

* `RouteExtractor`: lo unico que la capa de UI (Telegram) conoce. Lo
  implementa `FallbackChain`.
* `VisionExtractorProvider`: lo que implementa cada adaptador de proveedor
  (Gemini, OpenAI, Anthropic, y futuros Kimi/DeepSeek). Es la plantilla para
  sumar un proveedor sin tocar el resto del sistema.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models import RawDeliveryEntry
from app.services.vision.models import ExtractionRequest, ExtractionResult


class RouteExtractor(Protocol):
    """Extrae las paradas de un lote de capturas. La UI depende solo de esto."""

    async def extract_entries(
        self,
        images: list[bytes],
        mime_type: str = "image/jpeg",
        screenshot_ids: list[str] | None = None,
    ) -> list[RawDeliveryEntry]: ...

    async def aclose(self) -> None: ...


class VisionExtractorProvider(Protocol):
    """Un proveedor de vision (una empresa/SDK) atendiendo varios modelos."""

    # Identificador estable en minusculas ("gemini", "openai", "anthropic");
    # es la llave del registro, del log y de las metricas.
    provider_id: str
    # Nombre para logs legibles ("Gemini", "OpenAI", "Anthropic").
    display_name: str

    def is_configured(self) -> bool:
        """True solo con credencial explicita en nuestra config; nunca por
        credenciales ambientales que el SDK pudiera descubrir."""
        ...

    async def extract(self, request: ExtractionRequest, *, model: str) -> ExtractionResult:
        """Una llamada a un modelo. Solo lanza `ProviderError` por fallas de
        API, transporte o respuesta; los errores de programacion se propagan."""
        ...

    async def aclose(self) -> None: ...
