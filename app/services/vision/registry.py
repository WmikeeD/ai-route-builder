"""Registro de proveedores: ids conocidos y como construir cada adaptador.

`KNOWN_PROVIDERS` es la lista de ids validos en `VISION_CHAIN`. Un id
desconocido (p. ej. un typo) falla al arrancar. Un id conocido cuyo adaptador
todavia no esta registrado (OpenAI y Anthropic en la Etapa 1) es un tier
inactivo, no un error: la cadena lo omite con un INFO.

`ProviderRegistry` es un contenedor sin SDKs: quien arma la app (la fabrica
en `app.adapters.vision`) registra ahi los constructores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.services.vision.ports import VisionExtractorProvider

# id -> nombre para logs legibles. Sumar un proveedor nuevo empieza aqui.
KNOWN_PROVIDERS: dict[str, str] = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}


def display_name_for(provider_id: str) -> str:
    return KNOWN_PROVIDERS.get(provider_id, provider_id.capitalize())


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    # Devuelve el adaptador, o None si no esta configurado (sin key).
    builder: Callable[[], VisionExtractorProvider | None]
    # Variable de entorno de la credencial (solo para mensajes de log).
    key_env_var: str


class ProviderRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}

    def register(
        self,
        provider_id: str,
        builder: Callable[[], VisionExtractorProvider | None],
        *,
        key_env_var: str,
    ) -> None:
        self._registrations[provider_id] = ProviderRegistration(builder, key_env_var)

    def get(self, provider_id: str) -> ProviderRegistration | None:
        return self._registrations.get(provider_id)
