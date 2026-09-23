"""Configuracion de la aplicacion via variables de entorno.

Usa pydantic-settings para validar y tipar la configuracion en un solo
lugar. Falla rapido (ValidationError) al arrancar si falta una variable
requerida, en vez de fallar a medias mas adelante dentro de un handler.

La extraccion por vision se configura con bloques anidados, uno por
proveedor (`gemini`, `openai`, `anthropic`), mas un bloque de cadena
(`vision`): nada se comparte entre proveedores. Cada bloque lee su propio
prefijo de entorno (`GEMINI_*`, `OPENAI_*`, `ANTHROPIC_*`, `VISION_*`) del
mismo `.env`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.services.vision.errors import FailureReason
from app.services.vision.models import TierSpec

# Cadena decidida (docs/ARQUITECTURA_FALLBACK_MULTIPROVEEDOR.md). Se usa
# cuando VISION_CHAIN no esta definida. Un tier cuyo proveedor no tiene key
# (o cuyo adaptador aun no existe) queda inactivo y la cadena lo omite.
DEFAULT_CHAIN: tuple[TierSpec, ...] = (
    TierSpec("gemini", "gemini-3.6-flash"),
    TierSpec("gemini", "gemini-3.8-flash"),
    TierSpec("openai", "gpt-5.6-terra"),
    TierSpec("anthropic", "claude-sonnet-5"),
)

# Razones tras las cuales la cadena pasa al siguiente tier. Solo
# SOLICITUD_INVALIDA detiene la cadena: el mismo payload rechazado por un
# proveedor no se arregla cambiando de modelo.
DEFAULT_ADVANCE_ON: frozenset[FailureReason] = frozenset(FailureReason) - {
    FailureReason.SOLICITUD_INVALIDA
}


def _env_config(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def parse_chain(value: str) -> tuple[TierSpec, ...]:
    """`"gemini:gemini-3.6-flash,openai:gpt-5.6-terra"` -> tiers en orden."""
    tiers: list[TierSpec] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        provider_id, sep, model = item.partition(":")
        if not sep or not provider_id.strip() or not model.strip():
            raise ValueError(
                f"Tier invalido {item!r} en VISION_CHAIN: se espera 'proveedor:modelo'"
            )
        tiers.append(TierSpec(provider_id.strip().lower(), model.strip()))
    return tuple(tiers)


class _ProviderSettings(BaseSettings):
    """Campos comunes a todo proveedor; cada bloque tiene su propia instancia
    y su propio prefijo de entorno."""

    api_key: SecretStr | None = None
    # TEMPORAL: 60 s es un valor de la etapa de evaluacion LOCAL, elegido para
    # medir el tiempo maximo real de espera de cada modelo en este entorno.
    # Al desplegar en Render debe bajar bastante (la latencia de red local no
    # es representativa). Es configuracion por proveedor (<PREFIJO>_TIMEOUT_SECONDS),
    # no una constante del adaptador.
    timeout_seconds: float = Field(default=60.0, gt=0)
    # Reintentos centralizados en la cadena (R2); los SDK van en 1 intento.
    max_attempts: int = Field(default=2, ge=1)
    retry_initial_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=8.0, ge=0)

    def has_api_key(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value().strip())


class GeminiSettings(_ProviderSettings):
    model_config = _env_config("GEMINI_")

    # Sin `temperature`: deprecado en los modelos Gemini 3.x (ver gemini.py).

    # 1 intento por tier (no el 2 del resto de proveedores): con demanda alta
    # sostenida (503 / ALTA_DEMANDA), el segundo intento contra el mismo modelo
    # casi nunca rescato nada en varias sesiones y solo demoraba el fallback.
    # Aplica a TODOS los tiers de Gemini: la politica de reintento es por
    # proveedor, no por tier.
    max_attempts: int = Field(default=1, ge=1)

    # --- Alias DEPRECADOS (variante B') ---------------------------------
    # GEMINI_MODEL / GEMINI_FALLBACK_MODEL mapean a los tiers 1 y 2 de la
    # cadena cuando VISION_CHAIN no esta definida. Usa VISION_CHAIN.
    model: str | None = None
    fallback_model: str | None = None
    # Hook de pruebas manuales (antes vivia en gemini_client.py): fija los
    # intentos por tier. Alias deprecado de GEMINI_MAX_ATTEMPTS.
    test_retry_attempts: int | None = Field(default=None, ge=1)

    @property
    def effective_max_attempts(self) -> int:
        return self.test_retry_attempts or self.max_attempts


class OpenAISettings(_ProviderSettings):
    model_config = _env_config("OPENAI_")

    # Esfuerzo de razonamiento del modelo ("low", "medium", "high"...; los
    # valores validos dependen del modelo). Vacio = el default del modelo.
    reasoning_effort: str | None = None
    # Detalle con el que OpenAI procesa cada imagen: mas detalle = mas tokens.
    image_detail: Literal["low", "high", "auto"] = "auto"
    # La API de Responses guarda por defecto una copia de cada solicitud en
    # OpenAI. Las capturas traen datos de clientes, asi que no se guardan
    # salvo que se active explicitamente.
    store_responses: bool = False

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _blank_effort_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AnthropicSettings(_ProviderSettings):
    """Tier 4: completo pero INACTIVO hasta que exista ANTHROPIC_API_KEY. El
    adaptador solo usa la key de este bloque; jamas credenciales ambientales
    del SDK (ANTHROPIC_AUTH_TOKEN, perfiles, ANTHROPIC_BASE_URL...)."""

    model_config = _env_config("ANTHROPIC_")

    # Tope de tokens de salida por solicitud. Los tokens de razonamiento (si el
    # modelo razona) cuentan contra este tope; un lote grande necesita margen.
    max_tokens: int = Field(default=16000, ge=1)
    # Esfuerzo de razonamiento (`output_config.effort`). Vacio = default del modelo.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Un 429 SIN `retry-after` y sin el codigo documentado de tope de gasto
    # (`enforced_spend_limit_reached`) se trata como credito/gasto agotado
    # (bloquea al proveedor hasta reiniciar el proceso). Es una heuristica NO
    # verificada con una respuesta real: si diera un falso positivo (p. ej. un
    # limite de aceleracion), desactivala y ese 429 pasa a LIMITE/DESCONOCIDO.
    treat_429_without_retry_after_as_credit: bool = True

    @field_validator("effort", mode="before")
    @classmethod
    def _blank_effort_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class VisionSettings(BaseSettings):
    model_config = _env_config("VISION_")

    # VISION_CHAIN="gemini:gemini-3.6-flash,gemini:gemini-3.8-flash,openai:gpt-5.6-terra,..."
    chain: Annotated[tuple[TierSpec, ...] | None, NoDecode] = None
    # Tope total de espera de toda la cadena. TEMPORAL: valor de la etapa
    # local; ajustar junto con los timeouts por proveedor al desplegar.
    chain_deadline_seconds: float = Field(default=300.0, gt=0)
    advance_on: Annotated[frozenset[FailureReason], NoDecode] = DEFAULT_ADVANCE_ON
    # Archivo JSONL de intentos (T2). Vacio desactiva el archivo.
    attempts_log_path: Path | None = Path("logs/vision_attempts.jsonl")
    # Cortacircuitos de 120 s: solo fallas TRANSITORIAS (alta demanda, timeout, red).
    circuit_breaker_enabled: bool = True
    # El contador cuenta SOLICITUDES que agotaron todos sus intentos en este
    # tier, no attempts individuales (chain.py registra la falla una sola vez
    # por tier por solicitud, con el ultimo error como representativo). Por
    # eso el umbral es independiente de max_attempts: 3 significa "3
    # solicitudes distintas tuvieron mala suerte con este tier", que es la
    # intencion original del cortacircuitos (varios conductores, no uno solo
    # agotando sus reintentos).
    circuit_breaker_failure_threshold: int = Field(default=3, ge=1)
    circuit_breaker_cooldown_seconds: float = Field(default=120.0, gt=0)
    # Guardia de cuota, SEPARADO del cortacircuitos: salta el tier al detectar
    # CUOTA_DIARIA (hasta el proximo reset; Gemini: medianoche hora del Pacifico)
    # y el proveedor entero al detectar CREDITO_O_GASTO (hasta reiniciar el
    # proceso). Para proveedores SIN reset diario documentado se usa esta
    # duracion fija; 6 h = como maximo unos 4 sondeos fallidos al dia.
    quota_guard_enabled: bool = True
    daily_quota_fallback_block_seconds: float = Field(default=6 * 3600.0, gt=0)

    @field_validator("chain", mode="before")
    @classmethod
    def _parse_chain(cls, value: object) -> object:
        if isinstance(value, str):
            return parse_chain(value) or None
        return value

    @field_validator("advance_on", mode="before")
    @classmethod
    def _parse_advance_on(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(
                FailureReason(item.strip().upper()) for item in value.split(",") if item.strip()
            )
        return value

    @field_validator("attempts_log_path", mode="before")
    @classmethod
    def _blank_path_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class FaultInjectionSettings(BaseSettings):
    """Hooks de PRUEBAS MANUALES (ver app/services/vision/fault_injection.py)."""

    model_config = _env_config("")

    force_error: str | None = Field(default=None, validation_alias="FORCE_VISION_ERROR")
    # Alias deprecado del hook original: FORCE_GEMINI_MODEL_ERROR=503.
    legacy_force_gemini_model_error: str | None = Field(
        default=None, validation_alias="FORCE_GEMINI_MODEL_ERROR"
    )


@dataclass(frozen=True, slots=True)
class ChainResolution:
    tiers: tuple[TierSpec, ...]
    # Avisos (deprecaciones) para loguear al arrancar.
    notices: tuple[str, ...]


def _clean(value: str | None) -> str | None:
    return value.strip() or None if value else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    telegram_bot_token: str
    pdf_region: str = "Región Metropolitana"
    pdf_country: str = "Chile"
    log_level: str = "INFO"

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    fault_injection: FaultInjectionSettings = Field(default_factory=FaultInjectionSettings)

    def resolve_chain(self) -> ChainResolution:
        """Cadena efectiva: VISION_CHAIN si existe; si no, la por defecto con
        los alias deprecados GEMINI_MODEL / GEMINI_FALLBACK_MODEL aplicados a
        los tiers 1 y 2 (variante B')."""
        legacy_model = _clean(self.gemini.model)
        legacy_fallback = _clean(self.gemini.fallback_model)
        notices: list[str] = []

        if self.vision.chain is not None:
            if legacy_model or legacy_fallback:
                notices.append(
                    "GEMINI_MODEL / GEMINI_FALLBACK_MODEL estan deprecadas y se ignoran "
                    "porque VISION_CHAIN esta definida."
                )
            return ChainResolution(self.vision.chain, tuple(notices))

        tiers = list(DEFAULT_CHAIN)
        if legacy_model:
            tiers[0] = TierSpec("gemini", legacy_model)
            notices.append(
                f"GEMINI_MODEL esta deprecada: se aplico como tier 1 ({legacy_model}). "
                "Usa VISION_CHAIN."
            )
        if legacy_fallback:
            tiers[1] = TierSpec("gemini", legacy_fallback)
            notices.append(
                f"GEMINI_FALLBACK_MODEL esta deprecada: se aplico como tier 2 "
                f"({legacy_fallback}). Usa VISION_CHAIN."
            )
        return ChainResolution(tuple(tiers), tuple(notices))


@lru_cache
def get_settings() -> Settings:
    """Carga y cachea la configuracion (una sola lectura de entorno/.env)."""
    return Settings()
