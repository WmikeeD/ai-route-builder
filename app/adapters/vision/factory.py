"""Fabrica de la cadena de vision: arma `FallbackChain` desde `Settings`.

Es la raiz de composicion: la unica que conoce a la vez la config, el
registro de adaptadores y la cadena. Resuelve los tiers (con los alias
deprecados GEMINI_MODEL / GEMINI_FALLBACK_MODEL), construye solo los
proveedores configurados, aplica la inyeccion de fallas de prueba si esta
activa, y valida al arrancar (fail-fast):

* un id de proveedor desconocido en VISION_CHAIN es un error;
* un proveedor conocido sin adaptador registrado o sin key deja el tier
  INACTIVO, nunca un error;
* debe quedar al menos un tier activo.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.adapters.vision.anthropic import AnthropicProvider
from app.adapters.vision.gemini import GeminiProvider
from app.adapters.vision.openai import OpenAIProvider
from app.config import Settings
from app.services.vision.chain import ChainPolicy, FallbackChain, RetryPolicy, Tier
from app.services.vision.circuit_breaker import CircuitBreaker
from app.services.vision.errors import ConfigurationError
from app.services.vision.fault_injection import FaultInjectingProvider, parse_fault_spec
from app.services.vision.models import TierSpec
from app.services.vision.ports import VisionExtractorProvider
from app.services.vision.quota_guard import QuotaGuard, fixed_duration, next_pacific_midnight
from app.services.vision.registry import KNOWN_PROVIDERS, ProviderRegistry, display_name_for
from app.services.vision.telemetry import AttemptRecorder, AttemptSink

logger = logging.getLogger(__name__)


def build_default_registry(settings: Settings) -> ProviderRegistry:
    """Adaptadores implementados. Un proveedor nuevo se registra aqui."""
    registry = ProviderRegistry()
    registry.register(
        "gemini",
        lambda: GeminiProvider.from_settings(settings.gemini),
        key_env_var="GEMINI_API_KEY",
    )
    registry.register(
        "openai",
        lambda: OpenAIProvider.from_settings(settings.openai),
        key_env_var="OPENAI_API_KEY",
    )
    registry.register(
        "anthropic",
        lambda: AnthropicProvider.from_settings(settings.anthropic),
        key_env_var="ANTHROPIC_API_KEY",
    )
    return registry


def _retry_policies(settings: Settings) -> dict[str, RetryPolicy]:
    def policy(max_attempts: int, initial: float, cap: float) -> RetryPolicy:
        return RetryPolicy(max_attempts, initial, cap)

    return {
        "gemini": policy(
            settings.gemini.effective_max_attempts,
            settings.gemini.retry_initial_delay_seconds,
            settings.gemini.retry_max_delay_seconds,
        ),
        "openai": policy(
            settings.openai.max_attempts,
            settings.openai.retry_initial_delay_seconds,
            settings.openai.retry_max_delay_seconds,
        ),
        "anthropic": policy(
            settings.anthropic.max_attempts,
            settings.anthropic.retry_initial_delay_seconds,
            settings.anthropic.retry_max_delay_seconds,
        ),
    }


def _log_deprecations(settings: Settings, notices: tuple[str, ...]) -> None:
    for notice in notices:
        logger.warning("DEPRECADO: %s", notice)
    if settings.gemini.test_retry_attempts is not None:
        logger.warning(
            "DEPRECADO: GEMINI_TEST_RETRY_ATTEMPTS es un alias de GEMINI_MAX_ATTEMPTS "
            "(intentos por tier centralizados en la cadena)."
        )
    if settings.fault_injection.legacy_force_gemini_model_error is not None:
        logger.warning(
            "DEPRECADO: FORCE_GEMINI_MODEL_ERROR es un alias de FORCE_VISION_ERROR "
            "(se aplica al modelo del primer tier de Gemini)."
        )


def build_fallback_chain(
    settings: Settings,
    *,
    registry: ProviderRegistry | None = None,
    recorder: AttemptSink | None = None,
) -> FallbackChain:
    resolution = settings.resolve_chain()
    _log_deprecations(settings, resolution.notices)
    registry = registry or build_default_registry(settings)

    unknown = sorted({t.provider_id for t in resolution.tiers} - set(KNOWN_PROVIDERS))
    if unknown:
        raise ConfigurationError(
            f"Proveedor(es) desconocido(s) en la cadena: {', '.join(unknown)}. "
            f"Conocidos: {', '.join(sorted(KNOWN_PROVIDERS))}"
        )

    # Un adaptador por proveedor, compartido por todos sus tiers.
    providers: dict[str, VisionExtractorProvider | None] = {}
    inactive_reasons: dict[str, str] = {}
    for provider_id in dict.fromkeys(t.provider_id for t in resolution.tiers):
        registration = registry.get(provider_id)
        if registration is None:
            providers[provider_id] = None
            inactive_reasons[provider_id] = "adaptador no implementado todavia"
            continue
        provider = registration.builder()
        providers[provider_id] = provider
        if provider is None:
            inactive_reasons[provider_id] = f"sin {registration.key_env_var}"

    _apply_fault_injection(settings, resolution.tiers, providers)

    tiers = [
        Tier(
            spec=spec,
            display_name=display_name_for(spec.provider_id),
            provider=providers[spec.provider_id],
            inactive_reason=inactive_reasons.get(spec.provider_id),
        )
        for spec in resolution.tiers
    ]
    if not any(tier.active for tier in tiers):
        raise ConfigurationError(
            "Ningun tier de la cadena esta activo: configura al menos una API key "
            "(p. ej. GEMINI_API_KEY)."
        )

    vision = settings.vision
    breaker = (
        CircuitBreaker(
            failure_threshold=vision.circuit_breaker_failure_threshold,
            cooldown_seconds=vision.circuit_breaker_cooldown_seconds,
        )
        if vision.circuit_breaker_enabled
        else None
    )
    chain = FallbackChain(
        tiers,
        policy=ChainPolicy(
            advance_on=vision.advance_on,
            chain_deadline_seconds=vision.chain_deadline_seconds,
            retry_policies=_retry_policies(settings),
        ),
        recorder=recorder or AttemptRecorder(vision.attempts_log_path),
        breaker=breaker,
        quota_guard=_build_quota_guard(settings) if vision.quota_guard_enabled else None,
    )
    _log_chain_summary(tiers)
    return chain


def _build_quota_guard(settings: Settings) -> QuotaGuard:
    """Reset diario por proveedor. Solo Gemini tiene un reset predecible
    documentado (medianoche hora del Pacifico); el resto usa una duracion fija
    configurable."""
    return QuotaGuard(
        reset_policies={"gemini": next_pacific_midnight},
        default_policy=fixed_duration(settings.vision.daily_quota_fallback_block_seconds),
    )


def _apply_fault_injection(
    settings: Settings,
    tiers: Sequence[TierSpec],
    providers: dict[str, VisionExtractorProvider | None],
) -> None:
    fault = settings.fault_injection
    spec = fault.force_error or fault.legacy_force_gemini_model_error
    if not spec:
        return
    first_gemini = next((t.model for t in tiers if t.provider_id == "gemini"), None)
    rules = parse_fault_spec(spec, legacy_gemini_model=first_gemini)
    for provider_id, provider in list(providers.items()):
        if provider is not None:
            providers[provider_id] = FaultInjectingProvider(provider, rules)
    logger.warning(
        "MODO DE PRUEBA: inyeccion de fallas activa (%s). NO usar en produccion.", spec
    )


def _log_chain_summary(tiers: list[Tier]) -> None:
    parts = []
    for index, tier in enumerate(tiers, start=1):
        state = "activo" if tier.active else f"inactivo: {tier.inactive_reason}"
        parts.append(f"{index}) {tier.display_name} {tier.spec.model} [{state}]")
    logger.info("Cadena de vision: %s", " ".join(parts))
