"""Orquestador de la cadena de fallback multi-proveedor.

Recorre los tiers en orden. Solo conoce el puerto `VisionExtractorProvider`
y el vocabulario canonico de errores: ninguna clase de SDK entra aqui.

Politica (decisiones aprobadas en docs/ARQUITECTURA_FALLBACK_MULTIPROVEEDOR.md):

* Reintento centralizado (R2): los SDK no reintentan; la cadena reintenta un
  mismo tier solo ante `ALTA_DEMANDA`, `TIMEOUT`, `CONEXION` y
  `LIMITE_ALCANZADO/RATE_MINUTO`, respetando `Retry-After` cuando el
  proveedor lo entrega. Nunca reintenta un limite de cuota diaria o de
  credito.
* `advance_on`: razones tras las cuales se pasa al siguiente tier. Una razon
  fuera de esa lista detiene la cadena (`AllProvidersFailedError.aborted_by`).
* Un tier sin proveedor (sin key, o adaptador aun no implementado) se omite
  con un INFO; nunca lanza excepcion por su ausencia.
* Cortacircuitos por tier y tope total de espera de la cadena.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NoReturn

from app.domain.models import RawDeliveryEntry
from app.services.vision.circuit_breaker import CircuitBreaker
from app.services.vision.errors import (
    AllProvidersFailedError,
    FailureReason,
    LimitKind,
    ProviderError,
    dominant_failure,
)
from app.services.vision.models import (
    AttemptOutcome,
    ExtractionRequest,
    ExtractionResult,
    ProviderAttempt,
    TierSpec,
    UsageInfo,
)
from app.services.vision.ports import VisionExtractorProvider
from app.services.vision.quota_guard import QuotaBlock, QuotaGuard
from app.services.vision.telemetry import AttemptSink

logger = logging.getLogger(__name__)

_TRANSIENT_REASONS: frozenset[FailureReason] = frozenset(
    {FailureReason.ALTA_DEMANDA, FailureReason.TIMEOUT, FailureReason.CONEXION}
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Reintento centralizado de un proveedor (los SDK quedan en 1 intento)."""

    max_attempts: int = 1
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0


@dataclass(frozen=True, slots=True)
class ChainPolicy:
    advance_on: frozenset[FailureReason]
    chain_deadline_seconds: float
    retry_policies: Mapping[str, RetryPolicy] = field(default_factory=dict)


@dataclass(slots=True)
class Tier:
    spec: TierSpec
    display_name: str
    provider: VisionExtractorProvider | None
    inactive_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.provider is not None


@dataclass(slots=True)
class _Run:
    """Estado de una solicitud en curso (una pasada por la cadena)."""

    chain_id: str
    request: ExtractionRequest
    deadline: float
    payload_bytes: int
    attempts: list[ProviderAttempt] = field(default_factory=list)
    failures: list[ProviderError] = field(default_factory=list)
    # Un ProviderError sintetico por cada tier omitido por bloqueo de cuota, para
    # que la causa dominante refleje el limite aunque no se haya llamado a nadie.
    blocked: list[ProviderError] = field(default_factory=list)
    skipped: int = 0
    attempted: bool = False
    deadline_hit: bool = False
    aborted_by: ProviderError | None = None


class FallbackChain:
    """Implementa `RouteExtractor` recorriendo los tiers configurados."""

    def __init__(
        self,
        tiers: Sequence[Tier],
        *,
        policy: ChainPolicy,
        recorder: AttemptSink,
        breaker: CircuitBreaker | None = None,
        quota_guard: QuotaGuard | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if not any(tier.active for tier in tiers):
            raise ValueError("La cadena necesita al menos un tier activo")
        self._tiers = tuple(tiers)
        self._policy = policy
        self._recorder = recorder
        self._breaker = breaker
        self._quota_guard = quota_guard
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter

    @property
    def tiers(self) -> tuple[Tier, ...]:
        return self._tiers

    @property
    def policy(self) -> ChainPolicy:
        return self._policy

    async def aclose(self) -> None:
        closed: set[int] = set()
        for tier in self._tiers:
            provider = tier.provider
            if provider is not None and id(provider) not in closed:
                closed.add(id(provider))
                await provider.aclose()

    async def extract_entries(
        self,
        images: list[bytes],
        mime_type: str = "image/jpeg",
        screenshot_ids: list[str] | None = None,
    ) -> list[RawDeliveryEntry]:
        if not images:
            return []
        if screenshot_ids is not None and len(screenshot_ids) != len(images):
            raise ValueError("screenshot_ids debe tener el mismo largo que images")
        request = ExtractionRequest(
            images=images, mime_type=mime_type, screenshot_ids=screenshot_ids
        )
        result = await self._run(request)
        return result.entries

    # ------------------------------------------------------------------ nucleo

    async def _run(self, request: ExtractionRequest) -> ExtractionResult:
        run = _Run(
            chain_id=uuid.uuid4().hex[:12],
            request=request,
            deadline=self._clock() + self._policy.chain_deadline_seconds,
            payload_bytes=sum(len(image) for image in request.images),
        )
        # Primera pasada respetando el cortacircuitos. Si por eso no se llego a
        # intentar ningun tier, segunda pasada ignorandolo: la cadena nunca debe
        # rendirse sin haber probado al menos un proveedor activo.
        result = await self._walk(run, respect_breaker=True)
        if result is None and not run.attempted and not run.deadline_hit:
            result = await self._walk(run, respect_breaker=False)
        if result is not None:
            return result
        self._raise_exhausted(run)

    async def _walk(self, run: _Run, *, respect_breaker: bool) -> ExtractionResult | None:
        for index, tier in enumerate(self._tiers, start=1):
            provider = tier.provider
            if provider is None:
                if respect_breaker:  # solo en la primera pasada: sin duplicar eventos
                    self._record_skip(
                        run, index, tier, AttemptOutcome.SALTADO_INACTIVO, tier.inactive_reason
                    )
                continue

            # Bloqueo por cuota diaria / credito: se respeta SIEMPRE, tambien en la
            # pasada forzada. Llamar a un tier cuyo limite sabemos agotado es
            # gasto sin posibilidad de exito. Va antes del cortacircuitos para no
            # reservar un sondeo en un tier que igual no se va a llamar.
            if self._quota_guard is not None:
                block = self._quota_guard.block_for(tier.spec.provider_id, tier.spec.model)
                if block is not None:
                    if respect_breaker:  # solo en la primera pasada: sin duplicar eventos
                        self._record_quota_skip(run, index, tier, block)
                    continue

            key = CircuitBreaker.key(tier.spec.provider_id, tier.spec.model)
            if respect_breaker and self._breaker is not None and not self._breaker.allow(key):
                wait = self._breaker.seconds_until_probe(key)
                self._record_skip(
                    run, index, tier, AttemptOutcome.SALTADO_CIRCUITO, f"reintento en {wait:.0f}s"
                )
                continue

            run.attempted = True
            result, error = await self._run_tier(run, index, tier, provider, key)
            if result is not None:
                return result
            if error is None:  # imposible: sin resultado siempre hay error
                raise RuntimeError("tier sin resultado ni error")
            # Una sola vez por tier por solicitud (no por attempt interno): el
            # umbral del cortacircuitos cuenta "solicitudes que agotaron todos
            # sus intentos", no attempts individuales. Se usa el ultimo error
            # de la solicitud como representativo. Como unico punto de
            # registro, esto tambien evita el caso borde de un sondeo
            # semiabierto con mas de un intento interno: antes, el primer
            # fallo limpiaba `probe_started_at` y un segundo fallo de la
            # misma solicitud caia por la rama equivocada.
            if self._breaker is not None:
                self._breaker.record_failure(key, error.reason)
            run.failures.append(error)
            if run.deadline_hit:
                return None
            if error.reason not in self._policy.advance_on:
                run.aborted_by = error
                return None
        return None

    async def _run_tier(
        self,
        run: _Run,
        index: int,
        tier: Tier,
        provider: VisionExtractorProvider,
        key: str,
    ) -> tuple[ExtractionResult | None, ProviderError | None]:
        retry = self._policy.retry_policies.get(tier.spec.provider_id, RetryPolicy())
        error: ProviderError | None = None

        for attempt in range(1, retry.max_attempts + 1):
            remaining = run.deadline - self._clock()
            if remaining <= 0:
                run.deadline_hit = True
                return None, error or self._timeout_error(tier, "se agoto el tiempo de la cadena")

            started = self._clock()
            deadline_guard = asyncio.timeout(remaining)
            try:
                async with deadline_guard:
                    result = await provider.extract(run.request, model=tier.spec.model)
            except ProviderError as exc:
                error = exc
            except TimeoutError:
                # Vence el tope de la cadena, o un proveedor dejo escapar un
                # TimeoutError crudo: en ambos casos es un TIMEOUT de este tier.
                # `expired()` es exacto; comparar relojes no lo es: en Windows
                # asyncio puede disparar el temporizador ~15 ms antes.
                if deadline_guard.expired():
                    run.deadline_hit = True
                    error = self._timeout_error(tier, "se agoto el tiempo de la cadena")
                else:
                    error = self._timeout_error(tier, "TimeoutError sin clasificar del proveedor")
            except BaseException:
                if self._breaker is not None:
                    self._breaker.release_probe(key)
                raise
            else:
                if self._breaker is not None:
                    self._breaker.record_success(key)
                if self._quota_guard is not None:
                    self._quota_guard.record_success(tier.spec.provider_id, tier.spec.model)
                self._emit(
                    run,
                    self._attempt(
                        run, index, tier, AttemptOutcome.EXITO, attempt, retry.max_attempts,
                        started, usage=result.usage,
                        provider_request_id=result.provider_request_id,
                    ),
                )
                return result, None

            if self._quota_guard is not None:
                new_block = self._quota_guard.record_failure(error)
                if new_block is not None:
                    self._log_new_block(tier, new_block)
            self._emit(
                run,
                self._attempt(
                    run, index, tier, AttemptOutcome.FALLO, attempt, retry.max_attempts,
                    started, error=error,
                ),
            )
            if run.deadline_hit or attempt >= retry.max_attempts:
                break
            delay = self._retry_delay(error, attempt, retry, run.deadline)
            if delay is None:
                break
            await self._sleep(delay)

        return None, error

    @staticmethod
    def _timeout_error(tier: Tier, message: str) -> ProviderError:
        return ProviderError(
            FailureReason.TIMEOUT,
            provider=tier.spec.provider_id,
            model=tier.spec.model,
            message=message,
        )

    def _retry_delay(
        self, error: ProviderError, attempt: int, retry: RetryPolicy, deadline: float
    ) -> float | None:
        """Espera antes de reintentar el mismo tier, o `None` si no conviene
        reintentar (falla no transitoria, `Retry-After` demasiado largo o sin
        tiempo en el tope de la cadena)."""
        transient = error.reason in _TRANSIENT_REASONS or (
            error.reason is FailureReason.LIMITE_ALCANZADO
            and error.limit_kind is LimitKind.RATE_MINUTO
        )
        if not transient:
            return None
        base = min(retry.max_delay_seconds, retry.initial_delay_seconds * 2 ** (attempt - 1))
        delay = base * (1 + 0.25 * self._jitter())
        if error.retry_after_seconds is not None:
            if error.retry_after_seconds > retry.max_delay_seconds:
                return None  # esperar tanto no vale la pena: mejor el siguiente tier
            delay = max(delay, error.retry_after_seconds)
        if self._clock() + delay >= deadline:
            return None
        return delay

    # ------------------------------------------------------------------ registro

    def _emit(self, run: _Run, attempt: ProviderAttempt) -> None:
        run.attempts.append(attempt)
        self._recorder.record(attempt)

    def _attempt(
        self,
        run: _Run,
        index: int,
        tier: Tier,
        outcome: AttemptOutcome,
        attempt: int,
        max_attempts: int,
        started: float,
        *,
        error: ProviderError | None = None,
        usage: UsageInfo | None = None,
        provider_request_id: str | None = None,
    ) -> ProviderAttempt:
        return ProviderAttempt(
            ts=datetime.now(UTC),
            chain_request_id=run.chain_id,
            tier=index,
            tiers_total=len(self._tiers),
            provider=tier.spec.provider_id,
            display_name=tier.display_name,
            model=tier.spec.model,
            outcome=outcome,
            attempt=attempt,
            max_attempts=max_attempts,
            reason=error.reason if error else None,
            limit_kind=error.limit_kind if error else None,
            http_status=error.http_status if error else None,
            provider_code=error.provider_code if error else None,
            latency_ms=round((self._clock() - started) * 1000),
            images=len(run.request.images),
            payload_bytes=run.payload_bytes,
            usage=usage,
            provider_request_id=error.provider_request_id if error else provider_request_id,
            detail=(error.message or None) if error else None,
        )

    def _record_skip(
        self, run: _Run, index: int, tier: Tier, outcome: AttemptOutcome, detail: str | None
    ) -> None:
        run.skipped += 1
        self._emit(
            run,
            ProviderAttempt(
                ts=datetime.now(UTC),
                chain_request_id=run.chain_id,
                tier=index,
                tiers_total=len(self._tiers),
                provider=tier.spec.provider_id,
                display_name=tier.display_name,
                model=tier.spec.model,
                outcome=outcome,
                images=len(run.request.images),
                payload_bytes=run.payload_bytes,
                detail=detail,
            ),
        )

    def _record_quota_skip(self, run: _Run, index: int, tier: Tier, block: QuotaBlock) -> None:
        run.skipped += 1
        run.blocked.append(
            ProviderError(
                FailureReason.LIMITE_ALCANZADO,
                provider=tier.spec.provider_id,
                model=tier.spec.model,
                message=f"tier omitido por bloqueo: {block.describe()}",
                limit_kind=block.kind,
            )
        )
        self._emit(
            run,
            ProviderAttempt(
                ts=datetime.now(UTC),
                chain_request_id=run.chain_id,
                tier=index,
                tiers_total=len(self._tiers),
                provider=tier.spec.provider_id,
                display_name=tier.display_name,
                model=tier.spec.model,
                outcome=AttemptOutcome.SALTADO_CUOTA,
                reason=FailureReason.LIMITE_ALCANZADO,
                limit_kind=block.kind,
                images=len(run.request.images),
                payload_bytes=run.payload_bytes,
                detail=block.describe(),
            ),
        )

    @staticmethod
    def _log_new_block(tier: Tier, block: QuotaBlock) -> None:
        """Una sola vez, al registrar el bloqueo (los omitidos posteriores van
        por el registro de intentos)."""
        if block.until is None:
            logger.error(
                "BLOQUEO_CREDITO_O_GASTO %s (todos sus modelos): %s",
                tier.spec.provider_id,
                block.describe(),
            )
        else:
            logger.warning(
                "BLOQUEO_CUOTA_DIARIA %s: %s; no se reintenta hasta entonces",
                block.scope,
                block.describe(),
            )

    def _raise_exhausted(self, run: _Run) -> NoReturn:
        reason, limit_kind = dominant_failure([*run.failures, *run.blocked])
        if run.aborted_by is not None:
            reason, limit_kind = run.aborted_by.reason, run.aborted_by.limit_kind
        per_provider: dict[str, int] = {}
        for failure in run.failures:
            per_provider[failure.provider] = per_provider.get(failure.provider, 0) + 1
        detail = ", ".join(f"{name}:{count}" for name, count in per_provider.items())
        logger.warning(
            "CADENA_AGOTADA chain=%s fallos=%d omitidos=%d dominante=%s%s (%s)%s",
            run.chain_id,
            len(run.failures),
            run.skipped,
            reason,
            f"/{limit_kind}" if limit_kind else "",
            detail or "sin intentos",
            " [detenida por falla no continuable]" if run.aborted_by else "",
        )
        raise AllProvidersFailedError(
            run.attempts,
            dominant_reason=reason,
            dominant_limit_kind=limit_kind,
            aborted_by=run.aborted_by,
        )
