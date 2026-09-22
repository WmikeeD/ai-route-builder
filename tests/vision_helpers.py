"""Dobles y constructores compartidos por los tests de la cadena de vision."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence

from app.config import DEFAULT_ADVANCE_ON
from app.domain.models import RawDeliveryEntry
from app.services.vision.chain import ChainPolicy, FallbackChain, RetryPolicy, Tier
from app.services.vision.circuit_breaker import CircuitBreaker
from app.services.vision.errors import FailureReason, LimitKind, ProviderError
from app.services.vision.models import (
    ExtractionRequest,
    ExtractionResult,
    ProviderAttempt,
    TierSpec,
    UsageInfo,
)
from app.services.vision.quota_guard import QuotaGuard
from app.services.vision.telemetry import AttemptSink


def ok_result(count: int = 1) -> ExtractionResult:
    entries = [RawDeliveryEntry(address=f"Calle Falsa {i}") for i in range(1, count + 1)]
    return ExtractionResult(entries=entries, usage=UsageInfo(finish_reason="STOP"))


def failure(
    reason: FailureReason,
    *,
    provider: str = "gemini",
    model: str = "m",
    limit_kind: LimitKind | None = None,
    retry_after: float | None = None,
    http_status: int | None = None,
    provider_code: str | None = None,
) -> ProviderError:
    return ProviderError(
        reason,
        provider=provider,
        model=model,
        limit_kind=limit_kind,
        retry_after_seconds=retry_after,
        http_status=http_status,
        provider_code=provider_code,
    )


class FakeProvider:
    """Proveedor guionado: cada llamada consume el siguiente elemento del guion
    (un `ExtractionResult`, una excepcion a lanzar, o una corrutina a esperar).
    Sin guion, responde OK."""

    def __init__(
        self,
        provider_id: str = "gemini",
        script: Sequence[object] = (),
        display_name: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.display_name = display_name or provider_id.capitalize()
        self._script = list(script)
        self.calls: list[str] = []
        self.closed = 0

    def is_configured(self) -> bool:
        return True

    async def extract(self, request: ExtractionRequest, *, model: str) -> ExtractionResult:
        self.calls.append(model)
        item = self._script.pop(0) if self._script else ok_result()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = await item()
        assert isinstance(item, ExtractionResult)
        return item

    async def aclose(self) -> None:
        self.closed += 1


class ListSink:
    """Recolecta los intentos que la cadena registra."""

    def __init__(self) -> None:
        self.attempts: list[ProviderAttempt] = []

    def record(self, attempt: ProviderAttempt) -> None:
        self.attempts.append(attempt)

    @property
    def outcomes(self) -> list[str]:
        return [str(a.outcome) for a in self.attempts]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def tier(
    provider: FakeProvider | None,
    model: str,
    *,
    provider_id: str | None = None,
    display_name: str | None = None,
    inactive_reason: str | None = None,
) -> Tier:
    pid = provider_id or (provider.provider_id if provider else "gemini")
    return Tier(
        spec=TierSpec(pid, model),
        display_name=display_name or pid.capitalize(),
        provider=provider,
        inactive_reason=inactive_reason,
    )


class ChainHarness:
    """Cadena armada con doubles deterministas (sin esperas reales)."""

    def __init__(
        self,
        tiers: Sequence[Tier],
        *,
        retry: Mapping[str, RetryPolicy] | None = None,
        advance_on: frozenset[FailureReason] = DEFAULT_ADVANCE_ON,
        deadline: float = 300.0,
        breaker: CircuitBreaker | None = None,
        clock: Callable[[], float] = time.monotonic,
        recorder: AttemptSink | None = None,
        quota_guard: QuotaGuard | None = None,
    ) -> None:
        self.sink = ListSink()
        self.sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.chain = FallbackChain(
            tiers,
            policy=ChainPolicy(advance_on, deadline, dict(retry or {})),
            recorder=recorder or self.sink,
            breaker=breaker,
            quota_guard=quota_guard,
            clock=clock,
            sleep=fake_sleep,
            jitter=lambda: 0.0,
        )

    async def run(self, images: int = 1) -> list[RawDeliveryEntry]:
        return await self.chain.extract_entries(images=[b"img"] * images)


Sleeper = Callable[[], Awaitable[ExtractionResult]]
