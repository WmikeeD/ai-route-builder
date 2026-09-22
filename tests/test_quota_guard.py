"""Tests del bloqueo por cuota diaria y por credito (`QuotaGuard`).

Es un mecanismo SEPARADO del cortacircuitos de 120 s: aqui se prueba tanto la
logica del guardia como su integracion con la cadena y, en particular, que es
independiente del cortacircuitos.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from app.services.vision.chain import RetryPolicy
from app.services.vision.circuit_breaker import CircuitBreaker
from app.services.vision.errors import (
    AllProvidersFailedError,
    FailureReason,
    LimitKind,
    ProviderError,
)
from app.services.vision.quota_guard import (
    QuotaGuard,
    fixed_duration,
    next_pacific_midnight,
)
from app.services.vision.telemetry import AttemptRecorder
from tests.vision_helpers import (
    ChainHarness,
    FakeClock,
    FakeProvider,
    failure,
    ok_result,
    tier,
)

R = FailureReason
DAILY = LimitKind.CUOTA_DIARIA
CREDIT = LimitKind.CREDITO_O_GASTO
ONE_ATTEMPT = {"gemini": RetryPolicy(1), "openai": RetryPolicy(1)}


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class FakeNow:
    """Hora de calendario controlable (el guardia no usa el reloj monotono)."""

    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **delta: float) -> None:
        self.value += timedelta(**delta)


def limit(
    kind: LimitKind, provider: str = "gemini", model: str = "gemini-3.6-flash"
) -> ProviderError:
    return failure(R.LIMITE_ALCANZADO, provider=provider, model=model, limit_kind=kind)


def make_guard(now: FakeNow, fallback_seconds: float = 6 * 3600.0) -> QuotaGuard:
    return QuotaGuard(
        reset_policies={"gemini": next_pacific_midnight},
        default_policy=fixed_duration(fallback_seconds),
        now=now,
    )


# ------------------------------------------------------- reset a medianoche del Pacifico


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Verano (PDT, UTC-7): 14:10 PDT -> medianoche PDT del dia siguiente.
        (utc(2026, 9, 21, 21, 10), utc(2026, 9, 22, 7, 0)),
        # Un minuto antes de la medianoche PDT.
        (utc(2026, 9, 22, 6, 59), utc(2026, 9, 22, 7, 0)),
        # Justo en la medianoche: el reset es el SIGUIENTE, no el mismo instante.
        (utc(2026, 9, 22, 7, 0), utc(2026, 9, 23, 7, 0)),
        # Invierno (PST, UTC-8).
        (utc(2026, 12, 1, 12, 0), utc(2026, 12, 2, 8, 0)),
        # Fin del horario de verano (dia de 25 h): 01:30 PDT del 1-nov-2026.
        (utc(2026, 11, 1, 8, 30), utc(2026, 11, 2, 8, 0)),
        # Inicio del horario de verano (dia de 23 h): 01:00 PST del 14-mar-2027.
        (utc(2027, 3, 14, 9, 0), utc(2027, 3, 15, 7, 0)),
    ],
)
def test_next_pacific_midnight(now: datetime, expected: datetime) -> None:
    result = next_pacific_midnight(now)

    assert result == expected
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)  # siempre un instante UTC


def test_fixed_duration_policy() -> None:
    assert fixed_duration(3600)(utc(2026, 9, 21, 12)) == utc(2026, 9, 21, 13)


# ------------------------------------------------------------------ logica del guardia


def test_a_daily_quota_failure_blocks_that_model_until_the_pacific_reset() -> None:
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    guard = make_guard(now)

    new = guard.record_failure(limit(DAILY))

    assert new is not None
    assert (new.kind, new.until) == (DAILY, utc(2026, 9, 22, 7, 0))
    assert new.scope == "gemini:gemini-3.6-flash"
    assert guard.block_for("gemini", "gemini-3.6-flash") is new
    assert guard.block_for("gemini", "gemini-3.8-flash") is None  # la cuota diaria es por modelo


def test_the_daily_block_lifts_exactly_at_the_reset() -> None:
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    guard = make_guard(now)
    guard.record_failure(limit(DAILY))

    now.value = utc(2026, 9, 22, 6, 59)
    assert guard.block_for("gemini", "gemini-3.6-flash") is not None
    now.value = utc(2026, 9, 22, 7, 0)
    assert guard.block_for("gemini", "gemini-3.6-flash") is None
    assert guard.active_blocks() == []


def test_providers_without_a_documented_reset_use_the_fixed_duration() -> None:
    now = FakeNow(utc(2026, 9, 21, 12, 0))
    guard = make_guard(now, fallback_seconds=6 * 3600)

    block = guard.record_failure(limit(DAILY, provider="openai", model="gpt-5.6-terra"))

    assert block is not None
    assert block.until == utc(2026, 9, 21, 18, 0)


def test_a_credit_failure_blocks_the_whole_provider_indefinitely() -> None:
    now = FakeNow(utc(2026, 9, 21, 12, 0))
    guard = make_guard(now)

    block = guard.record_failure(limit(CREDIT, provider="openai", model="gpt-5.6-terra"))

    assert block is not None
    assert block.until is None
    assert block.scope == "openai"
    now.advance(days=400)  # nunca se levanta solo: requiere intervencion humana
    assert guard.block_for("openai", "gpt-5.6-terra") is block
    assert guard.block_for("openai", "otro-modelo-de-openai") is block  # todos sus modelos
    assert guard.block_for("gemini", "gemini-3.6-flash") is None  # otro proveedor: libre


def test_a_second_credit_failure_is_not_reported_as_a_new_block() -> None:
    guard = make_guard(FakeNow(utc(2026, 9, 21)))
    assert guard.record_failure(limit(CREDIT, provider="openai")) is not None

    assert guard.record_failure(limit(CREDIT, provider="openai")) is None


@pytest.mark.parametrize(
    "error",
    [
        limit(LimitKind.RATE_MINUTO),
        limit(LimitKind.DESCONOCIDO),
        failure(R.ALTA_DEMANDA),
        failure(R.TIMEOUT),
        failure(R.MODELO_NO_ENCONTRADO),
        failure(R.AUTENTICACION),
    ],
)
def test_only_daily_quota_and_credit_block(error: ProviderError) -> None:
    guard = make_guard(FakeNow(utc(2026, 9, 21)))

    assert guard.record_failure(error) is None
    assert guard.active_blocks() == []


def test_a_success_clears_the_daily_block() -> None:
    guard = make_guard(FakeNow(utc(2026, 9, 21, 21, 10)))
    guard.record_failure(limit(DAILY))

    guard.record_success("gemini", "gemini-3.6-flash")

    assert guard.block_for("gemini", "gemini-3.6-flash") is None


def test_the_block_descriptions_say_what_to_do() -> None:
    guard = make_guard(FakeNow(utc(2026, 9, 21, 21, 10)))
    daily = guard.record_failure(limit(DAILY))
    credit = guard.record_failure(limit(CREDIT, provider="openai"))

    assert daily is not None and credit is not None
    assert daily.describe() == "cuota diaria agotada hasta 2026-09-22T07:00+00:00"
    assert "intervencion humana" in credit.describe()
    assert "reiniciar el proceso" in credit.describe()


# ------------------------------------------------------------------ integracion con la cadena


def _two_tiers(script1: list[object], script2: list[object] | None = None):
    p1, p2 = FakeProvider("gemini", script1), FakeProvider("gemini", script2 or [])
    return p1, p2, [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")]


def test_a_daily_quota_tier_is_skipped_on_the_following_requests_without_any_call() -> None:
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    p1, p2, tiers = _two_tiers([limit(DAILY)])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, quota_guard=make_guard(now))

    for _ in range(3):
        asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash"]  # una sola vez: las demas se ahorran
    assert p2.calls == ["gemini-3.8-flash"] * 3
    assert harness.sink.outcomes.count("SALTADO_CUOTA") == 2


def test_after_the_reset_the_tier_is_probed_again_and_a_success_clears_the_block() -> None:
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    p1, _p2, tiers = _two_tiers([limit(DAILY), ok_result()])
    guard = make_guard(now)
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, quota_guard=guard)
    asyncio.run(harness.run())  # bloquea el tier 1

    now.value = utc(2026, 9, 22, 7, 0)  # ya paso la medianoche del Pacifico
    asyncio.run(harness.run())  # sondeo natural: responde el tier 1

    assert p1.calls == ["gemini-3.6-flash"] * 2
    assert guard.active_blocks() == []


def test_a_new_daily_failure_after_the_reset_blocks_until_the_next_reset() -> None:
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    p1, _p2, tiers = _two_tiers([limit(DAILY), limit(DAILY)])
    guard = make_guard(now)
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, quota_guard=guard)
    asyncio.run(harness.run())

    now.value = utc(2026, 9, 22, 7, 0)
    asyncio.run(harness.run())  # el sondeo vuelve a fallar por cuota

    (block,) = guard.active_blocks()
    assert block.until == utc(2026, 9, 23, 7, 0)  # el reset SIGUIENTE
    assert len(p1.calls) == 2


def test_a_credit_limit_blocks_every_model_of_the_provider_within_the_same_request() -> None:
    now = FakeNow(utc(2026, 9, 21, 12))
    openai = FakeProvider("openai", [limit(CREDIT, provider="openai", model="gpt-a")])
    gemini = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(openai, "gpt-a"), tier(openai, "gpt-b"), tier(gemini, "gemini-3.6-flash")],
        retry=ONE_ATTEMPT,
        quota_guard=make_guard(now),
    )

    asyncio.run(harness.run())  # gpt-a falla por credito; gpt-b ya ni se intenta
    now.advance(days=30)
    asyncio.run(harness.run())  # un mes despues sigue bloqueado: hace falta intervencion

    assert openai.calls == ["gpt-a"]
    assert gemini.calls == ["gemini-3.6-flash"] * 2
    assert harness.sink.outcomes.count("SALTADO_CUOTA") == 3  # gpt-b + (gpt-a y gpt-b)


def test_when_every_active_tier_is_quota_blocked_the_chain_fails_without_any_call() -> None:
    """A diferencia del cortacircuitos, NO hay pasada forzada: llamar a un tier
    cuyo limite sabemos agotado es gasto sin posibilidad de exito."""
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    p1 = FakeProvider("gemini", [limit(DAILY)])
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash")], retry=ONE_ATTEMPT, quota_guard=make_guard(now)
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())  # primera vez: llama y falla por cuota diaria

    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(harness.run())  # segunda: ni siquiera se intenta

    assert p1.calls == ["gemini-3.6-flash"]
    assert info.value.dominant_reason is R.LIMITE_ALCANZADO
    assert info.value.dominant_limit_kind is DAILY  # el mensaje al usuario sigue siendo correcto


def test_a_credit_blocked_chain_reports_the_credit_kind_to_the_user_layer() -> None:
    now = FakeNow(utc(2026, 9, 21, 12))
    p1 = FakeProvider("openai", [limit(CREDIT, provider="openai")])
    harness = ChainHarness(
        [tier(p1, "gpt-5.6-terra")], retry=ONE_ATTEMPT, quota_guard=make_guard(now)
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())

    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(harness.run())

    assert info.value.dominant_limit_kind is CREDIT
    assert p1.calls == ["gpt-5.6-terra"]


def test_without_a_guard_the_tier_is_attempted_on_every_request() -> None:
    p1, _p2, tiers = _two_tiers([limit(DAILY), limit(DAILY)])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)  # sin quota_guard

    asyncio.run(harness.run())
    asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash"] * 2


def test_block_registration_is_logged_once_and_the_skips_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    openai = FakeProvider("openai", [limit(CREDIT, provider="openai")])
    _p1, p2, tiers = _two_tiers([limit(DAILY)])
    harness = ChainHarness(
        [*tiers, tier(openai, "gpt-5.6-terra")],
        retry=ONE_ATTEMPT,
        quota_guard=make_guard(now),
        recorder=AttemptRecorder(None),
    )
    p2._script = [failure(R.ALTA_DEMANDA)] * 3

    with caplog.at_level(logging.INFO):
        for _ in range(3):
            with pytest.raises(AllProvidersFailedError):
                asyncio.run(harness.run())

    daily_logs = [r for r in caplog.records if "BLOQUEO_CUOTA_DIARIA" in r.getMessage()]
    credit_logs = [r for r in caplog.records if "BLOQUEO_CREDITO_O_GASTO" in r.getMessage()]
    assert len(daily_logs) == 1 and daily_logs[0].levelno == logging.WARNING
    assert len(credit_logs) == 1 and credit_logs[0].levelno == logging.ERROR
    skips = [r for r in caplog.records if "omitido por LIMITE_ALCANZADO" in r.getMessage()]
    assert skips and all(r.levelno == logging.INFO for r in skips)
    assert any("kind=CUOTA_DIARIA: cuota diaria agotada hasta" in r.getMessage() for r in skips)
    assert any("kind=CREDITO_O_GASTO" in r.getMessage() for r in skips)


# ------------------------------------------------------- independencia del cortacircuitos


def test_the_quota_block_is_independent_of_the_120s_circuit_breaker() -> None:
    """Decision de diseno: son DOS mecanismos. El enfriamiento de 120 s del
    cortacircuitos (fallas transitorias) no levanta un bloqueo de cuota, y una
    falla de cuota no cuenta para abrir el cortacircuitos."""
    clock = FakeClock()
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=120.0, clock=clock)
    guard = make_guard(now)

    quota_tier = FakeProvider("gemini", [limit(DAILY)])
    flaky_tier = FakeProvider("openai", [failure(R.ALTA_DEMANDA, provider="openai")])
    steady = FakeProvider("anthropic")
    harness = ChainHarness(
        [
            tier(quota_tier, "gemini-3.6-flash"),
            tier(flaky_tier, "gpt-5.6-terra"),
            tier(steady, "claude-sonnet-5"),
        ],
        retry={"gemini": RetryPolicy(1), "openai": RetryPolicy(1)},
        breaker=breaker,
        quota_guard=guard,
        clock=clock,
    )
    asyncio.run(harness.run())  # tier 1: cuota diaria; tier 2: 503 (abre el cortacircuitos)

    # La falla de CUOTA no abrio el cortacircuitos (umbral 1); la de 503 si.
    assert breaker.is_open(CircuitBreaker.key("gemini", "gemini-3.6-flash")) is False
    assert breaker.is_open(CircuitBreaker.key("openai", "gpt-5.6-terra")) is True
    assert [b.scope for b in guard.active_blocks()] == ["gemini:gemini-3.6-flash"]

    clock.now += 121.0  # vencio el enfriamiento de 120 s
    asyncio.run(harness.run())

    assert flaky_tier.calls == ["gpt-5.6-terra"] * 2  # el cortacircuitos si sondeo
    assert quota_tier.calls == ["gemini-3.6-flash"]  # el bloqueo de cuota NO se levanto
    assert "SALTADO_CUOTA" in harness.sink.outcomes


def test_a_quota_blocked_tier_never_reserves_a_circuit_breaker_probe() -> None:
    clock = FakeClock()
    now = FakeNow(utc(2026, 9, 21, 21, 10))
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=120.0, clock=clock)
    p1 = FakeProvider("gemini", [limit(DAILY)])
    p2 = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")],
        retry=ONE_ATTEMPT,
        breaker=breaker,
        quota_guard=make_guard(now),
        clock=clock,
    )
    for _ in range(3):
        asyncio.run(harness.run())

    assert breaker.allow(CircuitBreaker.key("gemini", "gemini-3.6-flash")) is True  # sin estado
