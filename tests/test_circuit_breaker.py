"""Tests del cortacircuitos por tier (reloj inyectado, sin esperas reales)."""

from __future__ import annotations

import pytest

from app.services.vision.circuit_breaker import CircuitBreaker
from app.services.vision.errors import FailureReason
from tests.vision_helpers import FakeClock

R = FailureReason
KEY = CircuitBreaker.key("gemini", "gemini-3.6-flash")


def _breaker(clock: FakeClock, threshold: int = 3, cooldown: float = 60.0) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, cooldown_seconds=cooldown, clock=clock)


def test_stays_closed_below_the_threshold() -> None:
    breaker = _breaker(FakeClock())

    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    breaker.record_failure(KEY, R.TIMEOUT)

    assert breaker.allow(KEY) is True
    assert breaker.is_open(KEY) is False


def test_opens_at_the_threshold_and_blocks_calls() -> None:
    breaker = _breaker(FakeClock(), threshold=2)

    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    breaker.record_failure(KEY, R.CONEXION)

    assert breaker.is_open(KEY) is True
    assert breaker.allow(KEY) is False


@pytest.mark.parametrize(
    "reason",
    [
        R.MODELO_NO_ENCONTRADO,
        R.AUTENTICACION,
        R.SOLICITUD_INVALIDA,
        R.RESPUESTA_INVALIDA,
        R.LIMITE_ALCANZADO,
    ],
)
def test_only_transient_failures_count(reason: FailureReason) -> None:
    breaker = _breaker(FakeClock(), threshold=1)

    for _ in range(5):
        breaker.record_failure(KEY, reason)

    assert breaker.is_open(KEY) is False


def test_a_success_resets_the_failure_count() -> None:
    breaker = _breaker(FakeClock(), threshold=2)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)

    breaker.record_success(KEY)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)

    assert breaker.is_open(KEY) is False


def test_after_the_cooldown_exactly_one_probe_is_allowed() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    clock.now += 61.0

    assert breaker.allow(KEY) is True  # el sondeo
    assert breaker.allow(KEY) is False  # ya hay uno en vuelo


def test_a_successful_probe_closes_the_circuit() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    clock.now += 61.0
    breaker.allow(KEY)

    breaker.record_success(KEY)

    assert breaker.is_open(KEY) is False
    assert breaker.allow(KEY) is True


def test_a_failed_probe_reopens_the_cooldown() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    clock.now += 61.0
    breaker.allow(KEY)

    breaker.record_failure(KEY, R.TIMEOUT)

    assert breaker.allow(KEY) is False
    clock.now += 61.0
    assert breaker.allow(KEY) is True


def test_an_inconclusive_probe_does_not_reopen_and_allows_another_probe() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    clock.now += 61.0
    breaker.allow(KEY)

    breaker.record_failure(KEY, R.MODELO_NO_ENCONTRADO)  # no concluyente

    assert breaker.allow(KEY) is True


def test_an_abandoned_probe_is_released_or_expires() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    clock.now += 61.0
    breaker.allow(KEY)

    breaker.release_probe(KEY)
    assert breaker.allow(KEY) is True  # liberado
    clock.now += 61.0
    assert breaker.allow(KEY) is True  # vencido sin resultado


def test_circuits_are_independent_per_provider_and_model() -> None:
    breaker = _breaker(FakeClock(), threshold=1)
    other = CircuitBreaker.key("gemini", "gemini-3.8-flash")

    breaker.record_failure(KEY, R.ALTA_DEMANDA)

    assert breaker.allow(KEY) is False
    assert breaker.allow(other) is True


def test_seconds_until_probe_counts_down() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure(KEY, R.ALTA_DEMANDA)
    clock.now += 20.0

    assert breaker.seconds_until_probe(KEY) == pytest.approx(40.0)
    assert breaker.seconds_until_probe("otro") == 0.0


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
