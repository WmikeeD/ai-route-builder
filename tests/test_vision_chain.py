"""Tests del orquestador `FallbackChain`: orden de tiers, avance por razon,
reintento centralizado (R2), tope de espera, cortacircuitos y tiers inactivos.

Los proveedores son dobles guionados: cada test controla exactamente que
responde cada tier, sin red ni esperas reales.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.config import DEFAULT_ADVANCE_ON
from app.services.vision.chain import RetryPolicy
from app.services.vision.circuit_breaker import CircuitBreaker
from app.services.vision.errors import AllProvidersFailedError, FailureReason, LimitKind
from app.services.vision.models import ExtractionResult
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
ONE_ATTEMPT = {"gemini": RetryPolicy(max_attempts=1)}
# Sin delay entre intentos: los tests del cortacircuitos con 2 intentos por
# tier no necesitan avanzar el reloj para los reintentos internos, solo para
# el cooldown entre solicitudes.
TWO_ATTEMPTS_NO_DELAY = {
    "gemini": RetryPolicy(max_attempts=2, initial_delay_seconds=0.0, max_delay_seconds=0.0)
}


def _two_gemini_tiers(
    first: list[object], second: list[object] | None = None
) -> tuple[FakeProvider, FakeProvider, list]:
    p1, p2 = FakeProvider("gemini", first), FakeProvider("gemini", second or [])
    return p1, p2, [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")]


# ------------------------------------------------------------------ orden y avance


def test_success_on_first_tier_never_touches_the_rest() -> None:
    p1, p2, tiers = _two_gemini_tiers([ok_result(3)])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    entries = asyncio.run(harness.run())

    assert len(entries) == 3
    assert p1.calls == ["gemini-3.6-flash"]
    assert p2.calls == []
    assert harness.sink.outcomes == ["EXITO"]


def test_high_demand_on_first_tier_falls_back_to_the_second() -> None:
    p1, p2, tiers = _two_gemini_tiers([failure(R.ALTA_DEMANDA, http_status=503)])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    entries = asyncio.run(harness.run())

    assert len(entries) == 1
    assert p1.calls == ["gemini-3.6-flash"]
    assert p2.calls == ["gemini-3.8-flash"]
    assert harness.sink.outcomes == ["FALLO", "EXITO"]


@pytest.mark.parametrize(
    "reason",
    [
        R.ALTA_DEMANDA,
        R.LIMITE_ALCANZADO,
        R.MODELO_NO_ENCONTRADO,
        R.AUTENTICACION,  # cambio decidido: antes abortaba
        R.RESPUESTA_INVALIDA,  # cambio decidido: antes abortaba
        R.TIMEOUT,
        R.CONEXION,
        R.DESCONOCIDO,
    ],
)
def test_advance_on_defaults_continue_to_the_next_tier(reason: FailureReason) -> None:
    _p1, p2, tiers = _two_gemini_tiers([failure(reason)])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    asyncio.run(harness.run())

    assert p2.calls == ["gemini-3.8-flash"]


def test_invalid_request_stops_the_chain_without_trying_other_tiers() -> None:
    _p1, p2, tiers = _two_gemini_tiers([failure(R.SOLICITUD_INVALIDA, http_status=400)])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(harness.run())

    assert p2.calls == []
    assert info.value.dominant_reason is R.SOLICITUD_INVALIDA
    assert info.value.aborted_by is not None


def test_advance_on_is_configurable() -> None:
    _p1, p2, tiers = _two_gemini_tiers([failure(R.RESPUESTA_INVALIDA)])
    strict = DEFAULT_ADVANCE_ON - {R.RESPUESTA_INVALIDA}
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, advance_on=strict)

    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(harness.run())

    assert p2.calls == []
    assert info.value.dominant_reason is R.RESPUESTA_INVALIDA


def test_all_tiers_failing_raises_with_dominant_reason_and_attempts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _p1, _p2, tiers = _two_gemini_tiers(
        [failure(R.ALTA_DEMANDA)], [failure(R.ALTA_DEMANDA)]
    )
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    with caplog.at_level(logging.WARNING, logger="app.services.vision.chain"), pytest.raises(
        AllProvidersFailedError
    ) as info:
        asyncio.run(harness.run())

    error = info.value
    assert error.dominant_reason is R.ALTA_DEMANDA
    assert error.aborted_by is None
    assert len(error.attempts) == 2
    assert len({a.chain_request_id for a in error.attempts}) == 1
    assert any("CADENA_AGOTADA" in r.getMessage() for r in caplog.records)


def test_limit_takes_priority_over_high_demand_as_dominant_reason() -> None:
    gemini = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)])
    credit_limit = failure(
        R.LIMITE_ALCANZADO, provider="openai", limit_kind=LimitKind.CREDITO_O_GASTO
    )
    openai = FakeProvider("openai", [credit_limit])
    harness = ChainHarness(
        [tier(gemini, "gemini-3.6-flash"), tier(openai, "gpt-5.6-terra")],
        retry={"gemini": RetryPolicy(1), "openai": RetryPolicy(1)},
    )

    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(harness.run())

    assert info.value.dominant_reason is R.LIMITE_ALCANZADO
    assert info.value.dominant_limit_kind is LimitKind.CREDITO_O_GASTO


def test_unexpected_exception_propagates_without_falling_back() -> None:
    _p1, p2, tiers = _two_gemini_tiers([RuntimeError("bug")])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(harness.run())

    assert p2.calls == []


def test_empty_batch_returns_nothing_and_does_not_call_providers() -> None:
    p1, _p2, tiers = _two_gemini_tiers([])
    harness = ChainHarness(tiers)

    assert asyncio.run(harness.chain.extract_entries(images=[])) == []
    assert p1.calls == []


def test_mismatched_screenshot_ids_are_rejected() -> None:
    _p1, _p2, tiers = _two_gemini_tiers([])
    harness = ChainHarness(tiers)

    with pytest.raises(ValueError, match="screenshot_ids"):
        asyncio.run(harness.chain.extract_entries(images=[b"a", b"b"], screenshot_ids=["x"]))


def test_close_closes_a_shared_provider_only_once() -> None:
    shared = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(shared, "gemini-3.6-flash"), tier(shared, "gemini-3.8-flash")]
    )

    asyncio.run(harness.chain.aclose())

    assert shared.closed == 1


# ------------------------------------------------------------------ tiers inactivos


def test_inactive_tiers_are_skipped_with_the_exact_info_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _p1, _p2, tiers = _two_gemini_tiers([failure(R.ALTA_DEMANDA)], [failure(R.ALTA_DEMANDA)])
    tiers += [
        tier(None, "gpt-5.6-terra", provider_id="openai", display_name="OpenAI",
             inactive_reason="sin OPENAI_API_KEY"),
        tier(None, "claude-sonnet-5", provider_id="anthropic", display_name="Anthropic",
             inactive_reason="sin ANTHROPIC_API_KEY"),
    ]
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, recorder=AttemptRecorder(None))

    with caplog.at_level(logging.INFO, logger="vision.attempts"), pytest.raises(
        AllProvidersFailedError
    ):
        asyncio.run(harness.run())

    messages = [r.getMessage() for r in caplog.records if r.name == "vision.attempts"]
    assert "tier 3 (OpenAI) no configurado, se omite" in messages
    assert "tier 4 (Anthropic) no configurado, se omite" in messages
    skipped = [r for r in caplog.records if "no configurado" in r.getMessage()]
    assert all(r.levelno == logging.INFO for r in skipped)


def test_inactive_tiers_are_not_reached_when_an_earlier_tier_succeeds() -> None:
    p1 = FakeProvider("gemini", [ok_result()])
    harness = ChainHarness(
        [
            tier(p1, "gemini-3.6-flash"),
            tier(None, "claude-sonnet-5", provider_id="anthropic", inactive_reason="sin key"),
        ]
    )

    asyncio.run(harness.run())

    assert harness.sink.outcomes == ["EXITO"]


def test_skipped_inactive_tier_is_recorded_so_activation_can_be_measured() -> None:
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)])
    harness = ChainHarness(
        [
            tier(p1, "gemini-3.6-flash"),
            tier(None, "claude-sonnet-5", provider_id="anthropic", inactive_reason="sin key"),
        ],
        retry=ONE_ATTEMPT,
    )

    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())

    assert harness.sink.outcomes == ["FALLO", "SALTADO_INACTIVO"]
    assert harness.sink.attempts[1].provider == "anthropic"


def test_a_chain_without_any_active_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="al menos un tier activo"):
        ChainHarness([tier(None, "x", provider_id="openai", inactive_reason="sin key")])


# ------------------------------------------------------------------ reintento (R2)


def test_transient_failure_is_retried_on_the_same_tier_before_falling_back() -> None:
    p1, p2, tiers = _two_gemini_tiers([failure(R.ALTA_DEMANDA), ok_result()])
    harness = ChainHarness(tiers, retry={"gemini": RetryPolicy(2, 1.0, 8.0)})

    asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash", "gemini-3.6-flash"]
    assert p2.calls == []
    assert harness.sleeps == [1.0]
    assert [(a.attempt, a.max_attempts) for a in harness.sink.attempts] == [(1, 2), (2, 2)]


def test_backoff_is_exponential_and_capped_by_max_delay() -> None:
    fails = [failure(R.TIMEOUT) for _ in range(4)]
    p1, _p2, tiers = _two_gemini_tiers(fails)
    harness = ChainHarness(tiers, retry={"gemini": RetryPolicy(4, 1.0, 3.0)})

    asyncio.run(harness.run())  # el 5o intento no existe: cae al tier 2

    assert p1.calls.count("gemini-3.6-flash") == 4
    assert harness.sleeps == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "kind", [LimitKind.CUOTA_DIARIA, LimitKind.CREDITO_O_GASTO, LimitKind.DESCONOCIDO]
)
def test_daily_quota_and_credit_limits_are_never_retried(kind: LimitKind) -> None:
    p1, p2, tiers = _two_gemini_tiers([failure(R.LIMITE_ALCANZADO, limit_kind=kind)])
    harness = ChainHarness(tiers, retry={"gemini": RetryPolicy(3, 1.0, 8.0)})

    asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash"]  # una sola vez
    assert p2.calls == ["gemini-3.8-flash"]
    assert harness.sleeps == []


def test_per_minute_rate_limit_is_retried_honoring_a_short_retry_after() -> None:
    limit = failure(R.LIMITE_ALCANZADO, limit_kind=LimitKind.RATE_MINUTO, retry_after=5.0)
    p1, _p2, tiers = _two_gemini_tiers([limit, ok_result()])
    harness = ChainHarness(tiers, retry={"gemini": RetryPolicy(2, 1.0, 8.0)})

    asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash", "gemini-3.6-flash"]
    assert harness.sleeps == [5.0]  # Retry-After manda sobre el backoff de 1 s


def test_a_retry_after_longer_than_the_max_delay_advances_instead_of_waiting() -> None:
    limit = failure(R.LIMITE_ALCANZADO, limit_kind=LimitKind.RATE_MINUTO, retry_after=30.0)
    p1, p2, tiers = _two_gemini_tiers([limit])
    harness = ChainHarness(tiers, retry={"gemini": RetryPolicy(3, 1.0, 8.0)})

    asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash"]
    assert p2.calls == ["gemini-3.8-flash"]
    assert harness.sleeps == []


def test_non_transient_failures_are_not_retried() -> None:
    p1, _p2, tiers = _two_gemini_tiers([failure(R.MODELO_NO_ENCONTRADO, http_status=404)])
    harness = ChainHarness(tiers, retry={"gemini": RetryPolicy(3, 1.0, 8.0)})

    asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash"]
    assert harness.sleeps == []


# ------------------------------------------------------------------ tope de espera


def test_chain_deadline_stops_before_trying_the_next_tier() -> None:
    clock = FakeClock()

    async def slow_failure() -> ExtractionResult:
        clock.now += 100.0
        raise failure(R.ALTA_DEMANDA)

    _p1, p2, tiers = _two_gemini_tiers([slow_failure])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, deadline=50.0, clock=clock)

    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())

    assert p2.calls == []  # sin tiempo: no se prueba el segundo tier


def test_chain_deadline_cancels_a_hung_provider() -> None:
    async def hang() -> ExtractionResult:
        await asyncio.sleep(10)
        return ok_result()

    _p1, p2, tiers = _two_gemini_tiers([hang])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT, deadline=0.05)

    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(harness.run())

    assert info.value.dominant_reason is R.TIMEOUT
    assert p2.calls == []
    assert harness.sink.attempts[0].reason is R.TIMEOUT


def test_a_raw_timeout_error_from_a_provider_is_a_timeout_of_that_tier_only() -> None:
    _p1, p2, tiers = _two_gemini_tiers([TimeoutError("sdk")])
    harness = ChainHarness(tiers, retry=ONE_ATTEMPT)

    asyncio.run(harness.run())

    assert p2.calls == ["gemini-3.8-flash"]  # no se confunde con el tope de la cadena
    assert harness.sink.attempts[0].reason is R.TIMEOUT


# ------------------------------------------------------------------ cortacircuitos


def _breaker(
    clock: FakeClock, threshold: int = 2, cooldown_seconds: float = 60.0
) -> CircuitBreaker:
    return CircuitBreaker(
        failure_threshold=threshold, cooldown_seconds=cooldown_seconds, clock=clock
    )


def test_circuit_opens_after_repeated_failures_and_the_tier_is_skipped() -> None:
    clock = FakeClock()
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA), failure(R.ALTA_DEMANDA)])
    p2 = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")],
        retry=ONE_ATTEMPT,
        breaker=_breaker(clock),
        clock=clock,
    )

    for _ in range(3):
        asyncio.run(harness.run())

    assert p1.calls == ["gemini-3.6-flash"] * 2  # la 3a solicitud ya no lo intenta
    assert p2.calls == ["gemini-3.8-flash"] * 3
    assert "SALTADO_CIRCUITO" in harness.sink.outcomes


def test_after_the_cooldown_one_probe_call_closes_the_circuit_on_success() -> None:
    clock = FakeClock()
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA), failure(R.ALTA_DEMANDA), ok_result()])
    p2 = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")],
        retry=ONE_ATTEMPT,
        breaker=_breaker(clock),
        clock=clock,
    )
    for _ in range(3):
        asyncio.run(harness.run())

    clock.now += 61.0
    asyncio.run(harness.run())  # sondeo: el tier 1 responde bien
    asyncio.run(harness.run())  # cerrado: sigue intentandose

    assert p1.calls == ["gemini-3.6-flash"] * 4


def test_a_failed_probe_reopens_the_circuit() -> None:
    clock = FakeClock()
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)] * 3)
    p2 = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")],
        retry=ONE_ATTEMPT,
        breaker=_breaker(clock),
        clock=clock,
    )
    for _ in range(2):
        asyncio.run(harness.run())
    clock.now += 61.0
    asyncio.run(harness.run())  # sondeo (3a llamada real): falla y reabre
    asyncio.run(harness.run())  # de nuevo omitido

    assert len(p1.calls) == 3


def test_when_every_active_tier_is_open_the_chain_still_tries_one() -> None:
    clock = FakeClock()
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA), ok_result()])
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash")],
        retry=ONE_ATTEMPT,
        breaker=_breaker(clock, threshold=1),
        clock=clock,
    )

    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())  # abre el circuito
    entries = asyncio.run(harness.run())  # todo abierto: segunda pasada forzada

    assert len(entries) == 1
    assert len(p1.calls) == 2


def test_an_unexpected_exception_releases_a_reserved_probe() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, threshold=1)
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA), RuntimeError("bug")])
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash")], retry=ONE_ATTEMPT, breaker=breaker, clock=clock
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(harness.run())
    clock.now += 61.0

    with pytest.raises(RuntimeError):
        asyncio.run(harness.run())

    assert breaker.allow(CircuitBreaker.key("gemini", "gemini-3.6-flash")) is True


def test_circuit_breaker_counts_exhausted_requests_not_individual_attempts() -> None:
    """Con max_attempts=2, una solicitud que agota sus 2 intentos debe
    aportar 1 sola falla al cortacircuitos, no 2: chain.py registra la falla
    una vez por tier por solicitud (el ultimo error), no una vez por attempt
    interno. Con el conteo viejo (por attempt) el circuito ya habria abierto
    a mitad de la 2a solicitud y la 3a ni la habria intentado."""
    clock = FakeClock()
    # 3 solicitudes, cada una agota sus 2 intentos en tier 1 = 6 llamadas.
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)] * 6)
    p2 = FakeProvider("gemini")
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")],
        retry=TWO_ATTEMPTS_NO_DELAY,
        breaker=_breaker(clock, threshold=3),
        clock=clock,
    )

    for _ in range(3):
        asyncio.run(harness.run())

    # Las 3 solicitudes intentaron tier 1 de verdad (6 llamadas = 3 x 2
    # intentos): con el umbral contando attempts, ya se habria abierto antes.
    assert p1.calls == ["gemini-3.6-flash"] * 6

    # Recien tras la 3a solicitud agotada se cumple el umbral: la 4a lo salta.
    asyncio.run(harness.run())
    assert p1.calls == ["gemini-3.6-flash"] * 6  # sin llamada nueva
    assert "SALTADO_CIRCUITO" in harness.sink.outcomes


def test_a_multi_attempt_probe_reopens_the_circuit_exactly_once() -> None:
    """Caso borde corregido: durante un sondeo semiabierto, si el tier hace
    mas de 1 intento interno, el circuito debe reabrir con un solo evento
    (el ultimo error de la solicitud), no procesar el sondeo intento por
    intento. Antes, el primer fallo del sondeo ya limpiaba
    `probe_started_at` y un segundo fallo de la misma solicitud caia por la
    rama de conteo normal en vez de la de reapertura."""
    clock = FakeClock()
    # 3 solicitudes agotan tier 1 y abren el circuito (umbral 3); luego el
    # sondeo agota sus 2 intentos y debe reabrir.
    p1 = FakeProvider("gemini", [failure(R.ALTA_DEMANDA)] * (6 + 2))
    p2 = FakeProvider("gemini")
    # cooldown_seconds explicito igual al default real de produccion
    # (VisionSettings.circuit_breaker_cooldown_seconds = 120.0): si el test
    # usara el default de _breaker() (60.0) no detectaria una regresion
    # donde el cooldown de produccion cambia y deja de coincidir.
    breaker = _breaker(clock, threshold=3, cooldown_seconds=120.0)
    harness = ChainHarness(
        [tier(p1, "gemini-3.6-flash"), tier(p2, "gemini-3.8-flash")],
        retry=TWO_ATTEMPTS_NO_DELAY,
        breaker=breaker,
        clock=clock,
    )
    for _ in range(3):
        asyncio.run(harness.run())  # abre el circuito (ver test anterior)

    clock.now += 121.0  # supera el cooldown real de produccion (120.0)
    asyncio.run(harness.run())  # sondeo: agota sus 2 intentos y reabre
    assert p1.calls == ["gemini-3.6-flash"] * 8

    # Reabierto de inmediato (mismo instante): la proxima solicitud, sin
    # esperar el cooldown completo, lo salta.
    asyncio.run(harness.run())
    assert p1.calls == ["gemini-3.6-flash"] * 8  # sin llamada nueva
    assert "SALTADO_CIRCUITO" in harness.sink.outcomes
