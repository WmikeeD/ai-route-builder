"""Tests de la telemetria: formato canonico de las lineas, niveles y JSONL."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.services.vision.errors import FailureReason, LimitKind
from app.services.vision.models import AttemptOutcome, ProviderAttempt, UsageInfo
from app.services.vision.telemetry import AttemptRecorder, format_attempt, level_for

R = FailureReason


def _attempt(**overrides: object) -> ProviderAttempt:
    base = ProviderAttempt(
        ts=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
        chain_request_id="abc123",
        tier=2,
        tiers_total=4,
        provider="gemini",
        display_name="Gemini",
        model="gemini-3.8-flash",
        outcome=AttemptOutcome.FALLO,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_high_demand_line_names_reason_provider_and_model() -> None:
    line = format_attempt(
        _attempt(
            reason=R.ALTA_DEMANDA,
            http_status=503,
            provider_code="UNAVAILABLE",
            latency_ms=20400,
        )
    )

    assert line == (
        "ALTA_DEMANDA en Gemini (gemini-3.8-flash) [tier 2/4] "
        "http=503 code=UNAVAILABLE latency=20.4s"
    )


def test_limit_line_carries_the_kind() -> None:
    line = format_attempt(
        _attempt(
            tier=3,
            provider="openai",
            display_name="OpenAI",
            model="gpt-5.6-terra",
            reason=R.LIMITE_ALCANZADO,
            limit_kind=LimitKind.CREDITO_O_GASTO,
            http_status=429,
            provider_code="credit_balance_exhausted",
        )
    )

    assert line == (
        "LIMITE_ALCANZADO en OpenAI (gpt-5.6-terra) [tier 3/4] "
        "kind=CREDITO_O_GASTO http=429 code=credit_balance_exhausted"
    )


def test_success_line_includes_usage() -> None:
    line = format_attempt(
        _attempt(
            tier=1,
            outcome=AttemptOutcome.EXITO,
            latency_ms=1500,
            usage=UsageInfo(prompt_tokens=10, output_tokens=20, finish_reason="STOP"),
        )
    )

    assert line == (
        "EXITO en Gemini (gemini-3.8-flash) [tier 1/4] finish_reason=STOP "
        "prompt_tokens=10 output_tokens=20 latency=1.5s"
    )


def test_inactive_tier_line_is_the_agreed_text() -> None:
    line = format_attempt(
        _attempt(tier=4, provider="anthropic", display_name="Anthropic",
                 outcome=AttemptOutcome.SALTADO_INACTIVO)
    )

    assert line == "tier 4 (Anthropic) no configurado, se omite"


def test_circuit_skip_line_says_when_the_probe_comes() -> None:
    line = format_attempt(
        _attempt(outcome=AttemptOutcome.SALTADO_CIRCUITO, detail="reintento en 42s")
    )

    assert "omitido: cortacircuitos abierto (reintento en 42s)" in line


def test_retry_counter_only_appears_when_there_are_several_attempts() -> None:
    once = format_attempt(_attempt(reason=R.TIMEOUT))
    retried = format_attempt(_attempt(reason=R.TIMEOUT, attempt=2, max_attempts=3))

    assert "attempt=" not in once
    assert "attempt=2/3" in retried


@pytest.mark.parametrize(
    ("overrides", "level"),
    [
        ({"reason": R.ALTA_DEMANDA}, logging.WARNING),
        ({"reason": R.LIMITE_ALCANZADO, "limit_kind": LimitKind.RATE_MINUTO}, logging.WARNING),
        ({"reason": R.LIMITE_ALCANZADO, "limit_kind": LimitKind.CREDITO_O_GASTO}, logging.ERROR),
        ({"reason": R.MODELO_NO_ENCONTRADO}, logging.ERROR),
        ({"reason": R.AUTENTICACION}, logging.ERROR),
        ({"reason": R.SOLICITUD_INVALIDA}, logging.ERROR),
        ({"reason": R.RESPUESTA_INVALIDA}, logging.WARNING),
        ({"reason": R.TIMEOUT}, logging.WARNING),
        ({"reason": R.CONEXION}, logging.WARNING),
        ({"outcome": AttemptOutcome.EXITO}, logging.INFO),
        ({"outcome": AttemptOutcome.SALTADO_INACTIVO}, logging.INFO),
    ],
)
def test_log_level_follows_the_cause(overrides: dict[str, object], level: int) -> None:
    assert level_for(_attempt(**overrides)) == level


def test_record_emits_a_readable_line_and_a_json_line(caplog: pytest.LogCaptureFixture) -> None:
    recorder = AttemptRecorder(None)

    with caplog.at_level(logging.INFO, logger="vision.attempts"):
        recorder.record(_attempt(reason=R.ALTA_DEMANDA, http_status=503))

    by_logger = {r.name: r for r in caplog.records}
    assert by_logger["vision.attempts"].levelno == logging.WARNING
    assert by_logger["vision.attempts"].getMessage().startswith("ALTA_DEMANDA en Gemini")
    payload = json.loads(by_logger["vision.attempts.json"].getMessage())
    assert payload["reason"] == "ALTA_DEMANDA"
    assert payload["provider"] == "gemini"
    assert payload["model"] == "gemini-3.8-flash"


def test_jsonl_file_gets_one_parseable_line_per_attempt(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "vision_attempts.jsonl"
    recorder = AttemptRecorder(path)
    try:
        recorder.record(_attempt(reason=R.ALTA_DEMANDA, http_status=503, latency_ms=20400))
        recorder.record(_attempt(tier=3, outcome=AttemptOutcome.EXITO,
                                 usage=UsageInfo(prompt_tokens=7, finish_reason="STOP")))
    finally:
        recorder.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    first, second = (json.loads(line) for line in lines)
    assert first["reason"] == "ALTA_DEMANDA"
    assert (first["http_status"], first["latency_ms"]) == (503, 20400)
    assert second["outcome"] == "EXITO"
    assert (second["prompt_tokens"], second["finish_reason"]) == (7, "STOP")


def test_records_never_carry_capture_content() -> None:
    record = _attempt(reason=R.ALTA_DEMANDA, images=15, payload_bytes=2442041).to_record()

    forbidden = {"address", "customer_name", "package_id", "entries", "images_data"}
    assert forbidden.isdisjoint(record)
    assert record["images"] == 15
    assert record["payload_bytes"] == 2442041


def test_a_broken_file_sink_never_breaks_recording(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")

    with caplog.at_level(logging.INFO, logger="vision.attempts"):
        recorder = AttemptRecorder(blocker / "attempts.jsonl")  # no se puede crear
        recorder.record(_attempt(reason=R.TIMEOUT))  # no lanza

    assert any(r.name == "vision.attempts" for r in caplog.records)


def test_quota_skip_line_names_the_kind_and_the_reset() -> None:
    attempt = _attempt(
        outcome=AttemptOutcome.SALTADO_CUOTA,
        reason=R.LIMITE_ALCANZADO,
        limit_kind=LimitKind.CUOTA_DIARIA,
        detail="cuota diaria agotada hasta 2026-09-22T07:00+00:00",
    )

    assert format_attempt(attempt) == (
        "tier 2 (Gemini, gemini-3.8-flash) omitido por LIMITE_ALCANZADO kind=CUOTA_DIARIA: "
        "cuota diaria agotada hasta 2026-09-22T07:00+00:00"
    )
    assert level_for(attempt) == logging.INFO
