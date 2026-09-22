"""Tests de los tipos canonicos de falla y de la causa dominante."""

from __future__ import annotations

import pytest

from app.services.vision.errors import (
    AllProvidersFailedError,
    FailureReason,
    LimitKind,
    ProviderError,
    dominant_failure,
)
from tests.vision_helpers import failure

R = FailureReason
RATE = LimitKind.RATE_MINUTO


def test_a_limit_without_kind_defaults_to_unknown_kind() -> None:
    error = ProviderError(R.LIMITE_ALCANZADO, provider="openai", model="gpt-5.6-terra")

    assert error.limit_kind is LimitKind.DESCONOCIDO
    assert str(error) == "LIMITE_ALCANZADO en openai (gpt-5.6-terra)"


def test_limit_kind_is_dropped_for_reasons_that_are_not_limits() -> None:
    error = ProviderError(
        R.ALTA_DEMANDA, provider="gemini", model="m", limit_kind=LimitKind.RATE_MINUTO
    )

    assert error.limit_kind is None


def test_long_provider_messages_are_truncated() -> None:
    error = ProviderError(R.DESCONOCIDO, provider="gemini", model="m", message="x" * 5000)

    assert len(error.message) == 300


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        ([], (R.DESCONOCIDO, None)),
        ([failure(R.ALTA_DEMANDA)], (R.ALTA_DEMANDA, None)),
        ([failure(R.TIMEOUT), failure(R.ALTA_DEMANDA)], (R.ALTA_DEMANDA, None)),
        ([failure(R.MODELO_NO_ENCONTRADO), failure(R.TIMEOUT)], (R.TIMEOUT, None)),
        (
            [failure(R.ALTA_DEMANDA), failure(R.LIMITE_ALCANZADO, limit_kind=RATE)],
            (R.LIMITE_ALCANZADO, RATE),
        ),
        (
            [
                failure(R.LIMITE_ALCANZADO, limit_kind=RATE),
                failure(R.LIMITE_ALCANZADO, limit_kind=LimitKind.CUOTA_DIARIA),
                failure(R.LIMITE_ALCANZADO, limit_kind=LimitKind.CREDITO_O_GASTO),
            ],
            (R.LIMITE_ALCANZADO, LimitKind.CREDITO_O_GASTO),
        ),
        (
            [failure(R.LIMITE_ALCANZADO), failure(R.LIMITE_ALCANZADO, limit_kind=RATE)],
            (R.LIMITE_ALCANZADO, RATE),
        ),
    ],
)
def test_dominant_failure_priority(
    failures: list[ProviderError], expected: tuple[FailureReason, LimitKind | None]
) -> None:
    assert dominant_failure(failures) == expected


def test_all_providers_failed_error_describes_the_cause() -> None:
    error = AllProvidersFailedError(
        (), dominant_reason=R.LIMITE_ALCANZADO, dominant_limit_kind=LimitKind.CUOTA_DIARIA
    )

    assert "LIMITE_ALCANZADO/CUOTA_DIARIA" in str(error)
    assert error.aborted_by is None
    assert error.attempts == ()
