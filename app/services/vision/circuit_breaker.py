"""Cortacircuitos por tier de la cadena.

Problema que ataca: si un modelo lleva horas saturado, cada solicitud
gastaria ~65-80 s en esos tiers antes de llegar al siguiente. Tras N fallas
transitorias seguidas, el tier se omite durante un enfriamiento y luego se
reprueba con UNA sola llamada de sondeo (estado semiabierto): si responde, el
circuito se cierra; si falla, se reabre.

Solo cuentan las fallas transitorias del proveedor (`ALTA_DEMANDA`,
`TIMEOUT`, `CONEXION`). Un 404, una key mala o un JSON invalido no indican
saturacion y no abren el circuito.

El estado vive en memoria (se pierde al reiniciar) y el reloj es inyectable
para probarlo sin esperas.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from app.services.vision.errors import FailureReason

DEFAULT_COUNTED_REASONS: frozenset[FailureReason] = frozenset(
    {FailureReason.ALTA_DEMANDA, FailureReason.TIMEOUT, FailureReason.CONEXION}
)


@dataclass(slots=True)
class _Circuit:
    failures: int = 0
    opened_at: float | None = None
    # No es None mientras hay una llamada de sondeo en vuelo (semiabierto).
    probe_started_at: float | None = None


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 120.0,
        counted_reasons: frozenset[FailureReason] = DEFAULT_COUNTED_REASONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold debe ser >= 1")
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._counted = counted_reasons
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}

    @staticmethod
    def key(provider_id: str, model: str) -> str:
        return f"{provider_id}:{model}"

    def allow(self, key: str) -> bool:
        """True si se puede intentar este tier ahora. Si el enfriamiento ya
        vencio, reserva la unica llamada de sondeo (estado semiabierto)."""
        circuit = self._circuits.get(key)
        if circuit is None or circuit.opened_at is None:
            return True
        now = self._clock()
        if circuit.probe_started_at is not None:
            if now - circuit.probe_started_at < self._cooldown:
                return False  # ya hay un sondeo en vuelo
            circuit.probe_started_at = now  # sondeo abandonado: reintenta
            return True
        if now - circuit.opened_at >= self._cooldown:
            circuit.probe_started_at = now
            return True
        return False

    def record_success(self, key: str) -> None:
        self._circuits.pop(key, None)

    def record_failure(self, key: str, reason: FailureReason) -> None:
        circuit = self._circuits.setdefault(key, _Circuit())
        probing = circuit.probe_started_at is not None
        if reason not in self._counted:
            if probing:
                circuit.probe_started_at = None  # el sondeo no fue concluyente
            return
        now = self._clock()
        if probing:
            circuit.opened_at = now  # el sondeo fallo: reabre el enfriamiento
            circuit.probe_started_at = None
            return
        circuit.failures += 1
        if circuit.failures >= self._threshold and circuit.opened_at is None:
            circuit.opened_at = now

    def release_probe(self, key: str) -> None:
        """Libera un sondeo reservado que no llego a registrar resultado
        (p. ej. la tarea fue cancelada)."""
        circuit = self._circuits.get(key)
        if circuit is not None:
            circuit.probe_started_at = None

    def is_open(self, key: str) -> bool:
        circuit = self._circuits.get(key)
        return circuit is not None and circuit.opened_at is not None

    def seconds_until_probe(self, key: str) -> float:
        circuit = self._circuits.get(key)
        if circuit is None or circuit.opened_at is None:
            return 0.0
        return max(0.0, circuit.opened_at + self._cooldown - self._clock())
