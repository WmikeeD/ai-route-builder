"""Bloqueo de tiers por limites propios que un reintento NO arregla.

Es un mecanismo SEPARADO del cortacircuitos (`circuit_breaker.py`), con otra
razon de ser y otro ciclo de vida:

* El cortacircuitos (120 s) atiende fallas TRANSITORIAS del proveedor
  (`ALTA_DEMANDA`, `TIMEOUT`, `CONEXION`): se apaga solo tras un enfriamiento
  corto y se reprueba con una llamada de sondeo.
* Este guardia atiende dos causas de `LIMITE_ALCANZADO` que, sin bloqueo, se
  seguirian intentando en cada solicitud durante horas sin ningun ahorro:

  - `CUOTA_DIARIA`: el tier (proveedor + modelo, la cuota diaria es por
    modelo) se salta hasta el proximo reset conocido. Gemini: medianoche hora
    del Pacifico (documentado: los RPD se reinician a esa hora). Proveedores
    sin reset predecible documentado: una duracion fija configurable.
  - `CREDITO_O_GASTO`: se salta el PROVEEDOR completo (el credito y el tope
    de gasto son de la cuenta, no del modelo) hasta que se reinicie el
    proceso. Se asume que requiere intervencion humana: no tiene sentido
    reintentar solo.

Un bloqueo de cuota diaria que vence deja pasar la siguiente solicitud, que
hace de sondeo natural: si vuelve a fallar por cuota, se bloquea otra vez
hasta el siguiente reset; si responde, el bloqueo queda limpio.

El estado vive en memoria (se pierde al reiniciar, que es justo cuando el
bloqueo por credito debe levantarse). Usa hora de calendario (UTC), no el
reloj monotono de la cadena, porque el reset es una hora del dia. La fuente
de tiempo es inyectable para probarlo sin esperas.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from app.services.vision.errors import FailureReason, LimitKind, ProviderError

# Documentado por Google: "Requests per day (RPD) quotas reset at midnight
# Pacific time" (guia de rate limits de la API de Gemini).
PACIFIC_TZ = "America/Los_Angeles"

ResetPolicy = Callable[[datetime], datetime]
Now = Callable[[], datetime]


def next_pacific_midnight(now: datetime) -> datetime:
    """Proxima medianoche hora del Pacifico (PST/PDT), como instante UTC.

    Respeta el horario de verano: los dias de cambio duran 23 o 25 horas.
    """
    pacific = ZoneInfo(PACIFIC_TZ)
    local = now.astimezone(pacific)
    next_day = (local + timedelta(days=1)).date()
    return datetime(next_day.year, next_day.month, next_day.day, tzinfo=pacific).astimezone(UTC)


def fixed_duration(seconds: float) -> ResetPolicy:
    """Reset por duracion fija, para proveedores sin reset predecible documentado."""
    return lambda now: now + timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class QuotaBlock:
    """Un bloqueo vigente (o recien registrado)."""

    kind: LimitKind
    # None = indefinido: dura hasta que se reinicie el proceso.
    until: datetime | None
    # "proveedor" (credito) o "proveedor:modelo" (cuota diaria).
    scope: str

    def describe(self) -> str:
        """Texto para logs y para el detalle del intento omitido."""
        if self.until is None:
            return (
                "credito o gasto agotado: requiere intervencion humana "
                "(se levanta al reiniciar el proceso)"
            )
        return f"cuota diaria agotada hasta {self.until.isoformat(timespec='minutes')}"


class QuotaGuard:
    def __init__(
        self,
        *,
        reset_policies: Mapping[str, ResetPolicy],
        default_policy: ResetPolicy,
        now: Now = lambda: datetime.now(UTC),
    ) -> None:
        self._reset_policies = dict(reset_policies)
        self._default_policy = default_policy
        self._now = now
        # Credito/gasto: por proveedor, indefinido.
        self._credit: dict[str, QuotaBlock] = {}
        # Cuota diaria: por (proveedor, modelo), hasta un instante.
        self._daily: dict[tuple[str, str], QuotaBlock] = {}

    def block_for(self, provider_id: str, model: str) -> QuotaBlock | None:
        """El bloqueo vigente que afecta a este tier, o None si puede intentarse."""
        credit = self._credit.get(provider_id)
        if credit is not None:
            return credit
        key = (provider_id, model)
        daily = self._daily.get(key)
        if daily is None:
            return None
        if daily.until is not None and daily.until <= self._now():
            del self._daily[key]  # vencio: la proxima solicitud hace de sondeo
            return None
        return daily

    def record_failure(self, error: ProviderError) -> QuotaBlock | None:
        """Registra un bloqueo si la falla es de cuota diaria o de credito.

        Devuelve el bloqueo NUEVO (para loguearlo una sola vez) o None si la
        falla no bloquea o el proveedor ya estaba bloqueado por credito.
        """
        if error.reason is not FailureReason.LIMITE_ALCANZADO:
            return None
        if error.limit_kind is LimitKind.CREDITO_O_GASTO:
            if error.provider in self._credit:
                return None
            block = QuotaBlock(LimitKind.CREDITO_O_GASTO, None, error.provider)
            self._credit[error.provider] = block
            return block
        if error.limit_kind is LimitKind.CUOTA_DIARIA:
            policy = self._reset_policies.get(error.provider, self._default_policy)
            block = QuotaBlock(
                LimitKind.CUOTA_DIARIA,
                policy(self._now()),
                f"{error.provider}:{error.model}",
            )
            self._daily[(error.provider, error.model)] = block
            return block
        return None  # RATE_MINUTO / DESCONOCIDO: no bloquean

    def record_success(self, provider_id: str, model: str) -> None:
        """Una respuesta exitosa limpia un bloqueo diario que ya habia vencido."""
        self._daily.pop((provider_id, model), None)

    def active_blocks(self) -> list[QuotaBlock]:
        """Bloqueos vigentes (para diagnostico)."""
        blocks = list(self._credit.values())
        for provider_id, model in list(self._daily):
            block = self.block_for(provider_id, model)
            if block is not None:
                blocks.append(block)
        return blocks
