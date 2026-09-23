"""Prueba de conexion aislada contra la cadena de vision (lectura de UNA captura).

Standalone: no se integra al bot. Arma la cadena tal cual esta en produccion
(misma config, mismos tiers, mismos timeouts) y la prueba contra una captura
real, para confirmar que la conexion y el schema de salida
funcionan antes de probar el flujo end-to-end con Telegram.

GASTA CUOTA REAL. Por eso, por defecto se NIEGA a correr si la configuracion
permite mas de UNA llamada real en el peor caso (mas de un tier activo, o
reintentos por tier). Para una sola llamada:

    VISION_CHAIN=gemini:gemini-3.6-flash GEMINI_MAX_ATTEMPTS=1 \\
        python scripts/test_gemini_connection.py

`--multi` permite recorrer la cadena completa (varias llamadas posibles).

Uso: python scripts/test_gemini_connection.py [--multi]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.vision.factory import build_fallback_chain
from app.config import get_settings
from app.services.vision.chain import RetryPolicy
from app.services.vision.errors import AllProvidersFailedError, FailureReason
from app.services.vision.models import AttemptOutcome, ProviderAttempt
from app.services.vision.telemetry import AttemptRecorder, real_provider_calls

TEST_IMAGE_PATH = Path(__file__).resolve().parent.parent / "EjemploRutaDrivin.jpeg"


class _Collector:
    """Guarda los intentos y ademas los emite por el registro normal."""

    def __init__(self) -> None:
        self.attempts: list[ProviderAttempt] = []
        self._recorder = AttemptRecorder(None)

    def record(self, attempt: ProviderAttempt) -> None:
        self.attempts.append(attempt)
        self._recorder.record(attempt)


class _HttpCallCounter(logging.Handler):
    """Control cruzado: requests con respuesta HTTP segun el log de
    httpx/httpx2 (`generateContent` de Gemini, `/v1/responses` de OpenAI o
    `/v1/messages` de Anthropic). No ve las llamadas que vencen por timeout:
    el conteo oficial sale de los intentos de la cadena."""

    _LOGGERS = ("httpx", "httpx2")
    _MARKERS = ("generateContent", "/v1/responses", "/v1/messages")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if any(marker in message for marker in self._MARKERS):
            self.count += 1

    def attach(self) -> None:
        for name in self._LOGGERS:
            logging.getLogger(name).addHandler(self)

    def detach(self) -> None:
        for name in self._LOGGERS:
            logging.getLogger(name).removeHandler(self)


def _client_timeout_ms(provider: object) -> object:
    """Timeout HTTP efectivo del cliente del SDK, en ms (lectura defensiva de
    internos: Gemini lo guarda en ms; OpenAI y Anthropic en segundos)."""
    client = getattr(provider, "_client", None)
    try:
        return client._api_client._http_options.timeout  # Gemini
    except AttributeError:
        pass
    seconds = getattr(client, "timeout", None)
    if isinstance(seconds, int | float):
        return round(seconds * 1000)
    return "no disponible"


async def main(allow_multi: bool) -> int:
    settings = get_settings()
    collector = _Collector()
    chain = build_fallback_chain(settings, recorder=collector)

    print("Cadena configurada:")
    for index, tier in enumerate(chain.tiers, start=1):
        state = "activo" if tier.active else f"inactivo ({tier.inactive_reason})"
        print(f"  {index}) {tier.display_name} {tier.spec.model}: {state}")

    worst_case_calls = sum(
        chain.policy.retry_policies.get(t.spec.provider_id, RetryPolicy()).max_attempts
        for t in chain.tiers
        if t.active
    )
    print(f"Peor caso de llamadas reales: {worst_case_calls}")
    if worst_case_calls > 1 and not allow_multi:
        print()
        print("❌ NO SE HIZO NINGUNA LLAMADA. Esta configuracion permite mas de una llamada real")
        print("   en el peor caso. Para UNA sola:")
        print("   VISION_CHAIN=gemini:<modelo> GEMINI_MAX_ATTEMPTS=1")
        print("   (o usa --multi para recorrer la cadena completa).")
        await chain.aclose()
        return 2

    if not TEST_IMAGE_PATH.exists():
        print(f"❌ No se encontro imagen de prueba en {TEST_IMAGE_PATH}")
        await chain.aclose()
        return 1

    first_active = next(t for t in chain.tiers if t.active)
    timeout_ms = _client_timeout_ms(first_active.provider)
    print(f"Timeout HTTP efectivo del cliente: {timeout_ms} ms")
    image_bytes = TEST_IMAGE_PATH.read_bytes()
    print(f"Imagen de prueba: {TEST_IMAGE_PATH.name} ({len(image_bytes)} bytes)")
    print()

    counter = _HttpCallCounter()
    counter.attach()

    entries: list = []
    failure_text: str | None = None
    start = time.perf_counter()
    try:
        entries = await chain.extract_entries(
            images=[image_bytes], mime_type="image/jpeg", screenshot_ids=["test-1"]
        )
    except AllProvidersFailedError as exc:
        failure_text = str(exc)
    except Exception as exc:
        failure_text = f"Fallo inesperado: {type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - start
        counter.detach()
        await chain.aclose()

    print()
    print("=" * 70)
    print("RESULTADO")
    real_calls = real_provider_calls(collector.attempts)
    print(f"Llamadas reales al proveedor: {real_calls}")
    if counter.count != real_calls:
        print(
            f"  control cruzado: {counter.count} con respuesta HTTP en el log; "
            f"la diferencia son llamadas sin respuesta (timeout o error de red)"
        )
    print(f"Tiempo total: {elapsed:.2f} s (timeout configurado: {timeout_ms} ms)")
    for a in collector.attempts:
        if a.outcome in (AttemptOutcome.EXITO, AttemptOutcome.FALLO):
            usage = a.usage
            print(
                f"  intento: tier {a.tier} {a.display_name} {a.model} -> {a.outcome} "
                f"reason={a.reason} http={a.http_status} code={a.provider_code} "
                f"latency={a.latency_ms} ms"
            )
            if usage is not None:
                print(
                    f"           finish_reason={usage.finish_reason} "
                    f"prompt_tokens={usage.prompt_tokens} output_tokens={usage.output_tokens} "
                    f"reasoning_tokens={usage.reasoning_tokens}"
                )
            if a.detail:
                print(f"           detalle del proveedor (truncado): {a.detail}")

    timed_out = any(a.reason is FailureReason.TIMEOUT for a in collector.attempts)
    print(f"Se alcanzo el timeout: {'SI' if timed_out else 'no'}")

    if failure_text is not None:
        print(f"❌ La cadena no obtuvo resultado: {failure_text}")
        return 1

    winner = next(a for a in collector.attempts if a.outcome is AttemptOutcome.EXITO)
    print(f"Respondio: {winner.display_name} {winner.model}")
    print(f"JSON valido segun el schema: SI ({len(entries)} entradas)")
    print()
    print("Entradas extraidas:")
    for e in entries:
        print(
            f"  orden={e.order} | {e.address} | comuna={e.locality} | bulto={e.package_id} | "
            f"eta={e.eta} | ventana={e.time_window} | tipo={e.delivery_type} | "
            f"estado={e.status} | fuente={e.source_screenshot_id}"
        )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(asyncio.run(main("--multi" in sys.argv)))
