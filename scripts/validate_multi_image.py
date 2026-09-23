"""Validacion de extraccion con VARIAS capturas, igual que el bot (un solo request).

Standalone: no se integra al bot. Arma la cadena tal cual esta en produccion
(misma config, tiers, timeouts y prompt), envia TODAS las capturas en una sola
extraccion, como hace "Procesar Ruta", y reporta:

- las tarjetas extraidas por captura (para ver exactamente cuales se omiten),
- las paradas finales tras deduplicar y agrupar (el numero que llega al PDF),
- las llamadas reales al proveedor.

Las llamadas reales se cuentan por los INTENTOS que registra la cadena (exito o
fallo), no por las lineas de log de httpx: una llamada que vence por timeout no
deja ninguna linea de log, y los SDK de OpenAI y Anthropic loguean por `httpx2`.
El conteo por logs HTTP queda solo como control cruzado.

GASTA CUOTA REAL. Sin `--max-calls` NO hace ninguna llamada: solo muestra la
cadena y el peor caso de llamadas reales. Con `--max-calls N` corre solo si ese
peor caso es <= N. Para limitarlo a un tier:

    VISION_CHAIN=gemini:gemini-3.6-flash python scripts/validate_multi_image.py \\
        --max-calls 1 ../captura1.jpg ../captura2.jpg ...

Uso: python scripts/validate_multi_image.py [--max-calls N] [--expected N] IMAGEN...
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.vision.factory import build_fallback_chain
from app.config import get_settings
from app.domain.route_engine import deduplicate_and_group
from app.logging_setup import install_redaction_filter
from app.services.vision.chain import RetryPolicy
from app.services.vision.errors import AllProvidersFailedError
from app.services.vision.fault_injection import FAULT_MESSAGE_PREFIX
from app.services.vision.models import AttemptOutcome, ProviderAttempt
from app.services.vision.telemetry import AttemptRecorder

_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


class _Collector:
    """Guarda los intentos y ademas los emite por el registro normal."""

    def __init__(self) -> None:
        self.attempts: list[ProviderAttempt] = []
        self._recorder = AttemptRecorder(None)

    def record(self, attempt: ProviderAttempt) -> None:
        self.attempts.append(attempt)
        self._recorder.record(attempt)


def real_provider_calls(attempts: list[ProviderAttempt]) -> int:
    """Llamadas que llegaron al proveedor real: intentos con exito o fallo,
    incluidos los que vencieron por timeout. Excluye los tiers saltados
    (inactivo, circuito, cuota) y las fallas forzadas por FORCE_VISION_ERROR,
    que nunca salen del proceso."""
    return sum(
        1
        for a in attempts
        if a.outcome in (AttemptOutcome.EXITO, AttemptOutcome.FALLO)
        and not (a.detail or "").startswith(FAULT_MESSAGE_PREFIX)
    )


class _HttpCallCounter(logging.Handler):
    """Control cruzado: requests con respuesta HTTP segun el log INFO de
    `httpx` (Gemini) y `httpx2` (OpenAI, Anthropic). No ve los timeouts."""

    LOGGERS = ("httpx", "httpx2")
    _MARKERS = ("generateContent", "/v1/responses", "/v1/messages")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if any(marker in record.getMessage() for marker in self._MARKERS):
            self.count += 1


def _load_images(paths: list[Path]) -> tuple[list[bytes], str]:
    suffixes = {p.suffix.lower() for p in paths}
    mimes = {_MIME_BY_SUFFIX.get(s) for s in suffixes}
    if len(mimes) != 1 or None in mimes:
        raise SystemExit(f"Todas las imagenes deben ser del mismo tipo (jpg o png): {suffixes}")
    return [p.read_bytes() for p in paths], mimes.pop() or "image/jpeg"


async def main(paths: list[Path], max_calls: int | None, expected: int | None) -> int:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"❌ No existen: {', '.join(str(p) for p in missing)}")
        return 1

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
    print(f"Capturas: {len(paths)} (en UNA sola extraccion, como el bot)")
    print(f"Peor caso de llamadas reales: {worst_case_calls}")

    if max_calls is None or worst_case_calls > max_calls:
        print()
        if max_calls is None:
            print("NO SE HIZO NINGUNA LLAMADA (sin --max-calls: solo diagnostico).")
        else:
            print(f"❌ NO SE HIZO NINGUNA LLAMADA: el peor caso ({worst_case_calls}) supera")
            print(f"   --max-calls {max_calls}. Limita la cadena con VISION_CHAIN o sube el tope.")
        await chain.aclose()
        return 2

    images, mime_type = _load_images(paths)
    ids = [p.name for p in paths]
    print(f"Tamano total: {sum(len(i) for i in images):,} bytes ({mime_type})")
    print()

    counter = _HttpCallCounter()
    for name in counter.LOGGERS:
        logging.getLogger(name).addHandler(counter)
    entries: list = []
    failure_text: str | None = None
    start = time.perf_counter()
    try:
        entries = await chain.extract_entries(
            images=images, mime_type=mime_type, screenshot_ids=ids
        )
    except AllProvidersFailedError as exc:
        failure_text = str(exc)
    except Exception as exc:
        failure_text = f"Fallo inesperado: {type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - start
        for name in counter.LOGGERS:
            logging.getLogger(name).removeHandler(counter)
        await chain.aclose()

    print()
    print("=" * 70)
    print("RESULTADO")
    real_calls = real_provider_calls(collector.attempts)
    print(f"Llamadas reales al proveedor: {real_calls} (tope autorizado: {max_calls})")
    if counter.count != real_calls:
        print(
            f"  control cruzado: {counter.count} con respuesta HTTP en el log; "
            f"la diferencia son llamadas sin respuesta (timeout o error de red)"
        )
    print(f"Tiempo total: {elapsed:.2f} s")
    for a in collector.attempts:
        if a.outcome in (AttemptOutcome.EXITO, AttemptOutcome.FALLO):
            print(
                f"  intento: tier {a.tier} {a.display_name} {a.model} -> {a.outcome} "
                f"reason={a.reason} http={a.http_status} latency={a.latency_ms} ms"
            )
    if failure_text is not None:
        print(f"❌ La cadena no obtuvo resultado: {failure_text}")
        return 1

    winner = next(a for a in collector.attempts if a.outcome is AttemptOutcome.EXITO)
    print(f"Respondio: {winner.display_name} {winner.model}")
    print()

    per_image = Counter(e.source_screenshot_id for e in entries)
    print(f"Tarjetas extraidas: {len(entries)}")
    for name in ids:
        print(f"\n  [{name}] {per_image.get(name, 0)} tarjeta(s)")
        for e in (e for e in entries if e.source_screenshot_id == name):
            print(
                f"    {e.address} | comuna={e.locality} | ot={e.package_id} | "
                f"eta={e.eta} | ventana={e.time_window} | tipo={e.delivery_type}"
            )
    unknown = [e for e in entries if e.source_screenshot_id not in ids]
    if unknown:
        print(f"\n  ⚠️  {len(unknown)} tarjeta(s) sin captura de origen reconocible")

    stops = deduplicate_and_group(entries)
    print()
    print(f"PARADAS FINALES (tras deduplicar y agrupar, lo que va al PDF): {len(stops)}")
    if expected is not None:
        verdict = "✅ coincide" if len(stops) == expected else "❌ NO coincide"
        print(f"Esperadas: {expected} -> {verdict}")
    return 0 if expected is None or len(stops) == expected else 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", type=Path, help="capturas, en el orden de envio")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="tope de llamadas reales autorizadas; sin esto no se hace ninguna llamada",
    )
    parser.add_argument("--expected", type=int, default=None, help="paradas finales esperadas")
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    install_redaction_filter()
    raise SystemExit(asyncio.run(main(args.images, args.max_calls, args.expected)))
