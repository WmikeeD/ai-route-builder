"""Motor de deduplicacion y agrupacion de paradas.

Recibe las filas crudas extraidas por Gemini desde una o varias capturas de
la aplicacion de origen (posiblemente con filas repetidas por scroll superpuesto) y produce
la lista final de Stops: una parada por direccion unica (calle + comuna),
con todos los paquetes que corresponden a esa direccion.

Regla de negocio: el package_id es unico por paquete, pero una misma
direccion puede recibir paquetes de clientes distintos. Por eso la llave de
agrupacion de una Stop es la direccion normalizada (incluyendo comuna, para
no fusionar calles homonimas de comunas distintas) y NO el package_id; una
fila solo se descarta por duplicada cuando coincide en direccion Y en
package_id (o, si el package_id no vino en la captura, cuando la fila es
identica en el resto de sus campos) con una fila ya vista.

Capa de dominio pura: solo usa la libreria estandar de Python y las
entidades de app.domain.models.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from app.domain.models import Package, RawDeliveryEntry, Route, Stop

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[.,;:#°ºª]")

_DuplicateKey = tuple[str, str]
_ExactRowKey = tuple[str, str, str | None, str | None]


def normalize_text(value: str) -> str:
    """Minusculas, sin acentos, sin puntuacion, espacios colapsados."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    without_punctuation = _PUNCTUATION_RE.sub(" ", without_accents)
    collapsed = _WHITESPACE_RE.sub(" ", without_punctuation)
    return collapsed.strip().lower()


def normalize_address(address: str, locality: str | None = None) -> str:
    """Normaliza calle + comuna para usarlas como llave de agrupacion de
    Stops. Se incluye la comuna para no fusionar calles homonimas de
    comunas distintas (ej. dos "Panamericana Norte 1500" en sectores
    diferentes de Santiago)."""
    normalized = normalize_text(address)
    if locality:
        normalized = f"{normalized} | {normalize_text(locality)}"
    return normalized


def _duplicate_key(entry: RawDeliveryEntry, address_key: str) -> _DuplicateKey | None:
    """Llave (direccion, package_id) para detectar la misma fila repetida
    entre capturas. Devuelve None si no hay package_id: sin el, no se puede
    afirmar con certeza que dos filas sean el mismo paquete, y tratarlas
    como iguales arriesgaria perder un paquete real."""
    if not entry.package_id:
        return None
    return (address_key, entry.package_id.strip().lower())


def _exact_row_key(entry: RawDeliveryEntry, address_key: str) -> _ExactRowKey:
    customer_key = normalize_text(entry.customer_name) if entry.customer_name else ""
    return (address_key, customer_key, entry.eta, entry.time_window)


def _merge_duplicate(existing: RawDeliveryEntry, incoming: RawDeliveryEntry) -> RawDeliveryEntry:
    """Combina dos filas del mismo paquete visto en mas de una captura,
    completando campos vacios con el dato disponible en la otra fila."""
    return existing.model_copy(
        update={
            "order": existing.order if existing.order is not None else incoming.order,
            "customer_name": existing.customer_name or incoming.customer_name,
            "delivery_type": existing.delivery_type or incoming.delivery_type,
            "eta": existing.eta or incoming.eta,
            "time_window": existing.time_window or incoming.time_window,
            "source_screenshot_id": existing.source_screenshot_id or incoming.source_screenshot_id,
        }
    )


def deduplicate_entries(entries: list[RawDeliveryEntry]) -> list[RawDeliveryEntry]:
    """Elimina filas duplicadas (mismo paquete visto en mas de una captura),
    preservando el orden de primera aparicion."""
    index_by_key: dict[_DuplicateKey, int] = {}
    seen_exact_rows: set[_ExactRowKey] = set()
    result: list[RawDeliveryEntry] = []

    for entry in entries:
        address_key = normalize_address(entry.address, entry.locality)
        key = _duplicate_key(entry, address_key)

        if key is not None:
            existing_index = index_by_key.get(key)
            if existing_index is None:
                index_by_key[key] = len(result)
                result.append(entry)
            else:
                result[existing_index] = _merge_duplicate(result[existing_index], entry)
            continue

        exact_key = _exact_row_key(entry, address_key)
        if exact_key in seen_exact_rows:
            continue
        seen_exact_rows.add(exact_key)
        result.append(entry)

    return result


def group_into_stops(entries: list[RawDeliveryEntry]) -> list[Stop]:
    """Agrupa filas ya deduplicadas en Stops por direccion normalizada."""
    packages_by_group: dict[str, list[Package]] = {}
    address_by_group: dict[str, str] = {}
    locality_by_group: dict[str, str | None] = {}
    order_by_group: dict[str, int | None] = {}

    for entry in entries:
        group_key = normalize_address(entry.address, entry.locality)

        if group_key not in packages_by_group:
            packages_by_group[group_key] = []
            address_by_group[group_key] = entry.address
            locality_by_group[group_key] = entry.locality
            order_by_group[group_key] = entry.order
        elif entry.order is not None:
            current_order = order_by_group[group_key]
            if current_order is None or entry.order < current_order:
                order_by_group[group_key] = entry.order

        packages_by_group[group_key].append(
            Package(
                package_id=entry.package_id,
                customer_name=entry.customer_name,
                delivery_type=entry.delivery_type,
                eta=entry.eta,
                time_window=entry.time_window,
                status=entry.status,
                order=entry.order,
                source_screenshot_id=entry.source_screenshot_id,
            )
        )

    stops = [
        Stop(
            address=address_by_group[group_key],
            locality=locality_by_group[group_key],
            normalized_address=group_key,
            order=order_by_group[group_key],
            packages=tuple(
                sorted(packages, key=lambda pkg: (pkg.order is None, pkg.order or 0))
            ),
        )
        for group_key, packages in packages_by_group.items()
    ]

    stops.sort(key=lambda stop: (stop.order is None, stop.order or 0))
    return stops


def deduplicate_and_group(entries: list[RawDeliveryEntry]) -> list[Stop]:
    """Punto de entrada del motor: deduplica filas y las agrupa en Stops."""
    return group_into_stops(deduplicate_entries(entries))


def build_route(
    route_id: str,
    entries: list[RawDeliveryEntry],
    driver_name: str | None = None,
    route_date: date | None = None,
) -> Route:
    """Construye una Route completa a partir de las filas crudas extraidas."""
    stops = deduplicate_and_group(entries)
    return Route(
        route_id=route_id,
        driver_name=driver_name,
        route_date=route_date,
        stops=tuple(stops),
    )
