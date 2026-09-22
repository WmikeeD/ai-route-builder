"""Tests del generador de PDF de rutas (Paso 3)."""

from __future__ import annotations

from app.domain.models import DeliveryStatus, RawDeliveryEntry, Route
from app.domain.route_engine import build_route
from app.services.pdf_generator import (
    DEFAULT_COUNTRY,
    DEFAULT_REGION,
    _build_styles,
    _stop_row,
    generate_route_pdf,
)


def _assert_is_valid_pdf(pdf_bytes: bytes) -> None:
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")
    assert len(pdf_bytes) > 0


def test_generate_route_pdf_returns_valid_pdf_bytes() -> None:
    entries = [
        RawDeliveryEntry(
            order=54,
            package_id="696733800652",
            address="Panamericana Norte 8000",
            locality="Quilicura",
            time_window="07:00 - 21:00",
        ),
    ]
    route = build_route("RUTA-1", entries, driver_name="Carlos Soto")

    pdf_bytes = generate_route_pdf(route)

    _assert_is_valid_pdf(pdf_bytes)


def test_generate_route_pdf_empty_route_does_not_crash() -> None:
    route = build_route("RUTA-EMPTY", [])

    pdf_bytes = generate_route_pdf(route)

    _assert_is_valid_pdf(pdf_bytes)


def test_generate_route_pdf_multiple_packages_same_stop() -> None:
    entries = [
        RawDeliveryEntry(order=56, package_id="A1", address="Pasaje Colbun 1584", locality="Renca"),
        RawDeliveryEntry(order=56, package_id="A2", address="Pasaje Colbun 1584", locality="Renca"),
    ]
    route = build_route("RUTA-2", entries)

    pdf_bytes = generate_route_pdf(route)

    _assert_is_valid_pdf(pdf_bytes)


def test_generate_route_pdf_escapes_special_characters_without_crashing() -> None:
    entries = [
        RawDeliveryEntry(
            order=1,
            package_id="A1",
            address='Pasaje Colbun & Cia 1584, Depto "B"',
            locality="Renca",
            status=DeliveryStatus.DELIVERED,
        ),
    ]
    route = build_route("RUTA-3", entries)

    pdf_bytes = generate_route_pdf(route)

    _assert_is_valid_pdf(pdf_bytes)


def test_generate_route_pdf_accepts_custom_region_and_country() -> None:
    entries = [RawDeliveryEntry(order=1, package_id="A1", address="Calle 1", locality="Vitacura")]
    route = build_route("RUTA-4", entries)

    pdf_bytes = generate_route_pdf(route, region="Valparaíso", country="Chile")

    _assert_is_valid_pdf(pdf_bytes)


def _printed_order_column(route: Route) -> list[str]:
    """Texto de la columna 'Parada' tal como se imprimiria en el PDF, en el
    mismo orden en que `generate_route_pdf` arma la tabla."""
    styles = _build_styles()
    return [
        _stop_row(position, stop, DEFAULT_REGION, DEFAULT_COUNTRY, styles)[0].text
        for position, stop in enumerate(route.stops, start=1)
    ]


def test_generate_route_pdf_prints_sequential_position_not_raw_order() -> None:
    """El numero impreso en el PDF es la posicion secuencial (1, 2, 3...) de
    la parada en la lista final, no el `order` crudo del sistema de origen (que puede
    saltar, ej. 3 -> 24, porque una parada con varios bultos consume
    numeracion de la ruta original). El `order` real se sigue usando para
    decidir el ORDEN (route_engine.py), intacto."""
    entries = [
        RawDeliveryEntry(order=3, package_id="A1", address="Calle Uno 100", locality="Renca"),
        RawDeliveryEntry(order=24, package_id="A2", address="Calle Dos 200", locality="Renca"),
        RawDeliveryEntry(order=25, package_id="A3", address="Calle Tres 300", locality="Renca"),
    ]
    route = build_route("RUTA-ORDER", entries)

    # El order crudo del sistema de origen se conserva intacto: route_engine no cambia.
    assert [stop.order for stop in route.stops] == [3, 24, 25]

    pdf_bytes = generate_route_pdf(route)
    _assert_is_valid_pdf(pdf_bytes)

    assert _printed_order_column(route) == ["1", "2", "3"]


def test_generate_route_pdf_prints_sequential_position_even_without_raw_order() -> None:
    """Una parada sin `order` (la aplicacion de origen no lo entrego) tambien recibe un
    numero secuencial impreso, nunca "-": el hueco de `order` solo afecta
    el orden relativo (route_engine la manda al final), no el numero que ve
    el conductor."""
    entries = [
        RawDeliveryEntry(order=5, package_id="A1", address="Calle Uno 100", locality="Renca"),
        RawDeliveryEntry(package_id="A2", address="Calle Sin Orden 200", locality="Renca"),
    ]
    route = build_route("RUTA-NOORDER", entries)

    assert route.stops[-1].order is None  # la parada sin order queda al final
    assert _printed_order_column(route) == ["1", "2"]
