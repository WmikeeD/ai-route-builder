"""Tests del motor de deduplicacion y agrupacion de paradas (Paso 1)."""

from __future__ import annotations

from app.domain.models import DeliveryStatus, RawDeliveryEntry
from app.domain.route_engine import (
    build_route,
    deduplicate_and_group,
    normalize_address,
    normalize_text,
)


def test_normalize_text_strips_accents_case_and_punctuation() -> None:
    assert normalize_text("  Panamericana, Norte°.  ") == "panamericana norte"


def test_normalize_address_includes_locality_to_disambiguate() -> None:
    key_renca = normalize_address("Panamericana Norte 1500", "Renca")
    key_quilicura = normalize_address("Panamericana Norte 1500", "Quilicura")

    assert key_renca != key_quilicura


def test_duplicate_row_same_address_and_package_id_is_merged() -> None:
    entries = [
        RawDeliveryEntry(
            order=55,
            package_id="546401014692",
            address="Panamericana Norte 1500",
            locality="Renca",
            eta="12:59",
        ),
        # Misma parada vista de nuevo por scroll superpuesto entre capturas.
        RawDeliveryEntry(
            order=55,
            package_id="546401014692",
            address="panamericana norte 1500",
            locality="renca",
            time_window="07:00 - 21:00",
        ),
    ]

    stops = deduplicate_and_group(entries)

    assert len(stops) == 1
    assert stops[0].package_count == 1
    merged_package = stops[0].packages[0]
    assert merged_package.eta == "12:59"
    assert merged_package.time_window == "07:00 - 21:00"


def test_same_street_different_locality_are_not_merged() -> None:
    entries = [
        RawDeliveryEntry(
            order=1, package_id="A1", address="Panamericana Norte 1500", locality="Renca"
        ),
        RawDeliveryEntry(
            order=2, package_id="A2", address="Panamericana Norte 1500", locality="Quilicura"
        ),
    ]

    stops = deduplicate_and_group(entries)

    assert len(stops) == 2


def test_multiple_packages_same_address_are_grouped_into_one_stop() -> None:
    entries = [
        RawDeliveryEntry(
            order=56, package_id="713132491315", address="Pasaje Colbun 1584", locality="Renca"
        ),
        RawDeliveryEntry(
            order=56, package_id="713132491316", address="Pasaje Colbun 1584", locality="Renca"
        ),
    ]

    stops = deduplicate_and_group(entries)

    assert len(stops) == 1
    assert stops[0].package_count == 2
    package_ids = {pkg.package_id for pkg in stops[0].packages}
    assert package_ids == {"713132491315", "713132491316"}


def test_missing_package_id_is_not_merged_with_other_packages_at_same_address() -> None:
    entries = [
        RawDeliveryEntry(order=1, package_id=None, address="Calle Falsa 456", locality="Renca"),
        RawDeliveryEntry(
            order=2,
            package_id=None,
            address="Calle Falsa 456",
            locality="Renca",
            customer_name="Otro",
        ),
    ]

    stops = deduplicate_and_group(entries)

    assert len(stops) == 1
    assert stops[0].package_count == 2


def test_missing_package_id_exact_duplicate_row_is_still_deduplicated() -> None:
    entries = [
        RawDeliveryEntry(
            order=1, package_id=None, address="Calle Falsa 456", locality="Renca", eta="10:00"
        ),
        RawDeliveryEntry(
            order=1, package_id=None, address="calle falsa 456", locality="renca", eta="10:00"
        ),
    ]

    stops = deduplicate_and_group(entries)

    assert len(stops) == 1
    assert stops[0].package_count == 1


def test_stops_are_sorted_by_order() -> None:
    entries = [
        RawDeliveryEntry(order=3, package_id="C", address="Calle C"),
        RawDeliveryEntry(order=1, package_id="A", address="Calle A"),
        RawDeliveryEntry(order=2, package_id="B", address="Calle B"),
    ]

    stops = deduplicate_and_group(entries)

    assert [stop.order for stop in stops] == [1, 2, 3]


def test_build_route_computes_totals() -> None:
    entries = [
        RawDeliveryEntry(
            order=1, package_id="A", address="Calle A", status=DeliveryStatus.DELIVERED
        ),
        RawDeliveryEntry(order=2, package_id="B", address="Calle B", status=DeliveryStatus.FAILED),
        RawDeliveryEntry(order=2, package_id="C", address="Calle B", status=DeliveryStatus.PENDING),
    ]

    route = build_route("RUTA-1", entries, driver_name="Carlos")

    assert route.total_stops == 2
    assert route.total_packages == 3
    assert route.total_delivered == 1
    assert route.total_failed == 1
    assert route.total_pending == 1
    assert route.driver_name == "Carlos"
