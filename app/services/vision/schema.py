"""Schema de salida neutral y conversion a entidades de dominio.

`DeliveryEntryDTO` es la forma comun que todo proveedor debe producir. Cada
adaptador es dueno de adaptarla a su mecanismo de salida estructurada
(Gemini acepta una lista en la raiz; otros proveedores exigen un objeto raiz
y no soportan algunas restricciones como `minimum`). Vive en la capa de
servicios, no en el dominio: agrega `screenshot_index`, que solo sirve para
saber de que imagen del lote vino cada tarjeta.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.models import DeliveryStatus, RawDeliveryEntry


class DeliveryEntryDTO(BaseModel):
    """Una tarjeta de la aplicacion de origen tal como la devuelve el proveedor de vision."""

    screenshot_index: int = Field(ge=1)
    order: int | None = None
    package_id: str | None = None
    customer_name: str | None = None
    address: str
    locality: str | None = None
    delivery_type: str | None = None
    eta: str | None = None
    time_window: str | None = None
    status: DeliveryStatus = DeliveryStatus.UNKNOWN


def to_domain_entry(item: DeliveryEntryDTO, screenshot_ids: list[str] | None) -> RawDeliveryEntry:
    """Traduce el DTO a `RawDeliveryEntry`, resolviendo `source_screenshot_id`
    desde `screenshot_index` (base 1) si hay ids disponibles."""
    source_screenshot_id = None
    if screenshot_ids is not None and 1 <= item.screenshot_index <= len(screenshot_ids):
        source_screenshot_id = screenshot_ids[item.screenshot_index - 1]

    return RawDeliveryEntry(
        order=item.order,
        package_id=item.package_id,
        customer_name=item.customer_name,
        address=item.address,
        locality=item.locality,
        delivery_type=item.delivery_type,
        eta=item.eta,
        time_window=item.time_window,
        status=item.status,
        source_screenshot_id=source_screenshot_id,
    )
