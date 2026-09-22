"""Entidades de dominio de AI Route Builder.

Capa de dominio pura: no importa FastAPI, python-telegram-bot, google-genai
ni ReportLab. La unica dependencia externa permitida es Pydantic v2, tal
como exige BaseProyecto.md para la validacion estricta de datos.

Los campos reflejan lo que realmente se puede leer en la vista de "Visitas"
de la aplicacion de origen: numero de orden, direccion, comuna,
codigo de tracking, tipo de entrega, hora estimada y ventana horaria. No
hay nombre de cliente ni telefono en esa vista, por eso `customer_name` es
opcional y no existen campos de contacto.

Modelo de negocio:
    - RawDeliveryEntry: una fila tal como la extrae Gemini de UNA captura
      de pantalla de la aplicacion de origen. Puede haber filas duplicadas entre capturas
      distintas (por scroll superpuesto).
    - Package: un paquete/entrega individual ya deduplicado, asignado a
      una parada.
    - Stop: una parada fisica (direccion + comuna). El package_id es unico
      por paquete, pero una misma direccion puede recibir paquetes de
      clientes distintos, por eso una Stop agrupa 1..N Package.
    - Route: la ruta completa, compuesta por sus Stops.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RawDeliveryEntry(BaseModel):
    """Fila cruda extraida por Gemini de una captura de pantalla de la aplicacion de origen."""

    model_config = ConfigDict(frozen=True)

    order: int | None = Field(default=None, ge=1)
    package_id: str | None = None
    customer_name: str | None = None
    address: str = Field(min_length=1)
    locality: str | None = None
    delivery_type: str | None = None
    eta: str | None = None
    time_window: str | None = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    source_screenshot_id: str | None = None

    @field_validator("address", mode="before")
    @classmethod
    def _strip_required(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "package_id",
        "customer_name",
        "locality",
        "delivery_type",
        "eta",
        "time_window",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class Package(BaseModel):
    """Paquete individual, ya deduplicado, asignado a una Stop."""

    model_config = ConfigDict(frozen=True)

    package_id: str | None = None
    customer_name: str | None = None
    delivery_type: str | None = None
    eta: str | None = None
    time_window: str | None = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    order: int | None = Field(default=None, ge=1)
    source_screenshot_id: str | None = None


class Stop(BaseModel):
    """Parada fisica de la ruta (direccion + comuna).

    La llave de una Stop es la direccion normalizada (calle + comuna), no
    el package_id: dos paquetes de clientes distintos pueden compartir
    direccion, por lo que `packages` puede tener mas de un elemento.
    """

    model_config = ConfigDict(frozen=True)

    address: str = Field(min_length=1)
    locality: str | None = None
    normalized_address: str = Field(min_length=1)
    order: int | None = Field(default=None, ge=1)
    packages: tuple[Package, ...] = Field(min_length=1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def package_count(self) -> int:
        return len(self.packages)


class Route(BaseModel):
    """Ruta completa compuesta por las paradas ya deduplicadas y agrupadas."""

    model_config = ConfigDict(frozen=True)

    route_id: str = Field(min_length=1)
    driver_name: str | None = None
    route_date: date | None = None
    stops: tuple[Stop, ...] = Field(default_factory=tuple)

    @field_validator("driver_name", mode="before")
    @classmethod
    def _blank_driver_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_stops(self) -> int:
        return len(self.stops)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_packages(self) -> int:
        return sum(stop.package_count for stop in self.stops)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_pending(self) -> int:
        return sum(
            1
            for stop in self.stops
            for package in stop.packages
            if package.status == DeliveryStatus.PENDING
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_delivered(self) -> int:
        return sum(
            1
            for stop in self.stops
            for package in stop.packages
            if package.status == DeliveryStatus.DELIVERED
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_failed(self) -> int:
        return sum(
            1
            for stop in self.stops
            for package in stop.packages
            if package.status == DeliveryStatus.FAILED
        )
