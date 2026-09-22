"""Generacion de PDF de rutas para tu planificador de rutas.

Convierte una entidad de dominio `Route` (ver app.domain.models) en un PDF
vectorial en memoria usando ReportLab. Esta es la unica capa que conoce
ReportLab: el dominio no importa nada de este modulo, solo al reves.

Tabla en blanco y negro de alto contraste, una fila por Stop, con columnas:
Parada, Direccion Completa (calle + comuna + region + pais), Comuna,
Notas/Bultos (cantidad de paquetes y sus codigos de tracking) y Ventana
Horaria. Todas las celdas usan `Paragraph` para que el texto haga wrap en
vez de desbordar o truncarse.
"""

from __future__ import annotations

import io
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.domain.models import Route, Stop

DEFAULT_REGION = "Región Metropolitana"
DEFAULT_COUNTRY = "Chile"

_COLUMN_HEADERS = (
    "Parada",
    "Dirección Completa",
    "Comuna",
    "Notas / Bultos",
    "Ventana Horaria",
)
_COLUMN_WIDTH_FRACTIONS = (0.07, 0.34, 0.14, 0.25, 0.20)


def generate_route_pdf(
    route: Route,
    region: str = DEFAULT_REGION,
    country: str = DEFAULT_COUNTRY,
) -> bytes:
    """Genera el PDF de una ruta y devuelve sus bytes (documento en memoria)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Ruta {route.route_id}",
    )

    styles = _build_styles()
    story: list = [
        Paragraph(f"Ruta {xml_escape(route.route_id)}", styles["title"]),
        Paragraph(_build_subtitle(route), styles["subtitle"]),
        Spacer(1, 6 * mm),
    ]

    if route.stops:
        column_widths = [fraction * doc.width for fraction in _COLUMN_WIDTH_FRACTIONS]
        table_data = [_header_row(styles)] + [
            _stop_row(position, stop, region, country, styles)
            for position, stop in enumerate(route.stops, start=1)
        ]
        table = Table(table_data, colWidths=column_widths, repeatRows=1)
        table.setStyle(_table_style())
        story.append(table)
    else:
        story.append(Paragraph("Esta ruta no tiene paradas registradas.", styles["cell"]))

    doc.build(story)
    return buffer.getvalue()


def _build_subtitle(route: Route) -> str:
    parts = []
    if route.driver_name:
        parts.append(f"Conductor: {xml_escape(route.driver_name)}")
    if route.route_date:
        parts.append(f"Fecha: {route.route_date.isoformat()}")
    parts.append(f"Paradas: {route.total_stops}")
    parts.append(f"Bultos: {route.total_packages}")
    return " &nbsp;|&nbsp; ".join(parts)


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "RouteTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.black,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "RouteSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.black,
        ),
        "header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.white,
            alignment=TA_CENTER,
            leading=11,
        ),
        "cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            textColor=colors.black,
            alignment=TA_LEFT,
            leading=10.5,
        ),
        "cell_center": ParagraphStyle(
            "TableCellCenter",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.black,
            alignment=TA_CENTER,
            leading=13,
        ),
    }


def _header_row(styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(title, styles["header"]) for title in _COLUMN_HEADERS]


def _stop_row(
    position: int, stop: Stop, region: str, country: str, styles: dict[str, ParagraphStyle]
) -> list[Paragraph]:
    # `position` es el numero secuencial de esta parada en la lista final ya
    # ordenada y deduplicada (1, 2, 3... N), NO `stop.order`. El `order` crudo
    # del sistema de origen puede saltar (una parada con varios bultos consume numeracion
    # de la ruta original) y confunde al conductor; sigue siendo la base para
    # decidir el ORDEN de las paradas (ver route_engine.py), solo cambia el
    # numero que se IMPRIME.
    order_text = str(position)
    locality_text = _display_case(stop.locality) if stop.locality else "-"
    return [
        Paragraph(order_text, styles["cell_center"]),
        Paragraph(xml_escape(_full_address(stop, region, country)), styles["cell"]),
        Paragraph(xml_escape(locality_text), styles["cell"]),
        Paragraph(_notes_bultos(stop), styles["cell"]),
        Paragraph(xml_escape(_time_window(stop)), styles["cell"]),
    ]


def _display_case(text: str) -> str:
    return text.title()


def _full_address(stop: Stop, region: str, country: str) -> str:
    parts = [_display_case(stop.address)]
    if stop.locality:
        parts.append(_display_case(stop.locality))
    if region:
        parts.append(region)
    if country:
        parts.append(country)
    return ", ".join(parts)


def _notes_bultos(stop: Stop) -> str:
    suffix = "s" if stop.package_count != 1 else ""
    lines = [f"<b>{stop.package_count} bulto{suffix}</b>"]
    tracking_ids = [xml_escape(pkg.package_id) for pkg in stop.packages if pkg.package_id]
    if tracking_ids:
        lines.append(", ".join(tracking_ids))
    return "<br/>".join(lines)


def _time_window(stop: Stop) -> str:
    windows = sorted({pkg.time_window for pkg in stop.packages if pkg.time_window})
    return " / ".join(windows) if windows else "-"


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white]),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
            ("BOX", (0, 0), (-1, -1), 1.25, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
    )
