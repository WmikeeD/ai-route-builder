"""Criterio de descarte de tarjetas en el prompt compartido de vision.

Incidente (2026-09-23): con la regla "extrae solo tarjetas completas de
principio a fin", el modelo descartaba la ultima tarjeta de casi cada captura
(su linea inferior queda cortada encima del boton "Comenzar Ruta"): 43 paradas
de 51. El criterio correcto es la legibilidad de la direccion y del codigo de
tracking, no la completitud de la tarjeta.
"""

from __future__ import annotations

import re

from app.adapters.vision import anthropic, gemini, openai
from app.services.vision.prompt import SYSTEM_INSTRUCTION


def _rule(number: int) -> str:
    match = re.search(rf"^{number}\. (.*?)(?=^\d+\. |\Z)", SYSTEM_INSTRUCTION, re.M | re.S)
    assert match is not None, f"no se encontro la regla {number}"
    return " ".join(match.group(1).split())


def test_rule_1_extracts_cards_whose_address_and_tracking_code_are_legible() -> None:
    rule = _rule(1)

    assert "Extrae TODA tarjeta cuya direccion y cuyo codigo de tracking se lean completos" in rule
    # La posicion en pantalla no es motivo de descarte.
    for position in ("ultima fila visible", "Comenzar Ruta", "unica tarjeta de la captura"):
        assert position in rule
    # Los campos tapados van en null, no hacen descartar la tarjeta.
    assert "van en null" in rule


def test_rule_1_only_discards_cut_or_illegible_address_or_tracking_code() -> None:
    rule = _rule(1)

    assert "IGNORA una tarjeta SOLO si su direccion o su codigo de tracking estan" in rule
    assert "digitos del codigo cortados a la mitad" in rule
    assert "nunca adivines ni completes el codigo" in rule


def test_the_old_completeness_criterion_is_gone() -> None:
    flat = " ".join(SYSTEM_INSTRUCTION.split())

    assert "completas y legibles de principio a fin" not in flat
    assert "IGNORA por completo cualquier tarjeta" not in flat


def test_the_three_providers_share_the_same_prompt() -> None:
    assert gemini.SYSTEM_INSTRUCTION is SYSTEM_INSTRUCTION
    assert openai.SYSTEM_INSTRUCTION is SYSTEM_INSTRUCTION
    assert anthropic.SYSTEM_INSTRUCTION is SYSTEM_INSTRUCTION
