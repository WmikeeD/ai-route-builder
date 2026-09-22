"""Guarda contra deriva: los modelos estrictos por proveedor (`_OpenAIEntry`,
`_AnthropicEntry`) duplican a proposito los campos de `DeliveryEntryDTO` (cada
adaptador es dueno de su schema: sin `ge=1`, todo requerido). Si alguien agrega
o cambia un campo del DTO y olvida un adaptador, el proveedor dejaria de pedir
(o de validar) ese dato en silencio. Estos tests lo hacen fallar."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.adapters.vision.anthropic import _AnthropicEntry
from app.adapters.vision.openai import _OpenAIEntry
from app.services.vision.schema import DeliveryEntryDTO

STRICT_ENTRY_MODELS = [
    pytest.param(_OpenAIEntry, id="openai"),
    pytest.param(_AnthropicEntry, id="anthropic"),
]


@pytest.mark.parametrize("strict", STRICT_ENTRY_MODELS)
def test_a_strict_entry_model_has_exactly_the_dto_fields(strict: type[BaseModel]) -> None:
    assert set(strict.model_fields) == set(DeliveryEntryDTO.model_fields)


@pytest.mark.parametrize("strict", STRICT_ENTRY_MODELS)
def test_a_strict_entry_model_has_the_same_field_types_as_the_dto(strict: type[BaseModel]) -> None:
    for name, field in DeliveryEntryDTO.model_fields.items():
        assert strict.model_fields[name].annotation == field.annotation, name


@pytest.mark.parametrize("strict", STRICT_ENTRY_MODELS)
def test_a_strict_entry_model_requires_every_field_and_carries_no_constraints(
    strict: type[BaseModel],
) -> None:
    for name, field in strict.model_fields.items():
        assert field.is_required(), name  # modo estricto: nada opcional ni con default
        assert not field.metadata, name  # sin `ge=1` & co: el proveedor no los soporta
