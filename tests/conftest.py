"""Fixtures compartidas."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_PREFIXES = ("GEMINI_", "OPENAI_", "ANTHROPIC_", "VISION_", "FORCE_")


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Entorno sin `.env` ni variables del proceso.

    Los bloques anidados de `Settings` leen `.env` del directorio actual, asi
    que se ejecuta en un directorio vacio y se limpian las variables de la
    cadena de vision. Evita que un test lea las keys reales del desarrollador.
    """
    for name in list(os.environ):
        if name.upper().startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    return tmp_path
