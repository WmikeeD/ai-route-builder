"""Gestion en memoria de las sesiones de captura por chat de Telegram.

Cada chat_id acumula las capturas de pantalla de la aplicacion de origen que el usuario va
enviando hasta que decide procesarlas (boton "Procesar Ruta") o
descartarlas (boton "Borrar y reiniciar"). Es estado de trabajo
transitorio: no hay persistencia entre reinicios del proceso, y no es
responsabilidad de esta capa (no importa Telegram ni ningun framework).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class ChatSession:
    """Capturas pendientes de un chat, en el orden en que se recibieron."""

    screenshots: list[bytes] = field(default_factory=list)
    screenshot_ids: list[str] = field(default_factory=list)

    def add(self, data: bytes, screenshot_id: str) -> int:
        self.screenshots.append(data)
        self.screenshot_ids.append(screenshot_id)
        return len(self.screenshots)

    @property
    def count(self) -> int:
        return len(self.screenshots)

    @property
    def is_empty(self) -> bool:
        return not self.screenshots


class SessionManager:
    """Administra una ChatSession por chat_id, mas un lock por chat para
    evitar procesar la misma sesion dos veces si el usuario presiona
    "Procesar Ruta" mas de una vez antes de recibir respuesta."""

    def __init__(self) -> None:
        self._sessions: dict[int, ChatSession] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def get(self, chat_id: int) -> ChatSession:
        session = self._sessions.get(chat_id)
        if session is None:
            session = ChatSession()
            self._sessions[chat_id] = session
        return session

    def add_screenshot(self, chat_id: int, data: bytes, screenshot_id: str) -> int:
        return self.get(chat_id).add(data, screenshot_id)

    def lock_for(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    def clear(self, chat_id: int) -> None:
        self._sessions.pop(chat_id, None)
        self._locks.pop(chat_id, None)
