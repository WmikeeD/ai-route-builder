"""Tests del gestor de sesiones de captura por chat (Paso 4)."""

from __future__ import annotations

import asyncio

from app.services.session_manager import SessionManager


def test_get_creates_empty_session_for_new_chat() -> None:
    manager = SessionManager()

    session = manager.get(chat_id=1)

    assert session.is_empty
    assert session.count == 0


def test_get_returns_same_session_for_same_chat() -> None:
    manager = SessionManager()

    first = manager.get(chat_id=1)
    second = manager.get(chat_id=1)

    assert first is second


def test_add_screenshot_increments_count_and_preserves_order() -> None:
    manager = SessionManager()

    manager.add_screenshot(chat_id=1, data=b"img-1", screenshot_id="fid-1")
    count = manager.add_screenshot(chat_id=1, data=b"img-2", screenshot_id="fid-2")

    session = manager.get(chat_id=1)
    assert count == 2
    assert session.count == 2
    assert session.screenshots == [b"img-1", b"img-2"]
    assert session.screenshot_ids == ["fid-1", "fid-2"]
    assert not session.is_empty


def test_sessions_for_different_chats_are_independent() -> None:
    manager = SessionManager()

    manager.add_screenshot(chat_id=1, data=b"img-1", screenshot_id="fid-1")

    assert manager.get(chat_id=1).count == 1
    assert manager.get(chat_id=2).count == 0


def test_clear_removes_session_data() -> None:
    manager = SessionManager()
    manager.add_screenshot(chat_id=1, data=b"img-1", screenshot_id="fid-1")

    manager.clear(chat_id=1)

    assert manager.get(chat_id=1).is_empty


def test_lock_for_returns_same_lock_instance_for_same_chat() -> None:
    manager = SessionManager()

    lock_a = manager.lock_for(chat_id=1)
    lock_b = manager.lock_for(chat_id=1)

    assert lock_a is lock_b
    assert isinstance(lock_a, asyncio.Lock)


def test_lock_for_different_chats_returns_different_locks() -> None:
    manager = SessionManager()

    lock_chat_1 = manager.lock_for(chat_id=1)
    lock_chat_2 = manager.lock_for(chat_id=2)

    assert lock_chat_1 is not lock_chat_2


def test_clear_also_drops_the_chat_lock() -> None:
    manager = SessionManager()
    original_lock = manager.lock_for(chat_id=1)

    manager.clear(chat_id=1)
    new_lock = manager.lock_for(chat_id=1)

    assert new_lock is not original_lock
