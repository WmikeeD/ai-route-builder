"""Tests del adaptador de Telegram: consolidacion de mensajes en lotes de fotos.

Cuando el usuario envia varias capturas juntas (album), cada una debe
agregarse a la sesion en silencio y el bot debe responder o editar un unico
mensaje con el total acumulado, en vez de un mensaje por foto.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import NamedTuple, cast
from unittest.mock import AsyncMock

import pytest
from telegram import Update
from telegram.error import TimedOut
from telegram.ext import ContextTypes

from app.adapters import telegram_bot
from app.services.session_manager import SessionManager
from app.services.vision.errors import AllProvidersFailedError, FailureReason, LimitKind

# Los handlers solo acceden a un subconjunto de atributos de `Update` y
# `CallbackContext`; se usan dobles minimos (duck typing) en vez de
# instanciar las clases reales de la libreria, y se castean para que mypy
# los acepte en la firma de los handlers.


def _make_update(chat_id: int, file_unique_id: str) -> Update:
    photo = SimpleNamespace(file_id=f"fid-{file_unique_id}", file_unique_id=file_unique_id)
    message = SimpleNamespace(photo=[photo])
    chat = SimpleNamespace(id=chat_id)
    update = SimpleNamespace(effective_message=message, effective_chat=chat, effective_user=None)
    return cast(Update, update)


class _FakeCallbackUpdate(NamedTuple):
    """`update` va casteado a los handlers; `query` queda sin castear para
    que mypy reconozca los metodos de AsyncMock en las aserciones."""

    update: Update
    query: SimpleNamespace


def _make_callback_update(chat_id: int) -> _FakeCallbackUpdate:
    query = SimpleNamespace(answer=AsyncMock(), edit_message_text=AsyncMock())
    chat = SimpleNamespace(id=chat_id)
    raw_update = SimpleNamespace(callback_query=query, effective_chat=chat, effective_user=None)
    return _FakeCallbackUpdate(update=cast(Update, raw_update), query=query)


class _FakeContext(NamedTuple):
    """`context` va casteado a los handlers; `bot` queda sin castear para
    que mypy reconozca los metodos de AsyncMock en las aserciones."""

    context: ContextTypes.DEFAULT_TYPE
    bot: SimpleNamespace


def _make_context(
    session_manager: SessionManager, route_extractor: object | None = None
) -> _FakeContext:
    telegram_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"img"))
    )
    bot = SimpleNamespace(
        get_file=AsyncMock(return_value=telegram_file),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=999)),
        edit_message_text=AsyncMock(),
    )
    application = SimpleNamespace(
        create_task=lambda coro, update=None: asyncio.ensure_future(coro)
    )
    bot_data: dict[str, object] = {telegram_bot._BOT_DATA_SESSION_MANAGER: session_manager}
    if route_extractor is not None:
        bot_data[telegram_bot._BOT_DATA_ROUTE_EXTRACTOR] = route_extractor
    raw_context = SimpleNamespace(bot=bot, application=application, bot_data=bot_data)
    return _FakeContext(context=cast(ContextTypes.DEFAULT_TYPE, raw_context), bot=bot)


def test_album_photos_are_added_silently_and_produce_one_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)
        await telegram_bot.handle_photo(_make_update(1, "b"), fake.context)
        await telegram_bot.handle_photo(_make_update(1, "c"), fake.context)

        # Ninguna foto individual debe haber disparado una respuesta todavia.
        fake.bot.send_message.assert_not_called()
        fake.bot.edit_message_text.assert_not_called()

        await asyncio.sleep(0.05)

        assert session_manager.get(1).count == 3
        fake.bot.send_message.assert_awaited_once()
        _, kwargs = fake.bot.send_message.call_args
        assert kwargs["text"] == "Capturas agregadas: 3. ¿Qué deseas hacer?"
        fake.bot.edit_message_text.assert_not_called()

    asyncio.run(scenario())


def test_new_batch_edits_previous_status_message_instead_of_sending_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)
        await asyncio.sleep(0.05)
        fake.bot.send_message.assert_awaited_once()

        await telegram_bot.handle_photo(_make_update(1, "b"), fake.context)
        await asyncio.sleep(0.05)

        fake.bot.send_message.assert_awaited_once()
        fake.bot.edit_message_text.assert_awaited_once()
        _, kwargs = fake.bot.edit_message_text.call_args
        assert kwargs["message_id"] == 999
        assert kwargs["text"] == "Capturas agregadas: 2. ¿Qué deseas hacer?"

    asyncio.run(scenario())


def test_different_chats_do_not_share_the_debounce_or_the_status_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)
        await telegram_bot.handle_photo(_make_update(2, "b"), fake.context)

        await asyncio.sleep(0.05)

        assert session_manager.get(1).count == 1
        assert session_manager.get(2).count == 1
        assert fake.bot.send_message.await_count == 2

    asyncio.run(scenario())


def test_handle_photo_retries_on_timeout_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un TimedOut transitorio en el primer intento no debe perder la foto:
    el reintento debe agregarla igual a la sesion."""
    monkeypatch.setattr(telegram_bot, "_PHOTO_DOWNLOAD_RETRY_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)
        telegram_file = SimpleNamespace(
            download_as_bytearray=AsyncMock(return_value=bytearray(b"img"))
        )
        fake.bot.get_file.side_effect = [TimedOut(), telegram_file]

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)

        assert fake.bot.get_file.await_count == 2
        assert session_manager.get(1).count == 1
        fake.bot.send_message.assert_not_called()

    asyncio.run(scenario())


def test_handle_photo_notifies_user_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si la descarga falla en todos los intentos, se avisa al usuario y la
    foto no se agrega a la sesion (nada de perdida silenciosa)."""
    monkeypatch.setattr(telegram_bot, "_PHOTO_DOWNLOAD_RETRY_DELAY_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)
        fake.bot.get_file.side_effect = TimedOut()

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)

        assert fake.bot.get_file.await_count == telegram_bot._PHOTO_DOWNLOAD_MAX_ATTEMPTS
        assert session_manager.get(1).count == 0
        fake.bot.send_message.assert_awaited_once()
        _, kwargs = fake.bot.send_message.call_args
        assert kwargs["text"] == telegram_bot._PHOTO_DOWNLOAD_FAILED_TEXT

    asyncio.run(scenario())


def test_handle_photo_rejects_without_adding_while_route_is_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si _process_route tiene el lock del chat tomado, una foto nueva se
    rechaza de plano: no se descarga, no se agrega a la sesion, y el
    usuario recibe un aviso en vez de quedar sin respuesta."""
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)
        lock = session_manager.lock_for(1)
        await lock.acquire()
        try:
            await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)
        finally:
            lock.release()

        fake.bot.get_file.assert_not_called()
        assert session_manager.get(1).count == 0

        await asyncio.sleep(0.05)

        fake.bot.send_message.assert_awaited_once()
        _, kwargs = fake.bot.send_message.call_args
        assert kwargs["text"] == telegram_bot._PROCESSING_BUSY_TEXT

    asyncio.run(scenario())


def test_burst_of_rejected_photos_while_processing_produces_one_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Varias fotos seguidas mientras el bot procesa no deben generar un
    aviso de rechazo por foto: se agrupan en un unico mensaje, igual que
    "Capturas agregadas" agrupa las confirmaciones."""
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)
        lock = session_manager.lock_for(1)
        await lock.acquire()
        try:
            await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)
            await telegram_bot.handle_photo(_make_update(1, "b"), fake.context)
            await telegram_bot.handle_photo(_make_update(1, "c"), fake.context)
        finally:
            lock.release()

        fake.bot.send_message.assert_not_called()
        assert session_manager.get(1).count == 0

        await asyncio.sleep(0.05)

        fake.bot.send_message.assert_awaited_once()
        _, kwargs = fake.bot.send_message.call_args
        assert kwargs["text"] == telegram_bot._PROCESSING_BUSY_TEXT

    asyncio.run(scenario())


def test_handle_photo_after_processing_finishes_adds_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Una vez liberado el lock (procesamiento terminado), una foto nueva
    se agrega a la sesion sin cambios de comportamiento respecto a hoy."""
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)
        async with session_manager.lock_for(1):
            pass  # simula _process_route ya terminado: el lock quedo libre

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)

        assert fake.bot.get_file.await_count == 1
        assert session_manager.get(1).count == 1

        await asyncio.sleep(0.05)

        fake.bot.send_message.assert_awaited_once()
        _, kwargs = fake.bot.send_message.call_args
        assert kwargs["text"] == "Capturas agregadas: 1. ¿Qué deseas hacer?"

    asyncio.run(scenario())


def test_intermittent_rejections_each_get_their_own_notice_not_just_the_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si el usuario manda fotos sueltas y espaciadas (no una rafaga) durante
    un procesamiento largo, el aviso no debe esperar a que el usuario deje
    de mandar del todo: cada foto separada de la anterior por mas del
    debounce dispara su propio aviso, poco despues de esa foto. Sin esto,
    alguien mandando una foto cada rato durante 2-3 minutos de
    procesamiento real se quedaria sin ningun aviso hasta el final."""
    monkeypatch.setattr(telegram_bot, "ALBUM_DEBOUNCE_SECONDS", 0.01)

    async def scenario() -> None:
        session_manager = SessionManager()
        fake = _make_context(session_manager)
        lock = session_manager.lock_for(1)
        await lock.acquire()  # simula _process_route todavia en curso

        await telegram_bot.handle_photo(_make_update(1, "a"), fake.context)
        await asyncio.sleep(0.05)  # deja disparar el primer aviso
        fake.bot.send_message.assert_awaited_once()

        # Mucho despues (espaciada, no parte de la misma rafaga): otra foto
        # rechazada mientras el procesamiento sigue en curso.
        await telegram_bot.handle_photo(_make_update(1, "b"), fake.context)
        await asyncio.sleep(0.05)

        lock.release()

        assert fake.bot.send_message.await_count == 2
        for call in fake.bot.send_message.await_args_list:
            assert call.kwargs["text"] == telegram_bot._PROCESSING_BUSY_TEXT
        assert session_manager.get(1).count == 0

    asyncio.run(scenario())


def test_process_route_failure_offers_retry_and_reset_buttons() -> None:
    """Si extract_entries (u otro paso del pipeline) lanza, el usuario no
    debe quedar bloqueado: el mensaje de error debe traer botones para
    reintentar o borrar la sesion, no solo texto."""

    class _FailingExtractor:
        async def extract_entries(
            self, images: list[bytes], screenshot_ids: list[str]
        ) -> list[object]:
            raise RuntimeError("boom")

    async def scenario() -> None:
        session_manager = SessionManager()
        session_manager.add_screenshot(1, b"img", "fid-1")
        fake = _make_context(session_manager, route_extractor=_FailingExtractor())
        callback_update = _make_callback_update(1)

        await telegram_bot._process_route(callback_update.update, fake.context)

        fake.bot.send_message.assert_awaited_once()
        _, kwargs = fake.bot.send_message.call_args
        markup = kwargs["reply_markup"]
        buttons = [button for row in markup.inline_keyboard for button in row]
        assert [button.text for button in buttons] == [
            "🚀 Reintentar Procesamiento",
            "🗑️ Borrar y reiniciar",
        ]
        assert [button.callback_data for button in buttons] == [
            telegram_bot.CALLBACK_PROCESS_ROUTE,
            telegram_bot.CALLBACK_RESET,
        ]

    asyncio.run(scenario())


# Textos de referencia, escritos a mano a proposito: si alguien cambia el
# wording en telegram_bot.py, estos tests lo hacen visible.
HIGH_DEMAND_TEXT = (
    "⚠️ El servicio de lectura está experimentando alta demanda. "
    "Por favor, pulsa el botón Reintentar en unos segundos."
)
GENERIC_ERROR_TEXT = (
    "Ocurrió un error al procesar las capturas. La sesión no se "
    "borró: puedes reintentar o borrar y empezar de nuevo."
)


class _ProcessOutcome(NamedTuple):
    query: SimpleNamespace
    bot: SimpleNamespace
    session_manager: SessionManager


def _chain_failure(
    reason: FailureReason, kind: LimitKind | None = None
) -> AllProvidersFailedError:
    return AllProvidersFailedError((), dominant_reason=reason, dominant_limit_kind=kind)


def _process_with_failure(exc: Exception) -> _ProcessOutcome:
    """Corre `_process_route` con un extractor que lanza `exc`."""

    class _FailingExtractor:
        async def extract_entries(
            self, images: list[bytes], screenshot_ids: list[str]
        ) -> list[object]:
            raise exc

    async def scenario() -> _ProcessOutcome:
        session_manager = SessionManager()
        session_manager.add_screenshot(1, b"img", "fid-1")
        fake = _make_context(session_manager, route_extractor=_FailingExtractor())
        callback_update = _make_callback_update(1)

        await telegram_bot._process_route(callback_update.update, fake.context)
        return _ProcessOutcome(callback_update.query, fake.bot, session_manager)

    return asyncio.run(scenario())


def _button_data(outcome: _ProcessOutcome) -> list[str]:
    _, kwargs = outcome.query.edit_message_text.call_args
    markup = kwargs["reply_markup"]
    return [button.callback_data for row in markup.inline_keyboard for button in row]


def test_process_route_high_demand_edits_message_with_friendly_text_and_buttons() -> None:
    """Cuando toda la cadena de vision responde alta demanda (503/529/5xx), el
    bot edita el mensaje existente (no envia uno nuevo) con un texto amable y
    mantiene los botones visibles, sin borrar la sesion."""
    outcome = _process_with_failure(_chain_failure(FailureReason.ALTA_DEMANDA))

    # Primero edita a "Procesando...", luego edita ese mismo mensaje con el aviso.
    assert outcome.query.edit_message_text.await_count == 2
    args, _ = outcome.query.edit_message_text.call_args
    assert args[0] == HIGH_DEMAND_TEXT
    assert _button_data(outcome) == [
        telegram_bot.CALLBACK_PROCESS_ROUTE,
        telegram_bot.CALLBACK_RESET,
    ]
    outcome.bot.send_message.assert_not_called()
    # La sesion no se borra: el usuario puede reintentar con las mismas capturas.
    assert outcome.session_manager.get(1).count == 1


@pytest.mark.parametrize("reason", [FailureReason.TIMEOUT, FailureReason.CONEXION])
def test_process_route_timeouts_and_network_errors_are_treated_as_high_demand(
    reason: FailureReason,
) -> None:
    """Son transitorios y reintentar puede ayudar: mismo aviso y mismos botones."""
    outcome = _process_with_failure(_chain_failure(reason))

    args, _ = outcome.query.edit_message_text.call_args
    assert args[0] == HIGH_DEMAND_TEXT
    outcome.bot.send_message.assert_not_called()
    assert outcome.session_manager.get(1).count == 1


@pytest.mark.parametrize("kind", list(LimitKind))
def test_process_route_limit_reached_has_its_own_text_per_kind(kind: LimitKind) -> None:
    """Reemplaza al test que fijaba "un 429 se trata igual que un 503". Un limite
    propio (cuota diaria, credito) NO es alta demanda: reintentar de inmediato no
    lo arregla y el texto debe decirlo."""
    outcome = _process_with_failure(_chain_failure(FailureReason.LIMITE_ALCANZADO, kind))

    args, _ = outcome.query.edit_message_text.call_args
    assert args[0] == telegram_bot._LIMIT_TEXTS[kind]
    assert args[0] != HIGH_DEMAND_TEXT
    assert "límite" in args[0]
    assert _button_data(outcome) == [
        telegram_bot.CALLBACK_PROCESS_ROUTE,
        telegram_bot.CALLBACK_RESET,
    ]
    outcome.bot.send_message.assert_not_called()
    assert outcome.session_manager.get(1).count == 1


def test_process_route_limit_texts_say_whether_retrying_helps() -> None:
    per_minute = telegram_bot._LIMIT_TEXTS[LimitKind.RATE_MINUTO]
    assert "segundos" in per_minute
    assert "Reintentar" in per_minute

    for kind in (LimitKind.CUOTA_DIARIA, LimitKind.CREDITO_O_GASTO):
        assert "no ayudará" in telegram_bot._LIMIT_TEXTS[kind]

    assert "administra" in telegram_bot._LIMIT_TEXTS[LimitKind.CREDITO_O_GASTO]


def test_process_route_limit_without_a_kind_uses_the_unknown_kind_text() -> None:
    outcome = _process_with_failure(_chain_failure(FailureReason.LIMITE_ALCANZADO))

    args, _ = outcome.query.edit_message_text.call_args
    assert args[0] == telegram_bot._LIMIT_TEXTS[LimitKind.DESCONOCIDO]


@pytest.mark.parametrize(
    "reason",
    [
        FailureReason.SOLICITUD_INVALIDA,  # antes: ClientError 400
        FailureReason.MODELO_NO_ENCONTRADO,  # antes: ClientError 404
        FailureReason.RESPUESTA_INVALIDA,
        FailureReason.AUTENTICACION,
        FailureReason.DESCONOCIDO,
    ],
)
def test_process_route_real_errors_use_the_generic_message(reason: FailureReason) -> None:
    """Un error "real" (payload invalido, modelo retirado, key mala, respuesta
    inutilizable) no es saturacion: mensaje generico, no prometer que reintentar
    lo va a arreglar. Los botones de accion siguen visibles."""
    outcome = _process_with_failure(_chain_failure(reason))

    outcome.bot.send_message.assert_awaited_once()
    _, kwargs = outcome.bot.send_message.call_args
    assert kwargs["text"] == GENERIC_ERROR_TEXT
    assert "reply_markup" in kwargs
    assert outcome.session_manager.get(1).count == 1


def test_provider_failure_notice_ignores_exceptions_that_are_not_chain_failures() -> None:
    assert telegram_bot._provider_failure_notice(RuntimeError("boom")) is None
    assert telegram_bot._provider_failure_notice(ValueError("x")) is None
