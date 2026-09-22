"""Adaptador de Telegram para el bot de rutas de entrega.

Usa python-telegram-bot en modo asincrono (v21+). Esta es la unica capa
que conoce la libreria `telegram`: el dominio y los servicios no importan
nada de este modulo, solo al reves.

Flujo:
    1. El usuario envia una o mas capturas de la pantalla "Visitas" de
       la aplicacion de origen; cada una se agrega a su ChatSession (SessionManager).
    2. Un teclado inline con 3 botones acompana cada respuesta:
       "Agregar mas", "Procesar Ruta" y "Borrar y reiniciar".
    3. Al presionar "Procesar Ruta" se conecta el pipeline completo:
       SessionManager -> RouteExtractor (cadena de fallback de vision) -> route_engine.build_route
       -> pdf_generator.generate_route_pdf, se responde con el PDF y se
       limpia la sesion del chat.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Update,
)
from telegram.error import NetworkError, TelegramError, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.domain import build_route
from app.services.pdf_generator import generate_route_pdf
from app.services.session_manager import SessionManager
from app.services.vision.errors import AllProvidersFailedError, FailureReason, LimitKind
from app.services.vision.ports import RouteExtractor

logger = logging.getLogger(__name__)

CALLBACK_ADD_MORE = "add_more"
CALLBACK_PROCESS_ROUTE = "process_route"
CALLBACK_RESET = "reset"

_BOT_DATA_SESSION_MANAGER = "session_manager"
_BOT_DATA_ROUTE_EXTRACTOR = "route_extractor"
_BOT_DATA_ALBUM_BUFFERS = "album_buffers"

# Cuando el usuario envia varias fotos juntas (album), Telegram las entrega
# como updates independientes sin ninguna senal de "fin del lote". Se
# posterga la respuesta este tiempo tras cada foto recibida: si llega otra
# antes de que venza, se reprograma. Asi el lote completo produce un unico
# mensaje consolidado en vez de uno por foto.
ALBUM_DEBOUNCE_SECONDS = 1.5

# Reintentos ante fallas transitorias de red al descargar una foto (no ante
# errores de Gemini, que ya tienen su propio fallback en gemini_client.py).
_PHOTO_DOWNLOAD_MAX_ATTEMPTS = 3
_PHOTO_DOWNLOAD_RETRY_DELAY_SECONDS = 1.5
_PHOTO_DOWNLOAD_FAILED_TEXT = "No pude descargar una de tus fotos, por favor reenvíala."
_UNEXPECTED_ERROR_TEXT = "Ocurrió un error inesperado. Intenta de nuevo o usa /start."

# Si _process_route ya tiene el lock del chat tomado, una foto nueva se
# rechaza sin agregarse a la sesion (rechazo tajante, sin buffer ni
# reintento automatico: el usuario debe esperar y reenviar).
_PROCESSING_BUSY_TEXT = (
    "Tu ruta anterior todavía se está procesando. Espera a que termine y "
    "vuelve a enviar esta captura."
)

# Avisos por causa dominante de una cadena de vision agotada. Reintentar en
# segundos solo ayuda cuando la causa es transitoria: para un limite diario o
# de credito el texto debe dejar claro que reintentar de inmediato no sirve.
_HIGH_DEMAND_TEXT = (
    "⚠️ El servicio de lectura está experimentando alta demanda. "
    "Por favor, pulsa el botón Reintentar en unos segundos."
)
_LIMIT_TEXTS: dict[LimitKind, str] = {
    LimitKind.RATE_MINUTO: (
        "⚠️ Se alcanzó momentáneamente el límite de uso del servicio de lectura. "
        "Espera unos segundos y pulsa el botón Reintentar."
    ),
    LimitKind.CUOTA_DIARIA: (
        "⚠️ Se alcanzó el límite diario de uso del servicio de lectura. Reintentar "
        "ahora no ayudará: vuelve a intentarlo más tarde, cuando se renueve el cupo."
    ),
    LimitKind.CREDITO_O_GASTO: (
        "⚠️ Se alcanzó el límite de crédito o gasto del servicio de lectura. Reintentar "
        "ahora no ayudará: avisa a quien administra el bot para que amplíe el límite."
    ),
    LimitKind.DESCONOCIDO: (
        "⚠️ El servicio de lectura alcanzó un límite de uso. Es posible que reintentar "
        "de inmediato no ayude; inténtalo de nuevo más tarde."
    ),
}

# Timeouts al subir el PDF con send_document. Los defaults de PTB (5 s de
# lectura/escritura) son justos para un PDF de muchas paradas: un ReadTimeout
# ocurre DESPUES de que Telegram ya recibio el archivo, y el usuario ve un
# error aunque el PDF haya llegado (falso negativo).
_SEND_DOCUMENT_TIMEOUT_SECONDS = 30.0

WELCOME_TEXT = (
    "Hola! Envíame una o varias capturas de pantalla de tu ruta "
    "(pantalla \"Visitas\"). Cuando termines, presiona \"🚀 Procesar Ruta\" "
    "para recibir el PDF listo para tu planificador de rutas."
)


def _build_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("➕ Agregar más", callback_data=CALLBACK_ADD_MORE),
                InlineKeyboardButton("🚀 Procesar Ruta", callback_data=CALLBACK_PROCESS_ROUTE),
            ],
            [InlineKeyboardButton("🗑️ Borrar y reiniciar", callback_data=CALLBACK_RESET)],
        ]
    )


def _build_error_keyboard() -> InlineKeyboardMarkup:
    """Teclado para cuando falla el procesamiento: la sesion no se borra,
    asi que el usuario debe poder reintentar o reiniciar sin quedar
    bloqueado (sin esto, tendria que escribir /start a ciegas)."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🚀 Reintentar Procesamiento", callback_data=CALLBACK_PROCESS_ROUTE
                ),
                InlineKeyboardButton("🗑️ Borrar y reiniciar", callback_data=CALLBACK_RESET),
            ]
        ]
    )


def _session_manager(context: ContextTypes.DEFAULT_TYPE) -> SessionManager:
    return context.bot_data[_BOT_DATA_SESSION_MANAGER]


def _route_extractor(context: ContextTypes.DEFAULT_TYPE) -> RouteExtractor:
    return context.bot_data[_BOT_DATA_ROUTE_EXTRACTOR]


def _provider_failure_notice(exc: Exception) -> str | None:
    """Texto amable para una cadena de vision agotada por una causa del
    proveedor; `None` si es un error "real" (config rota, solicitud invalida,
    respuesta inutilizable...), que usa el mensaje generico.

    Los tiempos de espera y las caidas de red se tratan como alta demanda:
    son transitorios y reintentar puede ayudar.
    """
    if not isinstance(exc, AllProvidersFailedError):
        return None
    reason = exc.dominant_reason
    if reason is FailureReason.LIMITE_ALCANZADO:
        return _LIMIT_TEXTS[exc.dominant_limit_kind or LimitKind.DESCONOCIDO]
    if reason in (FailureReason.ALTA_DEMANDA, FailureReason.TIMEOUT, FailureReason.CONEXION):
        return _HIGH_DEMAND_TEXT
    return None


@dataclass
class _AlbumBuffer:
    """Estado (por chat) del debounce de fotos y del mensaje consolidado."""

    debounce_task: asyncio.Task[None] | None = None
    status_message_id: int | None = None
    # Timer independiente de debounce_task, NO uno compartido, a proposito:
    # ambos pueden estar pendientes al mismo tiempo para el mismo chat. Ej.:
    # llega una foto (se agenda debounce_task para "Capturas agregadas"),
    # el usuario presiona "Procesar Ruta" antes de que ese timer dispare (el
    # lock ya esta tomado) y manda otra foto (se necesita agendar el aviso
    # de "ocupado"). Si compartieran el campo, agendar uno cancelaria el
    # otro y el chat se quedaria sin el mensaje que le tocaba (la
    # confirmacion de las fotos ya agregadas, o el aviso de rechazo).
    busy_notice_task: asyncio.Task[None] | None = None


def _album_buffer(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> _AlbumBuffer:
    buffers: dict[int, _AlbumBuffer] = context.bot_data.setdefault(_BOT_DATA_ALBUM_BUFFERS, {})
    buffer = buffers.get(chat_id)
    if buffer is None:
        buffer = _AlbumBuffer()
        buffers[chat_id] = buffer
    return buffer


def _clear_album_buffer(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    buffers: dict[int, _AlbumBuffer] = context.bot_data.get(_BOT_DATA_ALBUM_BUFFERS, {})
    buffer = buffers.pop(chat_id, None)
    if buffer is None:
        return
    if buffer.debounce_task is not None:
        buffer.debounce_task.cancel()
    if buffer.busy_notice_task is not None:
        buffer.busy_notice_task.cancel()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is not None:
        await message.reply_text(WELCOME_TEXT, reply_markup=_build_keyboard())


async def _download_photo(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes | None:
    """Descarga una foto reintentando ante fallas transitorias de red.

    Devuelve None (en vez de propagar) si se agotan los intentos, para que
    el llamador pueda avisar al usuario en vez de perder la captura sin que
    se entere (ver incidente: TimedOut sin capturar descartaba fotos en
    silencio dentro de handle_photo).
    """
    for attempt in range(1, _PHOTO_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            telegram_file = await context.bot.get_file(file_id)
            return bytes(await telegram_file.download_as_bytearray())
        except (TimedOut, NetworkError) as exc:
            if attempt == _PHOTO_DOWNLOAD_MAX_ATTEMPTS:
                logger.warning(
                    "No se pudo descargar la foto (file_id=%s) tras %s intentos: %s",
                    file_id,
                    attempt,
                    exc,
                )
                return None
            await asyncio.sleep(_PHOTO_DOWNLOAD_RETRY_DELAY_SECONDS)
    return None


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None or not message.photo:
        return

    session_manager = _session_manager(context)
    if session_manager.lock_for(chat.id).locked():
        # _process_route tiene el lock de este chat: rechazo tajante, sin
        # descargar la foto ni tocar la sesion. El usuario debe esperar a
        # que termine y reenviar la captura.
        _notify_processing_busy(update, context, chat.id)
        return

    photo = message.photo[-1]  # ultima entrada = mayor resolucion disponible
    image_bytes = await _download_photo(context, photo.file_id)
    if image_bytes is None:
        await context.bot.send_message(chat_id=chat.id, text=_PHOTO_DOWNLOAD_FAILED_TEXT)
        return

    # Se agrega en silencio: si esta foto es parte de un lote, todavia no
    # sabemos si vienen mas. La respuesta se posterga via debounce.
    session_manager.add_screenshot(chat.id, image_bytes, photo.file_unique_id)

    buffer = _album_buffer(context, chat.id)
    if buffer.debounce_task is not None:
        buffer.debounce_task.cancel()

    buffer.debounce_task = context.application.create_task(
        _publish_album_summary(context, chat.id), update=update
    )


async def _publish_album_summary(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Espera a que el lote de fotos termine y publica un unico resumen.

    Se reprograma (cancela + reagenda) cada vez que llega una foto nueva del
    mismo chat, de forma que un album completo produzca una sola respuesta
    en vez de una por foto.
    """
    try:
        await asyncio.sleep(ALBUM_DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return

    buffer = _album_buffer(context, chat_id)
    buffer.debounce_task = None

    count = _session_manager(context).get(chat_id).count
    if count == 0:
        return

    text = f"Capturas agregadas: {count}. ¿Qué deseas hacer?"

    if buffer.status_message_id is not None:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=buffer.status_message_id,
                text=text,
                reply_markup=_build_keyboard(),
            )
            return
        except TelegramError:
            logger.debug(
                "No se pudo editar el mensaje de estado del chat %s; se envía uno nuevo.",
                chat_id,
            )

    sent = await context.bot.send_message(
        chat_id=chat_id, text=text, reply_markup=_build_keyboard()
    )
    buffer.status_message_id = sent.message_id


def _notify_processing_busy(
    update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int
) -> None:
    """Programa (o reprograma) el aviso de "ruta en procesamiento".

    Mismo mecanismo de debounce que `_publish_album_summary`: si el usuario
    manda varias fotos seguidas mientras el bot procesa, todas menos la
    ultima de la rafaga cancelan el aviso pendiente, y solo se envia un
    unico mensaje tras ALBUM_DEBOUNCE_SECONDS de silencio, no uno por foto.
    """
    buffer = _album_buffer(context, chat_id)
    if buffer.busy_notice_task is not None:
        buffer.busy_notice_task.cancel()

    buffer.busy_notice_task = context.application.create_task(
        _publish_busy_notice(context, chat_id), update=update
    )


async def _publish_busy_notice(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    try:
        await asyncio.sleep(ALBUM_DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return

    buffer = _album_buffer(context, chat_id)
    buffer.busy_notice_task = None

    await context.bot.send_message(chat_id=chat_id, text=_PROCESSING_BUSY_TEXT)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat = update.effective_chat
    if query is None or chat is None:
        return

    if query.data == CALLBACK_ADD_MORE:
        await query.answer()
        await query.edit_message_text(
            "Perfecto, envía las capturas que falten.",
            reply_markup=_build_keyboard(),
        )
        if query.message is not None:
            # Las proximas fotos deben editar este mismo mensaje en vez de
            # enviar uno nuevo.
            _album_buffer(context, chat.id).status_message_id = query.message.message_id
        return

    if query.data == CALLBACK_RESET:
        _session_manager(context).clear(chat.id)
        _clear_album_buffer(context, chat.id)
        await query.answer("Sesión reiniciada.")
        await query.edit_message_text(
            "Se borraron las capturas. Cuando quieras, envía una nueva captura "
            "de tu ruta para empezar de nuevo."
        )
        return

    if query.data == CALLBACK_PROCESS_ROUTE:
        await _process_route(update, context)
        return

    await query.answer()


async def _process_route(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat = update.effective_chat
    if query is None or chat is None:
        return

    session_manager = _session_manager(context)

    async with session_manager.lock_for(chat.id):
        session = session_manager.get(chat.id)

        if session.is_empty:
            await query.answer("No hay capturas para procesar.", show_alert=True)
            return

        await query.answer("Procesando ruta...")
        await query.edit_message_text(
            f"Procesando {session.count} captura(s)... esto puede tardar unos segundos."
        )

        try:
            entries = await _route_extractor(context).extract_entries(
                images=list(session.screenshots),
                screenshot_ids=list(session.screenshot_ids),
            )

            if not entries:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=(
                        "No se detectaron paradas válidas en las capturas enviadas. "
                        "Verifica que las tarjetas se vean completas y vuelve a intentar."
                    ),
                )
                return

            driver_name = update.effective_user.full_name if update.effective_user else None
            now = datetime.now(UTC)
            route_id = f"RUTA-{chat.id}-{now:%Y%m%d%H%M%S}"
            route = build_route(
                route_id, entries, driver_name=driver_name, route_date=now.date()
            )

            pdf_bytes = generate_route_pdf(route)
        except Exception as exc:
            # La cadena de vision ya probo los tiers configurados; aqui solo se
            # elige el mensaje segun la causa dominante, sin conocer ningun SDK.
            notice = _provider_failure_notice(exc)
            if notice is not None:
                assert isinstance(exc, AllProvidersFailedError)
                logger.warning(
                    "Cadena de visión sin resultado (%s%s) para el chat %s",
                    exc.dominant_reason,
                    f"/{exc.dominant_limit_kind}" if exc.dominant_limit_kind else "",
                    chat.id,
                )
                await query.edit_message_text(notice, reply_markup=_build_error_keyboard())
                return

            logger.exception("Fallo al procesar la ruta del chat %s", chat.id)
            await context.bot.send_message(
                chat_id=chat.id,
                text=(
                    "Ocurrió un error al procesar las capturas. La sesión no se "
                    "borró: puedes reintentar o borrar y empezar de nuevo."
                ),
                reply_markup=_build_error_keyboard(),
            )
            return

        await context.bot.send_document(
            chat_id=chat.id,
            document=InputFile(pdf_bytes, filename=f"{route_id}.pdf"),
            caption=(
                f"Ruta {route.route_id}\n"
                f"Paradas: {route.total_stops} | Bultos: {route.total_packages}"
            ),
            read_timeout=_SEND_DOCUMENT_TIMEOUT_SECONDS,
            write_timeout=_SEND_DOCUMENT_TIMEOUT_SECONDS,
        )

        session_manager.clear(chat.id)
        _clear_album_buffer(context, chat.id)


async def _handle_unexpected_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Red de seguridad global: sin esto, una excepcion no capturada en un
    handler solo queda en el log de PTB ("No error handlers are
    registered") y el usuario se queda sin ninguna respuesta."""
    logger.error("Excepción no manejada procesando un update", exc_info=context.error)

    chat = update.effective_chat if isinstance(update, Update) else None
    if chat is None:
        return
    try:
        await context.bot.send_message(chat_id=chat.id, text=_UNEXPECTED_ERROR_TEXT)
    except TelegramError:
        logger.exception("No se pudo notificar al chat %s del error", chat.id)


def create_application(
    token: str,
    route_extractor: RouteExtractor,
    session_manager: SessionManager | None = None,
) -> Application:
    """Construye la Application de python-telegram-bot con sus handlers.

    `route_extractor` y `session_manager` se guardan en `bot_data` para
    que los handlers (funciones simples, sin estado propio) los resuelvan
    en cada actualizacion.
    """
    application = ApplicationBuilder().token(token).build()
    application.bot_data[_BOT_DATA_SESSION_MANAGER] = session_manager or SessionManager()
    application.bot_data[_BOT_DATA_ROUTE_EXTRACTOR] = route_extractor

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_error_handler(_handle_unexpected_error)

    return application
