"""Entrypoint ASGI de AI Route Builder.

Expone un servidor FastAPI (health check para el hosting/orquestador) y,
en su `lifespan`, arranca y detiene el bot de Telegram (polling) dentro
del mismo proceso y event loop. Ejecutar con:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.telegram_bot import create_application
from app.adapters.vision.factory import build_fallback_chain
from app.config import get_settings
from app.logging_setup import setup_logging
from app.services.session_manager import SessionManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Siempre primero: silencia httpx/httpcore (loguean la URL con el token del
    # bot) e instala la redaccion de tokens antes de crear clientes HTTP.
    setup_logging(settings.log_level)

    # Cadena de fallback de vision: valida la config al arrancar (fail-fast) y
    # loguea que tiers quedaron activos.
    route_extractor = build_fallback_chain(settings)
    session_manager = SessionManager()
    telegram_app = create_application(
        token=settings.telegram_bot_token,
        route_extractor=route_extractor,
        session_manager=session_manager,
    )

    app.state.telegram_app = telegram_app
    app.state.route_extractor = route_extractor
    app.state.session_manager = session_manager

    await telegram_app.initialize()
    await telegram_app.start()
    if telegram_app.updater is not None:
        await telegram_app.updater.start_polling()
    logger.info("Bot de Telegram iniciado (polling).")

    try:
        yield
    finally:
        logger.info("Deteniendo bot de Telegram...")
        if telegram_app.updater is not None:
            await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
        await route_extractor.aclose()


app = FastAPI(title="AI Route Builder", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
