# Pendiente: migrar el bot de polling a webhook

**Estado:** diferido a propósito (decisión del 2026-09-23). Retomar cuando la
operación crezca más allá de la marcha blanca actual.

**Mitigación vigente mientras tanto:** deploys solo de madrugada, y shutdown
delay del servicio en Render subido a 300 s.

## Por qué hace falta

Render despliega con Zero Downtime, y eso **no se puede desactivar con un ajuste**:
según la documentación, solo se desactiva agregando un disco persistente. La
instancia vieja recibe `SIGTERM` recién **60 s después** de que la nueva queda
viva. Durante ese tiempo las dos hacen polling con el mismo token, y Telegram
responde `Conflict: terminated by other getUpdates request`. Se confirmó en
producción el 2026-09-23.

Efectos:
- ERRORs en el log durante más de un minuto en cada deploy. El proceso no se cae:
  python-telegram-bot reintenta el polling sin límite, con esperas de hasta 30 s.
- Las actualizaciones se reparten entre las dos instancias. Como las sesiones
  viven en memoria (`SessionManager`), un usuario que manda fotos durante el
  solape puede quedar con la sesión partida entre las dos.
- Riesgo menor: que la misma actualización se procese dos veces.

Con webhook, Telegram entrega cada actualización una sola vez, por HTTP, a la
instancia que Render tenga activa. Sin `getUpdates` no hay `Conflict`, y se
puede mantener Zero Downtime.

## Qué hay que implementar

1. **Ruta** `POST /telegram/webhook` en FastAPI. **Sin el token del bot en la
   URL**, por la lección del incidente de logs del 2026-09-23.
2. **Arranque:** construir la `Application` sin updater
   (`ApplicationBuilder().updater(None)`), hacer `initialize_with_retry` y
   `start()`, y registrar el webhook con
   `bot.set_webhook(url, secret_token=..., allowed_updates=[...])`.
3. **Apagado:** **no** llamar a `delete_webhook`. La instancia nueva ya lo
   apuntó a sí misma; si la vieja lo borra al recibir `SIGTERM`, deja a
   producción sin recibir mensajes.
4. **URL pública:** Render la expone en `RENDER_EXTERNAL_URL` (confirmar en el
   servicio) o se configura explícitamente. Telegram solo acepta los puertos
   443, 80, 88 y 8443; Render publica en 443.
5. **Modo seleccionable:** `TELEGRAM_MODE=webhook|polling`. Webhook en Render;
   polling solo en local.

## Mitigaciones de seguridad

1. **Secret token.** Generar un secreto aleatorio (`TELEGRAM_WEBHOOK_SECRET`,
   cargado solo en Render) y pasarlo en `set_webhook(secret_token=...)`. En cada
   request, comparar el header `X-Telegram-Bot-Api-Secret-Token` con
   `hmac.compare_digest`. Si no coincide, responder 403 sin procesar nada.
   Nunca loguear el secreto: agregarlo a la redacción de `app/logging_setup.py`.
2. **Filtrado por IP** de los rangos oficiales de Telegram. La documentación
   oficial (core.telegram.org/bots/webhooks) dice que los webhooks salen de
   `149.154.160.0/20` y `91.108.4.0/22`. **Punto a verificar antes de
   implementarlo:** detrás del proxy de Render, `request.client.host` es la IP del
   proxy. La IP real viene en `X-Forwarded-For`, y solo es confiable si
   uvicorn está configurado para confiar en el proxy de Render
   (`--proxy-headers` / `--forwarded-allow-ips`). Mal configurado, el filtro
   bloquea todo o se puede falsificar con el header. Es una capa extra; la
   defensa principal es el secret token.
3. **Límite de tamaño del body.** Las actualizaciones de Telegram son chicas
   (las fotos llegan como `file_id`, no como bytes). Rechazar con 413 todo body
   que supere un tope (propuesto: 1 MB), mirando `Content-Length` y cortando la
   lectura del stream si se excede. Así un request malicioso no puede agotar
   memoria antes de que se valide el secreto.
4. **Respuesta inmediata y tarea en segundo plano retenida.** Responder 200
   apenas se valida y encola la actualización
   (`application.update_queue.put(update)`), sin procesarla dentro del request:
   la extracción puede tardar minutos, y si Telegram no recibe respuesta a
   tiempo reintenta y duplica. Si en algún punto se usa `asyncio.create_task`
   directo, **guardar una referencia fuerte** a la tarea (por ejemplo, un `set`
   que se limpia en `add_done_callback`). El event loop solo guarda una
   referencia débil, y una tarea sin referencia puede desaparecer a mitad de
   ejecución. Preferir `application.create_task`, porque la `Application` las
   retiene y además `stop()` las espera al apagar.
5. **Validar el JSON** con `Update.de_json` y descartar lo que no parsee (400),
   sin loguear el body completo.

## Requisito previo: bot de desarrollo aparte

Crear en BotFather un bot de desarrollo con su propio token para trabajar en
local. En modo polling, python-telegram-bot llama a `deleteWebhook` al arrancar:
si alguien corre el bot local con el token de producción, **borra el webhook de
producción** y el bot deja de recibir mensajes sin ningún error visible. Hoy
ese descuido "solo" provoca `Conflict`; con webhook rompería producción en
silencio. Agregar la regla al `docs/PRE_DEPLOY_CHECKLIST.md`.

## Tests mínimos

- Un request sin el secret token o con uno incorrecto recibe 403 y no encola nada.
- Un body sobre el tope recibe 413.
- Un request válido recibe 200 de inmediato y encola la actualización.
- El arranque llama a `set_webhook` con el `secret_token`, y el apagado **no**
  llama a `delete_webhook`.
- El secreto no aparece en ningún log.

## Independiente de la migración

- **Sesiones en memoria:** se pierden o se parten en cada deploy, con polling o
  con webhook. Si importa, se resuelve aparte (sesiones persistentes o aviso al
  usuario).
- **Shutdown delay en 300 s:** es lo que permite que una extracción en curso
  termine antes del `SIGKILL`. `Application.stop()` espera las actualizaciones
  pendientes y las tareas creadas con `application.create_task`.
