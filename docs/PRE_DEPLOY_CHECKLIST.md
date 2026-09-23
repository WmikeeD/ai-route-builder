# Checklist de pre-deploy a producción

Revisar antes de **cualquier** deploy nuevo a producción, y antes de agregar una
dependencia, un proveedor o un secreto nuevo. Nace del incidente del primer deploy:
`httpx` logueaba en INFO la URL de cada request a Telegram, con el token del bot
adentro, y el token quedó en texto plano en los logs de Render.

## Logs

- [ ] Todos los loggers de librerías HTTP de terceros (`httpx`, `httpcore`, y
      cualquier cliente nuevo: `urllib3`, `aiohttp`, `requests`...) quedan en
      `WARNING` o superior **por código** (`app/logging_setup.py`), no por una
      variable de entorno que se pueda olvidar.
- [ ] Ningún secreto (token, API key) puede aparecer en una URL logueada. Si un
      proveedor nuevo pone el secreto en la URL (como Telegram), su patrón se
      agrega a la redacción de `app/logging_setup.py` **y** a
      `tests/test_logging_setup.py`, incluida la variante URL-encodeada.
- [ ] `tests/test_logging_setup.py` pasa en CI (no se saltea ni se marca `xfail`).
- [ ] Después del deploy: revisar las primeras líneas de log en Render y buscar
      `bot` seguido de dígitos y `api.telegram.org`. Solo debe aparecer
      `[TOKEN REDACTADO]`. Si aparece un token real: rotarlo en BotFather de inmediato;
      no alcanza con arreglar el código.
- [ ] `LOG_LEVEL` en producción es `INFO` (o superior). `DEBUG` solo de forma
      puntual y vigilada.

## Variables de entorno de producción

- [ ] No hay variables de prueba en el entorno de Render: nada con patrón `FORCE_*`
      (`FORCE_VISION_ERROR`, `FORCE_GEMINI_MODEL_ERROR`) ni `*_TEST_*`. Si el log
      de arranque muestra `MODO DE PRUEBA: inyeccion de fallas activa`, el deploy
      está mal configurado.
- [ ] Sin variables deprecadas (`GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`): usar
      `VISION_CHAIN`.
- [ ] Timeouts de producción, no los de evaluación local (60 s): los
      `*_TIMEOUT_SECONDS` configurados explícitamente para cada proveedor activo.
- [ ] Cada secreto de producción es distinto de los usados en local o en pruebas,
      y cualquier secreto que haya pasado por un output de herramienta, un log o
      un chat se rotó antes del deploy.
- [ ] Los secretos viven solo en el panel de Render y en los secrets de GitHub
      Actions, nunca en `render.yaml`, el workflow ni el repo.

## Repo y código (misma disciplina que la auditoría previa al push público)

- [ ] `.env` y `.env.*` siguen ignorados (excepto `.env.example`, que solo tiene
      nombres de variable, sin valores).
- [ ] `.env.example` está actualizado con toda variable nueva, sin valores reales.
- [ ] Barrido de secretos sobre el diff a deployar: formatos conocidos (`AIza...`,
      `sk-...`, `sk-ant-...`, `\d{8,10}:[A-Za-z0-9_-]{35}`) y asignaciones
      `api_key=` / `token=` / `secret=` con valores largos. Idealmente con
      `gitleaks detect --source . -v`.
- [ ] No se commitean datos reales de clientes (capturas, PDFs generados, logs
      en `logs/`), ni respaldos locales (`.backups/`), ni config local de
      herramientas (`.claude/`).
- [ ] Ningún mensaje de error o de log vuelca el contenido completo de una
      respuesta del proveedor ni datos de la captura (el registro de intentos
      guarda solo conteos, tiempos y detalle truncado).

## Operación

- [ ] CI en verde (ruff, mypy, pytest) sobre el commit exacto que se deploya.
- [ ] Un solo proceso de polling por token: con Zero Downtime activo, dos
      instancias pueden hacer polling a la vez durante el deploy (error 409 de
      Telegram). Confirmar que está resuelto (sin Zero Downtime o con webhook)
      antes de automatizar deploys frecuentes.
- [ ] Después del deploy: `/health` responde, el log muestra
      `Bot de Telegram iniciado (polling).` y los tiers de la cadena activos esperados.
