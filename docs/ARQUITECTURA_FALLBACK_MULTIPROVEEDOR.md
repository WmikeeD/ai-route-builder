# Arquitectura de la cadena de fallback multi-proveedor de visión

AI Route Builder · Auditoría y propuesta · fase de pruebas de la arquitectura multiproveedor
Estado: **borrador para revisión. No hay código escrito, no se instaló nada, no se tocó ningún archivo del proyecto ni se agregó ninguna API key.**

Cadena decidida (no está en discusión):

| Tier | Proveedor | Modelo | Estado al implementar |
|---|---|---|---|
| 1 | Google | `gemini-3.6-flash` | activo, sin cambios |
| 2 | Google | `gemini-3.8-flash` | activo, sin cambios |
| 3 | OpenAI | `gpt-5.6-terra` | NUEVO, activo |
| 4 | Anthropic | `claude-sonnet-5` | NUEVO, completo pero **inactivo** hasta que exista `ANTHROPIC_API_KEY` |

---

## 0. Resumen ejecutivo

1. **La lógica "es alta demanda" vive en dos sitios y conflaciona 429 con 5xx en ambos**, y los tests lo fijan a propósito:
   `app/adapters/telegram_bot.py:325-327` y `app/adapters/gemini_client.py:38, 215-243`. Además el reintento interno del SDK de Gemini también trata 429 igual que 503 (`google/genai/_api_client.py:552-559`).
2. **El acoplamiento a Gemini es total y está en 5 archivos de la app + 3 de soporte.** El punto más dañino: `telegram_bot.py` importa `google.genai.errors` (línea 25) y decide el mensaje al usuario con esos tipos. Sin cambiar eso, agregar OpenAI y Anthropic obligaría a duplicar la lógica de errores dentro de la capa de Telegram.
3. **`Settings` plano no escala.** Sirve para un proveedor; con cuatro tiers y tres proveedores mezcla conceptos (el "fallback model" es en realidad un tier de la cadena) y hace obligatoria la key de Gemini.
4. **Propuesta:** un puerto común (`VisionExtractorProvider`), un orquestador de cadena que solo conoce ese puerto, una taxonomía canónica de fallas (`FailureReason`), config anidada por proveedor y un registro estructurado de cada intento (`ProviderAttempt`) que permite contar fallas por proveedor y causa.
5. **Hallazgos que cambian el diseño** y que conviene decidir ahora: los SDK de OpenAI y Anthropic traen timeout por defecto de **10 minutos** y 2 reintentos (cuatro tiers sin límites explícitos podrían esperar decenas de minutos); los hooks de prueba de Gemini siguen en el código con C sin validar; la clasificación de 429 por tipo de límite depende de cuerpos de error que solo pude verificar en documentación, no contra respuestas reales.
6. **Hay una decisión que es tuya y no tomé:** compatibilidad hacia atrás (opción A) vs arquitectura limpia con más refactor (opción B). Ambas con sus trade-offs en la sección 4.

---

## 1. Auditoría del estado actual (Fase 0)

### 1.1 Qué se leyó

`app/adapters/gemini_client.py` (315 líneas, completo), `app/config.py` (35 líneas, completo), y para trazar acoplamiento: `app/adapters/telegram_bot.py` (tramos con Gemini), `app/main.py`, `tests/test_gemini_client.py`, `tests/test_telegram_bot.py`, `scripts/test_gemini_connection.py`, `.env.example`, `pyproject.toml`, `requirements.txt` y el código fuente del SDK instalado (`google/genai/errors.py`, `_api_client.py`, `types.py`).

### 1.2 Mapa de acoplamiento a Gemini

| # | Qué está atado a Gemini | Ubicación exacta | Qué rompe o duplica al sumar OpenAI y Anthropic |
|---|---|---|---|
| 1 | Import de tipos de error del SDK en la capa de Telegram | `telegram_bot.py:25` (`from google.genai.errors import ClientError, ServerError`) | La capa de UI debería conocer solo un tipo de error propio. Con tres SDK habría tres imports y tres ramas `isinstance`. |
| 2 | Clasificación "alta demanda" con tipos del SDK | `telegram_bot.py:325-327` | Ver 1.3. Solo reconoce excepciones de Gemini: un 503 de OpenAI caería en el mensaje genérico de error. |
| 3 | Tipado y nombre de la dependencia inyectada | `telegram_bot.py:43, 55, 117-118, 384, 395`; `main.py:18, 31-35, 39, 44, 61` (`GeminiRouteExtractor`, clave `bot_data["gemini_extractor"]`, `app.state.gemini_extractor`) | La UI depende de una clase concreta de un proveedor en vez de un puerto. |
| 4 | Lógica de fallback dentro del adaptador | `gemini_client.py:213-243` | El fallback conoce solo "modelo principal → modelo de respaldo del mismo SDK". No hay concepto de cadena ni de otro proveedor. |
| 5 | Códigos que disparan fallback | `gemini_client.py:38` (`_FALLBACK_CLIENT_ERROR_CODES = {404, 429}`) y `:215, :220` | Las categorías están implícitas en `isinstance` y códigos HTTP de Gemini, sin nombre ni vocabulario común. |
| 6 | Reintento del SDK configurado a nivel Gemini | `gemini_client.py:48` (`HttpRetryOptions(attempts=3, ...)`) | Cada SDK reintenta a su manera (ver 1.5). No hay política de reintento común. |
| 7 | Sin timeout de cliente | `gemini_client.py:164-167` (`HttpOptions` solo lleva `retry_options`) | Con el SDK en `timeout=None` una llamada quedó varios minutos colgada durante la fase de pruebas. Los otros dos SDK traen 10 min por defecto. |
| 8 | Prompt del sistema en el módulo del adaptador | `gemini_client.py:56-87` (`SYSTEM_INSTRUCTION`) | Es neutral respecto del proveedor y debe compartirse; hoy solo se importa desde el módulo Gemini. |
| 9 | Schema de salida con nombre y forma de Gemini | `gemini_client.py:94-112` (`_GeminiDeliveryEntry`) y `:289` (`response_schema=list[...]`) | Gemini acepta lista en la raíz; la documentación de OpenAI y Anthropic solo muestra objetos como raíz. `Field(ge=1)` (línea 103) es `minimum`, que Anthropic no soporta. |
| 10 | Excepción con nombre de proveedor | `gemini_client.py:90-91` (`GeminiExtractionError`), usada en `:246` y en `scripts/test_gemini_connection.py:26, 85` | JSON inválido o truncado también puede ocurrir en los otros dos proveedores. |
| 11 | Observabilidad atada al tipo de respuesta de Gemini | `gemini_client.py:115-137` (`_log_response_metrics`, usa `types.GenerateContentResponse`, `usage_metadata`, `thoughts_token_count`) | Cada proveedor expone tokens y motivo de fin con otros nombres. Hay que normalizarlos a una estructura común. |
| 12 | Parámetros de generación fijos | `gemini_client.py:286-291` (`temperature=0.0`) | Los modelos 4.7+ y 5 de Anthropic rechazan `temperature` (400). Cada proveedor necesita sus propios parámetros. |
| 13 | Conversión DTO → dominio | `gemini_client.py:296-315` (`_to_domain_entry`) | Es neutral y debe compartirse, pero hoy es método estático de la clase de Gemini y recibe `_GeminiDeliveryEntry`. |
| 14 | Hooks de prueba dentro del adaptador | `gemini_client.py:13, 50-54, 149-162, 170-174, 259-281` (`FORCE_GEMINI_MODEL_ERROR`, `GEMINI_TEST_RETRY_ATTEMPTS`, `_real_call_count`) | Solo actúan sobre Gemini. La validación de C sigue abierta, así que aún se necesitan. Ver sección 5. |
| 15 | Defaults de modelo duplicados | `gemini_client.py:25, 29` y `config.py:25-26`; el docstring del módulo (línea 4) también nombra `gemini-3.6-flash` | Dos fuentes de verdad. Los tests importan las constantes del adaptador (`test_gemini_client.py:28-29`). |
| 16 | Tests fabricando errores del SDK | `test_gemini_client.py:25, 37-50` y `test_telegram_bot.py:16, 247, 287, 317, 346` | Todos construyen `errors.ServerError` / `ClientError`. Habrá que reescribirlos contra el tipo canónico. |
| 17 | Script standalone atado a la clase concreta | `scripts/test_gemini_connection.py:24-27, 71` | Importa `MODEL_NAME`, `FALLBACK_MODEL_NAME`, `GeminiExtractionError`, `GeminiRouteExtractor`. |
| 18 | Variables de entorno | `.env.example:6-11`, `config.py:24-26` | `GEMINI_MODEL` y `GEMINI_FALLBACK_MODEL` no tienen equivalente en una cadena de 4 tiers. |

### 1.3 Dónde vive "es alta demanda" y qué conflaciona (con línea exacta)

**Ubicación 1: capa de Telegram**, `app/adapters/telegram_bot.py`

```
322   except Exception as exc:
325       is_high_demand = isinstance(exc, ServerError) or (
326           isinstance(exc, ClientError) and exc.code == 429
327       )
328       if is_high_demand:
329           logger.warning("Gemini con alta demanda (código %s) para el chat %s", ...)
334           "⚠️ El servicio de lectura está experimentando alta demanda. "
338           "Por favor, pulsa el botón Reintentar en unos segundos."
```

**Sí conflaciona 429 y 5xx**: ambos entran en `is_high_demand`, con el mismo log ("alta demanda") y el mismo mensaje al usuario ("pulsa Reintentar en unos segundos"). Para un 429 por cuota diaria ese mensaje es incorrecto: reintentar en segundos no lo arregla.

**Ubicación 2: dentro del adaptador**, `app/adapters/gemini_client.py`

```
38    _FALLBACK_CLIENT_ERROR_CODES = frozenset({404, 429})
215   except (errors.ServerError, errors.ClientError) as exc:
220       if isinstance(exc, errors.ClientError) and exc.code not in _FALLBACK_CLIENT_ERROR_CODES: raise
222       if isinstance(exc, errors.ClientError) and exc.code == 404:   # -> logger.error (línea 228)
235   else:
236       logger.warning("Modelo '%s' no disponible (código %s); ...")   # mismo texto para 5xx y 429
```

Aquí 5xx y 429 comparten rama, log y destino (el modelo de respaldo). Solo el 404 se distingue (ERROR).

**Ubicación 3: reintento interno del SDK**, `google/genai/_api_client.py:552-559` y `:577`. La lista por defecto de códigos reintentables es `(408, 429, 500, 502, 503, 504)`. El SDK ya reintenta un 429 igual que un 503 antes de que el error llegue a nuestro código. `HttpRetryOptions.http_status_codes` existe (`types.py:2617`) y `:577` lo usa (`options.http_status_codes or _RETRY_HTTP_STATUS_CODES`), así que es configurable, pero hoy no se configura (`gemini_client.py:48`).

**La conflación está fijada por tests**, no es un descuido:
- `tests/test_telegram_bot.py:279-306`: "Un 429 que sobrevive al fallback debe tratarse igual que un 503: mismo aviso amable".
- `tests/test_telegram_bot.py:237-276`, `309-334`, `337-363`: 503 → mensaje amable; 400 y 404 → mensaje genérico.
- `tests/test_gemini_client.py`: fallback para 5xx, 429 y 404; no para 400.

Cambiarlo requiere modificar esos tests de forma deliberada.

### 1.4 ¿Escala `Settings` plano a 4 tiers con 3 proveedores?

Hoy (`app/config.py:15-29`): `gemini_api_key: str` (obligatoria), `gemini_model`, `gemini_fallback_model`, más `telegram_bot_token`, `pdf_*`, `log_level`. `model_config` ya usa `extra="ignore"` (línea 20), así que agregar variables nuevas al `.env` no rompe la carga.

**No escala limpio.** Motivos concretos:

1. **`gemini_fallback_model` mezcla dos conceptos.** Hoy significa "el modelo de respaldo dentro de Gemini". En la cadena nueva el tier 2 es simplemente un eslabón; el campo queda redundante o ambiguo.
2. **Cada proveedor necesita opciones propias.** Auth distinta (Gemini y OpenAI usan key; Anthropic además puede resolver credenciales ambientales: variable, token o perfil de login), parámetros de generación distintos (temperature, razonamiento, thinking), timeouts y reintentos propios y schema propio. En un modelo plano aparecerían `openai_reasoning_effort`, `anthropic_effort`, `gemini_thinking...` mezclados con todo lo demás.
3. **`gemini_api_key: str` es obligatoria.** Hace imposible expresar "proveedor inactivo". Requisito: Anthropic debe existir en la cadena sin key.
4. **Defaults duplicados** (ver #15 del mapa).
5. **No hay noción de orden.** El orden de intento debe ser una lista configurable, no un efecto de nombres de campos.

**Conclusión:** conviene reestructurar en bloques anidados por proveedor más un bloque de cadena. Detalle y opciones en 3.4.

### 1.5 Hallazgos adicionales que afectan el diseño

| Hallazgo | Evidencia | Impacto |
|---|---|---|
| Timeouts por defecto de 10 min en Anthropic y OpenAI | Documentación oficial de ambos SDK (Python): timeout por defecto 10 minutos, `max_retries` 2 (reintentan conexión, 408, 409, 429 y ≥500) | Con 4 tiers sin límites explícitos, el peor caso teórico se cuenta en decenas de minutos. Vimos varios minutos colgados en Gemini con `timeout=None` durante la fase de pruebas. |
| Solo Gemini permite excluir códigos del reintento del SDK | `HttpRetryOptions.http_status_codes` en el SDK de Gemini. Los de OpenAI y Anthropic exponen solo `max_retries` (todo o nada) | Un 429 por crédito agotado (OpenAI) o tope de gasto (Anthropic) se reintenta 2 veces inútilmente. Decisión en 3.4 (política de reintento). |
| Errores de transporte de Gemini NO vienen envueltos | El probe realizado durante la fase de pruebas devolvió `ReadTimeout` de `httpx` con `code=None`, no un `errors.APIError` | El clasificador de Gemini debe mapear también `httpx.TimeoutException` y `httpx.TransportError`. |
| La llamada de Gemini que se cuelga no dispara fallback hasta que el servidor responde | Observado durante la fase de pruebas: varios minutos hasta un 503 | El fallback solo actúa si hay un error; un timeout explícito por tier es requisito para que la cadena avance. |
| Clasificación de Gemini en documentación vs. respuestas reales | La página de errores lista códigos tipo `rate_limit_exceeded`, `quota_exceeded`, `model_not_found`, `service_unavailable`; las respuestas reales observadas durante la fase de pruebas traen `status` clásico (`NOT_FOUND` en un 404 real del probe) | No se puede afirmar qué identificador devuelve un 429 real. El clasificador debe apoyarse en el código HTTP y registrar el cuerpo crudo truncado durante la marcha blanca para aprender los valores reales. |

---

## 2. Verificación de errores por proveedor (base del mapeo)

Fuentes consultadas durante esta auditoría, en documentación oficial:

| Proveedor | Qué se verificó | Salvedad |
|---|---|---|
| **Gemini** | Página de errores de la API y guía de rate limits. 429: límite por minuto/segundo (`rate_limit_exceeded`) o cuota diaria (`quota_exceeded`); reinicio del RPD a medianoche hora del Pacífico. 503: "temporalmente sobrecargado o caído". 404 con `model_not_found`. 504 `deadline_exceeded`. 402: crédito prepago agotado. Errores del SDK: `ServerError` (5xx) y `ClientError` (4xx) con `.code`, `.status`, `.message`, `.details` (código fuente local). | La documentación resumida dice que **no detalla campos del cuerpo** para distinguir tipos de cuota. Los identificadores textuales no están confirmados contra respuestas reales. |
| **OpenAI** | 429 con `error.code` ∈ {`slow_down`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded`} y `rate_limit_error` genérico. 503 con `error.type=service_unavailable_error` y `error.code=server_is_overloaded`. Clases del SDK: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `UnprocessableEntityError`, `RateLimitError`, `InternalServerError` (≥500), `APIConnectionError`, `APITimeoutError`; atributos `status_code`, `response`, `request_id`. | La ruta exacta del `error.code` en la excepción (`exc.code` vs `exc.body`) **no quedó confirmada**; se verifica contra el SDK instalado al implementar. |
| **Anthropic** | Overload = **529** `overloaded_error` (no 503). 429 `rate_limit_error` (un tope de gasto por tier da 429 **sin** `retry-after`). 402 `billing_error`. 400 también cuando se alcanza un límite de gasto configurado por la organización. 504 `timeout_error`. Cuerpo `{"type":"error","error":{"type","message"},"request_id"}`. SDK: 400 `BadRequestError`, 401 `AuthenticationError`, 403 `PermissionDeniedError`, 404 `NotFoundError`, 409 `ConflictError`, 422 `UnprocessableEntityError`, 429 `RateLimitError`, **≥500 `InternalServerError`** (incluye 529), sin conexión `APIConnectionError`, `APITimeoutError`. | 402 y 413 **no aparecen** en la tabla de clases del SDK: probablemente salen como `APIStatusError` genérico. Por eso el clasificador debe usar `status_code` y `error.type`, no solo la clase. Distinguir "400 por tope de gasto" de un 400 real requeriría leer el texto del mensaje (frágil). |

Regla de diseño que sale de aquí: **clasificar por (código HTTP, tipo/código del cuerpo), con la clase de excepción como apoyo, nunca solo por clase.**

---

## 3. Arquitectura propuesta (Fase 1)

### 3.1 Vista general

```
                     Telegram handler  (_process_route)
                              │  extract_entries(images, screenshot_ids)
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  RouteExtractor (Protocol)      ← lo único que la UI conoce │
        │  implementado por: FallbackChain                           │
        │    · recorre los tiers en orden                            │
        │    · salta los inactivos (INFO "tier N ... se omite")      │
        │    · llama provider.extract(request, model=...)            │
        │    · decide seguir o abortar según la razón canónica       │
        │    · emite un ProviderAttempt por intento                  │
        └──────────────┬──────────────────────────────────────────┘
                       │  solo conoce el Protocol, jamás un SDK
                       ▼
        VisionExtractorProvider (Protocol)
          ├─ GeminiProvider      (google-genai)
          ├─ OpenAIProvider      (openai)
          ├─ AnthropicProvider   (anthropic)
          └─ (futuro) KimiProvider / DeepSeekProvider
                 cada uno: schema propio · auth propia · reintento y timeout propios
                           traduce SDK-exception → ProviderError(FailureReason)

  Fallo de toda la cadena → AllProvidersFailedError(attempts, dominant_reason)
        → la UI mapea dominant_reason a un mensaje; no importa ningún SDK.
```

**Dónde vive cada cosa** (siguiendo la convención actual: las capas que conocen un SDK viven en `app/adapters/`; lo demás en `app/services/`; `app/domain/` no cambia):

```
app/services/vision/            # aplicación: no importa ningún SDK
    ports.py         RouteExtractor, VisionExtractorProvider (Protocols)
    errors.py        FailureReason, LimitKind, ProviderError, AllProvidersFailedError
    models.py        ExtractionRequest, ExtractionResult, UsageInfo, ProviderAttempt, TierSpec
    prompt.py        SYSTEM_INSTRUCTION (movido desde gemini_client.py:56-87)
    schema.py        DeliveryEntryDTO (movido/renombrado desde _GeminiDeliveryEntry)
                     + conversión DTO → RawDeliveryEntry (movida desde :296-315)
    chain.py         FallbackChain
    registry.py      id de proveedor → fábrica (adaptador + su bloque de settings)
    telemetry.py     emisión de ProviderAttempt (log estructurado + contadores)
app/adapters/vision/            # aquí sí se importan los SDK
    gemini.py        GeminiProvider
    openai.py        OpenAIProvider
    anthropic.py     AnthropicProvider
app/config.py                   # Settings + bloques anidados por proveedor
```

### 3.2 Interfaz común

Contrato de `VisionExtractorProvider` (firmas ilustrativas, no código final):

| Miembro | Responsabilidad |
|---|---|
| `provider_id: str` | Identificador estable y en minúsculas (`"gemini"`, `"openai"`, `"anthropic"`). Es la llave del registro, del log y de las métricas. |
| `display_name: str` | Nombre para logs legibles (`"Gemini"`, `"OpenAI"`, `"Anthropic"`). |
| `is_configured() -> bool` | `True` solo si hay credencial **explícita** en su bloque de settings. Nunca deriva de credenciales ambientales del SDK (ver nota abajo). |
| `extract(request, *, model) -> ExtractionResult` | Una llamada a un modelo. Devuelve entradas de dominio + `UsageInfo`. **Solo lanza `ProviderError`** para fallas de API, transporte o respuesta. Errores de programación (`ValueError`) se propagan tal cual. |
| `aclose()` | Cierra el cliente HTTP del SDK. |

- `ExtractionRequest`: `images`, `mime_type`, `screenshot_ids`. `ExtractionResult`: `entries`, `usage` (`prompt_tokens`, `output_tokens`, `reasoning_tokens`, `finish_reason` en forma común, reemplazando `_log_response_metrics`).
- **El modelo se pasa por llamada**, no se fija en el adaptador: un mismo `GeminiProvider` atiende los tiers 1 y 2. Cambiar un modelo por otro de la misma empresa es tocar una línea de la cadena.
- **El adaptador traduce sus propias excepciones.** Ninguna clase de SDK sale del adaptador. Esto elimina el acoplamiento #1, #2, #16 del mapa.
- **Plantilla para reutilizar código:** una clase base opcional (`BaseVisionProvider`) con el flujo común (validar entrada, armar el request, llamar, normalizar la respuesta, convertir DTO a dominio, emitir el log de uso) y cuatro ganchos que cada proveedor implementa: construir el schema propio, enviar, parsear y traducir errores. El `Protocol` sirve para tipado e inyección; la clase base evita copiar y pegar.
- **Nota de credenciales ambientales:** el SDK de Anthropic resuelve credenciales en cascada (`ANTHROPIC_API_KEY` → `ANTHROPIC_AUTH_TOKEN` → perfil de login). Si el cliente se construyera sin key explícita, podría activarse el tier 4 por accidente. Diseño: el cliente solo se construye si `is_configured()` (key explícita en nuestro bloque de config); si no, el tier queda inactivo.

**Plantilla para un futuro `KimiAdapter` / `DeepSeekAdapter`:**
1. Clase que implementa el `Protocol` (heredando de `BaseVisionProvider`).
2. Bloque `KimiSettings` (key, timeout, reintentos, parámetros del modelo).
3. Una entrada en `registry.py`.
4. Una línea en la cadena (`kimi:<modelo>`) y la variable de la key.
Nada más cambia: ni `chain.py`, ni Telegram, ni el dominio.
*Salvedad:* si la API del proveedor es compatible con la de OpenAI, un `OpenAICompatibleProvider` con `base_url` configurable podría cubrirlo casi solo con config, pero no verifiqué si Kimi o DeepSeek soportan visión ni salida JSON estricta. Cada tier debería declarar sus **capacidades** (visión, schema estricto, máximo de imágenes) y validarse al arrancar; si un modelo no soporta schema estricto, su adaptador debe compensar con validación local y devolver `RESPUESTA_INVALIDA` cuando falle.

### 3.3 Clasificación canónica de errores

`FailureReason` (vocabulario único; se usa en logs, métricas y decisiones):

| Razón | Significado | Transitoria | Nivel de log |
|---|---|---|---|
| `ALTA_DEMANDA` | El proveedor está sobrecargado o no disponible (5xx: 500, 502, 503, 504, 529). No es problema de nuestra cuota. | sí | WARNING |
| `LIMITE_ALCANZADO` | Límite propio de cuota (429, o 402 de crédito). Accionable de nuestro lado. Lleva un subtipo `LimitKind`. | depende | WARNING (ERROR si `CREDITO_O_GASTO`) |
| `MODELO_NO_ENCONTRADO` | 404: modelo retirado o renombrado, config desactualizada (mismo caso que hoy). | no | ERROR |
| `AUTENTICACION` | 401/403: key inválida, revocada o sin permiso. | no | ERROR |
| `SOLICITUD_INVALIDA` | 400/413/422: payload o schema rechazado. | no | ERROR |
| `RESPUESTA_INVALIDA` | 200 pero sin JSON utilizable: truncado (`MAX_TOKENS`, `max_tokens`, `incomplete`), rechazo del modelo o JSON malformado. Sustituye a `GeminiExtractionError`. | a veces | WARNING |
| `TIMEOUT` | Timeout del cliente (no llegó respuesta). Hoy inexistente por falta de timeout. | sí | WARNING |
| `CONEXION` | Error de red/transporte. | sí | WARNING |
| `DESCONOCIDO` | Cualquier otra excepción del SDK. Se registra con la clase y el mensaje truncado. | ? | ERROR |

`LimitKind` (subtipo de `LIMITE_ALCANZADO`, para saber qué acción tomar):

| Subtipo | Qué hacer |
|---|---|
| `RATE_MINUTO` | Límite por minuto o por rampa. Esperar segundos. |
| `CUOTA_DIARIA` | Cupo diario agotado. Esperar el reinicio (Gemini: medianoche hora del Pacífico). |
| `CREDITO_O_GASTO` | Crédito agotado o tope de gasto/uso. **Pagar, recargar o subir el límite.** |
| `DESCONOCIDO` | No distinguible con el cuerpo disponible. Se guarda el cuerpo crudo truncado. |

**Mapeo por proveedor (HTTP/excepción → categoría común):**

| Condición | Gemini (`google.genai`) | OpenAI (`openai`) | Anthropic (`anthropic`) |
|---|---|---|---|
| Sobrecarga / 5xx | `ServerError` (500, 502, 503, 504) → `ALTA_DEMANDA` | `InternalServerError` (≥500; 503 `server_is_overloaded`) → `ALTA_DEMANDA` | `InternalServerError` (≥500: 500, 502, 503, 504 y **529** `overloaded_error`) → `ALTA_DEMANDA` |
| 429 por rate limit | `ClientError` 429 → `LIMITE_ALCANZADO`; subtipo por cuerpo si es identificable, si no `DESCONOCIDO` | `RateLimitError` con `slow_down` o `rate_limit_error` sin código de gasto → `LIMITE_ALCANZADO/RATE_MINUTO` | `RateLimitError` con `retry-after` → `LIMITE_ALCANZADO/RATE_MINUTO` |
| Cuota o crédito agotado | 429 de cuota diaria → `CUOTA_DIARIA` (si el cuerpo lo revela); 402 crédito → `CREDITO_O_GASTO` | 429 con `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded` u `organization_usage_limit_exceeded` → `CREDITO_O_GASTO` | 402 `billing_error` → `CREDITO_O_GASTO`; 429 **sin** `retry-after` (tope de gasto por tier) → `CREDITO_O_GASTO` (heurística por confirmar) |
| Modelo inexistente | `ClientError` 404 → `MODELO_NO_ENCONTRADO` (visto durante la fase de pruebas: `NOT_FOUND`) | `NotFoundError` → `MODELO_NO_ENCONTRADO` | `NotFoundError` → `MODELO_NO_ENCONTRADO` |
| Autenticación | 401/403 → `AUTENTICACION` | `AuthenticationError`, `PermissionDeniedError` → `AUTENTICACION` | `AuthenticationError`, `PermissionDeniedError` → `AUTENTICACION` |
| Solicitud inválida | 400 (`INVALID_ARGUMENT`, `FAILED_PRECONDITION`) → `SOLICITUD_INVALIDA` | `BadRequestError`, `UnprocessableEntityError` → `SOLICITUD_INVALIDA` | 400, 413, 422 → `SOLICITUD_INVALIDA` (un 400 por tope de gasto de la organización solo se detectaría leyendo el texto: **no** se intenta) |
| Respuesta inutilizable | `response.parsed is None`, `finish_reason` ≠ `STOP` → `RESPUESTA_INVALIDA` | `refusal` o `status: incomplete` → `RESPUESTA_INVALIDA` | `stop_reason` ∈ {`max_tokens`, `refusal`} → `RESPUESTA_INVALIDA` |
| Timeout | `httpx.TimeoutException` (no viene envuelto por el SDK) → `TIMEOUT` | `APITimeoutError` → `TIMEOUT` | `APITimeoutError` → `TIMEOUT` |
| Red | `httpx.TransportError` → `CONEXION` | `APIConnectionError` → `CONEXION` | `APIConnectionError` → `CONEXION` |

**Política de continuar o abortar.** Cada eslabón que falla con una razón "continuable" pasa al siguiente tier; con una razón "no continuable" la cadena se detiene y se informa. Se propone una lista configurable (`advance_on`) con estos valores por defecto:

| Razón | Hoy (solo Gemini) | Por defecto propuesto | ¿Cambia el comportamiento actual? |
|---|---|---|---|
| `ALTA_DEMANDA` | continúa | continúa | no |
| `LIMITE_ALCANZADO` | continúa (429) | continúa | no |
| `MODELO_NO_ENCONTRADO` | continúa | continúa | no |
| `TIMEOUT`, `CONEXION` | no existen | continúa | nuevo |
| `AUTENTICACION` | **aborta** | **continúa** (ERROR en el log) | **sí**: una key mala de un proveedor no debe bloquear a los demás |
| `SOLICITUD_INVALIDA` | **aborta** | **aborta** | no |
| `RESPUESTA_INVALIDA` | **aborta** | **decisión tuya** | según lo que elijas: probar otro modelo cuesta una llamada extra pero puede rescatar el lote |

Los dos puntos marcados requieren tu decisión (sección 6).

**Cómo llegan al log.** Formato canónico único, con proveedor + modelo + causa (ver 3.6):

```
ALTA_DEMANDA en Gemini (gemini-3.8-flash) [tier 2/4] http=503 status=UNAVAILABLE latency=20.4s
LIMITE_ALCANZADO en OpenAI (gpt-5.6-terra) [tier 3/4] kind=CREDITO_O_GASTO http=429 code=credit_balance_exhausted
MODELO_NO_ENCONTRADO en Gemini (gemini-3.6-flash) [tier 1/4] http=404 status=NOT_FOUND
```

**Mensaje al usuario.** La UI recibe `AllProvidersFailedError.dominant_reason` y decide el texto. Hoy 429 y 5xx comparten el aviso de "alta demanda... Reintentar en unos segundos" y un test lo fija (`test_telegram_bot.py:279-306`). Con la clasificación nueva `LIMITE_ALCANZADO` puede tener un texto propio (p. ej., que reintentar no ayudará hasta el reinicio). Qué mostrar es decisión tuya y cambia ese test.

### 3.4 Configuración externalizada por proveedor

Estructura propuesta:

```
Settings                                   (variables actuales sin cambio: telegram_bot_token, pdf_*, log_level)
 ├─ vision: VisionSettings
 │     chain              lista ordenada de tiers
 │     chain_deadline_seconds   tope total de espera de toda la cadena
 │     advance_on         razones que continúan
 │     attempts_log_path  (opcional) archivo JSONL de intentos
 ├─ gemini:    GeminiSettings      prefijo GEMINI_     api_key, timeout, reintentos, retry_status_codes, thinking (sin temperature: deprecado en Gemini 3.x)
 ├─ openai:    OpenAISettings      prefijo OPENAI_     api_key, timeout, reintentos, esfuerzo de razonamiento, nivel de detalle de imagen
 └─ anthropic: AnthropicSettings   prefijo ANTHROPIC_  api_key, timeout, reintentos, effort/thinking
```

- **Nada compartido entre bloques.** Cada uno tiene su propia key opcional, timeouts, política de reintento y parámetros del modelo.
- **Activo/inactivo se deriva de la key**, no de una bandera separada: `is_configured()` = key explícita no vacía. Se puede permitir una bandera `enabled` opcional por tier solo para apagar uno a mano.
- **Claves como `SecretStr | None`** (hoy `gemini_api_key: str` es texto plano y obligatoria): evita filtrarlas en `repr` o logs. Requiere `get_secret_value()` en el punto de uso.
- **Validación al arrancar (fail-fast):** debe haber al menos un tier activo; los modelos declarados deben corresponder a un proveedor registrado; una key presente sin tier que la use genera aviso. Hoy `Settings()` ya falla rápido si falta una variable obligatoria; se conserva ese espíritu.

**Cómo se declara la cadena** (opciones):

| Opción | Forma | Ventajas | Desventajas |
|---|---|---|---|
| **C1. Una variable de entorno** | `VISION_CHAIN=gemini:gemini-3.6-flash,gemini:gemini-3.8-flash,openai:gpt-5.6-terra,anthropic:claude-sonnet-5` | Cambiar un modelo o el orden es una línea, sin redeploy de código. Es exactamente "una línea de configuración". | Formato propio que hay que parsear y validar. |
| **C2. Lista en `config.py`** | Constante tipada en código | Tipado fuerte, cero parseo. | Cambiar el orden o el modelo exige tocar código y redeployar. |
| **C3. Archivo estructurado** (YAML/JSON) | Un archivo aparte | Admite más campos por tier (timeouts propios, `enabled`). | Otro archivo que desplegar y mantener; más superficie. |

Recomendación de diseño: **C1 con default definido en código** (la cadena decidida), para que sin `.env` nuevo el sistema arranque con los 4 tiers y el 4 inactivo por falta de key.

**Política de reintento (decisión de diseño con trade-off):**

| Opción | Cómo | Pro | Contra |
|---|---|---|---|
| **R1. Dejar los reintentos de cada SDK, con límites explícitos** | `max_retries` y `timeout` por proveedor; en Gemini además `http_status_codes` sin 429 | Reutiliza el manejo de `Retry-After` de cada SDK; menos código | OpenAI y Anthropic solo permiten `max_retries` (todo o nada): reintentarían también los 429 por crédito agotado |
| **R2. Reintentos centralizados en la cadena, SDK en cero** | `max_retries=0` / `attempts=1` en los tres; la cadena reintenta solo `ALTA_DEMANDA`, `TIMEOUT`, `CONEXION` y `RATE_MINUTO` | Política uniforme; nunca reintenta un 429 no transitorio; el conteo de intentos en el log es exacto | Hay que implementar backoff y respetar `Retry-After` a mano |

Este punto lo dejo propuesto, no decidido. Con cualquiera de las dos, el timeout por tier y el `chain_deadline_seconds` son obligatorios.

### 3.5 Cadena de fallback configurable

Cada eslabón es un `TierSpec(provider_id, model)`. Comportamiento de `FallbackChain`:

1. Recorre los tiers en orden.
2. Si el proveedor del tier **no está configurado**, no falla ni lanza excepción: registra
   `INFO  tier 4 (Anthropic) no configurado, se omite`
   y un `ProviderAttempt` con `outcome=SALTADO_INACTIVO`, y sigue.
3. Si está activo, llama `provider.extract(...)`. Éxito → devuelve. Falla con `ProviderError` → registra el intento y, según `advance_on`, continúa o se detiene.
4. Si se agotan los tiers → `AllProvidersFailedError(attempts, dominant_reason)`.
5. Al arrancar, un único INFO resume la cadena resuelta:
   `Cadena de visión: 1) Gemini gemini-3.6-flash [activo] 2) Gemini gemini-3.8-flash [activo] 3) OpenAI gpt-5.6-terra [activo] 4) Anthropic claude-sonnet-5 [inactivo: sin ANTHROPIC_API_KEY]`

**Dato útil para decidir la activación del tier 4:** el evento `SALTADO_INACTIVO` del tier 4 cuenta cuántas veces la cadena **llegó** hasta ahí, es decir, cuántas veces también falló OpenAI. Ese número, sin gastar nada en Anthropic, es la evidencia directa para decidir si activarlo.

**Regla de `dominant_reason`** (para el mensaje al usuario): propuesta por prioridad, primero `LIMITE_ALCANZADO/CREDITO_O_GASTO` (acción del operador), luego `LIMITE_ALCANZADO`, luego `ALTA_DEMANDA`, luego el resto. A confirmar.

**Componente opcional a evaluar: cortacircuitos por tier.** Si Gemini lleva horas sobrecargado, cada solicitud gasta decenas de segundos en los tiers 1 y 2 antes de llegar a OpenAI. Un cortacircuitos (tras N fallas `ALTA_DEMANDA`/`TIMEOUT` seguidas, saltar ese tier durante un enfriamiento y reprobarlo con una llamada de sondeo) baja la latencia y el gasto. Costo: estado en memoria, más casos de prueba, y el riesgo de saltar un tier que ya se recuperó. No lo pediste; lo planteo porque es el problema real que motivó esta tarea.

### 3.6 Registro por proveedor consultable

Cada intento genera un registro canónico `ProviderAttempt`:

| Campo | Ejemplo | Para qué |
|---|---|---|
| `ts` | ISO 8601 UTC | series de tiempo |
| `chain_request_id` | uuid corto | agrupar los intentos de una misma solicitud |
| `tier`, `tiers_total` | `3`, `4` | posición en la cadena |
| `provider`, `model` | `openai`, `gpt-5.6-terra` | identificar |
| `outcome` | `EXITO` / `FALLO` / `SALTADO_INACTIVO` | contar |
| `reason`, `limit_kind` | `LIMITE_ALCANZADO`, `CUOTA_DIARIA` | el requisito central |
| `http_status`, `provider_code` | `429`, `credit_balance_exhausted` | diagnóstico fino sin depender del texto |
| `latency_ms`, `images`, `payload_bytes` | `20400`, `15`, `2442041` | rendimiento (ejemplo ilustrativo de payload) |
| `prompt_tokens`, `output_tokens`, `reasoning_tokens`, `finish_reason` | | en éxitos; reemplaza `GEMINI_RESPONSE` (ya decidido permanente) |
| `provider_request_id` | `req_011...` | abrir tickets con el proveedor |

Regla de privacidad: **nunca** se registra contenido (direcciones, nombres, imágenes). Solo conteos, códigos y tiempos, como hace hoy `_log_response_metrics`.

**Opciones de almacenamiento** (elegir según dónde correrá la marcha blanca):

| Opción | Cómo se consulta | Pro | Contra |
|---|---|---|---|
| **T1. Log estructurado a stdout** (una línea JSON por intento + texto legible) | `jq`, o la búsqueda de logs del hosting | Funciona en cualquier entorno, cero infraestructura | Depende de la retención de logs del hosting; no se agrega solo |
| **T2. T1 + archivo JSONL rotativo** (`attempts_log_path`) | `jq`, pandas, DuckDB | Conserva la historia local; ideal para una marcha blanca | En hostings con disco efímero (p. ej., el despliegue a Render que planeas) el archivo se pierde en cada reinicio |
| **T3. T1 + tabla SQLite** (`provider_attempts`) | SQL: `GROUP BY provider, reason` | Consultas directas | Mismo problema de persistencia en disco efímero; agrega un componente de datos |
| **T4. T1 + contadores en memoria** expuestos en un endpoint (`GET /internal/vision-stats`) | Conteos por (proveedor, modelo, razón) | Respuesta inmediata | Se reinician con el proceso; un endpoint nuevo requiere decidir su protección |

Ejemplo de consulta con T1/T2 (conteo por proveedor y causa, solo ilustrativo): filtrar líneas con `outcome=FALLO` y agrupar por `provider` y `reason`. Con T3 es un `GROUP BY` directo.

Pregunta clave para decidir: **¿dónde correrá la marcha blanca, local o en Render?** Si es local, T2 basta. Si es en Render, T1 más la retención de logs del hosting, o T4 si quieres un conteo rápido.

### 3.7 Impacto en el resto del sistema

| Pieza | Cambio |
|---|---|
| `telegram_bot.py` | Deja de importar `google.genai`. Depende de `RouteExtractor` (Protocol) y de `AllProvidersFailedError`. El bloque `is_high_demand` (líneas 325-327) se reemplaza por una decisión sobre `dominant_reason`. La clave de `bot_data` deja de llamarse `gemini_extractor`. |
| `main.py` | Construye los proveedores desde `Settings`, arma la cadena y la inyecta. `aclose()` cierra todos los proveedores activos. |
| `config.py` | Bloques anidados y la cadena (ver 3.4). |
| `requirements.txt` | Agrega `openai` y `anthropic`. **Aunque el tier 4 esté inactivo, `anthropic` debe estar instalado** para que el adaptador se importe y se pruebe. |
| `.env.example` | Documenta `OPENAI_*`, `ANTHROPIC_*`, `VISION_CHAIN`. **No se agregan keys reales.** |
| Tests | Se reorganizan por capa (ver 3.8). |
| `scripts/test_gemini_connection.py` | Se ajusta al puerto nuevo o se generaliza a "probar un proveedor". |
| Dominio | Sin cambios. |

### 3.8 Estrategia de pruebas (sin costo de cuota)

- **Contrato del Protocol**: proveedores falsos que fabrican cada `FailureReason`; la cadena se prueba sin ningún SDK.
- **Mapeo de errores por adaptador**: pruebas parametrizadas con excepciones sintéticas de cada SDK (`errors.ServerError(503, {...})`, `openai.RateLimitError`, `anthropic.InternalServerError` con `status_code=529`, etc.), una fila por celda de la tabla de 3.3. El tier 4 se prueba así completo sin key.
- **Cadena**: tier inactivo se salta con el INFO exacto; todos fallan → `AllProvidersFailedError`; razón no continuable aborta; `chain_deadline_seconds` corta.
- **Telemetría**: cada intento produce exactamente un `ProviderAttempt` con la combinación proveedor + modelo + razón.
- **Inyección de fallas independiente del proveedor**: un decorador `FaultInjectingProvider` envolvería cualquier proveedor para simular 429/503/404 sin tocar los adaptadores (ver sección 5 sobre el destino de los hooks actuales).
- **Regresión de comportamiento actual**: los escenarios de `test_gemini_client.py` se conservan como pruebas de `GeminiProvider` + cadena de 2 tiers.

---

## 4. Opciones que requieren tu decisión: compatibilidad hacia atrás vs. arquitectura más limpia

Ambas opciones comparten lo inevitable: la taxonomía canónica, el cambio en `telegram_bot.py:325-327`, el destino de los hooks de prueba, las dos dependencias nuevas y la reescritura de los 4 tests de Telegram que fabrican errores del SDK. Difieren en cuánto se conserva del diseño actual.

### Opción A: aditiva, prioriza compatibilidad

- Se conserva `GeminiRouteExtractor` con su firma `extract_entries(...)` como **fachada** que delega en la cadena; `MODEL_NAME` y `FALLBACK_MODEL_NAME` siguen exportados.
- Variables `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL` siguen funcionando: si no hay `VISION_CHAIN`, la cadena por defecto se arma a partir de ellas (tiers 1 y 2) y se agregan los tiers 3 y 4.
- Solo OpenAI y Anthropic nacen con bloques anidados; Gemini conserva sus campos planos (más una vista anidada).
- La mayoría de `test_gemini_client.py` sigue verde con cambios mínimos.

| Pros | Contras |
|---|---|
| Menos archivos tocados y menor riesgo de regresión | Conviven dos estilos de configuración (plano para Gemini, anidado para el resto) |
| `.env` existente y despliegues actuales siguen válidos | Gemini queda como caso especial permanente: la promesa de "solo agregar un adaptador" se cumple, pero con una excepción |
| Se puede entregar en etapas con la app siempre desplegable | El nombre `GeminiRouteExtractor` seguirá siendo engañoso cuando realmente orquesta 3 proveedores |
| Los tests existentes sirven como red de seguridad | Deuda técnica: habrá que limpiar la fachada y los campos planos más adelante |

### Opción B: limpia, reestructuración completa

- Paquete nuevo, `GeminiRouteExtractor` desaparece y se reemplaza por `GeminiProvider` + `FallbackChain`; la dependencia de la UI pasa a llamarse de forma neutral (`route_extractor`).
- Configuración anidada para los tres proveedores; `GEMINI_MODEL` y `GEMINI_FALLBACK_MODEL` quedan obsoletas (la cadena las reemplaza). Variante B': mantenerlas como alias con aviso de deprecación.
- Los tests se reescriben por capa.

| Pros | Contras |
|---|---|
| Un solo estilo y ningún proveedor privilegiado: agregar Kimi o DeepSeek es exactamente adaptador + bloque + línea | Mucho más churn: reescribir `gemini_client.py`, `main.py`, `config.py`, hasta 21 tests (los 11 de `test_gemini_client.py` y los 10 de `test_telegram_bot.py`, de los cuales 4 fabrican errores del SDK), el script standalone y `.env.example` |
| Sin fachada ni alias que limpiar después | Mayor riesgo de regresión en un flujo que ya validaste (A y B) |
| La cadena y el registro nacen simétricos para los 3 proveedores | Los `.env` y las variables del hosting hay que migrarlos (con B': solo avisos) |
| Los hooks de prueba se rediseñan una sola vez | Más tiempo hasta el primer despliegue funcional, salvo que se entregue por etapas igualmente |

**No recomiendo una de las dos.** Si tu prioridad es llegar rápido a la marcha blanca con el flujo estable, A la favorece; si tu prioridad es no arrastrar excepciones cuando llegue el cuarto o quinto proveedor, B la favorece. Una mezcla razonable (B con alias) existe, pero es tu decisión cuánto refactor asumir.

---

## 5. Riesgos y secuenciación

1. **C sigue sin validarse y sus hooks viven en `gemini_client.py`** (líneas 13, 50-54, 149-162, 170-174, 259-281). Cualquiera de las dos opciones mueve o elimina ese archivo. Opciones: (a) terminar C antes de refactorizar (bloqueado por la sobrecarga de Google durante la fase de pruebas), (b) portar los hooks a un `FaultInjectingProvider` genérico y validar C después, (c) cerrar C como parcial. `GEMINI_RESPONSE` ya se decidió permanente y se generaliza como `UsageInfo`.
2. **Latencia acumulada.** Sin timeouts explícitos, 4 tiers pueden esperar decenas de minutos (defaults de 10 min en dos SDK). `timeout_seconds` por proveedor y `chain_deadline_seconds` son obligatorios en el diseño. Los valores concretos están abiertos (sección 6).
3. **Coste no acotado por pulsación.** Cada "Reintentar" del usuario puede recorrer los tiers 1-3 (los pagos son 3 y 4). Conviene acotar reintentos por chat o usar el cortacircuitos de 3.5. Es una decisión de producto.
4. **Datos de clientes salen a más proveedores.** Con el tier 3 activo, las capturas (direcciones, números de bulto) viajan a OpenAI cuando Gemini falla. Lo decidiste; lo dejo anotado por si afecta acuerdos con tu operación.
5. **Verificaciones pendientes contra el SDK real** (no se instala nada en esta fase): ruta exacta de `error.code` en OpenAI, clase exacta para 402 y 413 en Anthropic, y el cuerpo real de un 429 de Gemini. El mapeo debe registrar el cuerpo crudo truncado durante la marcha blanca para corregir la clasificación con datos.
6. **Formato de imagen y schema por proveedor.** `screenshot_index` con `minimum` no lo soporta Anthropic; raíz tipo lista no confirmada en OpenAI/Anthropic. Cada adaptador es dueño de su schema y valida localmente contra el DTO original.
7. **Sin cobertura de calidad todavía.** La fidelidad de extracción de OpenAI/Anthropic frente a Gemini no está medida (la comparación de fidelidad quedó en pausa). Activar el tier 3 ahora es una decisión tomada; conviene que la marcha blanca registre resultados para poder compararlos.

## 6. Preguntas abiertas para tu revisión

1. **Opción A o B** (o B con alias) para el refactor.
2. **`AUTENTICACION` continúa** al siguiente tier (cambio respecto de hoy) o aborta.
3. **`RESPUESTA_INVALIDA`** (JSON truncado o rechazo del modelo): ¿continúa al siguiente tier o aborta como hoy?
4. **Mensaje al usuario** para `LIMITE_ALCANZADO`: ¿texto propio o el de alta demanda? (cambia un test fijado hoy).
5. **Política de reintento:** R1 (SDK con límites) o R2 (centralizada).
6. **Timeouts:** valor por proveedor y `chain_deadline_seconds`; y si quieres el cortacircuitos.
7. **Almacenamiento de intentos:** T1, T2, T3 o T4, según dónde corra la marcha blanca.
8. **Formato de la cadena:** C1 (variable de entorno), C2 (código) o C3 (archivo).
9. **Hooks de prueba y validación de C:** (a), (b) o (c) de la sección 5.
10. **`SecretStr`** para las keys.
11. **Dónde guardar este documento:** hoy vive fuera del repo (scratchpad). ¿Lo copio a `docs/` del proyecto?

## 7. Plan de implementación propuesto (para autorizar después)

| Etapa | Contenido | Puerta de calidad |
|---|---|---|
| 0 | Decisiones de la sección 6 | tu aprobación |
| 1 | Tipos canónicos, puerto, cadena, telemetría, `GeminiProvider` con el comportamiento actual + timeouts | tests de comportamiento actual verdes; ruff y mypy |
| 2 | `OpenAIProvider` (tier 3 activo) con mapeo de errores y pruebas | pruebas de mapeo por celda; una prueba real solo con tu autorización |
| 3 | `AnthropicProvider` completo pero inactivo | pruebas de mapeo y de "salta limpio sin key" |
| 4 | Config, `.env.example`, `requirements.txt`, documentación | arranque con y sin keys |
| 5 | Marcha blanca con registro de intentos | conteo de `FALLO` por proveedor y causa; conteo de `SALTADO_INACTIVO` del tier 4 |

Cada etapa se entrega con diff a la vista; sin commit; sin exponer keys; sin activar facturación de Google Cloud.
