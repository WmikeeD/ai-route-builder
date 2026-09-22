# Directrices de Desarrollo - AI Route Builder

## Visión General
Bot de Telegram para el procesamiento de capturas de pantalla de rutas de entrega, extracción estructurada de rutas logísticas vía Gemini Flash y exportación en PDF para tu planificador de rutas.

## Stack Tecnológico
- Python 3.11+
- FastAPI
- python-telegram-bot (Async)
- google-genai (Gemini 1.5/2.5 Flash)
- ReportLab (PDF)
- Pydantic v2, Pytest, Ruff, Mypy

## Comandos Principales

### Entorno y Ejecución Local
- Crear venv: `python -m venv venv && source venv/bin/activate`
- Instalar dependencias: `pip install -r requirements.txt`
- Ejecutar backend/bot: `uvicorn app.main:app --reload`

### Calidad de Código y Formateo
- Linter y Formateador: `ruff check . --fix`
- Verificación de tipos: `mypy app/`

### Pruebas Unitarias
- Ejecutar tests: `pytest`
- Tests con cobertura: `pytest --cov=app tests/`

## Estándares de Arquitectura y Estilo
- **Clean Architecture:** Mantener la separación estricta entre `/adapters`, `/domain` y `/services`. La lógica de dominio no debe importar frameworks ni librerías externas.
- **Tipado:** Uso obligatorio de *type hints* en todas las funciones y clases.
- **Validación:** Modelos de datos definidos estrictamente con Pydantic v2.
- **Formato:** Adherencia a PEP 8 vía Ruff.

## Convenciones de Commits (Conventional Commits)
- `feat:` Nuevas funcionalidades (ej. `feat(bot): add inline keyboard flow`)
- `fix:` Corrección de errores (ej. `fix(dedup): prevent keyerror on missing package id`)
- `refactor:` Mejoras de código sin cambiar funcionalidad
- `test:` Inclusión o ajuste de pruebas unitarias