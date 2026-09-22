"""Extraccion de rutas por vision con una cadena de proveedores en fallback.

Capa de aplicacion: define el puerto (`ports`), los tipos canonicos de falla
(`errors`), el orquestador (`chain`) y la telemetria. No importa ningun SDK de
proveedor: eso vive en `app.adapters.vision`.

Ver docs/ARQUITECTURA_FALLBACK_MULTIPROVEEDOR.md.
"""
