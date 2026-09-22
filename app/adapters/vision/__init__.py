"""Adaptadores de proveedores de vision (los unicos que importan un SDK).

Cada modulo implementa `VisionExtractorProvider` sobre el SDK de un
proveedor y traduce sus excepciones a `ProviderError`. `factory` arma la
cadena a partir de `Settings`.
"""
