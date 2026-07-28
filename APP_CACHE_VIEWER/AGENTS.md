# AGENTS.md - Cache Viewer

## Vision General

Aplicacion independiente para inspeccionar visualmente la cache del dataset generada por MyIAModelChat.

## Roles

### Developer
Implementa las funcionalidades del visor PyQt5.

- Crear y mantener `cache_viewer.py`
- Implementar las 4 pestañas del visor
- Mantener la aplicacion independiente

### Reviewer
Revisa calidad de codigo y funcionalidad.

- Validar imports y dependencias
- Verificar que la aplicacion funcione de forma independiente
- Revisar estilo de codigo

## Tareas por Agente

| Agente | Tarea | Archivos |
|--------|-------|----------|
| Developer | Crear ventana principal | `cache_viewer.py` |
| Developer | Implementar pestaña Resumen | `cache_viewer.py` |
| Developer | Implementar pestaña Estadisticas | `cache_viewer.py` |
| Developer | Implementar pestaña Muestras | `cache_viewer.py` |
| Developer | Implementar pestaña Vocabulario | `cache_viewer.py` |
| Developer | Mantener requirements.txt | `requirements.txt` |
| Reviewer | Validar dependencias | `requirements.txt` |
| Reviewer | Probar ejecucion independiente | Testing |

## Convenciones

- Python 3.12+
- Type hints obligatorios en funciones publicas
- Docstrings en clases y funciones publicas
- Sin emojis en el codigo

## Estructura del Proyecto

```
APP_CACHE_VIEWER/
├── __init__.py             # Exporta CacheViewer
├── cache_viewer.py         # Aplicacion principal
├── requirements.txt        # Dependencias independientes
├── AGENTS.md               # Este archivo
└── ROAPMAP_CACHE_VIEWER.md # Plan de desarrollo
```

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar con ruta explicita
python cache_viewer.py /ruta/a/dataset_cache

# Ejecutar con selector de carpetas
python cache_viewer.py
```

## Notas de Implementacion

- La aplicacion es completamente independiente de MyIAModelChat
- No requiere archivos del proyecto padre para funcionar
- Solo necesita la carpeta `dataset_cache/` con los archivos generados
- El visor NO modifica la cache, solo la lee
