# AGENTS.md - Cache Viewer

## Vision General

Aplicacion grafica independiente para inspeccionar visualmente la cache del dataset generada por MyIAModelChat. Desarrollada con PyQt5, incluye carga lazy loading con QThread para manejar datasets grandes de forma eficiente.

## Roles

### Developer
Implementa y mantiene las funcionalidades del visor PyQt5.

- Crear y mantener `cache_viewer.py`
- Implementar las 4 pestañas del visor
- Mantener la aplicacion independiente
- Optimizar rendimiento con lazy loading

### Reviewer
Revisa calidad de codigo y funcionalidad.

- Validar imports y dependencias
- Verificar que la aplicacion funcione de forma independiente
- Revisar estilo de codigo
- Probar con datasets grandes

## Tareas por Agente

| Agente | Tarea | Archivos |
|--------|-------|----------|
| Developer | Crear ventana principal | `cache_viewer.py` |
| Developer | Implementar pestaña Summary | `cache_viewer.py` |
| Developer | Implementar pestaña Statistics | `cache_viewer.py` |
| Developer | Implementar pestaña Samples (lazy loading) | `cache_viewer.py` |
| Developer | Implementar pestaña Vocabulary | `cache_viewer.py` |
| Developer | Implementar filtrado por fuente | `cache_viewer.py` |
| Developer | Mantener requirements.txt | `requirements.txt` |
| Reviewer | Validar dependencias | `requirements.txt` |
| Reviewer | Probar ejecucion independiente | Testing |

## Convenciones

- Python 3.12+
- Type hints obligatorios en funciones publicas
- Docstrings en clases y funciones publicas
- Sin emojis en el codigo
- Seguir estilo PEP 8

## Estructura del Proyecto

```
APP_CACHE_VIEWER/
├── __init__.py                    # Exporta CacheViewer
├── cache_viewer.py                # Aplicacion principal (1025 lineas)
├── requirements.txt               # Dependencias independientes
├── AGENTS.md                      # Este archivo
└── ROADMAP_CACHE_VIEWER.md        # Plan de desarrollo
```

## Arquitectura

```
CacheViewer (QMainWindow)
├── Header (cache path + change button)
├── QTabWidget
│   ├── Tab 1: Summary
│   │   ├── QGroupBox (cache info)
│   │   └── QTableWidget (files)
│   ├── Tab 2: Statistics
│   │   ├── QGroupBox (general stats)
│   │   └── QTableWidget (source breakdown)
│   ├── Tab 3: Samples
│   │   ├── QComboBox (source filter)
│   │   ├── VirtualScrollTable (lazy loading)
│   │   └── QTextEdit (full text detail)
│   └── Tab 4: Vocabulary
│       ├── QLineEdit (search)
│       ├── QComboBox (filter)
│       └── QTableWidget (tokens)
└── StatusBar (loading messages)

Workers:
├── DatasetWorker (initial block load)
└── ContinuousLoaderWorker (scroll-based loading)
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

## Dependencias

| Paquete | Version | Uso |
|---------|---------|-----|
| PyQt5 | >=5.15.0 | Ventana grafica |
| datasets | >=2.0.0 | Cargar Arrow files |

## Notas de Implementacion

- La aplicacion es completamente independiente de MyIAModelChat
- No requiere archivos del proyecto padre para funcionar
- Solo necesita la carpeta `dataset_cache/` con los archivos generados
- El visor NO modifica la cache, solo la lee
- Utiliza lazy loading con QThread para manejar datasets grandes
- Filtrado por fuente (AIML, HF, PDF, EPUB, Web, CSV)
- Pre-calculo de longitudes de tokens para mejor rendimiento
