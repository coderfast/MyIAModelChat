# ROADMAP - Cache Viewer

## Vision General

Aplicacion grafica independiente para inspeccionar visualmente la cache del dataset generada por MyIAModelChat. Desarrollada con PyQt5, incluye carga lazy loading con QThread para manejar datasets grandes de forma eficiente.

## Estructura de la Cache

```
dataset_cache/
├── prepared_dataset/              # HuggingFace Arrow format
│   ├── data-00000-of-00001.arrow  # Datos columnares
│   ├── dataset_info.json          # Schema de features
│   └── state.json                 # Estado del dataset
├── cache_metadata.pkl             # Info tokenizer BPE
├── dataset_stats.pkl              # Estadisticas generales
├── sentencepiece.model            # Modelo BPE binario
└── sentencepiece.vocab            # Vocabulario texto
```

## Estados

### Completado
- [x] Crear carpeta `APP_CACHE_VIEWER/`
- [x] Crear `__init__.py`
- [x] Crear `requirements.txt` independiente
- [x] Crear `ROADMAP_CACHE_VIEWER.md`
- [x] Crear `AGENTS.md`
- [x] Implementar `cache_viewer.py` con 4 pestañas (1025 lineas)
- [x] Implementar lazy loading con QThread
- [x] Implementar filtrado por fuente (AIML, HF, PDF, EPUB, Web, CSV)
- [x] Pre-calculo de longitudes de tokens para mejor rendimiento
- [x] VirtualScrollTable con carga progresiva al hacer scroll

### Pendiente
- [ ] Agregar grafico de barras en pestaña Statistics (matplotlib)
- [ ] Agregar export de datos a CSV/JSON
- [ ] Agregar mas filtros en pestaña Vocabulario (rango de scores)
- [ ] Agregar atajos de teclado
- [ ] Agregar modo oscuro/claro
- [ ] Agregar busqueda avanzada en Samples

## Funcionalidades

### Pestaña Summary
- Indicador de existencia de cache (verde/rojo)
- Tamano total de archivos (B/KB/MB)
- Fecha de ultima modificacion
- Tabla con archivos: nombre, tamano, fecha, tipo

### Pestaña Statistics
- Total de samples
- Estadisticas de longitudes de texto (avg, min, max)
- Desglose por fuente (AIML, HF, PDF, EPUB, Web, CSV)

### Pestaña Samples
- Filtro por fuente con QComboBox
- VirtualScrollTable con lazy loading (bloques de 500)
- Barra de progreso de carga
- Panel lateral con texto completo al hacer clic
- Soporte para datasets grandes (>100K samples)

### Pestaña Vocabulary
- Campo de busqueda por token o ID
- Filtro: All, Special Tokens, Regular Tokens
- Conteo total y filtros activos
- Tabla con ID, Token, Score

## Dependencias

| Paquete | Version | Uso |
|---------|---------|-----|
| PyQt5 | >=5.15.0 | Ventana grafica |
| datasets | >=2.0.0 | Cargar Arrow files |

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar con ruta explicita
python cache_viewer.py /ruta/a/dataset_cache

# Ejecutar con selector de carpetas
python cache_viewer.py
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

## Clases Principales

| Clase | Responsabilidad |
|-------|-----------------|
| `CacheViewer` | Ventana principal, coordina UI y datos |
| `DatasetWorker` | Carga inicial de bloques en background |
| `ContinuousLoaderWorker` | Carga continua al hacer scroll |
| `LazyTableModel` | Modelo de tabla con lazy loading |
| `VirtualScrollTable` | QTableView con senal de scroll |
