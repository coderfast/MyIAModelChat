# ROADMAP - Cache Viewer

## Vision General

Aplicacion grafica independiente para inspeccionar la cache generada por DataPreparer.

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
- [x] Crear `ROAPMAP_CACHE_VIEWER.md`
- [x] Crear `AGENTS.md`
- [x] Implementar `cache_viewer.py` con 4 pestañas

### Pendiente
- [ ] Agregar grafico de barras en pestaña Statistics
- [ ] Agregar export de datos a CSV/JSON
- [ ] Agregar mas filtros en pestaña Vocabulario
- [ ] Agregar atajos de teclado

## Funcionalidades

### Pestaña Resumen
- Tamaño total de archivos
- Tabla con archivos: nombre, tamaño, fecha, tipo
- Indicador de existencia de cache

### Pestaña Estadisticas
- Total de samples
- Desglose por fuente (AIML, HF, PDF, EPUB, Web, CSV)
- Estadisticas de longitudes de texto

### Pestaña Muestras
- Selector de rango (primeros/ultimos/aleatorios)
- Tabla con texto truncado y conteo de tokens
- Panel lateral con texto completo

### Pestaña Vocabulario
- Campo de busqueda por token o ID
- Filtro de tokens especiales
- Conteo total y paginacion

## Dependencias

| Paquete | Version | Uso |
|---------|---------|-----|
| PyQt5 | >=5.15.0 | Ventana grafica |
| datasets | >=2.0.0 | Cargar Arrow files |

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python cache_viewer.py [ruta_a_dataset_cache]
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
│   │   ├── QComboBox (range selector)
│   │   ├── QTableWidget (dataset preview)
│   │   └── QTextEdit (full text)
│   └── Tab 4: Vocabulary
│       ├── QLineEdit (search)
│       ├── QComboBox (filter)
│       └── QTableWidget (tokens)
└── StatusBar (loading messages)
```
