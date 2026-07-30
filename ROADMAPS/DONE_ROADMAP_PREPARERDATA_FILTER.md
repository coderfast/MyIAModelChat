# ROADMAP: Sistema de Filtrado de Datos - dataset_preparer

## Estado Actual del Pipeline

```
[1/7] Load sources
[2/7] Combine + standardize
[3/7] Source validation (--validate-sources)
[4/7] Thinking generation (--generate_thinking)
[5/7] Deduplication (--enable-dedup)
[6/7] Quality filtering (--enable-quality-filter)
[7/7] Language filtering (--enable-lang-filter)
```

## Problemas de Contaminación Identificados

| # | Problema | Ubicación | Impacto |
|---|----------|-----------|---------|
| 1 | Oversampling CSV 20x sin control | `data_preparer.py:1001` | Skew extremo hacia datos CSV |
| 2 | Sin deduplicación cross-source | `data_preparer.py:1789` | Mismo texto en AIML+PDF+Web se mantiene |
| 3 | Filtrado de basura incompleto | `data_preparer.py:251` | No detecta: URLs, emails, código, números telefónicos |
| 4 | Filtrado de calidad superficial | `data_preparer.py:282` | Solo alpha ratio, no detecta: boilerplate, headers, nav text |
| 5 | Paso de 'unknown' en filtro de idioma | `data_preparer.py:460` | Textos en idiomas no deseados pasan sin filtro |
| 6 | Sin detección de data/test leakage | No existe | Modelos memorizan en vez de generalizar |
| 7 | Thinking duplication | `data_preparer.py:1765-1766` | Ambas versiones (original+thinking) se mantienen, duplicando |
| 8 | Sin control de balance cross-source | No existe | Un source puede dominar el dataset combinado |

---

## Arquitectura Propuesta

```
dataset_preparer/
├── contamination/                    # NUEVO módulo
│   ├── __init__.py
│   ├── filters.py                   # Filtros de calidad (mejorado)
│   ├── dedup.py                     # Deduplicación cross-source
│   ├── balance.py                   # Control de balance entre fuentes
│   ├── leakage.py                   # Detección de test/train leakage
│   ├── audit.py                     # Sistema de reportes de auditoría
│   └── reports/                     # Directorio de reportes generados
│       └── .gitkeep
├── data_preparer.py                 # Modificado: integrar pipeline de filtrado
└── source_validators.py             # Modificado: mejorar validadores existentes
```

---

## Detalle de Componentes

### 1. `contamination/filters.py` - Filtros de Calidad Mejorados

Extiende `filter_by_quality()` existente con detección de:

- **Basura/Noise**: URLs, emails, números telefónicos, códigos de error, paths de sistema
- **Boilerplate**: Headers de navegación, footers de copyright, textos de cookie/privacy
- **Código**: Bloques de código fuente, comentarios de programación, snippets
- **Contenido vacío**: Solo números, solo puntuación, solo caracteres especiales
- **Formateo corrupto**: Caracteres de reemplazo U+FFFD, secuencias de escape, NUL bytes
- **Repetición excesiva**: Mismo patrón de palabras repetido >3 veces
- **Longitud anómala**: Textos extremadamente largos (>500 palabras) o cortos (<3 palabras)

### 2. `contamination/dedup.py` - Deduplicación Cross-Source

Mejora el sistema actual:

- **Exact dedup**: Texto idéntico normalizado (ya existe)
- **Near dedup**: MinHash LSH con umbrales configurables (ya existe)
- **Cross-source dedup**: NUEVO - detecta duplicados entre fuentes diferentes
- **Semantic dedup**: Opcional - agrupa textos con significado similar
- **Preservación de metadata**: Marca el source de origen al deduplic
- **Configuración**: `--dedup-mode exact|near|cross|all` (default: `all`)

### 3. `contamination/balance.py` - Control de Balance

- **Ratio máximo por source**: Evita que un source domine (ej: CSV max 30% del total)
- **Undersampling inteligente**: Reduce oversampling manteniendo variedad
- **Muestreo estratificado**: Distribuye uniformemente entre fuentes
- **Configuración**: `--max-source-ratio 0.3` (default: 0.3 = 30%)

### 4. `contamination/leakage.py` - Detección de Data Leakage

- **N-gram overlap**: Detecta si samples comparten n-grams largos (>5 palabras)
- **Train/test split check**: Opcional - verifica que no haya overlap con test set
- **Repetition detection**: Detecta fragmentos que se repiten en muchos samples
- **Configuración**: `--leakage-threshold 0.5` (default: 0.5)

### 5. `contamination/audit.py` - Sistema de Reportes

Genera reportes JSON estructurados:

```json
{
  "timestamp": "2026-07-30T10:00:00",
  "pipeline_version": "1.0",
  "filters_applied": ["quality", "dedup", "balance", "leakage"],
  "per_source": {
    "aiml": {
      "original_count": 5000,
      "after_quality": 4800,
      "after_dedup": 4500,
      "discarded": {
        "empty": 50,
        "too_short": 80,
        "noise": 40,
        "boilerplate": 30,
        "code": 0,
        "encoding": 0
      }
    }
  },
  "cross_source_duplicates": 120,
  "balance_report": {
    "aiml": 0.35,
    "pdf": 0.25,
    "csv": 0.15,
    "hf": 0.15,
    "web": 0.10
  },
  "final_count": 12000,
  "total_discarded": 1500,
  "retention_rate": 0.85
}
```

Ubicación: `dataset_preparer/contamination/reports/filter_report_{timestamp}.json`

---

## Pipeline Modificado

```
[1/7]  Load sources (sin cambio)
[2/7]  Combine + standardize (sin cambio)
[3/7]  NEW: Pre-filter (antes de validación)
         - Limpiar basura, URLs, código, boilerplate
         - Corregir encoding issues
         - Normalizar unicode
[4/7]  Source validation (mejorado con más checks)
[5/7]  Cross-source deduplication
[6/7]  Balance control
[7/7]  Thinking generation + quality validation
[8/7]  Language filtering (sin paso de 'unknown')
[9/7]  Leakage detection
[10/7] Generate audit report
```

---

## Nuevos Flags CLI

```bash
# Filtros de calidad
--filter-noise              # Activar filtro de basura/ruido
--noise-categories          # Categorías a filtrar: urls,emails,phones,code,boilerplate

# Control de balance
--filter-balance            # Activar control de balance entre fuentes
--max-source-ratio          # Ratio máximo por source (default: 0.3)

# Deduplicación mejorada
--dedup-mode                # Modo: exact|near|cross|all (default: all)

# Detección de leakage
--leakage-detect            # Activar detección de data leakage
--leakage-threshold         # Umbral de overlap (default: 0.5)

# Reporte de auditoría
--audit-report              # Generar reporte de auditoría
--audit-dir                 # Directorio para reportes (default: dataset_preparer/contamination/reports/)
```

---

## Orden de Implementación

| Fase | Archivo | Descripción | Estado |
|------|---------|-------------|--------|
| 1 | `contamination/__init__.py` | Crear módulo vacío | COMPLETADA |
| 2 | `contamination/filters.py` | Filtros de calidad mejorados | COMPLETADA |
| 3 | `contamination/dedup.py` | Deduplicación cross-source | COMPLETADA |
| 4 | `contamination/balance.py` | Control de balance | COMPLETADA |
| 5 | `contamination/leakage.py` | Detección de data leakage | COMPLETADA |
| 6 | `contamination/audit.py` | Sistema de reportes | COMPLETADA |
| 7 | `data_preparer.py` | Integrar pipeline completo | COMPLETADA |
| 8 | `source_validators.py` | Mejorar validadores existentes | COMPLETADA |
| 9 | `main.py` | Agregar nuevos flags CLI | COMPLETADA |

**Estado: 9/9 fases completadas**

---

## Ejemplo de Uso Completo

```bash
# Pipeline completo de filtrado (recomendado)
python main.py --prepare-data --aiml --hf --csv --pdf --epub --web \
  --validate-sources \
  --filter-noise \
  --filter-balance --max-source-ratio 0.3 \
  --enable-dedup --dedup-mode all \
  --leakage-detect --leakage-threshold 0.5 \
  --enable-lang-filter --allowed-languages es en \
  --generate-thinking --thinking-depth adaptive \
  --audit-report
```

---

## Ejemplo de Reporte Generado

```json
{
  "timestamp": "2026-07-30T10:00:00",
  "pipeline_version": "1.0",
  "filters_applied": ["quality", "dedup", "balance", "leakage", "language", "thinking"],
  "per_source": {
    "aiml": {
      "original_count": 5000,
      "after_noise_filter": 4920,
      "after_quality": 4800,
      "after_dedup": 4500,
      "after_balance": 4200,
      "after_language": 4100,
      "discarded": {
        "empty": 50,
        "too_short": 80,
        "noise": 40,
        "boilerplate": 30,
        "urls": 15,
        "emails": 5,
        "code": 0,
        "encoding": 0,
        "language": 100,
        "duplicates_exact": 200,
        "duplicates_near": 100
      }
    },
    "pdf": {
      "original_count": 3000,
      "after_noise_filter": 2900,
      "after_quality": 2800,
      "after_dedup": 2600,
      "after_balance": 2600,
      "after_language": 2500,
      "discarded": {
        "empty": 20,
        "too_short": 50,
        "noise": 30,
        "boilerplate": 20,
        "page_numbers": 40,
        "headers_repeated": 30,
        "encoding": 10,
        "language": 100,
        "duplicates_exact": 100,
        "duplicates_near": 100
      }
    },
    "csv": {
      "original_count": 400,
      "after_noise_filter": 400,
      "after_quality": 395,
      "after_dedup": 390,
      "after_balance": 390,
      "after_language": 380,
      "discarded": {
        "empty": 0,
        "too_short": 5,
        "noise": 0,
        "boilerplate": 0,
        "encoding": 0,
        "language": 10,
        "duplicates_exact": 5,
        "duplicates_near": 0
      }
    }
  },
  "cross_source_duplicates": 120,
  "balance_report": {
    "aiml": 0.38,
    "pdf": 0.22,
    "csv": 0.15,
    "hf": 0.15,
    "web": 0.10
  },
  "thinking_stats": {
    "total_samples": 10000,
    "with_thinking": 7500,
    "thinking_quality_pass": 7200,
    "thinking_quality_fail": 300
  },
  "final_count": 10000,
  "total_discarded": 2500,
  "retention_rate": 0.80,
  "processing_time_seconds": 145.3
}
```

---

## Dependencias Nuevas Requeridas

```txt
# Ya existentes (no nuevas)
datasketch        # MinHash LSH
langdetect        # Language detection
spacy             # NLP processing

# No se requieren dependencias adicionales
# Todo se implementa con stdlib + dependencias existentes
```

---

## Métricas de Éxito

| Métrica | Objetivo |
|---------|----------|
| Retention rate post-filtrado | >= 75% |
| Duplicados cross-source eliminados | >= 90% |
| Basura/noise eliminado | >= 95% |
| Balance entre sources (max ratio) | <= 35% |
| Tiempo de procesamiento adicional | < 30% del tiempo actual |
| Reportes de auditoría generados | 100% de ejecuciones con --audit-report |

---

*Última actualización: 2026-07-30 - 9/9 fases completadas*
