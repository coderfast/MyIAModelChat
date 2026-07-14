# PLAN_TEST_DATASET.md - Análisis de Preparación de Datasets

## Resumen Ejecutivo

Este documento analiza cómo se preparan los datasets en los tres formatos soportados (`--aiml`, `--pdf`, `--epub`) y evalúa su coherencia con los estándares de la industria de IA.

**Veredicto**: La implementación es funcional pero tiene áreas de mejora significativas en comparación con los estándares modernos de la industria.

---

## 1. FORMATO AIML (`--aiml`)

### Flujo de Procesamiento

```
Archivos .aiml (XML) → aimlloder.py → Dataset HuggingFace → data_preparer.py → input_ids
```

### Muestra de Datos AIML Original (`aiml/ai.aiml`)

```xml
<category>
  <pattern>WHAT IS AI</pattern>
  <template>Artificial intelligence is the branch of engineering and 
  science devoted to constructing machines that think.</template>
</category>

<category>
  <pattern>WHO CREATED AIML</pattern>
  <template>Dr. Richard S. Wallace created AIML.</template>
</category>
```

### Transformación Paso a Paso

| Paso | Entrada | Salida |
|------|---------|--------|
| 1. Extracción XML | `<pattern>WHAT IS AI</pattern>` | `'input': 'WHAT IS AI'` |
| 2. Extracción template | `<template>Artificial intelligence...</template>` | `'output': 'Artificial intelligence...'` |
| 3. Creación Dataset | `{'input': ..., 'output': ...}` | `Dataset(features: ['input', 'output'])` |
| 4. Serialización | Dataset | `pickle.dump()` → archivo `.datasets` |
| 5. Conversión input_ids | `'input' + ' ' + 'output'` | `'input_ids': 'WHAT IS AI Artificial intelligence...'` |

### Muestra Final Procesada

```python
# Dataset resultante:
Dataset({
    features: ['input_ids'],
    num_rows: 673  # Ejemplo estimado para ai.aiml
})

# Contenido de una muestra:
'input_ids': 'WHAT IS AI Artificial intelligence is the branch of engineering and science devoted to constructing machines that think.'
```

### Código de Transformación (`data_preparer.py:370-393`)

```python
# Paso 5: Conversión a input_ids
if 'input' in ds.column_names and 'output' in ds.column_names:
    ds = ds.map(
        lambda x: {
            'input_ids': (x['input'] + ' ' + x['output']).strip()
            if x.get('output') else x['input']
        },
        batched=False
    )
```

### Análisis de Coherencia con Estándares de la Industria

| Aspecto | Implementación Actual | Estándar de la Industria | ¿Coherente? |
|---------|----------------------|-------------------------|--------------|
| **Formato entrada** | XML AIML 1.0 | Estándar ALICE A.I. Foundation | ✅ Sí |
| **Serialización** | Python pickle | HuggingFace Dataset API | ✅ Sí |
| **Estructura datos** | Pares input/output | Patrón estándar NLP supervisado | ✅ Sí |
| **Concatenación** | input + ' ' + output | Técnica común en Language Modeling | ⚠️ Parcial |
| **Manejo de wildcards** | Sin procesamiento especial | Requiere normalización | ❌ No |
| **Limpieza de XML** | Sin limpieza de templates AIML | Eliminar tags `<set>`, `<br/>`, etc. | ❌ No |

### Problemas Detectados

1. **Tags AIML en output**: El template puede contener tags como `<set name="topic">`, `<br/>`, `<thinking>` que pasan al dataset sin limpieza
2. **Concatenación simple**: No distingue entre pregunta y respuesta en el entrenamiento
3. **Sin normalización de mayúsculas**: "WHAT IS AI" vs "What is ai" se tratan como distintos

---

## 2. FORMATO PDF (`--pdf`)

### Flujo de Procesamiento

```
Archivos .pdf → PyPDF2 → Extracción por página → Concatenación → Split por '.' → Dataset
```

### Muestra de Transformación

```
PASO 1 - Extracción por página (PyPDF2):
text = "Machine Learning is a powerful technique... Chapter 1: Introduction The field of AI..."

PASO 2 - Split por '.':
sentences = [
    "Machine Learning is a powerful technique...",
    " Chapter 1: Introduction The field of AI...",
    ...
]

PASO 3 - Strip + filtro (>10 chars):
pdf_texts = [
    {'input_ids': 'Machine Learning is a powerful technique'},
    {'input_ids': ' Chapter 1: Introduction The field of AI'},
    ...
]

PASO 4 - Dataset resultante:
Dataset({
    features: ['input_ids'],
    num_rows: N
})
```

### Código de Extracción (`data_preparer.py:587-654`)

```python
def _load_pdf_data(self) -> Dataset:
    # Extracción de texto
    with open(file_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text += page.extract_text() + " "  # Concatena todo
            except Exception as e:
                logger.warning(f"Error extracting page {page_num}")
    
    # División por punto
    if text.strip():
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filtra fragmentos cortos
                pdf_texts.append({'input_ids': sentence})
```

### Análisis de Coherencia con Estándares de la Industria

| Aspecto | Implementación Actual | Estándar de la Industria | ¿Coherente? |
|---------|----------------------|-------------------------|--------------|
| **Librería PDF** | PyPDF2 | PyPDF2, pdfplumber, pymupdf | ✅ Sí |
| **Extracción** | Concatenación ciega de páginas | Preservar estructura de párrafos | ❌ No |
| **Sentence splitting** | `text.split('.')` | spaCy, NLTK sent_tokenize | ❌ No |
| **Limpieza** | Solo `strip()` | Regex multilinea, Unicode normalization | ❌ No |
| **Filtrado** | `len > 10` caracteres | Filtro por tokens, calidad | ⚠️ Parcial |
| **Chunking** | Oraciones individuales | Ventanas de 512-2048 tokens | ❌ No |
| **Deduplicación** | Ninguna | SimHash, MinHash | ❌ No |

### Problemas Detectados

1. **Split por punto ingenuo**: Rompe URLs, abreviaciones ("Dr."), decimales ("3.14")
2. **Sin normalización Unicode**: Caracteres especiales quedan sin procesar
3. **Sin eliminación de headers/footers**: Encabezados de página contaminan el texto
4. **Sin límite de longitud**: Oraciones muy largas pueden causar problemas
5. **Sin manejo de PDFs escaneados**: PyPDF2 no extrae texto de imágenes

### Comparación con Estándares Modernos

| Aspecto | Industria (2024-2026) | Este Proyecto |
|---------|----------------------|---------------|
| **Sentence tokenization** | `spacy.load('es_core_news_sm')` | `text.split('.')` |
| **Unicode** | `unicodedata.normalize('NFKC', text)` | Sin normalización |
| **Chunking** | Overlapping windows (512-2048 tokens) | Oraciones individuales |
| **Quality filtering** | Perplexity scoring, deduplicación | Solo filtro por longitud |
| **Metadata** | Preservar estructura de documentos | Sin metadata |

---

## 3. FORMATO EPUB (`--epub`)

### Flujo de Procesamiento

```
Archivos .epub → ebooklib → Extracción HTML → Limpieza regex → Split por '.' → Dataset
```

### Muestra de Transformación

```
ENTRADA (HTML interno del EPUB):
<html><body>
<h1>Introducción</h1>
<p>El desarrollo de la inteligencia artificial ha transformado la sociedad moderna.</p>
<p>Desde los primeros experimentos en los años 1950, hemos visto avances extraordinarios.</p>
</body></html>

DESPUÉS DE LIMPIEZA:
"El desarrollo de la inteligencia artificial ha transformado la sociedad moderna. 
 Desde los primeros experimentos en los años 1950, hemos visto avances extraordinarios."

DESPUÉS DE SPLIT:
[
    {'input_ids': 'El desarrollo de la inteligencia artificial ha transformado la sociedad moderna'},
    {'input_ids': 'Desde los primeros experimentos en los años 1950'},
    {'input_ids': 'hemos visto avances extraordinarios'}
]
```

### Código de Procesamiento (`data_preparer.py:656-733`)

```python
def _load_epub_data(self) -> Dataset:
    book = epub.read_epub(file_path)
    text = ""
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8', errors='ignore')
            
            # Limpieza de HTML
            content = re.sub(r'<[^>]+>', '', content)
            content = re.sub(r'\s+', ' ', content)
            text += content + " "
    
    # División por punto
    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:
            epub_texts.append({'input_ids': sentence})
```

### Análisis de Coherencia con Estándares de la Industria

| Aspecto | Implementación Actual | Estándar de la Industria | ¿Coherente? |
|---------|----------------------|-------------------------|--------------|
| **Librería EPUB** | ebooklib | ebooklib, epub2txt | ✅ Sí |
| **Extracción HTML** | Regex simple `<[^>]+>` | BeautifulSoup, lxml | ⚠️ Parcial |
| **Limpieza whitespace** | `re.sub(r'\s+', ' ', ...)` | ✅ Estándar | ✅ Sí |
| **Sentence splitting** | `text.split('.')` | spaCy, NLTK | ❌ No |
| **Preservación metadata** | Sin metadata | Título, autor, capítulo | ❌ No |
| **Filtrado calidad** | `len > 10` caracteres | Filtro por perplexity | ⚠️ Parcial |

### Problemas Detectados

1. **HTML stripping incompleto**: No maneja atributos, scripts inline, estilos
2. **Sin preservar estructura**: Se pierde información de capítulos/secciones
3. **Split por punto**: Mismos problemas que en PDF
4. **Sin detección de idioma**: Asume un solo idioma por archivo
5. **Espacio adicional**: `text += content + " "` agrega espacio extra al final

---

## 4. COMPARACIÓN GENERAL ENTRE FORMATOS

### Tabla Comparativa de Implementación

| Característica | AIML | PDF | EPUB |
|---------------|------|-----|------|
| **Librería principal** | xml.etree + aiml | PyPDF2 | ebooklib |
| **Limpieza de texto** | Sin limpieza | Solo strip() | Regex HTML + whitespace |
| **Sentence splitting** | No aplica (ya es por pares) | `split('.')` | `split('.')` |
| **Filtrado longitud** | No aplica | `len > 10` chars | `len > 10` chars |
| **Normalización Unicode** | ❌ No | ❌ No | ❌ No |
| **Manejo de metadata** | ❌ No | ❌ No | ❌ No |
| **Deduplicación** | ❌ No | ❌ No | ❌ No |
| **Quality filtering** | ❌ No | ❌ No | ❌ No |

### Formato de Salida Estándar

Todos los formatos convergen a:
```python
Dataset({
    features: ['input_ids'],
    num_rows: N
})
# Cada muestra: {'input_ids': 'texto plano concatenado'}
```

---

## 5. ESTÁNDARES DE LA INDUSTRIA vs IMPLEMENTACIÓN ACTUAL

### Lo que SÍ se cumple

1. ✅ **Formato HuggingFace Dataset**: Uso correcto de la API estándar
2. ✅ **Serialización pickle**: Estándar para persistencia de datasets
3. ✅ **Manejo de errores**: Try/except con logging apropiado
4. ✅ **Filtrado básico**: Eliminación de fragmentos vacíos
5. ✅ **Tokenización BPE**: SentencePiece con vocabulario configurable

### Lo que NO se cumple

1. ❌ **Sentence tokenization profesional**: No usa spaCy/NLTK para detección contextual
2. ❌ **Unicode NFKC normalization**: No normaliza caracteres especiales
3. ❌ **Chunking por tokens**: No implementa ventanas de contexto fijo
4. ❌ **Quality filtering**: No hay scoring de perplexity ni filtrado por calidad
5. ❌ **Deduplicación**: No elimina ejemplos duplicados
6. ❌ **Metadata preservation**: No preserva estructura de documentos
7. ❌ **Language detection**: No detecta ni filtra por idioma

### Brecha con la Industria (2024-2026)

| Aspecto | Industria Moderna | Este Proyecto | Brecha |
|---------|-------------------|---------------|--------|
| **Procesamiento** | Pipeline modular con validates | Script monolítico | Alta |
| **Calidad datos** | Data质量 pipelines (Great Expectations) | Validación básica | Alta |
| **Escalabilidad** | Streaming, lazy loading | Carga completa en memoria | Media |
| **Reproducibilidad** | DVC, MLflow tracking | Sin versionado de datos | Alta |
| **Testing** | Data tests (pandera, pytest) | Sin tests de datos | Alta |

---

## 6. RECOMENDACIONES DE MEJORA

### Prioridad Alta

1. **Implementar Sentence Tokenization profesional**
   ```python
   # Reemplazar: text.split('.')
   # Por: 
   import spacy
   nlp = spacy.load('es_core_news_sm')
   doc = nlp(text)
   sentences = [sent.text for sent in doc.sents]
   ```

2. **Agregar Unicode Normalization**
   ```python
   import unicodedata
   text = unicodedata.normalize('NFKC', text)
   ```

3. **Implementar Chunking por Tokens**
   ```python
   # Dividir en ventanas de 512 tokens con overlap de 50
   ```

### Prioridad Media

4. **Deduplicación con MinHash**
5. **Quality Filtering con perplexity scoring**
6. **Preservación de metadata** (título, autor, capítulo)
7. **Detección de idioma** con langdetect

### Prioridad Baja

8. **Manejo de PDFs escaneados** (OCR con Tesseract)
9. **Testing de datos** con pytest + pandera
10. **Tracking de versiones** con DVC

---

## 7. COMANDOS PARA PRUEBAS

### Preparar datos de cada formato

```bash
# AIML
python main.py --prepare-data --aiml

# PDF
python main.py --prepare-data --pdf

# EPUB
python main.py --prepare-data --epub

# Todos
python main.py --prepare-data --aiml --pdf --epub

# Con caché
python main.py --prepare-data --aiml --use-cache
```

### Verificar resultados

```bash
# Ver estadísticas en logs
# Buscar en salida: "DATASET PREPARATION SUMMARY"

# Verificar archivos de caché
ls -la dataset_cache/
```

---

## 8. CONCLUSIÓN

**Coherencia General**: ⚠️ PARCIAL

La implementación es funcional y sigue patrones básicos de la industria (HuggingFace Dataset, serialización pickle), pero tiene brechas significativas con los estándares modernos de:

1. **Procesamiento de texto**: Split por punto es demasiado ingenuo
2. **Calidad de datos**: Sin filtrado avanzado ni deduplicación
3. **Reproducibilidad**: Sin versionado ni testing de datos
4. **Escalabilidad**: Carga completa en memoria, sin streaming

**Impacto en el modelo**: Estas limitaciones pueden resultar en:
- Datos ruidosos que afectan la calidad del entrenamiento
- Oraciones truncadas que pierden contexto
- Duplicación que sesga el modelo
- Falta de diversidad en el vocabulario

**Próximos pasos sugeridos**: Implementar las mejoras de prioridad alta para acercarse a los estándares de la industria.

---

## 9. IMPLEMENTACIONES REALIZADAS ✅

### Resumen de Cambios (2026-07-14)

Se implementaron todas las mejoras prioritarias del análisis:

### 9.1 Sentence Tokenization Profesional
**Archivo**: `data_preparer.py`
- **Función**: `split_sentences(text)`
- **Implementación**: Usa spaCy (si está disponible) o regex mejorado como fallback
- **Beneficio**: Elimina problemas con abreviaciones ("Dr."), decimales ("3.14"), URLs
- **Uso**: Reemplaza `text.split('.')` en PDF y EPUB

### 9.2 Unicode NFKC Normalization
**Archivo**: `data_preparer.py`
- **Función**: `normalize_unicode(text)`
- **Implementación**: `unicodedata.normalize('NFKC', text)`
- **Beneficio**: Normaliza caracteres especiales (ligaduras, guiones largos)

### 9.3 Limpieza de Texto
**Archivo**: `data_preparer.py`
- **Función**: `clean_text(text)`
- **Implementación**: Unicode normalization + limpieza de whitespace
- **Beneficio**: Texto más consistente y limpio

### 9.4 Chunking por Tokens
**Archivo**: `data_preparer.py`
- **Función**: `chunk_text_by_tokens(text, max_tokens, overlap_tokens)`
- **Implementación**: División en ventanas con overlap configurable
- **Argumentos CLI**:
  - `--enable-chunking`: Activar chunking
  - `--chunk-max-tokens`: Máximo tokens por chunk (default: 512)
  - `--chunk-overlap`: Overlap entre chunks (default: 50)

### 9.5 Deduplicación con MinHash
**Archivo**: `data_preparer.py`
- **Función**: `deduplicate_texts(texts, threshold)`
- **Implementación**: MinHash LSH (con fallback a deduplicación exacta)
- **Argumentos CLI**:
  - `--enable-dedup`: Activar deduplicación
  - `--dedup-threshold`: Umbral de similitud (default: 0.8)

### 9.6 Quality Filtering
**Archivo**: `data_preparer.py`
- **Función**: `filter_by_quality(texts, min_words, max_words)`
- **Implementación**: Filtro por longitud, ratio alfabético, detección de spam
- **Argumentos CLI**:
  - `--enable-quality-filter`: Activar filtrado
  - `--min-words`: Mínimo de palabras (default: 5)
  - `--max-words`: Máximo de palabras (default: 1000)

### 9.7 Preservación de Metadata
**Archivo**: `data_preparer.py`
- **Funciones**: `extract_pdf_metadata()`, `extract_epub_metadata()`
- **Implementación**: Extrae título, autor, idioma, capítulos/páginas
- **Argumento CLI**:
  - `--preserve-metadata`: Activar preservación de metadata

### 9.8 Detección de Idioma
**Archivo**: `data_preparer.py`
- **Funciones**: `detect_language()`, `filter_by_language()`
- **Implementación**: langdetect (con fallback a heurísticas simples)
- **Argumentos CLI**:
  - `--enable-lang-filter`: Activar filtrado por idioma
  - `--allowed-languages`: Idiomas permitidos (default: es en)

---

## 10. NUEVOS ARGUMENTOS CLI

```bash
# Chunking
--enable-chunking              # Activar chunking por tokens
--chunk-max-tokens 512         # Máximo tokens por chunk
--chunk-overlap 50             # Overlap entre chunks

# Deduplicación
--enable-dedup                 # Activar deduplicación
--dedup-threshold 0.8          # Umbral de similitud

# Quality Filter
--enable-quality-filter        # Activar filtrado de calidad
--min-words 5                  # Mínimo de palabras
--max-words 1000               # Máximo de palabras

# Metadata
--preserve-metadata            # Preservar metadata del documento

# Language
--enable-lang-filter           # Activar filtrado por idioma
--allowed-languages es en      # Idiomas permitidos
```

---

## 11. EJEMPLO DE USO COMPLETO

```bash
# Preparar datos con todas las mejoras
python main.py --prepare-data --aiml --pdf --epub \
    --enable-chunking \
    --enable-dedup \
    --enable-quality-filter \
    --preserve-metadata \
    --enable-lang-filter \
    --allowed-languages es en

# Preparar solo PDF con chunking y deduplicación
python main.py --prepare-data --pdf \
    --enable-chunking \
    --chunk-max-tokens 256 \
    --enable-dedup \
    --dedup-threshold 0.9

# Preparar EPUB con filtrado de calidad
python main.py --prepare-data --epub \
    --enable-quality-filter \
    --min-words 10 \
    --max-words 500
```

---

## 12. VERIFICACIÓN DE COMPILACIÓN

Ambos archivos principales compilan correctamente:
- ✅ `data_preparer.py` - Sin errores de sintaxis
- ✅ `main.py` - Sin errores de sintaxis