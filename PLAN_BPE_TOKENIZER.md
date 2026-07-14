# Plan: Unificar tokenizadores — Eliminar word-level, BPE multilingüe

## Objetivo

1. Eliminar completamente los tokenizadores word-level (`BilingualTokenizer`, `WordTokenizer`, `Trie`, `TrieNode`)
2. Dejar únicamente `SentencePieceTokenizerWrapper` (BPE) como el único tokenizador
3. Hacer el modelo **multilingüe** (no solo EN/ES) — soportar cualquier idioma que el dataset incluya

## Archivos afectados

| Archivo | Acción |
|---------|--------|
| `word_tokenizer.py` | **ELIMINAR** — contiene las 4 clases word-level |
| `tests/test_word_tokenizer.py` | **ELIMINAR** — tests del tokenizador word-level |
| `main_chat.py` | **EDITAR** — eliminar imports, reemplazar fallbacks, limpiar safe_globals |
| `main_train.py` | **EDITAR** — eliminar import, reemplazar fallbacks |
| `dialogmanager.py` | **EDITAR** — limpiar código muerto (word2idx, trie) |
| `chatmodel.py` | **EDITAR** — actualizar docstrings |
| `bpe_tokenizer.py` | **EDITAR** — actualizar docstrings, eliminar referencia a bilingual |
| `data_preparer.py` | **EDITAR** — configurar BPE multilingüe, habilitar datasets multilingües |
| `tests/test_bpe_tokenizer.py` | **EDITAR** — tests genéricos (no EN/ES específicos) |
| `README.md` | **EDITAR** — actualizar árbol de archivos |

---

## Paso 1: Eliminar `word_tokenizer.py`

**Acción:** Borrar el archivo completo.

**Clases eliminadas:**
- `TrieNode` (línea 15)
- `Trie` (línea 21)
- `BilingualTokenizer` (línea 44) — word-level con detección de idioma
- `WordTokenizer` (línea 526) — subclase sin tokens de idioma

**Razón:** Estas clases usan tokenización por palabras completas (word-level), que es obsoleta para modelos de lenguaje modernos. BPE es el estándar de la industria.

---

## Paso 2: Eliminar `tests/test_word_tokenizer.py`

**Acción:** Borrar el archivo completo.

**Tests eliminados:**
- `test_load_vocabulary_ignores_sentencepiece_metadata` — testeaba `WordTokenizer.load_vocabulary()`
- `test_sentencepiece_wrapper_decodes_ids_to_text` — testeaba el wrapper BPE (mover a `test_bpe_tokenizer.py` si es necesario)

---

## Paso 3: Editar `main_chat.py`

### 3a. Eliminar import (línea 18)

```python
# ANTES:
from word_tokenizer import WordTokenizer

# DESPUÉS: (eliminar la línea)
```

### 3b. Eliminar fallbacks a WordTokenizer

Hay 4 instancias donde se crea `WordTokenizer()` como fallback:

**Línea 122** — fallback cuando vocab_file no tiene sentencepiece_model:
```python
# ANTES:
self.tokenizer = WordTokenizer()

# DESPUÉS:
raise RuntimeError(
    f"No SentencePiece model found in {vocab_file}. "
    "Run: python main.py --prepare-data --aiml --hf --use-bpe"
)
```

**Línea 127** — fallback cuando error al cargar vocab file:
```python
# ANTES:
self.tokenizer = WordTokenizer()

# DESPUÉS:
raise RuntimeError(
    f"Error loading tokenizer from {vocab_file}. "
    "Run: python main.py --prepare-data --aiml --hf --use-bpe"
)
```

**Línea 144** — fallback cuando no hay tokenizer en checkpoint:
```python
# ANTES:
self.tokenizer = WordTokenizer()

# DESPUÉS:
raise RuntimeError(
    "No tokenizer found in checkpoint or vocab file. "
    "Run: python main.py --prepare-data --aiml --hf --use-bpe"
)
```

**Línea 166** — fallback cuando checkpoint loading falla:
```python
# ANTES:
self.tokenizer = WordTokenizer()

# DESPUÉS:
raise RuntimeError(
    f"Error loading model checkpoint: {e}. "
    "Ensure the model was trained with BPE tokenizer."
)
```

### 3c. Limpiar safe_globals (líneas 257-266)

```python
# ANTES:
try:
    import word_tokenizer
    allowed = [
        word_tokenizer.Trie,
        word_tokenizer.BilingualTokenizer,
        word_tokenizer.WordTokenizer,
    ]
    if SentencePieceTokenizerWrapper is not None:
        allowed.append(SentencePieceTokenizerWrapper)
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=False)
except Exception:
    pass

# DESPUÉS:
try:
    allowed = []
    if SentencePieceTokenizerWrapper is not None:
        allowed.append(SentencePieceTokenizerWrapper)
    if allowed:
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location=device, weights_only=False)
except Exception:
    pass
```

---

## Paso 4: Editar `main_train.py`

### 4a. Eliminar import (línea 13)

```python
# ANTES:
from word_tokenizer import WordTokenizer

# DESPUÉS: (eliminar la línea)
```

### 4b. Eliminar fallbacks a WordTokenizer

**Línea 135** — fallback cuando SentencePiece init falla:
```python
# ANTES:
self.tokenizer = WordTokenizer()

# DESPUÉS:
raise RuntimeError(
    f"Could not initialize SentencePiece tokenizer: {e}. "
    "Install sentencepiece: pip install sentencepiece"
)
```

**Línea 139** — fallback cuando sentencepiece no está instalado:
```python
# ANTES:
self.tokenizer = WordTokenizer()

# DESPUÉS:
raise RuntimeError(
    "cache_metadata indicates a BPE model but 'sentencepiece' is not installed. "
    "Install it: pip install sentencepiece"
)
```

---

## Paso 5: Editar `dialogmanager.py`

### 5a. Eliminar código muerto (líneas 96-103)

Estas líneas verifican `word2idx` y `trie` que ya no existirán:

```python
# LÍNEAS A ELIMINAR (dead code después de remover word_tokenizer):
elif hasattr(self.tokenizer, "word2idx"):
    vocab_dict = self.tokenizer.word2idx
elif hasattr(self.tokenizer, "trie"):
    vocab_dict = {}
```

**Nota:** El código ya funciona sin estas líneas porque primero verifica `hasattr(self.tokenizer, "vocab")` (línea 94), que `SentencePieceTokenizerWrapper` sí tiene como property.

---

## Paso 6: Editar `chatmodel.py`

### 6a. Actualizar docstrings

**Línea 8** — cambiar `word_tokenizer` por `bpe_tokenizer`:
```python
# ANTES:
- `word_tokenizer`: A custom tokenizer module (not shown in the provided code).

# DESPUÉS:
- `bpe_tokenizer`: SentencePiece BPE tokenizer wrapper.
```

**Línea 37** — cambiar `word_tokenizer` por `bpe_tokenizer`:
```python
# ANTES:
- `word_tokenizer`: Un módulo de tokenizador personalizado (no se muestra en el código proporcionado).

# DESPUÉS:
- `bpe_tokenizer`: Wrapper de tokenizador BPE con SentencePiece.
```

---

## Paso 7: Actualizar `bpe_tokenizer.py` — Hacer multilingüe

### 7a. Actualizar docstrings

**Línea 1** — cambiar docstring del módulo:
```python
# ANTES:
"""SentencePiece BPE tokenizer wrapper with API compatible with WordTokenizer."""

# DESPUÉS:
"""SentencePiece BPE tokenizer wrapper — multilingual, language-agnostic."""
```

**Línea 12** — cambiar docstring de la clase:
```python
# ANTES:
"""Wrapper around SentencePiece BPE model exposing the same interface as WordTokenizer.

# DESPUÉS:
"""Wrapper around SentencePiece BPE model — multilingual, language-agnostic.

Supports: encode, decode, batch_encode, get_pad_index, get_unk_index,
get_eos_index, convert_ids_to_tokens, fit (no-op), save_vocabulary.
Also exposes vocab and idx2word dicts for DialogueManager compatibility.
"""
```

### 7b. Verificar compatibilidad multilingüe

El `SentencePieceTokenizerWrapper` ya es **agnóstico al idioma** — no tiene lógica específica de EN/ES. SentencePiece maneja cualquier idioma nativamente.

**No se necesitan cambios funcionales** en la interfaz del wrapper.

Métodos verificados:

| Método | ¿Existe? | Multilingüe? | Usado por |
|--------|----------|--------------|-----------|
| `encode(text)` | ✅ | ✅ Cualquier idioma | main_train, dialogmanager |
| `decode(indices)` | ✅ | ✅ Cualquier idioma | dialogmanager |
| `batch_encode(texts)` | ✅ | ✅ Cualquier idioma | main_train |
| `get_pad_index()` | ✅ | ✅ | main_train, dialogmanager |
| `get_unk_index()` | ✅ | ✅ | dialogmanager |
| `get_eos_index()` | ✅ | ✅ | dialogmanager |
| `convert_ids_to_tokens(ids)` | ✅ | ✅ | dialogmanager |
| `vocab_size` (property) | ✅ | ✅ | main_chat, main_train, chatmodel |
| `vocab` (property) | ✅ | ✅ | dialogmanager |
| `idx2word` (property) | ✅ | ✅ | dialogmanager |
| `fit(texts)` | ✅ (no-op) | ✅ | main_train |
| `save_vocabulary(path)` | ✅ | ✅ | main_train |

---

## Paso 8: Actualizar `data_preparer.py` — BPE multilingüe

### 8a. Eliminar tokens `<EN>,<ES>` del entrenamiento SP (línea 743)

SentencePiece es **agnóstico al idioma** — no necesita tokens de idioma. El modelo aprende subwords de cualquier idioma presente en el corpus.

```python
# ANTES (línea 743):
spm_cmd = f"--input={tmp_corpus} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 --user_defined_symbols=<EN>,<ES>"

# DESPUÉS:
spm_cmd = f"--input={tmp_corpus} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=0.9995"
```

**Cambios:**
- Eliminar `--user_defined_symbols=<EN>,<ES>` — SentencePiece no necesita tokens de idioma
- Cambiar `--character_coverage=1.0` a `0.9995` — estándar para multilingüe (1.0 puede fallar con caracteres raros)

### 8b. Habilitar datasets multilingües (líneas 427-431)

```python
# ANTES:
hf_dataset_names = [
    'wikitext',  # Wikipedia text
    # 'common_voice',  # Speech data (uncomment for more data)
    # 'opus_100',  # Multi-language data (uncomment for multilingual)
]

# DESPUÉS:
hf_dataset_names = [
    'wikitext',  # Wikipedia text (multilingual via different configs)
    # 'opus_100',  # Multi-language translation data (100+ languages)
    # 'common_voice',  # Speech data (multiple languages)
]
```

**Nota:** El dataset `wikitext` ya soporta multilingüe via configuraciones como `wikitext-2` (EN), `wikitext-103-uncased` (multi). Para más idiomas, descomentar `opus_100` que tiene 100+ idiomas.

### 8c. Actualizar `common_voice` para multilingüe (línea 442)

```python
# ANTES:
ds = load_dataset('common_voice', '2024-08', split='train[:1000]', languages=["en"])

# DESPUÉS:
ds = load_dataset('common_voice', '2024-08', split='train[:1000]', languages=["en", "es", "fr", "de", "pt", "it", "nl", "ru", "zh", "ja"])
```

**Nota:** Solo se ejecuta si el usuario descomenta esta línea. Los idiomas se pueden configurar según la necesidad.

---

## Paso 9: Actualizar `tests/test_bpe_tokenizer.py` — Tests genéricos

### 9a. Cambiar test de "spanish" a genérico

```python
# ANTES:
def test_encode_spanish(self, sp_tokenizer):
    result = sp_tokenizer.encode("Hola, como estas?")

# DESPUÉS:
def test_encode_non_english(self, sp_tokenizer):
    result = sp_tokenizer.encode("Hola, como estas?")  # Spanish — any language works
```

```python
# ANTES:
def test_encode_decode_roundtrip_spanish(self, sp_tokenizer):
    original = "hola mundo"

# DESPUÉS:
def test_encode_decode_roundtrip_non_english(self, sp_tokenizer):
    original = "hola mundo"  # Spanish — any language works
```

**Nota:** Los tests ya funcionan con cualquier idioma. Solo cambiar nombres para reflejar que no son específicos de EN/ES.

---

## Paso 10: Actualizar `README.md`

**Línea 83** — cambiar en el árbol de archivos:

```markdown
# ANTES:
├── word_tokenizer.py      # Word-level tokenizer (BilingualTokenizer, WordTokenizer)
├── bpe_tokenizer.py       # SentencePiece BPE tokenizer wrapper

# DESPUÉS:
├── bpe_tokenizer.py       # SentencePiece BPE tokenizer (multilingual)
```

---

## Resumen de cambios

### Archivos eliminados (2):
1. `word_tokenizer.py` — 530 líneas (Trie, BilingualTokenizer, WordTokenizer)
2. `tests/test_word_tokenizer.py` — 58 líneas

### Archivos editados (8):
1. `main_chat.py` — eliminar import, 4 fallbacks → raise, limpiar safe_globals
2. `main_train.py` — eliminar import, 2 fallbacks → raise
3. `dialogmanager.py` — eliminar 4 líneas de dead code
4. `chatmodel.py` — actualizar 2 docstrings
5. `bpe_tokenizer.py` — actualizar docstrings (multilingüe)
6. `data_preparer.py` — configurar BPE multilingüe, habilitar datasets multilingües
7. `tests/test_bpe_tokenizer.py` — renombrar tests genéricos
8. `README.md` — actualizar árbol de archivos

### Líneas de código eliminadas: ~534
### Líneas de código modificadas: ~50

### Cambios clave para multilingüe:
- **SentencePiece** es nativamente multilingüe — no necesita tokens de idioma
- `character_coverage=0.9995` — estándar para corpus multilingües
- Eliminar `<EN>,<ES>` — el modelo aprende subwords de cualquier idioma
- Habilitar `opus_100` — dataset con 100+ idiomas (opcional, por defecto comentado)

---

## Verificación

1. **Tests:**
   ```bash
   python -m pytest tests/ -v
   ```
   Todos los tests deben pasar (se eliminan los tests word-level, quedan los BPE y los smoke tests).

2. **Import check:**
   ```bash
   python -c "from bpe_tokenizer import SentencePieceTokenizerWrapper; print('OK')"
   ```

3. **Verificar que no hay imports rotos:**
   ```bash
   grep -r "word_tokenizer\|BilingualTokenizer\|WordTokenizer\|SimpleTokenizer" --include="*.py" .
   ```
   No debe retornar resultados.

4. **Verificar tokenizer multilingüe:**
   ```bash
   python -c "
   from bpe_tokenizer import SentencePieceTokenizerWrapper
   t = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
   # Test multiple languages
   for text in ['Hello world', 'Hola mundo', 'Bonjour le monde', 'Hallo Welt', 'Ciao mondo']:
       ids = t.encode(text)
       decoded = t.decode(ids)
       print(f'{text} -> {decoded}')
   "
   ```

5. **Entrenamiento:**
   ```bash
   python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 2000 --refresh-cache
   python main.py --train --use-cache --epochs 1
   ```

6. **Inferencia:**
   ```bash
   python main.py --chat
   ```
   Verificar que carga el tokenizador BPE (log: "Loaded SentencePiece BPE tokenizer").

---

## Orden de ejecución recomendado

1. Eliminar `word_tokenizer.py`
2. Eliminar `tests/test_word_tokenizer.py`
3. Editar `main_train.py` (import + fallbacks)
4. Editar `main_chat.py` (import + fallbacks + safe_globals)
5. Editar `dialogmanager.py` (dead code)
6. Editar `chatmodel.py` (docstrings)
7. Editar `bpe_tokenizer.py` (docstrings multilingüe)
8. Editar `data_preparer.py` (BPE multilingüe + datasets)
9. Editar `tests/test_bpe_tokenizer.py` (tests genéricos)
10. Editar `README.md` (árbol de archivos)
11. Ejecutar verificación completa
