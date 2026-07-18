# MODEL_TECHNICALSTACK.md — Stack Técnico del Proyecto

## Visión General

MyIAModelChat es un sistema de IA conversacional multilingüe basado en GPT-2, con entrenamiento personalizado, tokenización BPE, y un servidor API compatible con Ollama/OpenAI.

---

## Lenguaje y Runtime

| Componente | Detalle |
|---|---|
| Lenguaje | Python 3.8+ |
| Entorno virtual | `envMyIAModelChat/` |

---

## Framework de Deep Learning

| Componente | Versión/Detalle | Uso |
|---|---|---|
| **PyTorch** | 2.0+ | Framework principal de entrenamiento e inferencia |
| **torchvision** | — | Dependencia de PyTorch |
| **torchaudio** | — | Dependencia de PyTorch |
| **torchtext** | — | Dependencia de PyTorch |
| **ONNX** | — | Exportación de modelos (disponible en requirements) |

### Arquitectura del Modelo

- **Arquitectura**: `GPT2LMHeadModel` (HuggingFace Transformers) — se usa la estructura de GPT-2 como backbone
- **Entrenamiento**: Desde cero (random weights) con los datos del proyecto — **no usa pesos pre-entrenados de GPT-2**
- **Configuración custom**: `GPT2Config` con `vocab_size` del tokenizador SentencePiece, `n_embd=256`, `n_head=4`, `n_layer=4`, `n_positions=512`
- **Vocabulario propio**: SentencePiece BPE (no el vocabulario original de GPT-2)
- **Archivo**: `chatmodel.py`

### Ventajas del modelo custom

- **Ideal para proyecto local**: ~20M de parámetros, ligero y ejecutable en CPU
- **Rápido de entrenamiento**: entrena en minutos/horas (no días como modelos grandes)
- **Multilingüe**: SentencePiece BPE soporta cualquier idioma sin configuración adicional
- **Sin dependencia de GPUs potentes**: funciona bien en CPU, acelerable con CUDA si disponible
- **Control total**: configuración ajustable según necesidades del proyecto

### Soporte Multilingue

El tokenizador SentencePiece BPE es **agnóstico a idiomas** — soporta cualquier escritura (Latín, Cyrillico, CJK, Árabe, etc.) gracias a `character_coverage=0.9995`.

**Detección de idioma** (`data_preparer.py`):
- Usa `langdetect` si está instalado
- Fallback heurístico: cuenta palabras indicadoras (ES vs EN)

**Filtrado por idioma** (desactivado por defecto):

| Flag | Descripción | Valor por defecto |
|------|-------------|-------------------|
| `--enable-lang-filter` | Activa el filtrado por idioma | `false` |
| `--allowed-languages` | Códigos de idioma permitidos | `es en` |

**Ejemplos de uso:**

```bash
# Solo español
python main.py --prepare-data --aiml --hf --enable-lang-filter --allowed-languages es

# Español, inglés y francés
python main.py --prepare-data --aiml --hf --enable-lang-filter --allowed-languages es en fr

# Sin filtro (todos los idiomas)
python main.py --prepare-data --aiml --hf
```

---

## Tokenización

| Componente | Detalle |
|---|---|
| **SentencePiece BPE** | Tokenizador único del proyecto (multilingüe, language-agnostic) |
| **Wrapper** | `SentencePieceTokenizerWrapper` en `bpe_tokenizer.py` |
| **Vocabulario por defecto** | 8,000 tokens (`--bpe-vocab-size`) |
| **Cobertura de caracteres** | 0.9995 (estándar multilingüe) |
| **Modelo entrenado** | `dataset_cache/sentencepiece.model` |
| **Checkpoints** | `checkpoints/tokenizer_vocab.json` |

### Tokens especiales

- `<pad>` (ID 0), `<unk>` (ID 1), `<s>` / BOS, `</s>` / EOS
- `<think>` / `</think>` (chain-of-thought reasoning)

---

## Modelos de NLP (HuggingFace Transformers)

| Modelo | Tarea | Localización |
|---|---|---|
| `nlptown/bert-base-multilingual-uncased-sentiment` | Análisis de sentimiento (1-5 estrellas) | `models/sentiment/` |
| `nlptown/bert-base-multilingual-uncased-sentiment` | Clasificación de intención (reutilizado) | `models/intent/` |
| **Pipelines HF** | `pipeline('text-classification')`, `pipeline('sentiment-analysis')` | `main_chat.py` |

### Descarga de modelos

- `model_downloader.py` → `ensure_model_local()` descarga vía `huggingface_hub.snapshot_download()`
- Primera ejecución: descarga. Ejecuciones siguientes: carga local.

---

## API Server

| Componente | Detalle |
|---|---|
| **Framework** | FastAPI |
| **Servidor** | Uvicorn |
| **Validación** | Pydantic (modelos `ChatRequest`, `GenerateRequest`, `EmbeddingRequest`) |
| **Compatibilidad** | API compatible con Ollama y OpenAI |
| **Puerto por defecto** | 11434 |

### Endpoints

| Endpoint | Método | Descripción |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/api/chat/completions` | POST | Chat completions (alias) |
| `/api/chat` | POST | Chat endpoint |
| `/api/generate` | POST | Text generation |
| `/v1/embeddings` | POST | Embeddings |
| `/v1/models` | GET | Lista de modelos |
| `/v1/version` | GET | Versión del modelo |
| `/v1/health` | GET | Health check |
| `/health` | GET | Health check (root) |
| `/` | GET | Root info |

### Streaming

- Soporta `stream: true` en chat completions
- Chunk size: 64 caracteres

---

## Pipeline de Datos

| Fuente | Formato | Herramienta |
|---|---|---|
| **AIML** | Archivos `.aiml` → pickled HF Datasets | `aimlloder.py` |
| **PDF** | Extracción de texto | PyPDF2 |
| **EPUB** | Extracción de texto | ebooklib |
| **HuggingFace Datasets** | Datasets pre-entrenados | `datasets` library |
| **CSV** | Fine-tuning post-entrenamiento | Carga directa |

### Procesamiento de texto avanzado

| Feature | Herramienta | Flag CLI |
|---|---|---|
| Tokenización de oraciones | spaCy (`es_core_news_sm`, `en_core_web_sm`) | Automático |
| Normalización Unicode | NFKC | Automático |
| Chunking de textos largos | Custom (`chunk_text_by_tokens()`) | `--enable-chunking` |
| Deduplicación | MinHash LSH (datasketch) | `--enable-dedup` |
| Filtro de calidad | Custom (longitud, alpha ratio, spam) | `--enable-quality-filter` |
| Preservación de metadata | Custom | `--preserve-metadata` |
| Detección de idioma | langdetect (50+ idiomas) | `--enable-lang-filter` |

---

## Entrenamiento

| Parámetro | Valor por defecto |
|---|---|
| batch_size | 4 |
| accumulation_steps | 8 |
| learning_rate | 1e-3 |
| embed_size | 256 |
| hidden_size | 512 |
| grad_clip_norm | 1.0 |
| epochs | 30 |
| Optimizador | Adam |
| Scheduler | StepLR (step_size=epochs/3, gamma=0.1) |
| Loss function | CrossEntropyLoss (ignora `<pad>`) |

### Optimizaciones de entrenamiento

- **Mixed precision**: `torch.amp.GradScaler` + `torch.autocast` (GPU)
- **Gradient accumulation**: 8 pasos por actualización
- **Gradient checkpointing**: Para GPUs con poca memoria
- **Warm-up**: 10% del dataset, máx 100 batches
- **Dataset caching**: 12x más rápido con caché pre-tokenizado
- **CPU threading**: Configurable vía `OMP_NUM_THREADS`, `MKL_NUM_THREADS`

### Hardware soportado

- **CPU**: Entrenamiento completo en CPU
- **GPU CUDA**: Mixed precision + gradient checkpointing
- **Tesla K80**: Configuración optimizada específica
- **MPS (Apple Silicon)**: Soporte en inferencia

---

## Chain-of-Thought (<think>)

- Formato de entrenamiento: `<think>razonamiento</think>respuesta`
- Tokens especiales en SentencePiece: `<think>`, `</think>`
- Detección en entrenamiento: `_detect_thinking_data()` (metadata + heurística)
- Métricas: accuracy de tokens thinking cada 50 batches
- Inferencia: parsing en `main_chat.py`, no en `dialogmanager.py`

---

## Gestión del Diálogo

| Componente | Archivo | Función |
|---|---|---|
| `DialogueManager` | `dialogmanager.py` | Loop de generación autoregresiva |
| Filtrado | Top-k (50), Top-p (0.9), Temperature (0.7) | Control de generación |
| Penalización | N-gram dinámica (size=3, penalty=5.0) | Evita repeticiones |
| Min length | 3 tokens | Respuestas no vacías |
| Persona | Configurable (nombre, edad, ocupación) | Personalidad del bot |
| Sentiment → Temperature | ≤2 estrellas: -0.2, ≥4 estrellas: +0.1 | Ajuste adaptativo |

---

## Dependencias Principales

```
torch, torchvision, torchaudio, torchtext
transformers (HuggingFace)
sentencepiece
fastapi, uvicorn, pydantic
datasets (HuggingFace)
PyPDF2
ebooklib
numpy
python-aiml
keyboard
huggingface_hub
onnx
psutil (runtime, no en requirements.txt)
```

### Dependencias opcionales (procesamiento avanzado)

```
spacy (+ es_core_news_sm, en_core_web_sm)
langdetect
datasketch
```

---

## Estructura de Archivos Clave

```
MyIAModelChat/
├── main.py                    # CLI entry point (--prepare-data, --train, --chat)
├── main_train.py              # Pipeline de entrenamiento completo
├── main_chat.py               # Servidor API + chat por consola
├── chatmodel.py               # Arquitectura GPT-2 (ChatModel)
├── bpe_tokenizer.py           # SentencePieceTokenizerWrapper
├── dialogmanager.py           # Loop de generación + intent/sentiment
├── data_preparer.py           # Carga multi-fuente + BPE training
├── chatdataset.py             # Dataset PyTorch legacy
├── aimlloder.py               # Procesador de archivos AIML
├── model_downloader.py        # Descarga de modelos HF
├── generate_thinking_data.py  # Generación de datos <think>
├── manual_test.py             # Tests manuales
├── envAIModels/
│   ├── app.py                 # FastAPI app (Ollama-compatible)
│   ├── model.py               # Configuración del modelo
│   ├── routers_api.py         # Routers API
│   ├── routers_v1.py          # Routers v1
│   ├── schemas.py             # Modelos Pydantic
│   └── utils.py               # Utilidades
├── checkpoints/               # Modelos checkpoint
├── dataset_cache/             # Caché de datasets tokenizados
├── models/                    # Modelos HF descargados
└── tests/                     # Suite de tests
```
