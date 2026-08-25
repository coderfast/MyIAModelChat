# FICHA_HUGGINGFACE

## Nombre del modelo
- `myiamodelchat-local`

## Descripción
- Modelo local basado en GPT-2 adaptado para chat conversacional.
- Entrenado/fine-tuned para respuestas en español e inglés.
- Arquitectura Transformer decoder-only (GPT2LMHeadModel).

## Tipo de modelo
- `GPT2LMHeadModel` (Transformer decoder-only)
- Modelo de lenguaje autoregresivo para generación de texto.

## Tamaño del modelo
- Parámetros totales: **5,338,624** (~5.34M)
- Configuración de arquitectura:
  - `vocab_size`: 8000 (default BPE, configurable)
  - `n_embd`: 256
  - `n_layer`: 4
  - `n_head`: 4
  - `n_positions`: 512
- Desglose:
  - Embedding token: 2,048,000 (vocab_size × n_embd)
  - Embedding posición: 131,072 (n_positions × n_embd)
  - 4 capas Transformer: ~3,013,376
  - Layer norm final: 512

## Tokenizador
- Tokenizador BPE multilingüe: `SentencePieceTokenizerWrapper` (módulo `commons/tokenizer/bpe_tokenizer.py`)
- Único tokenizador del proyecto — no hay fallback a word-level
- Soporta Español e Inglés de forma nativa, extensible a más idiomas
- Modelo BPE entrenado en `dataset_cache/sentencepiece.model`
- Vocabulario cargado desde `checkpoints/tokenizer_vocab.json`
- Tamaño del vocabulario configurable (default: 8000, flag `--bpe-vocab-size`)
- Si `sentencepiece` no está instalado, raise `ImportError` (sin fallback)

## Uso principal
- Chatbot conversacional local
- Servidor compatible con API Ollama y OpenAI
- Endpoint de inferencia HTTP y chat de consola

## Entrenamiento
- Arquitectura definida en `commons/model/chatmodel.py`
- Pipeline de entrenamiento en `training/trainer.py`
- Datos preparados con `python main.py --prepare-data`
- Soporta cache de datasets para entrenamiento rápido

## Inferencia
- Servidor FastAPI en `ServerFastAPI/app.py`
- Endpoints disponibles:
  - `POST /v1/chat/completions` — Chat completion (compatible OpenAI)
  - `POST /api/chat/completions` — Chat completion alternativo
  - `POST /v1/embeddings` — Embeddings de texto
  - `POST /api/generate` — Generación de texto
  - `GET /v1/models` — Lista modelos disponibles
  - `GET /v1/health` — Health check
  - `GET /v1/version` — Versión del servidor
- Soporte para **chain-of-thought reasoning** mediante sistema dual de tokens:
  - Mode tokens: `<|thinking|>`, `<|context|>`, `<|answer|>`
  - Content tags: `<thinking>`, `</thinking>`
  - Parámetro `include_thinking` en endpoints API
- Streaming de respuestas soportado

## Requisitos principales
- Python 3.12+ (probado en 3.14)
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- Uvicorn
- psutil
- sentencepiece

## Archivos clave
- `main.py` — punto de entrada principal (CLI)
- `commons/model/chatmodel.py` — definición del modelo GPT2LMHeadModel
- `commons/dialogue/dialogmanager.py` — gestión de diálogo y generación de respuestas
- `commons/tokenizer/bpe_tokenizer.py` — tokenizador BPE (SentencePiece, único tokenizador)
- `commons/dataset/chatdataset.py` — Dataset de PyTorch para training
- `commons/registry/model_registry.py` — registro y listado de modelos
- `commons/registry/model_merge.py` — fusión de modelos por promedio de pesos
- `commons/registry/model_export.py` — exportación a GGUF/ONNX
- `commons/registry/model_downloader.py` — descarga de modelos HuggingFace
- `dataset_preparer/data_preparer.py` — preparación de datos multi-fuente
- `dataset_preparer/aiml/loader.py` — carga de archivos AIML
- `dataset_preparer/web/scraper.py` — scraping web para datos
- `dataset_preparer/thinking_generators.py` — generador de datos con thinking (mode tokens)
- `training/trainer.py` — pipeline de entrenamiento
- `inference/chat_engine.py` — lógica de inferencia
- `ServerFastAPI/app.py` — servidor FastAPI
- `checkpoints/tokenizer_vocab.json` — vocabulario del tokenizador
- `dataset_cache/sentencepiece.model` — modelo BPE entrenado
- `checkpoints/` — checkpoints de modelos entrenados

## Licencia
- Indicar la licencia de uso del proyecto si aplica.

## Notas adicionales
- El modelo funciona localmente y es compatible con Docker/entornos aislados si se adapta el entorno.
- Se puede mejorar con fine-tuning adicional y mayores fuentes de datos en español.
- La tokenización BPE permite soporte multilingüe sin cambiar la arquitectura del modelo.
- El modelo soporta chain-of-thought reasoning mediante sistema dual de tokens:
  - `<|thinking|>` / `<|context|>` / `<|answer|>` (mode tokens)
  - `<thinking>` / `</thinking>` (content tags para razonamiento)
- El servidor es compatible con clientes Ollama y OpenAI API (formato de mensajes idéntico).
- Opciones de streaming y campo `reasoning` disponibles en todos los endpoints.

---

*Updated: 2026-07-28 - Reflects new modular project structure*
