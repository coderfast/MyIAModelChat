# FICHA_HUGGINGFACE

## Nombre del modelo
- `myiamodelchat-local`

## Descripción
- Modelo local basado en GPT-2 pequeño adaptado para chat.
- Entrenado/fine-tuned para respuestas conversacionales en español.

## Tipo de modelo
- `GPT2LMHeadModel`
- Modelo de lenguaje autoregresivo para generación de texto.

## Tamaño del modelo
- Parámetros totales: aproximadamente 4.5 millones
- Configuración de arquitectura:
  - `vocab_size`: 4792
  - `n_embd`: 256
  - `n_layer`: 4
  - `n_head`: 4

## Tokenizador
- Tokenizador personalizado: `SimpleTokenizer`
- Vocabulario cargado desde `checkpoints/tokenizer_vocab.json`
- Tamaño del vocabulario: 4792

## Uso principal
- Chatbot conversacional local
- Servicio compatible con API Ollama
- Endpoint de inferencia HTTP y chat de consola

## Entrenamiento
- No se especifica el dataset exacto en esta ficha
- Arquitectura definida en `chatmodel.py`
- El entrenamiento puede realizarse con `main_train.py`

## Inferencia
- Servicio FastAPI en `main_chat.py`
- Endpoints disponibles:
  - `/v1/chat/completions`
  - `/api/chat/completions`
  - `/v1/embeddings`
  - `/api/generate`

## Requisitos principales
- Python 3.11+ / 3.14
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- psutil

## Archivos clave
- `main_chat.py` — servidor y lógica de inferencia
- `chatmodel.py` — definición del modelo
- `dialogmanager.py` — gestión de diálogo y generación de respuestas
- `simpletokenizer.py` — tokenizador personalizado
- `checkpoints/tokenizer_vocab.json` — vocabulario del tokenizador
- `chat_model.pth` — checkpoint del modelo

## Licencia
- Indicar la licencia de uso del proyecto si aplica.

## Notas adicionales
- El modelo funciona localmente y es compatible con Docker/entornos aislados si se adapta el entorno.
- Se puede mejorar con fine-tuning adicional y mayores fuentes de datos en español.
