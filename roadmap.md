# Roadmap

## Objetivo general
Agregar soporte para rethinking en el modelo y en la inferencia, de forma que el sistema pueda:
- aprender a aplicar rethinking durante el entrenamiento
- ofrecer la opción de activar o desactivar rethinking en tiempo de inferencia

## Tareas pendientes

### 1. Preparación en caché de datasets con BPE y multilingüe ✅ (COMPLETADA)
- Crear o mejorar el pipeline de preparación de datos para que genere un cache previo de datasets.
- Incluir tokenización BPE durante la fase de preprocesamiento antes de guardar en caché.
- Permitir que la caché soporte datos en Español e Inglés desde el inicio y pueda ampliarse a más idiomas.
- Asegurar que `main_train.py` cargue los datos tokenizados y cacheados directamente para acelerar el entrenamiento.
- Documentar la ruta de `prepare_datasets_for_training` y la forma correcta de regenerar la caché cuando se añaden nuevos idiomas.

### 2. Tokenizador BPE y soporte multilingüe (Máxima prioridad) ✅ (COMPLETADO)
- Cambiar el tokenizador actual a un tokenizador estándar basado en BPE. ✅
- Soportar múltiples idiomas durante el entrenamiento e inferencia. ✅
- Comenzar con Español e Inglés, pero diseñar el pipeline para añadir más idiomas en el futuro. ✅
- Asegurar que el vocabulario y la tokenización sean compatibles con las rutas de entrenamiento y los endpoints de `main_chat.py`. ✅
- Validar que la inferencia multilingüe funcione correctamente con la opción de rethinking opcional. ✅
- Ejecutar pruebas de verificación BPE + caché: ✅
  1. `python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 2000 --refresh-cache`
  2. `python main.py --train --use-cache --epochs 1`
- Objetivo: confirmar que `sentencepiece.model` se crea/carga correctamente, `dataset_cache/prepared_dataset` contiene `token_ids`, y `main_train.py` usa los `token_ids` pre-tokenizados sin volver a tokenizar. ✅
- Estado: **COMPLETADO** — Prioridad: MÁXIMA

### 3. ~~Rethinking en entrenamiento~~ ✅ NO NECESARIO — El rethinking se implementa en la capa de orquestación (agente o endpoint), no en el modelo ni en el entrenamiento. El modelo base se mantiene sin cambios.

### 4. ~~Rethinking en inferencia~~ ✅ NO NECESARIO — El rethinking se resuelve en la capa de inferencia/orquestación. El agente o endpoint decide cuándo y cuántas veces re-pasar la respuesta al modelo para refinement. No requiere cambios en el modelo.

### 5. ~~Arquitectura y código~~ ✅ NO NECESARIO — No se toca `chatmodel.py`, `dialogmanager.py` ni `main_train.py`. El rethinking se implementa como lógica de orquestación en el endpoint o agente externo con un número máximo de iteraciones configurables.

### 6. Mejora de compatibilidad y legibilidad del servidor FastAPI ✅ (COMPLETADO)
- Refactorizar `envAIModels/app.py` para eliminar duplicación entre endpoints. ✅
- Extraer helpers comunes para:
  - llamadas a modelo y compatibilidad de firma ✅
  - streaming y respuestas por chunk ✅
  - normalización de `stop` ✅
  - construcción de respuestas OpenAI/Ollama ✅
- Usar Pydantic para los endpoints `v1` y `api` cuando sea posible en lugar de parsear JSON manualmente. ✅
- Normalizar el comportamiento de `/api/chat/completions`, `/v1/chat/completions`, `/api/generate` y `/v1/completions`. ✅
- Agrupar endpoints por tipo: metadata, API regular y compatibilidad `v1`. ✅
- Mover la inspección y diagnóstico del modelo a un módulo o función opcional para no mezclarlo con la carga básica del servicio. ✅
- Si la versión de Python lo permite, usar `asyncio.to_thread` en vez de `loop.run_in_executor` y preferir `JSONResponse` cuando sea viable. ✅

### Cambios recientes ✅ (COMPLETADO)

- `main_train.py`: ahora carga `dataset_cache/cache_metadata.pkl` y detecta si la caché incluye `token_ids`. Si existen `token_ids`, el pipeline de entrenamiento los usa directamente para evitar re-tokenización. ✅ (COMPLETADO)
- Añadido `SentencePieceTokenizerWrapper` y soporte opcional para cargar `dataset_cache/sentencepiece.model` cuando `cache_metadata.pkl` contiene `bpe_model_path`; si `sentencepiece` está instalado el wrapper se inicializa automáticamente. ✅ (COMPLETADO)
- `bpe_tokenizer.py`: módulo standalone con `SentencePieceTokenizerWrapper` para uso compartido entre entrenamiento e inferencia. ✅ (COMPLETADO)
- `main_chat.py`: detecta `sentencepiece_model` en `tokenizer_vocab.json` y carga el modelo BPE en inferencia. ✅ (COMPLETADO)
- `data_preparer.py`: añadidos tokens `<EN>`, `<ES>` al entrenamiento BPE via `--user_defined_symbols`. ✅ (COMPLETADO)
- `word_tokenizer.py`: renombrado `simpletokenizer.py` → `word_tokenizer.py`, clase `SimpleTokenizer` → `WordTokenizer`. ✅ (COMPLETADO)

### Tarea prioritaria para mañana (Máxima prioridad) ✅ (COMPLETADO)

- Ejecutar pruebas de verificación BPE + caché:
  1. `python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 2000 --refresh-cache`
  2. `python main.py --train --use-cache --epochs 1`

  Objetivo: confirmar que `sentencepiece.model` se crea/carga correctamente, `dataset_cache/prepared_dataset` contiene `token_ids`, y `main_train.py` usa los `token_ids` pre-tokenizados sin volver a tokenizar.

### 7. Documentación del tokenizador BPE ✅ (COMPLETADO)
- Añadir sección en `README.md` o `FICHA_HUGGINGFACE.MD` sobre la tokenización BPE multilingüe. ✅
- Documentar los flags de entrenamiento e inferencia relacionados con BPE y multiidioma. ✅
- Crear ejemplos de uso para:
  - entrenamiento con BPE ✅
  - inferencia con BPE ✅
  - entrenamiento e inferencia multilingüe con BPE ✅
  - regeneración de caché al añadir nuevos idiomas ✅

## Notas de diseño
- El rethinking **no se implementa en el modelo ni en el entrenamiento**. Se resuelve en la capa de orquestación (agente o endpoint).
- El agente/endpoint recibe el input, lo pasa al modelo, evalúa la salida, y si no cumple, vuelve a llamar con contexto de crítica.
- Se configura un **número máximo de iteraciones** de rethinking para evitar ciclos infinitos.
- El modelo base se mantiene sin cambios. No se toca `chatmodel.py`, `dialogmanager.py` ni `main_train.py`.
