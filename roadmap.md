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

### 2. Tokenizador BPE y soporte multilingüe (Máxima prioridad)
- Cambiar el tokenizador actual a un tokenizador estándar basado en BPE.
- Soportar múltiples idiomas durante el entrenamiento e inferencia.
- Comenzar con Español e Inglés, pero diseñar el pipeline para añadir más idiomas en el futuro.
- Asegurar que el vocabulario y la tokenización sean compatibles con las rutas de entrenamiento y los endpoints de `main_chat.py`.
- Validar que la inferencia multilingüe funcione correctamente con la opción de rethinking opcional.
- Ejecutar pruebas de verificación BPE + caché:
  1. `python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 2000 --refresh-cache`
  2. `python main.py --train --use-cache --epochs 1`
- Objetivo: confirmar que `sentencepiece.model` se crea/carga correctamente, `dataset_cache/prepared_dataset` contiene `token_ids`, y `main_train.py` usa los `token_ids` pre-tokenizados sin volver a tokenizar.
- Estado: PENDIENTE — Prioridad: MÁXIMA

### 3. Rethinking en entrenamiento
- Definir la lógica de rethinking dentro del pipeline de entrenamiento.
- Diseñar un mecanismo que permita al modelo evaluar y reescribir internamente sus salidas parciales durante el entrenamiento.
- Integrar la generación de pasos intermedios (pensamiento / rethinking) en los datos de entrenamiento o en la función de pérdida.
- Añadir métricas que verifiquen si el rethinking mejora la calidad y coherencia de las respuestas.
- Probar con conjuntos de datos conversacionales para validar el comportamiento auto-repensante.

### 4. Rethinking en inferencia
- Implementar una opción de configuración para activar/desactivar rethinking en inferencia.
- Extender el servidor y/o los endpoints de `main_chat.py` para aceptar un parámetro tipo `use_rethinking: true/false`.
- Asegurar que el flujo de inferencia sin rethinking se mantenga compatible con los endpoints actuales.
- Documentar el comportamiento de la inferencia con y sin rethinking.

### 5. Arquitectura y código
- Revisar `chatmodel.py` y `dialogmanager.py` para definir dónde se inyecta el rethinking.
- Actualizar `main_train.py` para incluir nuevos flags o parámetros de rethinking.
- Actualizar `main_chat.py` y cualquier API wrapper para exponer la opción de uso de rethinking.
- Añadir tests básicos que verifiquen la activación/desactivación de rethinking en inferencia y entrenamiento.

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

### Tarea prioritaria para mañana (Máxima prioridad) ✅ (COMPLETADO)

- Ejecutar pruebas de verificación BPE + caché:
  1. `python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 2000 --refresh-cache`
  2. `python main.py --train --use-cache --epochs 1`

  Objetivo: confirmar que `sentencepiece.model` se crea/carga correctamente, `dataset_cache/prepared_dataset` contiene `token_ids`, y `main_train.py` usa los `token_ids` pre-tokenizados sin volver a tokenizar.

### 6. Documentación y guías
- Añadir sección en `README.md` o `FICHA_HUGGINGFACE.MD` sobre la nueva capacidad de rethinking y la tokenización BPE multilingüe.
- Documentar los flags de entrenamiento e inferencia relacionados con rethinking y multiidioma.
- Crear ejemplos de uso para:
  - entrenamiento con rethinking activado
  - inferencia sin rethinking
  - inferencia con rethinking activado
  - entrenamiento e inferencia multilingüe con BPE

## Notas de diseño
- El rethinking debe ser opcional en inferencia, para no penalizar el rendimiento cuando no sea necesario.
- En entrenamiento, el rethinking puede implementarse como una segunda pasada de evaluación interna que refine la predicción.
- Priorizar una integración limpia que no rompa el pipeline existente de `main_train.py` y `main_chat.py`.
