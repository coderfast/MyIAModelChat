# Roadmap

## Objetivo general
Agregar soporte para rethinking en el modelo y en la inferencia, de forma que el sistema pueda:
- aprender a aplicar rethinking durante el entrenamiento
- ofrecer la opción de activar o desactivar rethinking en tiempo de inferencia

## Tareas pendientes

### 1. Rethinking en entrenamiento
- Definir la lógica de rethinking dentro del pipeline de entrenamiento.
  - Identificar si el rethinking se aplica por ejemplo, por bloque de tokens, por paso de decodificación, o por iteración completa de salida.
  - Delimitar si el rethinking se usa para corregir coherencia, corregir hechos o reforzar estilo conversacional.
- Diseñar un mecanismo que permita al modelo evaluar y reescribir internamente sus salidas parciales durante el entrenamiento.
  - Establecer un módulo o función que genere “pasos intermedios” de razonamiento.
  - Decidir si el modelo produce una salida inicial y luego una salida refinada, o si reescribe la misma secuencia internamente antes de emitirla.
- Integrar la generación de pasos intermedios (pensamiento / rethinking) en los datos de entrenamiento o en la función de pérdida.
  - Incluir ejemplos marcados con instrucciones paso a paso cuando el dataset lo permita.
  - Ajustar la función de pérdida para penalizar la inconsistencia entre la salida inicial y la final, o para reforzar la salida refinada deseada.
- Añadir métricas que verifiquen si el rethinking mejora la calidad y coherencia de las respuestas.
  - Medir calidad con métricas de similitud, exactitud y consistencia entre repuestas consecutivas.
  - Medir ganancia de coherencia comparando resultados con rethinking activado y desactivado en el mismo batch.
- Probar con conjuntos de datos conversacionales para validar el comportamiento auto-repensante.
  - Preparar un conjunto de pruebas con ejemplos de diálogo donde el rethinking pueda corregir omisiones, ambigüedades o errores de contexto.
  - Guardar ejemplos de inferencia y métricas para comparar iteraciones del modelo.

### 2. Rethinking en inferencia
- Implementar una opción de configuración para activar/desactivar rethinking en inferencia.
  - Definir un parámetro global de sistema y otro de request para activarlo en runtime.
  - Mantener el valor por defecto en `false` para no penalizar la inferencia estándar.
- Extender el servidor y/o los endpoints de `main_chat.py` para aceptar un parámetro tipo `use_rethinking: true/false`.
  - Ajustar el parsing de solicitudes JSON/CLI para reconocer el nuevo flag.
  - Garantizar que los clientes existentes no se rompan si no envían el parámetro.
- Asegurar que el flujo de inferencia sin rethinking se mantenga compatible con los endpoints actuales.
  - Implementar una ruta de ejecución clara: inferencia normal cuando `use_rethinking` está desactivado, y flujo avanzado cuando está activado.
  - Registrar el modo usado en los logs para facilitar diagnóstico.
- Documentar el comportamiento de la inferencia con y sin rethinking.
  - Incluir ejemplos de entrada/salida, diferencias de latencia y casos de uso recomendados.

### 3. Arquitectura y código
- Revisar `chatmodel.py` y `dialogmanager.py` para definir dónde se inyecta el rethinking.
  - Identificar la capa de inferencia donde se pueden insertar los pasos adicionales.
  - Diseñar interfaces limpias que permitan habilitar o deshabilitar el rethinking sin cambios mayores en el resto del código.
- Actualizar `main_train.py` para incluir nuevos flags o parámetros de rethinking.
  - Añadir parámetros como `--use_rethinking`, `--rethinking_steps`, `--rethinking_mode`.
  - Asegurar que dichos parámetros se propaguen a la configuración del modelo y del dataset.
- Actualizar `main_chat.py` y cualquier API wrapper para exponer la opción de uso de rethinking.
  - Permitir que el frontend o API REST envíen el flag `use_rethinking`.
  - Documentar el formato de request y el comportamiento esperado.
- Añadir tests básicos que verifiquen la activación/desactivación de rethinking en inferencia y entrenamiento.
  - Crear tests unitarios sobre el código de configuración y la lógica de flags.
  - Crear tests de integración que ejecuten un ciclo de inferencia con y sin rethinking para comparar resultados.

### 4. Documentación y guías
- Añadir sección en `README.md` o `FICHA_HUGGINGFACE.MD` sobre la nueva capacidad de rethinking.
  - Definir qué es rethinking en el contexto del proyecto.
  - Explicar cuándo conviene activarlo y cuándo no.
- Documentar los flags de entrenamiento e inferencia relacionados con rethinking.
  - Describir cada flag, su propósito, valores posibles y ejemplos de uso.
- Crear ejemplos de uso para:
  - entrenamiento con rethinking activado.
  - inferencia sin rethinking.
  - inferencia con rethinking activado.
  - comparativas de resultados y tiempos de respuesta.

### 5. Tokenizador BPE y soporte multilingüe
- Cambiar el tokenizador actual a un tokenizador estándar basado en BPE.
  - Evaluar bibliotecas compatibles y elegir una implementación estable.
  - Definir el vocabulario inicial y la estrategia de merges.
- Soportar múltiples idiomas durante el entrenamiento e inferencia.
  - Preparar datos de entrenamiento en Español e Inglés.
  - Asegurar que el pipeline pueda extenderse a más lenguajes sin refactorizaciones grandes.
- Comenzar con Español e Inglés, pero diseñar el pipeline para añadir más idiomas en el futuro.
  - Mantener metadatos de idioma en los datasets y en la caché.
- Asegurar que el vocabulario y la tokenización sean compatibles con las rutas de entrenamiento y los endpoints de `main_chat.py`.
  - Verificar que el tokenizador usado en `main_train.py` sea el mismo que en `main_chat.py`.
- Validar que la inferencia multilingüe funcione correctamente con la opción de rethinking opcional.
  - Probar inferencia en ambos idiomas con rethinking activado y desactivado.

### 6. Preparación en caché de datasets con BPE y multilingüe
- Crear o mejorar el pipeline de preparación de datos para que genere un cache previo de datasets.
  - Diseñar un formato de caché que incluya tokens, atención, límites y metadatos de idioma.
  - Añadir validación de integridad al generar la caché.
- Incluir tokenización BPE durante la fase de preprocesamiento antes de guardar en caché.
  - Tokenizar el texto completo y almacenar las secuencias ya procesadas.
  - Incluir la versión del tokenizador para poder invalidar cachés antiguos.
- Permitir que la caché soporte datos en Español e Inglés desde el inicio y pueda ampliarse a más idiomas.
  - Organizar la caché por idioma y por dataset para facilitar la ampliación.
- Asegurar que `main_train.py` cargue los datos tokenizados y cacheados directamente para acelerar el entrenamiento.
  - Incluir lógica para detectar el cache existente y recargarlo.
  - Permitir regenerar el cache cuando cambie el tokenizador o el conjunto de datos.
- Documentar la ruta de `prepare_datasets_for_training` y la forma correcta de regenerar la caché cuando se añaden nuevos idiomas.
  - Añadir instrucciones claras para borrar y reconstruir el cache.
  - Incluir ejemplos de comandos y advertencias sobre compatibilidad.

## Notas de diseño
- El rethinking debe ser opcional en inferencia, para no penalizar el rendimiento cuando no sea necesario.
- En entrenamiento, el rethinking puede implementarse como una segunda pasada de evaluación interna que refine la predicción.
- Priorizar una integración limpia que no rompa el pipeline existente de `main_train.py` y `main_chat.py`.

## Consideraciones de modelo ligero
- El modelo debe ser pequeño y eficiente para poder ejecutarse en hardware limitado como Orange Pi 4 Plus (3 TOPS) o solo CPU.
- Priorizar modelos de 1.3B a 3B parámetros, preferiblemente con soporte para cuantización en 4-bit o 8-bit.
- Mantener una arquitectura de `decoder-only` simple, evitando diseños complejos que aumenten la carga de inferencia.
- Complementar el modelo con un pipeline de retrieval/RAG sobre el contenido EPUB para reducir la necesidad de memorizar información en los pesos.
- Usar fine-tuning o adaptadores ligeros (LoRA) en lugar de entrenar un modelo grande desde cero.
- Asegurar compatibilidad con runtimes optimizados para ARM/CPU, como `llama.cpp` u otros backends livianos.
