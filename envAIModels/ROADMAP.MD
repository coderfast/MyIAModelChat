# Roadmap de refactorización para `envAIModels`

## Objetivo
Separar el monolito actual en componentes Python independientes y bien organizados:
- endpoints `v1` en un módulo separado
- endpoints Ollama/OpenAI en otro módulo separado
- funciones de extracción / análisis de metadatos del modelo en un módulo independiente
- mantener `app.py` como punto de arranque y registro de rutas generales

## Propuesta de archivos

- `envAIModels/app.py`
  - Mantener solo la inicialización de FastAPI, carga de configuración y registro de routers.
  - Importar routers externos de los módulos de endpoints.
  - Importar el cliente/modelo compartido y las utilidades comunes.

- `envAIModels/routers_api.py`
  - Definir los endpoints propios de la API de la aplicación:
    - `/api/generate`
    - `/api/chat`
    - `/api/chat/completions`
  - Incluir rutas y lógica específicas de la API local.

- `envAIModels/routers_v1.py`
  - Definir los endpoints de compatibilidad OpenAI/v1:
    - `/v1/health`
    - `/v1/models`
    - `/v1/version`
    - `/v1/completions`
    - `/v1/chat/completions`
  - Mantener aquí solo la lógica necesaria para compatibilidad de payloads y respuestas OpenAI.

- `envAIModels/model_metadata.py`
  - Extraer toda la lógica de inspección y diagnóstico del modelo de IA.
  - Incluir funciones como:
    - lectura de `model_params`, `context_params`, `metadata`
    - inferencia de capas totales
    - enumeración de tensores, bloques, capas, embeddings y parámetros
    - generación de reportes o dicts reutilizables
  - Este módulo no debería exponer endpoints, solo utilidades para debugging y metadata.

- `envAIModels/utils.py`
  - Guardar helpers compartidos entre endpoints:
    - `normalize_stop`, `get_chat_stop`
    - `truncate_text_by_stop`
    - `build_text_completion_response`, `build_chat_completion_response`
    - `extract_v1_prompt`
    - `run_model_sync` / `run_model_async`
    - `safe_get_text_from_item`
    - lógica de streaming común

- `envAIModels/schemas.py`
  - Definir los modelos Pydantic usados por todos los endpoints:
    - `GenerateRequest`
    - `ChatMessage`
    - `ChatRequest`
  - Esto facilita compartir los esquemas sin duplicación.

## Orden sugerido de código dentro de `envAIModels`

1. `app.py`
   - imports básicos
   - variables de configuración y rutas base
   - carga del modelo compartido
   - registro de routers: `app.include_router(...)`
   - endpoints muy simples si son necesarios (salud, metadata global)

2. `schemas.py`
   - Pydantic models y validaciones comunes

3. `utils.py`
   - Helpers puramente de aplicación y compatibilidad
   - no incluir lógica del modelo ni inspectores de metadatos

4. `model_metadata.py`
   - Funciones de inspección y diagnóstico del modelo
   - solo lógica de lectura/análisis de metadatos

5. `routers_api.py`
   - Router con basename `/api`
   - endpoints locales orientados a la API actual

6. `routers_v1.py`
   - Router con basename `/v1`
   - endpoints de compatibilidad OpenAI / Ollama

## Beneficios

- Mejora la legibilidad y permite pruebas unitarias más fáciles.
- Facilita añadir nuevas variantes de rutas sin aumentar la complejidad de `app.py`.
- Aísla la lógica de metadatos del modelo en un módulo reutilizable y testeable.
- Hace más sencillo mantener compatibilidad con OpenAI/v1 y con las rutas propias de la app.

## Notas extra

- Mantener `runserver.bat` o `readme.md` actualizados con el nuevo arranque si cambian los imports.
- Si `envAIModels` crece más adelante, se puede crear un paquete `envAIModels/routers/__init__.py`.
- Evitar mover código hasta tener una primera versión de la estructura en el roadmap.
