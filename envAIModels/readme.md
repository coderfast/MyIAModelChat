# 🤖 envAIModels - FastAPI LLM Server

Servidor FastAPI modular para ejecutar modelos de lenguaje locales (Llama, Qwen, etc.) con soporte nativo y OpenAI-compatible (v1).

## 📋 Arquitectura Modular

El proyecto está refactorizado en módulos independientes para mejor mantenibilidad:

### Estructura de Módulos

| Módulo | Propósito | Líneas |
|--------|-----------|--------|
| **`app.py`** | Punto de entrada, inicialización FastAPI, registro de routers | 24 |
| **`routers_api.py`** | Endpoints nativos `/api/*` (generate, chat, completions) | 174 |
| **`routers_v1.py`** | Endpoints OpenAI-compatible `/v1/*` (drop-in replacement para OpenAI) | 166 |
| **`model.py`** | Carga lazy del modelo para evitar overhead en startup | 18 |
| **`schemas.py`** | Modelos Pydantic compartidos (validación de requests/responses) | 20 |
| **`utils.py`** | Funciones centralizadas: builders, streaming, normalizaciones | 310+ |
| **`model_metadata.py`** | Inspección y metadatos del modelo (capas, tamaño, params) | 200+ |

### Endpoints Disponibles

#### 🌐 API Nativa (`/api/*`)
- `POST /api/generate` — Generación de texto
- `POST /api/chat` — Chat bidireccional
- `POST /api/chat/completions` — Completaciones de chat (alias)
- `GET /api/models` — Listar modelos disponibles
- `GET /api/version` — Versión del servidor
- `GET /api/tags` — Metadatos adicionales

#### 🔄 API OpenAI-Compatible (`/v1/*`)
- `POST /v1/completions` — Completación de texto
- `POST /v1/chat/completions` — Completación de chat (OpenAI format)
- `GET /v1/models` — Listar modelos (OpenAI format)
- `GET /v1/health` — Health check
- `GET /v1/version` — Información de versión

### Características
- ✅ **Lazy Loading**: El modelo se carga solo en el primer uso
- ✅ **Streaming**: Soporte para respuestas en tiempo real (NDJSON)
- ✅ **OpenAI Compatible**: Usar cualquier cliente de OpenAI/Ollama sin cambios
- ✅ **Async/Sync**: Manejo híbrido de requests sync y async
- ✅ **Validación**: Pydantic schemas para todos los inputs
- ✅ **Tests**: Suite de tests unitarios y manuales

## 🚀 Inicio Rápido

### Prerrequisitos
- Python 3.10+
- Modelo cuantizado disponible (ej: Qwen2.5-1.5B-Instruct-Q4_0)
- Dependencias: FastAPI, Uvicorn, llama-cpp-python, Pydantic

### Instalación
```bash
# Instalar dependencias
pip install -r ../requirements.txt

# Descargar/localizar modelo (GGUF format)
# Ubicar en: ./model_name.gguf o ajustar MODEL_PATH en model.py
```

### Ejecutar Servidor

**Windows (PowerShell):**
```bash
# Con reload (desarrollo)
python -m uvicorn envAIModels.app:app --reload --port 11434

# O usar el batch script
.\runserver.bat
```

**Linux/Mac:**
```bash
python -m uvicorn envAIModels.app:app --reload --port 11434
```

El servidor estará disponible en `http://127.0.0.1:11434`

## 📖 Ejemplos de Uso

### Opción A: cURL (línea de comandos)

**Generación de texto (API nativa):**
```bash
curl -X POST "http://127.0.0.1:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"¿Cuál es la capital de Francia?","max_tokens":50}'
```

**Chat (OpenAI-compatible):**
```bash
curl -X POST "http://127.0.0.1:11434/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Hola, ¿quién eres?"}],
    "max_tokens":100,
    "temperature":0.7
  }'
```

**Streaming (respuesta en tiempo real):**
```bash
curl -X POST "http://127.0.0.1:11434/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Cuéntame una historia"}],
    "stream":true
  }'
```

> **Nota para Windows**: usar `curl.exe` en lugar de `curl` (alias de PowerShell)

### Opción B: Python (Requests)

```python
import requests
import json

# Generación
response = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={
        "prompt": "¿Qué es la inteligencia artificial?",
        "max_tokens": 100,
        "temperature": 0.7
    }
)
print(response.json()["response"])

# Streaming
response = requests.post(
    "http://127.0.0.1:11434/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Hola"}],
        "stream": True
    },
    stream=True
)
for line in response.iter_lines():
    if line:
        data = json.loads(line)
        if "choices" in data:
            print(data["choices"][0]["delta"]["content"], end="")
```

### Opción C: Cliente OpenAI (Drop-in Replacement)

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://127.0.0.1:11434/v1"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hola"}],
    max_tokens=100,
    stream=False
)
print(response.choices[0].message.content)
```

## 🧪 Testing

### Tests Unitarios
```bash
# Ejecutar suite completa
python -m pytest tests/test_envaimodels_smoke.py -v

# Resultado esperado: 8/8 tests passed ✅
```

### Tests Manuales (servidor en vivo)
```bash
# Requiere server corriendo en otra terminal
python manual_test.py

# Prueba: /api/generate, /v1/chat/completions, streaming, /v1/completions
```

## ⚙️ Configuración

Editar variables en `model.py`:

```python
MODEL_PATH = "path/to/model.gguf"       # Ruta del modelo
MODEL_NAME = "local-Qwen2.5-1.5B"       # Nombre mostrado
OLLAMA_VERSION = "1.0"                  # Versión compatible
MAX_TOKENS = 1024                       # Límite de tokens
```

En `utils.py`:
```python
STOP_TOKENS = ["</s>", "Human:", "AI:"]  # Tokens de parada
```

## 📊 Monitoreo de Modelo

Inspeccionar metadatos del modelo:

```python
from envAIModels.model_metadata import gather_model_metadata

metadata = gather_model_metadata()
print(f"Capas: {metadata['layer_count']}")
print(f"Contexto: {metadata['context_length']}")
print(f"Parámetros: {metadata['parameters_estimate']}")
```

## 📝 Formatos de Respuesta

### `/api/generate` (nativa)
```json
{
  "response": "París es la capital de Francia.",
  "prompt_eval_count": 12,
  "eval_count": 8,
  "total_duration": 1234567890
}
```

### `/v1/chat/completions` (OpenAI-compatible)
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "local-Qwen2.5-1.5B-Instruct-Q4_0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hola, soy un asistente de IA."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming (NDJSON)
```
{"choices":[{"delta":{"content":"París"}}]}
{"choices":[{"delta":{"content":" es"}}]}
{"choices":[{"delta":{"content":" la"}}]}
...
{"choices":[{"delta":{},"finish_reason":"stop"}]}
```

## 🔧 Solución de Problemas

| Problema | Solución |
|----------|----------|
| Puerto 11434 en uso | `netstat -ano \| findstr :11434` (Windows) o cambiar puerto |
| Modelo no encontrado | Verificar `MODEL_PATH` en `model.py` |
| Import error | Activar venv: `. envMyIAModelChat/Scripts/activate` |
| Timeout en generación | Aumentar `max_tokens` o reducir `temperature` |

## 📚 Referencias

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Ollama API Docs](https://github.com/jmorganca/ollama/blob/main/api)
- [llama.cpp Docs](https://github.com/ggerganov/llama.cpp)
