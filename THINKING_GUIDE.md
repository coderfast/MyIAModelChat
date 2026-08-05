# Chain-of-Thought Reasoning (`<thinking>`)

## Qué es

El modelo soporta **razonamiento encadenado** (chain-of-thought) usando la etiqueta `<thinking>`. Cuando el modelo genera una respuesta, puede incluir su proceso de pensamiento interno antes de la respuesta final.

### Formato

```
<thinking>
El usuario pregunta sobre X. Voy a analizar la información disponible...
Voy a dar una respuesta clara y directa.
</thinking>
La respuesta es Y.
```

- **Dentro de `<thinking>`**: razonamiento interno, pasos intermedios, autocorrección
- **Fuera de `<thinking>`**: la respuesta limpia que ve el usuario

### Formato de datos de entrenamiento

Cada par de entrenamiento tiene esta estructura:

```
<|user|>pregunta del usuario<|end|>
<thinking>razonamiento interno del modelo...</thinking>respuesta final limpia<|end|>
```

---

## Flujo completo

```
1. Preparar datos con thinking (Ollama teacher o NLP)
   python main.py --prepare-data --aiml --hf --thinking-mode ollama --refresh-cache

2. Entrenar el modelo
   python main.py --train --use-cache --epochs 30

3. Inferencia
   python main.py --chat --show-thinking
```

---

## Generar datos con thinking

### Opción A: Ollama Teacher (recomendada)

Usa un LLM externo para generar `<thinking>` de alta calidad:

```bash
# Configurar modelo en config.py (OLLAMA_MODEL)
# Asegurarse de que Ollama esté corriendo en localhost:11434

python main.py --prepare-data --aiml --hf --thinking-mode ollama --refresh-cache
```

### Opción B: NLP-based (sin dependencias externas)

Genera thinking usando análisis NLP con ThinkingEngine:

```bash
python main.py --prepare-data --aiml --hf --thinking-mode nlp --refresh-cache
```

### Opción C: Script independiente

```bash
python dataset_preparer/generate_thinking_data.py --source all
```

### Desde CSV y AIML

```bash
python dataset_preparer/generate_thinking_data.py --source all
```

Esto genera `datasets/thinking/thinking_data.csv` con el formato:

```csv
input,output,thinking,thinking_text,category
"¿Quién es tu creador?","Mi creador es Eduardo Piñera Aznárez.","Pregunta de identidad. Respondo de forma clara y directa.","<thinking>Pregunta de identidad. Respondo de forma clara y directa.</thinking>Mi creador es Eduardo Piñera Aznárez.",identity
```

### Desde un CSV personalizado

```bash
python dataset_preparer/generate_thinking_data.py --source csv --input mi_dataset.csv --output datasets/thinking/mi_thinking.csv
```

### Desde directorio AIML

```bash
python dataset_preparer/generate_thinking_data.py --source aiml --input aiml_dev
```

---

## Preparar datos

La preparación incluye automáticamente los datos de thinking si se usa `--thinking-mode`.

```bash
python main.py --prepare-data --aiml --hf --thinking-mode ollama --refresh-cache
```

### Qué hace:
- Genera datos thinking para cada fuente (AIML, PDF, HF, etc.)
- Crea DOS samples por cada entrada original (uno sin thinking, uno con thinking)
- Entrena modelo BPE con tokens `<thinking>` y `</thinking>`
- Tokeniza todos los datos (incluyendo thinking)
- Guarda en `dataset_cache/`

### Tokens especiales en el BPE

`<thinking>` y `</thinking>` se añaden como `user_defined_symbols` en el entrenamiento SentencePiece, asegurando que no se fragmenten (tokens indivisibles).

### Archivos generados:
```
dataset_cache/
├── prepared_dataset/          # Dataset con token_ids
├── sentencepiece.model        # Modelo BPE (incluye tokens <thinking>/</thinking>)
├── cache_metadata.pkl         # Incluye has_thinking_tokens: true
└── dataset_stats.pkl          # Estadísticas
```

---

## Entrenar con thinking

```bash
python main.py --train --use-cache --epochs 30
```

### Qué hace:
- Detecta si el dataset contiene `<thinking>` (via `cache_metadata.pkl` o heurística)
- Entrena el modelo para generar `<thinking>...</thinking>...`
- Registra métricas de thinking durante entrenamiento

### Generación de secuencias de entrenamiento

Cada secuencia de entrenamiento se construye así:

```python
input_ids:  [..., token_antes_de_thinking]
target_ids: [..., <thinking>, razonamiento, </thinking>, respuesta]
```

- El modelo recibe contexto previo y debe predecir la secuencia completa
- Se usa teacher forcing: alimentar la secuencia real como input, predecir el siguiente token
- El modelo GPT-2 genera tokens secuencialmente, sin cambios necesarios en `commons/model/chatmodel.py`

### Función de pérdida

Se usa la estrategia simple: tratar `<thinking>...</thinking>...` como texto continuo, con loss sobre todos los tokens. No hay ponderación adicional por el momento.

### Métricas de entrenamiento

Se registran las siguientes métricas:

- `%` de ejemplos donde el modelo genera `<thinking>` correctamente
- `%` de ejemplos donde `</thinking>` cierra correctamente
- Coherencia del razonamiento generado (evaluación manual o con LLM)

### Output esperado:
```
✓ Thinking data detected (from cache metadata)
✓ Thinking data: ENABLED (model will learn <thinking>...</thinking> structure)
Training batch 50/inf in progress...
  Thinking Metrics Summary:
    thinking_token_accuracy: 0.0234
    thinking_open_accuracy: 0.0189
    thinking_close_accuracy: 0.0278
```

Las métricas de thinking empiezan bajas y mejoran con más epochs.

---

## Inferencia con thinking

### Consola

```bash
# Con thinking visible
python main.py --chat --show-thinking

# Salida:
# You: Hola
# Bot [thinking]: El usuario me saluda. Debo responder de forma amigable.
# Bot: ¡Hola! ¿En qué puedo ayudarte?

# Sin thinking (respuesta limpia)
python main.py --chat

# Salida:
# You: Hola
# Bot: ¡Hola! ¿En qué puedo ayudarte?
```

### Lógica de parsing

```python
from inference.chat_engine import parse_thinking_response

result = parse_thinking_response(raw_output)
# Returns: {'thinking': str or None, 'response': str}
thinking = result['thinking']
response = result['response']
```

Por defecto el thinking NO se muestra (solo respuesta). Usar `--show-thinking` para verlo.

---

## Endpoints API

Todos los endpoints soportan `include_thinking`:

### `/v1/chat/completions`

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "¿Qué es Python?"}],
    "include_thinking": true
  }'
```

### `/api/chat/completions`

```bash
curl -X POST http://localhost:11434/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hola"}],
    "include_thinking": true
  }'
```

### `/api/generate`

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "¿Qué es inteligencia artificial?",
    "include_thinking": true
  }'
```

### Respuesta con thinking

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "La inteligencia artificial es...",
      "reasoning": "El usuario pregunta sobre IA. Voy a dar una definición clara."
    }
  }]
}
```

### Respuesta sin thinking

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "La inteligencia artificial es..."
    }
  }]
}
```

> **Nota:** El campo `reasoning` no existe en la API estándar de OpenAI. Es una extensión no estándar del proyecto. Solo se incluye cuando `include_thinking: true` está en el request.

---

## Archivos y ubicaciones

### Archivos nuevos
| Archivo | Descripción |
|---------|-------------|
| `config.py` | Configuración centralizada (OLLAMA_MODEL, OLLAMA_URL) |
| `dataset_preparer/generate_thinking_data.py` | Genera datos con `<thinking>` |
| `dataset_preparer/thinking_engine.py` | Motor NLP para generación de thinking |
| `datasets/thinking/thinking_data.csv` | Datos generados |

### Archivos modificados
| Archivo | Descripción |
|---------|-------------|
| `commons/tokenizer/bpe_tokenizer.py` | Helpers: `has_thinking()`, `split_thinking()`, `extract_response()` |
| `dataset_preparer/data_preparer.py` | Carga datos thinking, tokens `<thinking>`/`</thinking>` en BPE |
| `dataset_preparer/thinking_generators.py` | Base ThinkingGenerator + OllamaTeacher |
| `training/trainer.py` | Detecta thinking, métricas de entrenamiento |
| `inference/chat_engine.py` | Parsing de thinking en inferencia |
| `main.py` | Flag `--thinking-mode`, `--thinking-model`, `--show-thinking` |
| `envAIModels/schemas.py` | Campo `include_thinking` en requests |
| `envAIModels/utils.py` | `parse_thinking_response()` |
| `envAIModels/routers_v1.py` | Endpoints v1 con thinking |
| `envAIModels/routers_api.py` | Endpoints api con thinking |

---

## Solución de problemas

### El modelo no genera `<thinking>`

- Verificar que el dataset contiene datos thinking: `head datasets/thinking/thinking_data.csv`
- Verificar que el BPE fue entrenado con los tokens: `cache_metadata.pkl` debe tener `has_thinking_tokens: true`
- Entrenar más epochs (el modelo necesita tiempo para aprender la estructura)

### El thinking no aparece en la API

- Verificar que `include_thinking: true` está en el request
- Verificar que la respuesta del modelo contiene `<thinking>` (puede que aún no lo genere)

### Tokens `<thinking>` aparecen como `⁇`

- El modelo BPE fue entrenado sin los tokens
- Ejecutar: `python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --refresh-cache`
- Reentrenar el modelo

### Verificación end-to-end completa

```bash
# 1. Preparar datos con thinking (NLP, sin Ollama)
python main.py --prepare-data --aiml --hf --thinking-mode nlp --refresh-cache

# 2. Entrenar con datos de thinking
python main.py --train --use-cache --epochs 30

# 3. Inferencia sin thinking
python main.py --chat
# → Respuesta limpia solamente

# 4. Inferencia con thinking
python main.py --chat --show-thinking
# → Muestra razonamiento + respuesta

# 5. Test endpoint
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "¿Qué es Python?"}], "include_thinking": true}'
# → Respuesta con campo "reasoning"
```

---

*Updated: 2026-07-28 - Reflects new modular project structure, `<thinking>` tags, and Ollama teacher support*
