# Chain-of-Thought Reasoning (<think>)

## Qué es

El modelo soporta **razonamiento encadenado** (chain-of-thought) usando la etiqueta `<think>`. Cuando el modelo genera una respuesta, puede incluir su proceso de pensamiento interno antes de la respuesta final.

### Formato

```
<think>
El usuario pregunta sobre X. Voy a analizar la información disponible...
Voy a dar una respuesta clara y directa.
</think>
La respuesta es Y.
```

- **Dentro de `<think>`**: razonamiento interno, pasos intermedios, autocorrección
- **Fuera de `<think>`**: la respuesta limpia que ve el usuario

### Formato de datos de entrenamiento

Cada par de entrenamiento tiene esta estructura:

```
<|user|>pregunta del usuario<|end|>
<think>razonamiento interno del modelo...</think>respuesta final limpia<|end|>
```

---

## Flujo completo

```
1. Generar datos con thinking
   python dataset_preparer/generate_thinking_data.py --source all

2. Preparar datos (incluye thinking en el dataset)
   python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache

3. Entrenar el modelo
   python main.py --train --use-cache --epochs 30

4. Inferencia
   python main.py --chat --show-thinking
```

---

## Generar datos con thinking

El script `dataset_preparer/generate_thinking_data.py` lee el dataset actual (AIML, HuggingFace, etc.) y genera ejemplos enriquecidos con `<think>`.

### Estrategia

- **Opción A (recomendada):** Usar un LLM externo (DeepSeek-R1, GPT-4, Claude) para generar `<think>` a partir de pares pregunta-respuesta existentes
- **Opción B:** Escribir razonamiento manualmente para datos críticos
- **Opción C:** Mezcla de ambas

### Desde CSV y AIML

```bash
python dataset_preparer/generate_thinking_data.py --source all
```

Esto genera `datasets/thinking/thinking_data.csv` con el formato:

```csv
input,output,thinking,thinking_text,category
"¿Quién es tu creador?","Mi creador es Eduardo Piñera Aznárez.","Pregunta de identidad. Respondo de forma clara y directa.","<think>Pregunta de identidad. Respondo de forma clara y directa.</think>Mi creador es Eduardo Piñera Aznárez.",identity
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

La preparación incluye automáticamente los datos de thinking si existen en `datasets/thinking/thinking_data.csv`.

```bash
python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
```

### Qué hace:
- Crea datos thinking desde CSV y AIML
- Entrena modelo BPE con tokens `<think>` y `</think>`
- Tokeniza todos los datos (incluyendo thinking)
- Guarda en `dataset_cache/`

### Tokens especiales en el BPE

`<think>` y `</think>` se añaden como `user_defined_symbols` en el entrenamiento SentencePiece, asegurando que no se fragmenten (tokens indivisibles).

### Archivos generados:
```
dataset_cache/
├── prepared_dataset/          # Dataset con token_ids
├── sentencepiece.model        # Modelo BPE (incluye tokens <think>/</think>)
├── cache_metadata.pkl         # Incluye has_thinking_tokens: true
└── dataset_stats.pkl          # Estadísticas
```

---

## Entrenar con thinking

```bash
python main.py --train --use-cache --epochs 30
```

### Qué hace:
- Detecta si el dataset contiene `<think>` (via `cache_metadata.pkl` o heurística)
- Entrena el modelo para generar `<think>...</think>...`
- Registra métricas de thinking durante entrenamiento

### Generación de secuencias de entrenamiento

Cada secuencia de entrenamiento se construye así:

```python
input_ids:  [..., token_antes_de_thinking]
target_ids: [..., <think>, razonamiento, </think>, respuesta]
```

- El modelo recibe contexto previo y debe predecir la secuencia completa
- Se usa teacher forcing: alimentar la secuencia real como input, predecir el siguiente token
- El modelo GPT-2 genera tokens secuencialmente, sin cambios necesarios en `commons/model/chatmodel.py`

### Función de pérdida

Se usa la estrategia simple: tratar `<think>...</think>...` como texto continuo, con loss sobre todos los tokens. No hay ponderación adicional por el momento.

### Métricas de entrenamiento

Se registran las siguientes métricas:

- `%` de ejemplos donde el modelo genera `<think>` correctamente
- `%` de ejemplos donde `</think>` cierra correctamente
- Coherencia del razonamiento generado (evaluación manual o con LLM)

### Output esperado:
```
✓ Thinking data detected (from cache metadata)
✓ Thinking data: ENABLED (model will learn <think>...</think> structure)
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
from inference.chat_engine import ChatEngine

def parse_thinking_response(raw_output):
    if '<think>' in raw_output and '</think>' in raw_output:
        thinking = raw_output.split('<think>')[1].split('</think>')[0]
        response = raw_output.split('</think>')[1].strip()
        return thinking, response
    return None, raw_output
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
| `dataset_preparer/generate_thinking_data.py` | Genera datos con `<think>` |
| `datasets/thinking/thinking_data.csv` | Datos generados |

### Archivos modificados
| Archivo | Descripción |
|---------|-------------|
| `commons/tokenizer/bpe_tokenizer.py` | Helpers: `has_thinking()`, `split_thinking()`, `extract_response()` |
| `dataset_preparer/data_preparer.py` | Carga datos thinking, tokens `<think>`/`</think>` en BPE |
| `training/trainer.py` | Detecta thinking, métricas de entrenamiento |
| `inference/chat_engine.py` | Parsing de thinking en inferencia |
| `main.py` | Flag `--show-thinking` |
| `envAIModels/schemas.py` | Campo `include_thinking` en requests |
| `envAIModels/utils.py` | `parse_thinking_response()` |
| `envAIModels/routers_v1.py` | Endpoints v1 con thinking |
| `envAIModels/routers_api.py` | Endpoints api con thinking |

---

## Solución de problemas

### El modelo no genera `<think>`

- Verificar que el dataset contiene datos thinking: `head datasets/thinking/thinking_data.csv`
- Verificar que el BPE fue entrenado con los tokens: `cache_metadata.pkl` debe tener `has_thinking_tokens: true`
- Entrenar más epochs (el modelo necesita tiempo para aprender la estructura)

### El thinking no aparece en la API

- Verificar que `include_thinking: true` está en el request
- Verificar que la respuesta del modelo contiene `<think>` (puede que aún no lo genere)

### Tokens `<think>` aparecen como `⁇`

- El modelo BPE fue entrenado sin los tokens
- Ejecutar: `python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache`
- Reentrenar el modelo

### Verificación end-to-end completa

```bash
# 1. Preparar datos con thinking
python dataset_preparer/generate_thinking_data.py --source aiml --output datasets/thinking/

# 2. Entrenar con datos de thinking
python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
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

*Updated: 2026-07-28 - Reflects new modular project structure*
