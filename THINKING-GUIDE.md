# THINKING-GUIDE.md — Guía de Chain-of-Thought Reasoning

## Qué es el Thinking

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

---

## Flujo completo

```
1. Generar datos con thinking
   python generate_thinking_data.py --source all

2. Preparar datos (incluye thinking en el dataset)
   python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache

3. Entrenar el modelo
   python main.py --train --use-cache --epochs 30

4. Inferencia
   python main.py --chat --show-thinking
```

---

## 1. Generar datos con thinking

### Desde CSV y AIML

```bash
python generate_thinking_data.py --source all
```

Esto genera `datasets/thinking/thinking_data.csv` con el formato:

```csv
input,output,thinking,thinking_text,category
"¿Quién es tu creador?","Mi creador es Eduardo Piñera Aznárez.","Pregunta de identidad. Respondo de forma clara y directa.","<think>Pregunta de identidad. Respondo de forma clara y directa.</think>Mi creador es Eduardo Piñera Aznárez.",identity
```

### Desde un CSV personalizado

```bash
python generate_thinking_data.py --source csv --input mi_dataset.csv --output datasets/thinking/mi_thinking.csv
```

### Desde directorio AIML

```bash
python generate_thinking_data.py --source aiml --input aiml_dev
```

---

## 2. Preparar datos

La preparación incluye automáticamente los datos de thinking si existen en `datasets/thinking/thinking_data.csv`.

```bash
python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
```

### Qué hace:
- Crea datos thinking desde CSV y AIML
- Entrena modelo BPE con tokens `<think>` y `效益`
- Tokeniza todos los datos (incluyendo thinking)
- Guarda en `dataset_cache/`

### Archivos generados:
```
dataset_cache/
├── prepared_dataset/          # Dataset con token_ids
├── sentencepiece.model        # Modelo BPE (incluye tokens <think>/效益)
├── cache_metadata.pkl         # Incluye has_thinking_tokens: true
└── dataset_stats.pkl          # Estadísticas
```

---

## 3. Entrenar el modelo

```bash
python main.py --train --use-cache --epochs 30
```

### Qué hace:
- Detecta datos de thinking en el dataset
- Entrena el modelo para generar `<think>...效益...`
- Registra métricas de thinking durante entrenamiento

### Output esperado:
```
✓ Thinking data detected (from cache metadata)
✓ Thinking data: ENABLED (model will learn <think>...效益 structure)
Training batch 50/inf in progress...
  Thinking Metrics Summary:
    thinking_token_accuracy: 0.0234
    thinking_open_accuracy: 0.0189
    thinking_close_accuracy: 0.0278
```

Las métricas de thinking empiezan bajas y mejoran con más epochs.

---

## 4. Inferencia

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

### API — Endpoints

Todos los endpoints soportan `include_thinking`:

#### `/v1/chat/completions`

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "¿Qué es Python?"}],
    "include_thinking": true
  }'
```

#### `/api/chat/completions`

```bash
curl -X POST http://localhost:11434/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hola"}],
    "include_thinking": true
  }'
```

#### `/api/generate`

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

---

## Archivos del sistema thinking

| Archivo | Descripción |
|---------|-------------|
| `generate_thinking_data.py` | Genera datos con `<think>` |
| `datasets/thinking/thinking_data.csv` | Datos generados |
| `bpe_tokenizer.py` | Helpers: `has_thinking()`, `split_thinking()`, `extract_response()` |
| `data_preparer.py` | Carga datos thinking, tokens `<think>`/`效益` en BPE |
| `main_train.py` | Detecta thinking, métricas de entrenamiento |
| `main_chat.py` | Parsing de thinking en inferencia |
| `envAIModels/schemas.py` | Campo `include_thinking` en requests |
| `envAIModels/utils.py` | `parse_thinking_response()` |
| `envAIModels/routers_v1.py` | Endpoints v1 con thinking |
| `envAIModels/routers_api.py` | Endpoints api con thinking |

---

## Troubleshooting

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
