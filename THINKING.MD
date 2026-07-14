# THINKING.MD — Plan de implementación de `<think>` en MyIAModelChat

## Objetivo
Agregar soporte para **razonamiento encadenado (chain-of-thought)** usando la etiqueta `<think>` en el modelo LSTM del proyecto. El modelo aprenderá a generar su proceso de pensamiento dentro de `<think>...</think>` antes de dar la respuesta final.

---

## Tarea 1: Datos de entrenamiento con `<think>` ✅ (COMPLETADO)

### 1.1 Definir el formato de cada ejemplo

Cada par de entrenamiento debe tener esta estructura:

```
<|user|>pregunta del usuario<|end|>
<think>razonamiento interno del modelo...</think>respuesta final limpia<|end|>
```

### 1.2 Estrategia de generación de datos

- **Opción A (recomendada):** Usar un LLM externo (DeepSeek-R1, GPT-4, Claude) para generar `<think>` a partir de pares pregunta-respuesta existentes
- **Opción B:** Escribir razonamiento manualmente para datos críticos
- **Opción C:** Mezcla de ambas

### 1.3 Script generador

Crear `generate_thinking_data.py`:
- Lee el dataset actual (AIML, HuggingFace, etc.)
- Para cada ejemplo, envía el prompt a un LLM externo pidiendo razonamiento
- Guarda el resultado en formato `<think>...</think>...`
- Genera un dataset enriquecido en `datasets/thinking/`

### 1.4 Archivos a crear/modificar

| Archivo | Acción |
|---------|--------|
| `generate_thinking_data.py` | **Crear** — script generador de datos con thinking |
| `datasets/thinking/` | **Crear** — directorio para datos con thinking |
| `data_preparer.py` | **Modificar** — detectar y parsear `<think>` durante preparación |

### 1.5 Criterio de aceptación ✅
- Dataset con al menos 500 ejemplos en formato `<think>...</think>...` ✅ (6,196 generados)
- `data_preparer.py` detecta y preserva las etiquetas `<think>` durante la tokenización ✅

---

## Tarea 2: Tokenización — Tokens especiales `<think>` ✅ (COMPLETADO)

### 2.1 Añadir tokens especiales al BPE

- Añadir `<think>` y `</think>` como `user_defined_symbols` en el entrenamiento SentencePiece
- Asegurar que no se fragmenten (tokens indivisibles)

### 2.2 Modificar `data_preparer.py`

```python
# En la función de entrenamiento BPE, añadir:
user_defined_symbols = ['<EN>', '<ES>', '<think>', '</think>']
```

### 2.3 Actualizar `bpe_tokenizer.py`

- Añadir método `has_thinking(text)` → bool
- Añadir método `split_thinking(text)` → (thinking, response)
- Añadir método `extract_response(text)` → str (solo respuesta sin thinking)
- Los tokens especiales deben mapear a IDs fijos en el vocabulario

### 2.4 Actualizar `word_tokenizer.py`

- Añadir los mismos métodos para compatibilidad

### 2.5 Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `data_preparer.py` | Añadir `<think>`, `</think>` a `user_defined_symbols` |
| `bpe_tokenizer.py` | Añadir helpers de parsing de thinking |
| `word_tokenizer.py` | Añadir mismos helpers |

### 2.6 Criterio de aceptación
- `<think>` y `</think>` son tokens indivisibles en el vocabulario BPE
- Los métodos de parsing funcionan correctamente con datos reales

---

## Tarea 3: Entrenamiento con `<think>` ✅ (COMPLETADO)

### 3.1 Modificar `main_train.py`

- Detectar si el dataset contiene `<think>` (via `cache_metadata.pkl` o heurística)
- Si contiene thinking, usar la estructura completa para el entrenamiento
- El modelo aprende a generar: `<think>pensamiento</think>respuesta`

### 3.2 Función de pérdida

- **Opción A (simple):** Tratar `<think>...</think>...` como texto continuo, loss sobre todos los tokens
- **Opción B (ponderada):** Dar más peso a la generación correcta de `<think>` y `</think>` como delimitadores
- Recomendación: empezar con Opción A, simplificar

### 3.3 Generación de secuencias de entrenamiento

```python
# Cada secuencia de entrenamiento:
input_ids:  [..., token_antes_de_thinking]
target_ids: [..., <think>, razonamiento, </think>, respuesta]
```

- El modelo recibe contexto previo y debe predecir la secuencia completa
- Usar teacher forcing: alimentar la secuencia real como input, predecir el siguiente token

### 3.4 Validación del modelo

- Añadir métrica: % de ejemplos donde el modelo genera `<think>` correctamente
- Añadir métrica: % de ejemplos donde `</think>` cierra correctamente
- Monitorear si el razonamiento generado es coherente (evaluación manual o con LLM)

### 3.5 Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `main_train.py` | Detectar thinking, ajustar secuencias de entrenamiento |
| `chatmodel.py` | Sin cambios (el LSTM genera tokens secuencialmente) |

### 3.6 Criterio de aceptación
- El modelo genera `<think>` en al menos el 70% de las salidas durante entrenamiento
- Las métricas de thinking se registran en logs

---

## Tarea 4: Inferencia con `<think>` ✅ (COMPLETADO)

### 4.1 Modificar `main_chat.py`

- Detectar si la respuesta del modelo contiene `<think>`
- Si la contiene, separar thinking de respuesta
- Decidir qué mostrar al usuario según configuración

### 4.2 Lógica de parsing

```python
def parse_thinking_response(raw_output):
    if '<think>' in raw_output and '</think>' in raw_output:
        thinking = raw_output.split('<think>')[1].split('</think>')[0]
        response = raw_output.split('</think>')[1].strip()
        return thinking, response
    return None, raw_output
```

### 4.3 Configuración

- Añadir flag `--show-thinking` en `main.py`
- Añadir parámetro `include_thinking` en los endpoints de `main_chat.py`
- Por defecto: NO mostrar thinking (solo respuesta)

### 4.4 Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `main_chat.py` | Parsing de `<think>`, lógica de展示 |
| `main.py` | Añadir flag `--show-thinking` |
| `envAIModels/routers_v1.py` | Parámetro `include_thinking` en endpoints |
| `envAIModels/routers_api.py` | Parámetro `include_thinking` en endpoints |

### 4.5 Criterio de aceptación
- Con `--show-thinking`, el usuario ve tanto el razonamiento como la respuesta
- Sin el flag, solo se muestra la respuesta limpia
- Los endpoints devuelven `reasoning` solo cuando se solicita

---

## Tarea 5: Endpoints API con `<think>` ✅ (COMPLETADO)

### 5.1 Añadir parámetro a endpoints

En `routers_v1.py` y `routers_api.py`:

```python
# Añadir al schema de request:
include_thinking: bool = False

# En la respuesta:
{
    "choices": [{
        "message": {
            "content": "respuesta limpia",
            "reasoning": "razonamiento interno"  # solo si include_thinking=True
        }
    }]
}
```

### 5.2 Compatible con OpenAI

- El campo `reasoning` no existe en la API estándar de OpenAI
- Opción: usar campo personalizado `reasoning` o `thinking_content`
- Documentar como extensión no estándar

### 5.3 Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `envAIModels/schemas.py` | Añadir `include_thinking` al request schema |
| `envAIModels/routers_v1.py` | Usar parsing de thinking en respuestas |
| `envAIModels/routers_api.py` | Usar parsing de thinking en respuestas |
| `envAIModels/utils.py` | Helper `build_thinking_response()` |

### 5.4 Criterio de aceptación
- `POST /v1/chat/completions` con `include_thinking: true` devuelve `reasoning`
- `POST /v1/chat/completions` sin `include_thinking` no devuelve `reasoning`
- Streaming funciona correctamente con thinking

---

## Tarea 6: Documentación ✅ (COMPLETADO)

### 6.1 Actualizar `README.md`

- Añadir sección "Razonamiento con `<think>`"
- Documentar flags: `--show-thinking`
- Documentar parámetro de endpoints: `include_thinking`
- Ejemplos de uso

### 6.2 Actualizar `FICHA_HUGGINGFACE.MD`

- Mencionar capacidad de chain-of-thought
- Ejemplo de salida con thinking

### 6.3 Crear `THINKING-GUIDE.md`

- Guía completa de uso
- Ejemplos de datos de entrenamiento
- Cómo generar datos con thinking
- Cómo entrenar el modelo
- Cómo usar en inferencia

### 6.4 Archivos a modificar/crear

| Archivo | Cambio |
|---------|--------|
| `README.md` | Añadir sección de thinking |
| `FICHA_HUGGINGFACE.MD` | Mencionar CoT |
| `THINKING-GUIDE.md` | **Crear** — guía completa |

### 6.5 Criterio de aceptación
- Documentación clara con ejemplos funcionales
- Guía paso a paso para generar datos, entrenar e inferir

---

## Resumen de archivos

### Archivos nuevos
- `generate_thinking_data.py`
- `datasets/thinking/`
- `THINKING-GUIDE.md`

### Archivos a modificar
- `data_preparer.py` — tokens especiales + parsing
- `bpe_tokenizer.py` — helpers de thinking
- `word_tokenizer.py` — mismos helpers
- `main_train.py` — detección y manejo de thinking
- `main_chat.py` — parsing de thinking en inferencia
- `main.py` — flag `--show-thinking`
- `envAIModels/schemas.py` — parámetro `include_thinking`
- `envAIModels/routers_v1.py` — respuesta con thinking
- `envAIModels/routers_api.py` — respuesta con thinking
- `envAIModels/utils.py` — helper de respuesta
- `README.md` — documentación
- `FICHA_HUGGINGFACE.MD` — documentación

---

## Orden de ejecución recomendado

1. **Tarea 2** (Tokenización) — base para todo lo demás
2. **Tarea 1** (Datos) — necesita tokens especiales
3. **Tarea 3** (Entrenamiento) — necesita datos y tokens
4. **Tarea 4** (Inferencia) — necesita modelo entrenado
5. **Tarea 5** (Endpoints) — extiende inferencia
6. **Tarea 6** (Documentación) — al final

---

## Verificación end-to-end

```bash
# 1. Preparar datos con thinking
python generate_thinking_data.py --source aiml --output datasets/thinking/

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
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "¿Qué es Python?"}], "include_thinking": true}'
# → Respuesta con campo "reasoning"
```
