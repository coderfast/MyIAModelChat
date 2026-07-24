# NLP & Transformers en MyIAModelChat

## Modelos utilizados

El proyecto usa **un solo modelo** de HuggingFace para ambas tareas NLP:

### nlptown/bert-base-multilingual-uncased-sentiment

- **Tipo**: BERT multilingual fine-tuned para clasificación de sentimiento por estrellas
- **Capacidades**: Soporta 6 idiomas (inglés, español, francés, alemán, italiano, neerlandés)
- **Salida**: Clasificación de 1 a 5 estrellas (1 star, 2 stars, 3 stars, 4 stars, 5 stars)
- **Tamaño**: ~110M parámetros (BERT-base)
- **Uso en el proyecto**: Ambas pipelines (intención y sentimiento) usan este mismo modelo

```python
from transformers import pipeline

# Ambas pipelines cargan el mismo modelo base
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
intent_classifier = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
```

### Descarga y caché local

Los modelos se descargan una vez y se almacenan localmente:

```python
from model_downloader import ensure_model_local

sentiment_model_path = ensure_model_local(
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'models/sentiment'  # Caché local
)
intent_model_path = ensure_model_local(
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'models/intent'  # Caché local (copia separada)
)
```

**Archivos locales:**
```
models/
├── sentiment/    # Modelo para análisis de sentimiento
└── intent/       # Modelo para clasificación de intención
```

## Pipeline deDialogueManager

### Flujo completo

```
Entrada del usuario
    ↓
┌─────────────────────────┐
│ Intent Classification   │  → label: "4 stars"
│ (text-classification)   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Sentiment Analysis      │  → label: "4 stars"
│ (sentiment-analysis)    │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Ajuste de temperatura   │  → temp: 0.7 → 0.8
│ (según sentimiento)     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Prompt enriquecido      │  → "Pregunta: ...\nContexto: ..."
│ (con contexto NLP)      │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Generación GPT-2        │  → Tokens generados
│ (autoregressive)        │
└─────────────────────────┘
    ↓
Respuesta final
```

### Análisis de intención

El clasificador de intención usa el modelo de sentimiento como proxy para entender la intención del usuario. Las estrellas se mapean a descripciones en español:

| Estrellas | Label del modelo | Descripción en prompt |
|-----------|------------------|----------------------|
| 1 star | `1 star` | El usuario está muy molesto |
| 2 stars | `2 stars` | El usuario está frustrado |
| 3 stars | `3 stars` | El usuario tiene una consulta neutral |
| 4 stars | `4 stars` | El usuario está satisfecho |
| 5 stars | `5 stars` | El usuario está muy contento |

**Código de mapeo** (`dialogmanager.py:161-170`):
```python
intent_map = {
    'POSITIVE': 'El usuario expresa algo positivo',
    'NEGATIVE': 'El usuario expresa queja o frustración',
    '1 star': 'El usuario está muy molesto',
    '2 stars': 'El usuario está frustrado',
    '3 stars': 'El usuario tiene una consulta neutral',
    '4 stars': 'El usuario está satisfecho',
    '5 stars': 'El usuario está muy contento',
}
```

### Análisis de sentimiento y ajuste de temperatura

El sentimiento detectado ajusta la temperatura de generación para adaptar el tono de la respuesta:

| Sentimiento | Temperatura | Efecto |
|-------------|-------------|--------|
| 1-2 estrellas (negativo) | `temp - 0.2` (mín. 0.3) | Respuestas más conservadoras, cuidadosas |
| 3 estrellas (neutral) | Sin cambio | Temperatura base |
| 4-5 estrellas (positivo) | `temp + 0.1` (máx. 1.0) | Respuestas más variadas, expresivas |

**Código de ajuste** (`dialogmanager.py:144-153`):
```python
if sentiment_label:
    stars = sentiment_label.replace(' stars', '').replace(' star', '').strip()
    stars = int(stars)
    if stars <= 2:
        adjusted_temperature = max(0.3, self.temperature - 0.2)
    elif stars >= 4:
        adjusted_temperature = min(1.0, self.temperature + 0.1)
```

### Prompt enriquecido

El análisis NLP se inyecta en el prompt antes de la generación:

```
Pregunta: ¿Cómo estás?
Contexto: El usuario está satisfecho
Respuesta:
```

Esto le da al modelo GPT-2 contexto sobre el tono emocional del usuario para generar respuestas más adecuadas.

## Parámetros de generación

### Configuración por defecto

```python
DialogueManager(
    top_k=50,              # Muestra de los 50 tokens más probables
    top_p=0.9,             # Nucleus sampling: 90% de probabilidad acumulada
    temperature=0.7,       # Temperatura base (ajustada por sentimiento)
    max_len=128,           # Longitud máxima de respuesta en tokens
    min_length=3,          # Longitud mínima de respuesta
    no_repeat_ngram_size=3 # Penalización de n-gramas repetidos
)
```

### Filtrado Top-K / Top-P

El `DialogueManager` implementa filtrado combinado:

1. **Top-K**: Elimina tokens fuera de los K más probables
2. **Top-P (Nucleus)**: Elimina tokens cuya probabilidad acumulada supera P

```python
def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0):
    # 1. Filtrar por top_k
    kth_vals, _ = torch.topk(logits, top_k)
    min_kth = kth_vals[..., -1, None]
    logits[logits < min_kth] = -Inf

    # 2. Filtrar por top_p
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    logits[indices_to_remove] = -Inf
```

### Penalización de n-gramas repetidos

El modelo aplica una penalización dinámica para evitar repeticiones:

```python
if self.no_repeat_ngram_size and len(generated) >= self.no_repeat_ngram_size - 1:
    penalty = 5.0  # Penalización fija
    for token_id in range(logits.size(-1)):
        cand_seq = generated + [token_id]
        if self._is_repeated_ngram(cand_seq, self.no_repeat_ngram_size):
            logits[0, token_id] -= penalty
```

## Arquitectura del modelo principal

### GPT-2 Transformer (chatmodel.py)

El modelo de generación es un GPT-2 personalizado:

```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=8000,      # Tamaño del vocabulario BPE
    n_embd=256,           # Dimensión de embeddings
    n_head=4,             # Cabezas de atención
    n_layer=4,            # Capas Transformer
    n_positions=512       # Longitud máxima de secuencia
)
model = GPT2LMHeadModel(config)
```

**Parámetros totales**: ~9.2M (ligero, entrenable en GPU modestas)

### Flujo del forward pass

```
Token IDs → Embedding → Transformer (4 capas) → LM Head → Logits
                                                      ↓
                                              Softmax → Probabilidades
                                                      ↓
                                              Top-K/Top-P Sampling
                                                      ↓
                                              Siguiente token
```

## Dependencias

| Paquete | Propósito | Obligatorio |
|---------|-----------|-------------|
| `transformers` | Pipelines NLP + GPT-2 | Sí |
| `torch` | Inferencia del modelo | Sí |
| `model_downloader.py` | Descarga y caché de modelos | Sí (interno) |

## Rendimiento

| Métrica | Valor |
|---------|-------|
| Tiempo de inferencia NLP | ~50-100ms por request |
| Tiempo de generación GPT-2 | ~200-500ms por respuesta |
| Memoria NLP (BERT) | ~280MB por modelo (2 modelos = ~560MB) |
| Memoria GPT-2 | ~37MB |
| **Total RAM inferencia** | **~700MB** |

## Solución de problemas

### Modelo NLP no encontrado
```bash
# Los modelos se descargan automáticamente la primera vez
# Si falla, verificar conexión a internet y reintentar
# O descargar manualmente:
python -c "from model_downloader import ensure_model_local; ensure_model_local('nlptown/bert-base-multilingual-uncased-sentiment', 'models/sentiment')"
```

### Sentimiento no detectado
- Verificar que el texto tiene suficientes caracteres (mínimo ~3)
- El modelo BERT funciona mejor con oraciones completas
- Los textos muy cortos o con solo emoticonos pueden no clasificarse bien

### Temperatura no se ajusta
- El ajuste solo funciona si el sentimiento es "1 star", "2 stars", "4 stars" o "5 stars"
- "3 stars" no cambia la temperatura (neutral)
- Los labels "POSITIVE"/"NEGATIVE" no activan el ajuste de temperatura
