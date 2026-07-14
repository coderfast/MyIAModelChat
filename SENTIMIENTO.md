# Plan: Modelos de Sentimiento e Intención con efecto real en inferencia

## Objetivo

1. Descargar los modelos de sentimiento/intención al equipo local si no existen
2. Cargarlos desde el equipo local si ya existen
3. Hacer que tengan **efecto real** en la generación de respuestas

## Modelo actual

- `nlptown/bert-base-multilingual-uncased-sentiment` — clasifica texto en 1-5 estrellas (sentimiento)
- Se usa para `intent_classifier` y `sentiment_analyzer` (mismo modelo, duplicado)
- **Problema:** Los resultados se descartan o se añaden al prompt sin efecto

## Arquitectura propuesta

```
models/
├── sentiment/                    # Modelo de sentimiento (1-5 estrellas)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── intent/                       # Modelo de intención (pregunta/saludo/queja/etc)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## Paso 1: Crear función de descarga de modelos

**Archivo:** `model_downloader.py` (nuevo)

Crear una función que:
1. Verifique si el modelo existe en `models/sentiment/` o `models/intent/`
2. Si no existe, lo descargue desde HuggingFace con `huggingface_hub.snapshot_download()`
3. Si existe, retorne la ruta local
4. Maneje errores de red con fallback

```python
def ensure_model_local(model_id: str, local_dir: str) -> str:
    """
    Descarga modelo de HuggingFace si no existe localmente.
    Retorna la ruta local del modelo.
    """
```

---

## Paso 2: Actualizar `main_chat.py` — Carga local de modelos

**Cambiar** las líneas 194-195:

```python
# ANTES:
intent_classifier = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# DESPUÉS:
from model_downloader import ensure_model_local

sentiment_model_path = ensure_model_local(
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'models/sentiment'
)
intent_model_path = ensure_model_local(
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'models/intent'
)

intent_classifier = pipeline('text-classification', model=intent_model_path)
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model_path)
```

**Nota:** Ambos apuntan al mismo modelo por ahora. Cuando se añada un modelo real de intención, solo hay que cambiar el `model_id`.

---

## Paso 3: Hacer que sentimiento afecte la temperatura en `dialogmanager.py`

**Idea:** Si el usuario está enojado (sentimiento bajo), usar temperatura más baja (respuestas más cuidadosas). Si está contento, temperatura normal.

**Líneas ~126-146** — Añadir lógica de ajuste:

```python
# Después de obtener sentiment_label:
adjusted_temperature = self.temperature
if sentiment_label:
    stars = sentiment_label.replace(' stars', '').replace(' star', '').strip()
    try:
        stars = int(stars)
        if stars <= 2:
            # Usuario enojado → respuestas más cuidadosas
            adjusted_temperature = max(0.3, self.temperature - 0.2)
        elif stars >= 4:
            # Usuario contento → respuestas más libres
            adjusted_temperature = min(1.0, self.temperature + 0.1)
    except ValueError:
        pass
```

Usar `adjusted_temperature` en lugar de `self.temperature` al generar tokens.

---

## Paso 4: Hacer que intención afecte el prompt en `dialogmanager.py`

**Idea:** Enriquecer el prompt con contexto de intención para guiar al modelo.

```python
# Prompt mejorado:
prompt_parts = [f"Pregunta: {user_text.strip()}"]

if intent_label:
    # Mapear labels del modelo a descripciones legibles
    intent_map = {
        'POSITIVE': 'El usuario expresa algo positivo',
        'NEGATIVE': 'El usuario expresa queja o frustración',
        '1 star': 'El usuario está muy molesto',
        '2 stars': 'El usuario está frustrado',
        '3 stars': 'El usuario tiene una consulta neutral',
        '4 stars': 'El usuario está satisfecho',
        '5 stars': 'El usuario está muy contento',
    }
    intent_desc = intent_map.get(intent_label, f'Tono: {intent_label}')
    prompt_parts.append(f"Contexto: {intent_desc}")

prompt_text = "\n".join(prompt_parts) + "\nRespuesta:"
```

---

## Paso 5: Añadir logging detallado en `dialogmanager.py`

**Línea ~138** — Log después del análisis:

```python
logger.info(f"Intent: {intent_label}, Sentiment: {sentiment_label}, Adjusted temp: {adjusted_temperature:.2f}")
```

---

## Paso 6: Añadir `.gitignore` para modelos

**Archivo:** `.gitignore` — añadir:

```
# Modelos descargados (muy pesados, no subir a git)
models/
```

---

## Archivos a modificar

| Archivo | Acción |
|---------|--------|
| `model_downloader.py` | **CREAR** — función de descarga local |
| `main_chat.py` | **EDITAR** — usar carga local de modelos |
| `dialogmanager.py` | **EDITAR** — sentimiento ajusta temperatura, intención enriquece prompt |
| `.gitignore` | **EDITAR** — ignorar carpeta `models/` |
| `requirements.txt` | **EDITAR** — añadir `huggingface_hub` |

---

## Verificación

1. Ejecutar `python main.py --chat` — debe descargar modelos la primera vez
2. Verificar que `models/sentiment/` y `models/intent/` se crean
3. Ejecutar de nuevo — debe cargar desde local (sin descarga)
4. Enviar mensaje negativo: `"Esto es terrible, no funciona"` — verificar que temperatura baja
5. Enviar mensaje positivo: `"Excelente trabajo"` — verificar que temperatura sube
6. Verificar logs: `Intent: ..., Sentiment: ..., Adjusted temp: 0.50`

---

## Orden de ejecución

1. Crear `model_downloader.py`
2. Crear carpeta `models/` y añadir a `.gitignore`
3. Editar `main_chat.py` — carga local
4. Editar `dialogmanager.py` — efecto real en inferencia
5. Actualizar `requirements.txt`
6. Ejecutar verificación
