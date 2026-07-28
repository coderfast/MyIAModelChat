# MODELS.md - Arquitecturas de Modelos para Chat

## Tipos de arquitectura

### Decoder-only (generacion de texto)

Los modelos decoder-only generan texto token por token, de izquierda a derecha. Son los mas comunes para chatbots y asistentes.

| Modelo | Autor | Params | Ventaja |
|---|---|---|---|
| **GPT-2** (actual) | OpenAI | 15-124M | Ligero, rapido de entrenar |
| **GPT-3** | OpenAI | 175B | Alta calidad,requiere API |
| **LLaMA** | Meta | 7B-70B | Moderno, alta calidad, open source |
| **Mistral** | Mistral AI | 7B | Eficiente, open source |
| **Qwen** | Alibaba | 1.5B-72B | Multilingue, buen Espanol |
| **Phi** | Microsoft | 1.3B-14B | Pequeno pero capaz |
| **Gemma** | Google | 2B-7B | Open source, eficiente |

### Encoder-decoder (seq2seq)

Los modelos encoder-decoder procesan una secuencia completa y generan otra. Ideales para traduccion y resumen.

| Modelo | Autor | Params | Ventaja |
|---|---|---|---|
| **T5** | Google | 60M-11B | Flexible, multilingue |
| **mBART** | Meta | 680M | Multilingue nativo, buen Espanol |
| **BART** | Meta | 140M-400M | Bueno para resumen y generacion |
| **UL2** | Google | 20B | Unificado, multiple tareas |

### Encoder-only (clasificacion)

Los modelos encoder-only entienden texto pero no generan. Ideales para clasificacion, sentiment analysis, NER.

| Modelo | Autor | Params | Ventaja |
|---|---|---|---|
| **BERT** | Google | 110M | Basico para clasificacion |
| **RoBERTa** | Meta | 125M | Mejor que BERT en muchas tareas |
| **DistilBERT** | HuggingFace | 66M | Version ligera de BERT |
| **ALBERT** | Google | 12M-235M | Optimizado en parametros |
| **DeBERTa** | Microsoft | 135M-1.5B | Estado del arte en clasificacion |

---

## Para el proyecto MyIAModelChat

### Arquitectura actual: GPT-2 custom

- **Tipo**: Decoder-only
- **Params**: ~5.34M (custom, no 124M estandar)
- **Entrenamiento**: Desde cero con datos propios
- **Tokenizer**: SentencePiece BPE (multilingue)
- **Archivo**: `commons/model/chatmodel.py`

### Ventajas del modelo actual

- Ideal para proyecto local (~20M params)
- Rapido de entrenamiento (minutos/horas)
- Multilingue nativo
- Sin dependencia de GPUs potentes
- Control total de la configuracion

### Alternativas consideradas

| Modelo | Params | Pros | Contras |
|---|---|---|---|
| **Qwen2.5-1.5B** | 1.5B | Multilingue, buen Espanol | Mas lento, requiere mas RAM |
| **Phi-2** | 2.7B | Muy capaz para su tamano | Solo ingles nativo |
| **mBART** | 680M | Multilingue nativo | Encoder-decoder, diferente arquitectura |
| **LLaMA-7B** | 7B | Alta calidad | Demasiado grande para local |

---

## Glosario

| Termino | Significado |
|---|---|
| **Backbone** | Arquitectura base que se usa como estructura del modelo |
| **Decoder-only** | Modelo que solo genera texto (izquierda a derecha) |
| **Encoder-decoder** | Modelo que primero entiende la entrada y luego genera la salida |
| **Encoder-only** | Modelo que solo entiende texto, no genera |
| **Params** | Numero de parametros entrenables del modelo |
| **GGUF** | Formato de cuantizacion para inference eficiente (llama.cpp) |
| **BPE** | Byte-Pair Encoding, algoritmo de tokenizacion |
| **SentencePiece** | Libreria de tokenizacion agnostica a idioma |
