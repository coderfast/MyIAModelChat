# MyIAModelChat
My First IA Chat Model - Advanced Conversational AI with Multi-Source Data Support

## 🚀 Features

MyIAModelChat is a sophisticated conversational AI system built with PyTorch, featuring:

### Core Capabilities
- **Neural Chat Model**: GPT-2 Transformer architecture for natural language generation
- **Bilingual Support**: English/Spanish tokenization with accent handling
- **Intent Recognition**: BERT-based intent classification for user queries
- **Sentiment Analysis**: BERT-based star rating detection (1-5 stars)
- **Context Management**: Maintains conversation history and coherence
- **Persona Modeling**: Customizable AI personality traits
- **Model Library**: Train, combine, and export multiple independent models

### Data Sources & Processing
- **AIML Pattern Matching**: Traditional rule-based responses from AIML files
- **PDF Text Extraction**: Automatic content extraction from PDF documents
- **EPUB E-Book Support**: Full e-book parsing and integration
- **Hugging Face Datasets**: Integration with pre-trained datasets
- **Dataset Caching**: 12x faster training with intelligent caching system

### Advanced Features
- **Dynamic N-gram Penalization**: Prevents repetitive responses
- **Min-Length Enforcement**: Ensures meaningful response lengths
- **Autoregressive Generation**: High-quality text generation with sampling controls
- **Multi-threading Support**: Optimized for CPU training
- **Model Checkpointing**: Best model saving and tokenizer persistence
- **Chain-of-Thought Reasoning**: Optional `<think>` reasoning in training and inference
- **Model Library**: Multiple trained models with merge and export capabilities
- **GGUF/ONNX Export**: Export models to Ollama, llama.cpp, ONNX Runtime

### Advanced Text Processing (NEW)
- **Professional Sentence Tokenization**: spaCy-based sentence splitting (replaces naive `split('.')`)
- **Unicode Normalization**: NFKC normalization for consistent text
- **Text Chunking**: Split long documents into overlapping token windows
- **Deduplication**: MinHash LSH for removing duplicate/near-duplicate texts
- **Quality Filtering**: Filter low-quality texts by length, alpha ratio, spam detection
- **Metadata Preservation**: Extract and preserve document metadata (title, author, etc.)
- **Language Detection**: Filter texts by detected language (supports 50+ languages)

## 🚀 Quick Start

### 1. Prepare Your Data
```bash
# Prepare datasets from multiple sources
python main.py --prepare-data --aiml --pdf --epub
```

### 2. Train the Model
```bash
# Train with cached data for speed
python main.py --train --use-cache --epochs 30
```

### 3. Start Chatting
```bash
# Launch interactive chat
python main.py --chat
```

## 📚 Model Library

Train multiple independent models and use them individually or combined.

### Train a model
```bash
# Train with a custom name and dataset
python main.py --train --dataset datasets_source/ciencias/ --checkpoint-name ciencias_naturales --aiml --hf --epochs 30
python main.py --train --dataset datasets_source/programacion/ --checkpoint-name programacion --aiml --hf --epochs 30
```

### List available models
```bash
python main.py --list-models
```

### Chat with a specific model
```bash
python main.py --chat --model ciencias_naturales
python main.py --chat --model programacion
```

### Combine models (merge)
```bash
python main.py --chat --model ciencias_naturales+programacion
```

### Export models
```bash
# Export to GGUF for Ollama/llama.cpp
python main.py --export ciencias_naturales --formats gguf

# Export to ONNX
python main.py --export ciencias_naturales --formats onnx,onnx_int8

# Export merged model
python main.py --export ciencias_naturales+programacion --formats gguf,onnx
```

### Use with Ollama
```bash
# After exporting to GGUF
ollama create mi-ciencias -f Modelfile.ciencias_naturales
ollama run mi-ciencias
```

## 📖 Examples

### Basic Conversation
```
User: Hello, how are you?
AI: Hello! I'm doing well, thank you for asking. How can I help you today?

User: Tell me about machine learning
AI: Machine learning is a fascinating field of artificial intelligence that focuses on creating algorithms that can learn from data...
```

### Bilingual Support
```
User: Hola, ¿cómo estás?
AI: ¡Hola! Estoy bien, gracias por preguntar. ¿En qué puedo ayudarte?

User: ¿Qué es el aprendizaje automático?
AI: El aprendizaje automático es un campo fascinante de la inteligencia artificial...
```

## 🏗️ Project Structure

```
MyIAModelChat/
├── main.py                 # Main entry point with all CLI commands
├── main_train.py          # Training pipeline
├── main_chat.py           # Chat interface (API server)
├── dialogmanager.py       # Response generation with intent/sentiment
├── data_preparer.py       # Multi-source data loading (AIML, PDF, EPUB, HF)
├── bpe_tokenizer.py       # SentencePiece BPE tokenizer (multilingual)
├── chatmodel.py           # GPT-2 model architecture
├── chatdataset.py         # PyTorch Dataset loader
├── aimlloder.py           # AIML file processing
├── model_downloader.py    # HuggingFace model downloader
├── model_registry.py      # Model discovery, listing, validation
├── model_merge.py         # Model merging by weight averaging
├── model_export.py        # Export to GGUF, ONNX, ONNX quantized
├── generate_thinking_data.py  # Chain-of-thought data generation
├── manual_test.py         # Manual testing utilities
├── checkpoints/           # Model checkpoints (legacy)
├── models/                # Trained model checkpoints (.pth)
│   └── exported/          # Exported models (.gguf, .onnx)
├── datasets_source/       # User-prepared datasets
├── dataset_cache/         # Cached datasets
├── aiml_dev/              # AIML development files
├── envAIModels/           # API server (FastAPI, GGUF/llama_cpp)
└── tests/                 # Test suite
```

## 📚 Data Sources

### AIML Files
- Traditional chatbot patterns from A.L.I.C.E. project
- Rule-based responses for common queries
- Located in `aiml/` directory

### PDF Documents
- Automatic text extraction using PyPDF2
- Supports complex layouts and formatting
- Place PDFs in `pdfs/` directory

### EPUB E-books
- Full e-book parsing with ebooklib
- Chapter-by-chapter content extraction
- Supports metadata and structure
- Place EPUBs in `epub/` directory

## ⚙️ Configuration

### Training Parameters
```python
# Actual configuration (main_train.py)
batch_size = 4                # Micro batch size
accumulation_steps = 8        # Gradient accumulation (effective batch = 32)
learning_rate = 1e-3
embed_size = 256              # Embedding dimension
hidden_size = 512             # Hidden state dimension
num_layers = 4                # Transformer layers
n_head = 4                    # Attention heads
n_positions = 512             # Max sequence length
vocab_size = 8000             # BPE vocabulary size (default)
```

### Generation Settings
```python
# Chat parameters (dialogmanager.py)
top_k = 50
top_p = 0.9
temperature = 0.7
min_length = 3
no_repeat_ngram_size = 3
```

## 📊 Performance

- **Dataset Caching**: 12x faster training iterations
- **Memory Efficient**: Optimized for CPU training
- **Multi-threaded**: Parallel data processing
- **GPU Support**: CUDA acceleration available

## 🔧 Recent Improvements

- Model library system with merge and export capabilities
- GGUF/ONNX export for Ollama, llama.cpp, ONNX Runtime
- SentencePiece BPE multilingual tokenizer
- Chain-of-thought reasoning support
- Dynamic n-gram penalization
- Advanced text processing (chunking, dedup, quality filter)

## 📖 Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training setup
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) - Technical architecture
- [THINKING_GUIDE.md](THINKING_GUIDE.md) - Chain-of-thought reasoning
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Troubleshooting

## 🗂️ Tokenizador BPE multilingüe

El proyecto usa un tokenizador **SentencePiece BPE** (Byte-Pair Encoding) como único sistema de tokenización. SentencePiece es nativamente multilingüe — soporta cualquier idioma (Español, Inglés, Francés, etc.) sin configuración adicional.

### Cómo funciona

1. Durante `--prepare-data`, se entrena un modelo BPE con `sentencepiece`
2. El modelo se guarda en `dataset_cache/sentencepiece.model`
3. Los datos se tokenizan y cachean con `token_ids` pre-calculados
4. En entrenamiento e inferencia, se cargan los `token_ids` directamente (sin re-tokenizar)

### Fases de uso

**Fase 1 — Preparación de datos (entrena el BPE y genera caché):**
```bash
python main.py --prepare-data --aiml --hf --bpe-vocab-size 8000
```

**Fase 2 — Entrenamiento (usa caché tokenizada):**
```bash
python main.py --train --use-cache --epochs 30
```

**Fase 3 — Inferencia (carga modelo BPE automáticamente):**
```bash
python main.py --chat
```

### Flags de BPE

| Flag | Descripción | Valor por defecto |
|------|-------------|-------------------|
| `--bpe-vocab-size` | Tamaño del vocabulario BPE | `8000` |
| `--refresh-cache` | Reconstruir caché desde cero | (no aplica) |
| `--use-cache` | Cargar dataset cacheado si existe | (no aplica) |

### Archivos generados en caché

```
dataset_cache/
├── prepared_dataset/          # Dataset tokenizado (contiene token_ids)
├── dataset_stats.pkl          # Estadísticas del dataset
├── cache_metadata.pkl         # Metadata: bpe_model_path, vocab_size
└── sentencepiece.model        # Modelo BPE entrenado
```

### Regenerar caché

Cuando se añaden nuevos idiomas o fuentes de datos, regenerar la caché:
```bash
python main.py --prepare-data --aiml --hf --pdf --epub --bpe-vocab-size 8000 --refresh-cache
```

### Archivos del tokenizer

| Archivo | Descripción |
|---------|-------------|
| `bpe_tokenizer.py` | Wrapper de SentencePiece (`SentencePieceTokenizerWrapper`) |
| `checkpoints/tokenizer_vocab.json` | Vocabulario del tokenizador (incluye `sentencepiece_model` path) |
| `dataset_cache/sentencepiece.model` | Modelo BPE entrenado |

### Smoke test rápido

```bash
# Preparar datos con vocabulario pequeño para prueba rápida
python main.py --prepare-data --aiml --hf --bpe-vocab-size 2000 --refresh-cache

# Entrenar 1 epoch usando caché
python main.py --train --use-cache --epochs 1
```

Verificar que:
- `dataset_cache/sentencepiece.model` existe
- `dataset_cache/prepared_dataset` contiene `token_ids`
- `main_train.py` carga el modelo BPE desde caché automáticamente

## 🧠 Chain-of-Thought Reasoning (`<think>`)

El modelo soporta **razonamiento encadenado** usando la etiqueta `<think>`. El modelo aprende a generar su proceso de pensamiento antes de la respuesta final.

### Formato de salida

```
<think>
Razonamiento interno del modelo...
</think>
Respuesta final limpia
```

### Generar datos con thinking

```bash
# Generar desde CSV y AIML
python generate_thinking_data.py --source all

# Salida: datasets/thinking/thinking_data.csv
```

### Entrenar con thinking

```bash
# Preparar datos (incluye thinking automáticamente)
python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache

# Entrenar
python main.py --train --use-cache --epochs 30
```

### Inferencia con thinking

**Consola:**
```bash
# Con thinking visible
python main.py --chat --show-thinking

# Sin thinking (respuesta limpia)
python main.py --chat
```

**API:**
```bash
# Con thinking
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hola"}],
    "include_thinking": true
  }'

# Respuesta incluye campo "reasoning"
```

### Flags relacionados

| Flag | Descripción |
|------|-------------|
| `--show-thinking` | Muestra reasoning en consola |
| `include_thinking` | Parámetro en endpoints API |

Ver [THINKING_GUIDE.md](THINKING_GUIDE.md) para documentación completa.

## 🔬 Advanced Text Processing

MyIAModelChat includes professional text processing features for high-quality dataset preparation.

### Key Features

| Feature | Description | Flag |
|---------|-------------|------|
| **Sentence Tokenization** | spaCy-based sentence splitting | Automatic (uses spaCy if available) |
| **Unicode Normalization** | NFKC normalization for consistency | Automatic |
| **Text Chunking** | Split long texts into overlapping windows | `--enable-chunking` |
| **Deduplication** | Remove duplicate texts with MinHash LSH | `--enable-dedup` |
| **Quality Filtering** | Filter low-quality texts | `--enable-quality-filter` |
| **Metadata Preservation** | Extract document metadata | `--preserve-metadata` |
| **Language Detection** | Filter by detected language | `--enable-lang-filter` |

### Usage Examples

```bash
# Full advanced workflow with all features
python main.py --prepare-data --aiml --pdf --epub \
    --enable-chunking \
    --enable-dedup \
    --enable-quality-filter \
    --preserve-metadata \
    --enable-lang-filter \
    --allowed-languages es en

# Prepare PDF with chunking for long documents
python main.py --prepare-data --pdf \
    --enable-chunking \
    --chunk-max-tokens 256 \
    --chunk-overlap 50

# Prepare data with quality filtering
python main.py --prepare-data --pdf --epub \
    --enable-quality-filter \
    --min-words 10 \
    --max-words 500

# Prepare data with deduplication
python main.py --prepare-data --aiml --pdf \
    --enable-dedup \
    --dedup-threshold 0.9

# Prepare data with language filtering (Spanish only)
python main.py --prepare-data --pdf \
    --enable-lang-filter \
    --allowed-languages es
```

### Text Processing Pipeline

```
Raw Text → Unicode Normalization → Sentence Tokenization → Quality Filtering
    → Deduplication → Language Filtering → Chunking (optional) → Dataset
```

### Benefits

1. **Higher Quality Data**: Professional sentence splitting avoids breaking URLs, abbreviations, decimals
2. **Consistent Formatting**: Unicode normalization ensures consistent text across sources
3. **Reduced Redundancy**: Deduplication removes duplicate texts that could bias the model
4. **Cleaner Dataset**: Quality filtering removes noise, spam, and low-quality texts
5. **Better Context**: Chunking preserves context with overlapping windows
6. **Multilingual Support**: Language detection enables filtering by language

### Dependencies (Optional)

```bash
# For professional sentence tokenization
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

# For language detection
pip install langdetect

# For MinHash deduplication
pip install datasketch
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- PyPDF2
- ebooklib
- datasets
- sentencepiece (optional, required for BPE/tokenization during preparation)
- numpy, pandas
- spacy (optional, for professional sentence tokenization)
- langdetect (optional, for language detection)
- datasketch (optional, for MinHash deduplication)

## 🚀 Installation

```bash
# Clone repository
git clone <repository-url>
cd MyIAModelChat

# Create virtual environment
python -m venv envMyIAModelChat
envMyIAModelChat\Scripts\activate  # Windows
# source envMyIAModelChat/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- A.L.I.C.E. AIML project for pattern data
- Hugging Face for transformers and datasets
- PyTorch community for neural network framework