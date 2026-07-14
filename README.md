# MyIAModelChat
My First IA Chat Model - Advanced Conversational AI with Multi-Source Data Support

## 🚀 Features

MyIAModelChat is a sophisticated conversational AI system built with PyTorch, featuring:

### Core Capabilities
- **Neural Chat Model**: LSTM-based architecture for natural language generation
- **Bilingual Support**: English/Spanish tokenization with accent handling
- **Intent Recognition**: BERT-based intent classification for user queries
- **Sentiment Analysis**: DistilBERT-powered emotion detection
- **Context Management**: Maintains conversation history and coherence
- **Persona Modeling**: Customizable AI personality traits

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

### Advanced Text Processing (NEW)
- **Professional Sentence Tokenization**: spaCy-based sentence splitting (replaces naive `split('.')`)
- **Unicode Normalization**: NFKC normalization for consistent text
- **Text Chunking**: Split long documents into overlapping token windows
- **Deduplication**: MinHash LSH for removing duplicate/near-duplicate texts
- **Quality Filtering**: Filter low-quality texts by length, alpha ratio, spam detection
- **Metadata Preservation**: Extract and preserve document metadata (title, author, etc.)
- **Language Detection**: Filter texts by detected language (supports 50+ languages)

## 📋 Quick Start

### 1. Prepare Your Data
```bash
# Prepare datasets from multiple sources
python main.py --prepare-data --aiml --pdf --epub
```
```bash
# Prepare datasets with custom BPE vocab size (default: 8000)
python main.py --prepare-data --aiml --pdf --epub --bpe-vocab-size 8000
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
├── main.py                 # Main entry point
├── main_train.py          # Training pipeline
├── main_chat.py           # Chat interface (API server)
├── dialogmanager.py       # Response generation with intent/sentiment
├── data_preparer.py       # Multi-source data loading (AIML, PDF, EPUB, HF)
├── bpe_tokenizer.py       # SentencePiece BPE tokenizer (multilingual)
├── chatmodel.py           # GPT-2 model architecture
├── chatdataset.py         # PyTorch Dataset loader
├── aimlloder.py           # AIML file processing
├── model_downloader.py    # HuggingFace model downloader
├── generate_thinking_data.py  # Chain-of-thought data generation
├── manual_test.py         # Manual testing utilities
├── checkpoints/           # Model checkpoints
├── dataset_cache/         # Cached datasets
├── datasets/              # Raw data storage
├── models/                # Downloaded ML models (sentiment, intent)
├── aiml_dev/              # AIML development files
├── envAIModels/           # API server (FastAPI)
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
# Recommended settings
epochs = 30
batch_size = 32
learning_rate = 1e-3
max_len = 128
vocab_size = 50000
```

### Generation Settings
```python
# Chat parameters
top_k = 50
top_p = 0.9
temperature = 0.8
min_length = 5
no_repeat_ngram_size = 3
```

## 📊 Performance

- **Dataset Caching**: 12x faster training iterations
- **Memory Efficient**: Optimized for CPU training
- **Multi-threaded**: Parallel data processing
- **GPU Support**: CUDA acceleration available

## 🔧 Recent Improvements

See [CRITICAL-FIXES-APPLIED.md](CRITICAL-FIXES-APPLIED.md) for details on:
- Fixed LSTM forward pass issues
- Resolved tokenizer race conditions
- Improved autoregressive generation
- Added dynamic n-gram penalization

## 📖 Documentation

- [TRAINING-GUIDE.md](TRAINING-GUIDE.md) - Complete training setup
- [QUICK-REFERENCE.md](QUICK-REFERENCE.md) - Command reference
- [MODEL-ARCHITECTURE.md](MODEL-ARCHITECTURE.md) - Technical architecture
- [DATASET-CACHING-GUIDE.md](DATASET-CACHING-GUIDE.md) - Caching system
- [THINKING-GUIDE.md](THINKING-GUIDE.md) - Chain-of-thought reasoning
- [DEVELOPMENT-GUIDE.md](DEVELOPMENT-GUIDE.md) - Troubleshooting
- [PLAN_TEST_DATASET.md](PLAN_TEST_DATASET.md) - Dataset preparation analysis
- [PLAN_BPE_TOKENIZER.md](PLAN_BPE_TOKENIZER.md) - BPE tokenizer migration plan

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

Ver [THINKING-GUIDE.md](THINKING-GUIDE.md) para documentación completa.

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