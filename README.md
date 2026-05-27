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

## 📋 Quick Start

### 1. Prepare Your Data
```bash
# Prepare datasets from multiple sources
python main.py --prepare-data --aiml --pdf --epub
```
```bash
# Prepare datasets and train/apply BPE tokenizer (saves tokenized cache)
python main.py --prepare-data --aiml --pdf --epub --use-bpe --bpe-vocab-size 8000
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
├── main_chat.py           # Chat interface
├── dialogmanager.py       # Response generation
├── data_preparer.py       # Multi-source data loading
├── simpletokenizer.py     # Base tokenizer
├── chatmodel.py           # LSTM model architecture
├── aimlloder.py           # AIML file processing
├── checkpoints/           # Model checkpoints
├── dataset_cache/         # Cached datasets
├── datasets/              # Raw data storage
├── aiml/                  # AIML pattern files
├── pdfs/                  # PDF documents
├── epub/                  # EPUB e-books
└── docs/                  # Documentation
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
- [BILINGUAL-TOKENIZER-GUIDE.md](BILINGUAL-TOKENIZER-GUIDE.md) - Tokenizer features
- [DEVELOPMENT-GUIDE.md](DEVELOPMENT-GUIDE.md) - Troubleshooting

## 🗂️ Dataset cache and BPE

- When using `--use-bpe` during `--prepare-data`, the pipeline will train (or load) a SentencePiece BPE model and apply it to the prepared dataset. The cache will include:
	- `dataset_cache/prepared_dataset/` — prepared dataset (includes `bpe_text` and `token_ids` when BPE enabled)
	- `dataset_cache/dataset_stats.pkl` — statistics
	- `dataset_cache/cache_metadata.pkl` — metadata including `bpe_model_path` and vocab size
	- `dataset_cache/sentencepiece.model` — trained SentencePiece BPE model (if generated)

Use `--refresh-cache` to rebuild the cache after changing source files or tokenizer settings.

## Quick verification (BPE + cache smoke test)

To verify the BPE integration and cached training workflow quickly, run:

```bash
# Prepare datasets and build a BPE-tokenized cache (small vocab for quick test)
python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 2000 --refresh-cache

# Run a short training that uses the cached token_ids
python main.py --train --use-cache --epochs 1
```

If `cache_metadata.pkl` contains `bpe_model_path`, `main_train.py` will try to load the corresponding `sentencepiece.model` and use it for tokenization consistency; otherwise the training pipeline will prefer pre-tokenized `token_ids` saved in the cache to avoid re-tokenization and speed up startup.

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- PyPDF2
- ebooklib
- datasets
- sentencepiece (optional, required for BPE/tokenization during preparation)
- numpy, pandas

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