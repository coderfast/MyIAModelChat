# Training & Dataset Management Guide

## Dataset Structure

### Supported Dataset Types

1. **AIML-based Datasets**
   - Source: `datasets_source/aiml/` directory (~5 .aiml files, ~6000+ categories)
   - Processing: `dataset_preparer/aiml/parser.py` resolves `<srai>`, `<random>`, wildcards, HTML tags
   - Format: Normalized training samples with wildcard expansion and quality classification
   - Flag: `--aiml` enables AIML processing

2. **PDF Document Datasets**
   - Source: `datasets_source/pdf/` directory
   - Processing: `dataset_preparer/data_preparer.py` uses pypdf for text extraction
   - Format: Automatic page-by-page text extraction
   - Flag: `--pdf` enables PDF processing

3. **EPUB E-book Datasets**
   - Source: `datasets_source/epub/` directory
   - Processing: `dataset_preparer/data_preparer.py` uses ebooklib for parsing
   - Format: Chapter-by-chapter content extraction with metadata
   - Flag: `--epub` enables EPUB processing

4. **Hugging Face Datasets**
   - Loaded via `transformers.datasets`
   - Multiple public dialogue datasets available
   - Examples: wikitext, bookcorpus, common_voice, opus_100
   - Flag: `--hf` in training enables this

5. **CSV Datasets**
   - Source: `datasets_source/csv/` directory
   - Processing: `dataset_preparer/data_preparer.py` uses csv.reader
   - Format: Curated QA pairs with `input`/`output` columns
   - Flag: `--csv` enables CSV processing

6. **Web Scraping Datasets**
   - Source: `datasets_source/web/` directory
   - Processing: `dataset_preparer/web/scraper.py` crawls documentation
   - Format: Cleaned text from documentation sites
   - Flag: `--web` enables web scraping

### Dataset Caching System

The system includes intelligent caching for 12x faster training iterations:

```bash
# First time: Prepare and cache datasets
python main.py --prepare-data --aiml --pdf --epub

# Train model (always uses cached data)
python main.py --train --epochs 30

# Refresh cache after adding new data
python main.py --prepare-data --pdf --epub --refresh-cache

# Clear cache to free space
python main.py --clear-cache
```

## Build a BPE-tokenized cache (optional)

```bash
python main.py --prepare-data --aiml --pdf --epub --bpe-vocab-size 8000
```

SentencePiece (BPE) is trained on extracted text and the prepared cache will include a `token_ids` column for each sample as well as `dataset_cache/sentencepiece.model` and `dataset_cache/cache_metadata.pkl` containing tokenizer metadata. Note: BPE is applied to textual sources (AIML/PDF/EPUB/HF). If `sentencepiece` is not installed the pipeline will skip BPE and continue preparing a non-tokenized cache (a warning is emitted).

### BPE with Thinking Tokens

When using `--thinking-mode`, the BPE model automatically registers 5 special tokens as `user_defined_symbols`:

```
<thinking>, </thinking>, <|context|>, <|answer|>, <|thinking|>
```

These tokens are never fragmented into sub-tokens (always whole tokens). This ensures the model can learn to generate them correctly.

```bash
# Recommended: BPE with thinking
python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --refresh-cache
```

**Caching Benefits:**
- 12x faster dataset loading
- Consistent preprocessing across runs
- Automatic schema alignment for concatenation
- Memory-efficient storage

### ChatDataset Class (commons/dataset/chatdataset.py)

```python
from commons.dataset.chatdataset import ChatDataset

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns (input_ids, target_ids) pair
        # Input: dialogue context
        # Target: next token(s) to predict
```

### Dataset Combination Strategy

```python
# Load multiple datasets with caching
from dataset_preparer.data_preparer import DataPreparer

preparer = DataPreparer(args)

# Always use cached dataset (created by --prepare-data)
if os.path.exists('dataset_cache'):
    final_dataset = preparer.load_cached_dataset()
else:
    raise RuntimeError("No cached dataset found. Run: python main.py --prepare-data --aiml --hf")
```

## Training Configuration

### Command Line Arguments (main.py --train)

```bash
python main.py --train \
    --epochs 30 \
    --checkpoint-name my_model \
    --val-split 0.1 \
    --early-stopping-patience 5 \
    --cpu \
    --num_cores 4
```

| Argument | Purpose | Accepted Values | Default |
|----------|---------|-----------------|---------|
| `--epochs` | Number of training epochs | `1` - `1000` | `1` |
| `--checkpoint-name` | Name for saved checkpoint | string | `chat_model` |
| `--val-split` | Validation split ratio | `0.0` - `0.5` (0=no validation) | `0.1` |
| `--val-batches` | Max validation batches per epoch | `0` (all), `1` - `N` | `0` |
| `--early-stopping-patience` | Stop if no improvement in N epochs | `0` (disabled), `1` - `N` | `0` |
| `--no-metrics-csv` | Disable CSV metrics logging | flag | `False` |
| `--aiml` | Include AIML datasets (prepare-data only) | flag | `False` |
| `--pdf` | Include PDF document datasets (prepare-data only) | flag | `False` |
| `--epub` | Include EPUB e-book datasets (prepare-data only) | flag | `False` |
| `--hf` | Include Hugging Face datasets (prepare-data only) | flag | `False` |
| `--csv` | Include CSV datasets | flag | `False` |
| `--web` | Include web scraping datasets | flag | `False` |
| `--cpu` | Force CPU training | flag | `False` |
| `--gpu [N,N,...]` | Use GPU (auto or specific indices) | `auto`, `0`, `0,1`, etc. | `auto` |
| `--vulkan` | Force Vulkan backend | flag | `False` |
| `--cpu+gpu N,N,...` | CPU+GPU hybrid (split by VRAM) | `0`, `0,1`, etc. | - |
| `--gpu-enum` | Enumerate available GPUs and exit | flag | - |
| `--model-info NAME` | Show model layer details | model name | - |
| `--num_cores` | CPU cores for DataLoader workers | `0` (auto), `1` - `N` | `0` |
| `--num_threads` | Additional threads per worker | `0` (auto), `1` - `N` | `0` |
| `--dataset PATH` | Path to dataset directory | path string | `dataset_cache` |
| `--statistics` | Show detailed per-step timing stats | False |
| `--thinking-mode` | Generate thinking data (nlp, ollama) | none |
| `--thinking-model` | Ollama model for thinking (with ollama mode) | llama3.2 |
| `--show-thinking` | Show thinking in chat output | False |
| `--web-url URL` | Seed URL to scrape | reads from urls.txt |
| `--web-max-pages N` | Max pages to scrape | 50 |
| `--web-max-depth N` | Max link-following depth | 3 |
| `--draft-enabled` | Create draft model after training | False |
| `--draft-num-layers N` | Draft model layers (default: 2) | `1` - `4` |
| `--draft-embed-size N` | Draft embed size (default: 128) | `64` - `256` |
| `--draft-hidden-size N` | Draft hidden size (default: 256) | `128` - `512` |
| `--draft-n-head N` | Draft attention heads (default: 2) | `1` - `4` |
| `--draft-kd-enabled` | Train draft with Knowledge Distillation | False |
| `--draft-kd-temp T` | KD temperature (default: 2.0) | `1.0` - `5.0` |
| `--draft-kd-loss-weight W` | KD loss weight (default: 0.5) | `0.0` - `1.0` |
| `--draft-kd-epochs N` | Draft training epochs (default: 10) | `1` - `100` |
| `--tensorboard-enabled` | Enable TensorBoard logging | False |
| `--tensorboard-log-dir DIR` | TensorBoard log directory | `runs` |
| `--tensorboard-comment TEXT` | Comment suffix for run name | `""` |
| `--tensorboard-freq N` | Log every N epochs | `1` |

### Hyperparameter Tuning

**Learning Rate**
- Start with 0.001 for Adam optimizer
- Reduce if loss oscillates: 0.0001
- Increase if training is too slow: 0.01
- Use learning rate scheduling after epoch 5

**Batch Size**
- Small (8-16): More gradient updates, slower training
- Medium (32-64): Balanced (recommended for this project)
- Large (128+): Faster training, needs more memory

**Embedding Dimension**
- 256: Current default (fast, efficient)
- 512: Better for larger vocabularies
- <256: Faster inference, less expressive

**Hidden Size**
- 512: Current default
- 256: Faster, lighter model
- 1024+: More capacity, slower training

## Training with Thinking

### Overview

The model supports chain-of-thought reasoning using a dual-token system:

- **Mode tokens** (`<|thinking|>`, `<|context|>`, `<|answer|>`): Define sample structure
- **Content tags** (`<thinking>`, `</thinking>`): Wrap reasoning content

### Sample Formats

```
THINKING: <|thinking|>question<thinking>reasoning</thinking><|answer|>answer
CONTEXT:  <|context|>question<|answer|>answer
```

### Generating Thinking Data

```bash
# NLP-based thinking (no external dependencies)
python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --refresh-cache

# Ollama teacher (higher quality, requires Ollama)
python main.py --prepare-data --aiml --hf --thinking-mode ollama --bpe-vocab-size 8000 --refresh-cache

# Multilingual support (30 languages)
python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --allowed-languages es en fr de --refresh-cache
```

### Training with Thinking

```bash
python main.py --train --epochs 30
```

The trainer automatically:
1. Detects thinking data via mode tokens or content tags
2. Applies differentiated loss weighting:
   - Preamble (before `<thinking>`): weight 0.0
   - Reasoning content: weight 0.5
   - Answer tokens: weight 1.0
3. Tracks thinking metrics (delimiter accuracy, reasoning coverage)

### Expected Training Metrics

```
Epoch  1/30 | Train Loss: 4.2315 | Val Loss: 4.1892 | Perplexity: 68.90/65.90 | Gap: -0.0423 | LR: 1.00e-03 | Tokens/s: 12450 | Grad: 0.85
Epoch  2/30 | Train Loss: 3.8764 | Val Loss: 3.9201 | Perplexity: 48.30/50.40 | Gap: +0.0437 | LR: 9.20e-04 | Tokens/s: 12680 | Grad: 0.72
...
Epoch 15/30 | Train Loss: 2.1045 | Val Loss: 2.8934 | Perplexity: 8.20/18.10 | Gap: +0.7889 | LR: 2.10e-04 | Tokens/s: 12590 | Grad: 0.45
```

**Metrics explained:**
- **Train Loss**: Error on training data (lower = better)
- **Val Loss**: Error on validation data (lower = better, must track train loss)
- **Perplexity**: `e^loss` — "how many options the model considers" (8 = confident, 68 = uncertain)
- **Gap**: `val_loss - train_loss` (negative = good generalization, positive/growing = overfitting)
- **Grad Norm**: Gradient stability (spike >10 = gradient explosion)
- **Tokens/s**: Effective training speed
- **Thinking**: `thinking_accuracy`, `thinking_open_acc`, `thinking_close_acc`, `thinking_coverage`, `response_accuracy` — Thinking block metrics
- **Agent**: `agent_tool_call_acc`, `agent_observation_acc`, `agent_ratio` — Agentic token metrics
- **MoE**: `moe_gate_entropy_norm` (0=collapsed, 1=balanced), `moe_expert_N_util` — Expert utilization

**CSV logging**: Metrics saved to `checkpoints/{name}_metrics.csv` for visualization.

### HTML Report

After training, a `_report.html` file is generated alongside the CSV with:
- **Loss & Perplexity** charts (train vs val)
- **Gap & Learning Rate** charts
- **Thinking** section: accuracy, coverage, response accuracy with status indicators
- **Agent** section: tool_call_acc, observation_acc, ratio with status indicators
- **MoE** section: gate entropy, expert utilization stacked bars with status indicators
- **Status banner** at the bottom: OK (healthy), WARNING (overfitting), INFORMATION (stable)
- **Google Translate** banner (top-right) for translating the report to any language

**Status indicators:**
- OK: loss decreased >10% or accuracy > threshold
- WARNING: loss increased >10% or accuracy below threshold
- INFORMATION: loss changed <10% (stable/plateau)

### Chat with Thinking

```bash
# With thinking visible
python main.py --chat --show-thinking

# Clean response only
python main.py --chat
```

## Draft Model (Speculative Decoding)

### Que es Speculative Decoding?

Speculative decoding es una tecnica que acelera la inferencia **1.5x-3x** sin cambiar la calidad del modelo. Funciona asi:

1. Un **draft model** (pequeno y rapido) propone N tokens candidatos
2. El **target model** (grande y preciso) verifica todos en un solo forward pass
3. Los tokens aceptados se quedan; los rechazados se descartan
4. El resultado es **identico** a usar solo el target (lossless)

### Entrenar un Draft Model

```bash
# Draft con Knowledge Distillation (mejor calidad)
python main.py --train --epochs 30 --draft-enabled --draft-kd-enabled --draft-kd-epochs 10

# Draft sin KD (entrenamiento simple, mas rapido)
python main.py --train --epochs 30 --draft-enabled --draft-kd-epochs 10

# Draft con configuracion JSON
python main.py --train --config training_config.json

# Draft personalizado via CLI
python main.py --train --epochs 30 \
    --draft-enabled \
    --draft-num-layers 2 \
    --draft-embed-size 128 \
    --draft-hidden-size 256 \
    --draft-n-head 2 \
    --draft-kd-enabled \
    --draft-kd-temperature 2.0 \
    --draft-kd-loss-weight 0.5 \
    --draft-kd-epochs 10
```

**Archivos generados:**
```
checkpoints/
├── chat_model.pth                      # Modelo target
└── chat_model_draft.pth                # Modelo draft
```

### Exportar a GGUF

```bash
# Exportar target + draft
python main.py --export chat_model --formats gguf

# Genera:
# models/exported/chat_model_hf/          -> target GGUF
# models/exported/chat_model_draft_hf/    -> draft GGUF
```

## TensorBoard

TensorBoard permite visualizar el progreso del entrenamiento en tiempo real con graficas interactivas en el navegador.

### Instalacion

```bash
pip install tensorboard
```

### Uso

```bash
# Entrenar con TensorBoard habilitado
python main.py --train --tensorboard-enabled

# Con comment personalizado
python main.py --train --tensorboard-enabled --tensorboard-comment "mi_experimento"

# Loguear cada 5 epochs
python main.py --train --tensorboard-enabled --tensorboard-freq 5

# Con configuracion JSON
# tensorboard.enabled = true en training_config.json
python main.py --train --config training_config.json

# Ver TensorBoard en el navegador (en otra terminal)
tensorboard --logdir=runs
```

### Metricas Registradas

| Categoria | Metricas | Descripcion |
|-----------|----------|-------------|
| **Loss** | `loss/train`, `loss/val` | Error del modelo |
| **Perplexity** | `perplexity/train`, `perplexity/val` | `e^loss` |
| **General** | `gap`, `learning_rate`, `grad_norm`, `tokens_per_sec`, `best_loss` | Metricas generales |
| **Thinking** | `thinking/thinking_token_accuracy`, `thinking/thinking_coverage`, etc. | Si thinking enabled |
| **Agent** | `agent/agent_tool_call_accuracy`, `agent/agent_ratio`, etc. | Si agent enabled |
| **MoE** | `moe/moe_gate_entropy_norm`, `moe/moe_expert_*_util`, etc. | Si MoE enabled |
| **MTP** | `mtp/mtp_loss` | Si MTP enabled |

### Inference con llama.cpp

**Requisitos:** llama.cpp build 9200+ (soporte speculative decoding)

```bash
# Compilar llama.cpp (macOS/Linux)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release

# Ejecutar con draft model
./build/bin/llama-server \
  -m models/chat_model-q4_k_m.gguf \
  -md models/chat_model_draft-q4_k_m.gguf \
  --spec-type draft-simple \
  --spec-draft-n-max 5 \
  --port 8080
```

**Windows:**
```batch
llama-server.exe ^
  -m models\chat_model-q4_k_m.gguf ^
  -md models\chat_model_draft-q4_k_m.gguf ^
  --spec-type draft-simple ^
  --spec-draft-n-max 5 ^
  --port 8080
```

**Flags de llama.cpp:**

| Flag | Descripcion | Default |
|------|-------------|---------|
| `-m` | Modelo target (principal) | obligatorio |
| `-md` | Modelo draft (pequeno) | sin draft |
| `--spec-type` | Tipo: `draft-simple`, `draft-mtp`, `ngram-simple` | `none` |
| `--spec-draft-n-max` | Max tokens a proponer por paso | `3` |
| `--spec-draft-n-min` | Min tokens a proponer | `0` |
| `--spec-draft-p-min` | Probabilidad minima para seguir draftando | `0.00` |

**Uso con API (OpenAI-compatible):**
```bash
# El servidor expone la misma API que OpenAI
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat_model",
    "messages": [{"role": "user", "content": "Hola"}],
    "temperature": 0.7
  }'
```

### Inference con Ollama

Ollama no soporta draft models directamente, pero puedes:
1. Usar el modelo target sin draft (funciona normal)
2. Exportar a GGUF y usar con llama.cpp (ver arriba)

```bash
# Crear Modelfile (se genera automaticamente en la exportacion)
ollama create mi_modelo -f Modelfile

# Ejecutar
ollama run mi_modelo
```

### Inference con vLLM

vLLM no soporta draft models de terceros, pero soporta **MTP** (Multi-Token Prediction) si el modelo lo tiene habilitado:

```bash
# Instalar vLLM
pip install vllm

# Ejecutar (MTP se detecta automaticamente del GGUF)
vllm serve models/chat_model-q4_k_m.gguf \
  --tensor-parallel-size 1 \
  --port 8000
```

### Inference con HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar modelo target
model = AutoModelForCausalLM.from_pretrained("models/exported/chat_model_hf")
tokenizer = AutoTokenizer.from_pretrained("models/exported/chat_model_hf")

# Generar texto
inputs = tokenizer("Hola", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**Nota:** Transformers no soporta draft models. Solo usa el target.

### Cuándo usar Draft Model?

| Escenario | Usar draft? | Razon |
|-----------|-------------|-------|
| Chat interactivo | Si | Latencia reducida, mejor UX |
| Batch processing | Si | Throughput mayor |
| API server | Si | Mas requests por segundo |
| Fine-tuning | No | Solo sirve para inferencia |
| Evaluacion | No | Quieres resultados exactos del target |
| CPU only | No | Draft no acelera en CPU |

### Draft vs MTP

| Aspecto | Draft Model | MTP |
|---------|-------------|-----|
| **Archivo** | Separado (`_draft.pth`) | Embebido en el target |
| **Exportar** | Dos GGUFs separados | Un solo GGUF |
| **Uso llama.cpp** | `-md draft.gguf` | `--spec-type draft-mtp` |
| **Velocidad** | ~1.5-2x | ~1.8-2.5x |
| **Calidad** | Depende del KD | Depende del entrenamiento |
| **Complejidad** | Mas archivos | Menos archivos |

**Recomendacion:** Usar **MTP** si el modelo ya lo tiene (mejor integracion). Usar **Draft Model** si necesitas flexibilidad o el modelo no tiene MTP.

## Tokenization

### SentencePiece BPE Workflow

The system uses SentencePiece BPE for multilingual tokenization:

```python
# Training phase (during --prepare-data)
from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

# BPE model is trained automatically and saved to:
# dataset_cache/sentencepiece.model
# dataset_cache/cache_metadata.pkl

# Inference phase
tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
token_ids = tokenizer.encode("Hello world")  # English
token_ids_es = tokenizer.encode("Hola mundo")  # Spanish
token_ids_fr = tokenizer.encode("Bonjour le monde")  # French
text = tokenizer.decode(token_ids)
```

**BPE Features:**
- Multilingual support (any language)
- Subword tokenization (handles OOV words)
- Efficient encoding/decoding
- Vocabulary size configurable (default: 8000)

### Vocabulary Management

- Vocabulary size affects model parameters
- Larger vocab = more flexibility, more memory
- Common sizes: 2k, 8k, 16k, 32k tokens
- Special tokens (automatically registered in BPE):
  - `<pad>`, `<unk>`, `<s>`, `</s>` (structural)
  - `<thinking>`, `</thinking>` (content tags for reasoning)
  - `<|context|>`, `<|answer|>`, `<|thinking|>` (mode tokens)

### Issues & Solutions

**Out-of-vocabulary (OOV) tokens**
- Problem: Unseen words during inference
- Solution: BPE handles via subword tokenization
- Prevention: Use adequate vocabulary size (8000+ recommended)

**Tokenization mismatch**
- Problem: Different tokenization between train/inference
- Solution: Always use same SentencePiece model from cache
- Verify: Check token IDs for same input across runs

**Low vocabulary coverage**
- Problem: Too many unknown tokens
- Solution: Increase `--bpe-vocab-size` and retrain
- Prevention: Use 8000+ vocab size for multilingual data

## Training Best Practices

### Data Preprocessing with Multiple Sources
1. Clean text (remove special chars if needed)
2. Normalize case (lowercase or mixed)
3. Remove duplicates across all sources
4. Balance dataset categories if using classification
5. Split into train/validation/test (70/15/15)
6. Cache processed datasets for speed

### Training Monitoring with Caching
```python
# Track metrics per epoch
for epoch in range(epochs):
    if use_cache:
        # Load cached dataset (fast)
        dataset = load_cached_dataset()
    else:
        # Load fresh data (slow)
        dataset = load_all_sources()

    train_loss = run_training_epoch(dataset)
    val_loss = run_validation_epoch(dataset)

    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'loss': val_loss
        }, f'models/{checkpoint_name}_best_{timestamp}.pth')
```

### Common Training Issues

**Out of Memory (OOM)**
- Reduce batch size
- Use `--cpu` for CPU training
- Use `--cpu+gpu 0` to split layers across CPU and GPU
- Reduce sequence length (max_len parameter)
- Use gradient accumulation

**Loss not decreasing**
- Learning rate too high/low
- Model capacity insufficient
- Bad data quality from PDF/EPUB extraction
- Try different initialization seed

**Overfitting**
- Loss decreases but validation worsens
- Add dropout layers (currently 0.1)
- Use early stopping
- Increase regularization (L2)
- Add more diverse training data (PDFs/EPUBs)

**Slow training**
- Use cached dataset (from `--prepare-data`) for 12x speedup
- Increase `num_workers` in DataLoader
- Use GPU instead of CPU
- Profile code to find bottlenecks

**PDF/EPUB parsing errors**
- Verify pypdf and ebooklib are installed
- Check file formats are valid
- Use `--prepare-data` to test loading
- Check extracted text quality

## Dataset Splitting

### Train/Validation/Test Split

```python
from torch.utils.data import random_split

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size]
)
```

## Checkpointing & Recovery

### Saving Training State with Tokenizer

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'tokenizer': tokenizer,
    'model_name': checkpoint_name,
    'architecture': {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 4,
        'n_head': 4,
        'n_positions': 512,
        'vocab_size': tokenizer.vocab_size,
    },
    'dataset_source': dataset_source,
}
torch.save(checkpoint, f'models/{checkpoint_name}.pth')
```

### Resuming Training

Training automatically continues from the last checkpoint's epoch number. If `chat_model_epoch_5_*.pth` exists, running `--train` will start from epoch 6.

The resume logic handles architecture changes between sessions:
- **MoE detection**: reads actual state_dict keys (`mlp.experts`, `mlp.gate`) instead of metadata flags
- **Shape mismatch filtering**: `_filter_state_dict` skips keys with incompatible shapes (e.g., different expert count)
- **Graceful loading**: uses `strict=False` to load partial weights when architectures differ

```python
# Manual resume (if needed)
checkpoint = torch.load('checkpoints/chat_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch']  # Already includes offset
```

## Model Library

### Training Multiple Models

```bash
# Train each model with its own dataset and name
python main.py --train --dataset datasets_source/ciencias/ --checkpoint-name ciencias_naturales --aiml --hf --epochs 30
python main.py --train --dataset datasets_source/programacion/ --checkpoint-name programacion --aiml --hf --epochs 30
python main.py --train --dataset datasets_source/historia/ --checkpoint-name historia --aiml --hf --epochs 30
```

### Listing Models

```bash
python main.py --list-models
```

### Using Models

```bash
# Chat with a specific model
python main.py --chat --model ciencias_naturales

# Chat with merged models
python main.py --chat --model ciencias_naturales+programacion
```

### Exporting Models

```bash
# Export to GGUF for Ollama
python main.py --export ciencias_naturales --formats gguf

# Export to ONNX
python main.py --export ciencias_naturales --formats onnx,onnx_int8

# Export merged model
python main.py --export ciencias_naturales+programacion --formats gguf,onnx
```

### Checkpoint Format

Each checkpoint now includes metadata for validation and export:

```python
{
    'epoch': 30,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'loss': 0.1234,
    'tokenizer': ...,
    'model_name': 'ciencias_naturales',
    'architecture': {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 4,
        'n_head': 4,
        'n_positions': 512,
        'vocab_size': 8000,
    },
    'dataset_source': 'datasets_source/ciencias/',
}
```

**Epoch numbering**: Checkpoint filenames use correlated epoch numbers. If you trained 2 epochs previously (`chat_model_epoch_1_*.pth`, `chat_model_epoch_2_*.pth`), the next training session continues from epoch 3.

**Per-epoch overwrite**: `chat_model.pth` is overwritten after every epoch, always containing the latest model state. Epoch-specific checkpoints (`chat_model_epoch_N_*.pth`) are only saved when a new best loss is achieved.

**HTML report**: After training, a `_report.html` file is generated with Chart.js graphs for Loss, Perplexity, Gap, LR, Speed, Thinking, Agent, and MoE metrics. Python `None` values are serialized as JavaScript `null` for correct rendering.

## Performance Profiling

### Identifying Bottlenecks

```python
import torch.profiler as profiler

with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
    # Your training code
    pass

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Common Bottlenecks
1. Data loading (use pre-built cache from `--prepare-data` for 12x speedup)
2. PDF/EPUB parsing (cache extracted text)
3. GPU transfer (use pinned memory)
4. Tokenization (use batched encoding with SentencePieceTokenizerWrapper)
5. AIML parsing (cache parsed files)

---

*Updated: 2026-08-26 - Reflects new modular project structure*
