# PDF Data Loading - Quick Reference

## TL;DR

1. Put PDF files in `datasets_source/pdf/` directory
2. Run: `python main.py --prepare-data --pdf`
3. Train: `python main.py --train --aiml --pdf --epochs 10`

---

## Quick Commands

```bash
# Load only PDF data
python main.py --prepare-data --pdf

# Load PDFs + AIML dialogue patterns
python main.py --prepare-data --aiml --pdf

# Load PDFs + Hugging Face datasets
python main.py --prepare-data --hf --pdf

# Load all three sources (recommended)
python main.py --prepare-data --aiml --hf --pdf

# Train with PDF data
python main.py --train --aiml --pdf --epochs 10

# Refresh cache after adding new PDFs
python main.py --prepare-data --aiml --pdf --refresh-cache
```

---

## Directory Structure

```
project/
├── datasets_source/pdf/                    ← Put your PDF files here
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
├── main.py
├── data_preparer.py
└── requirements.txt
```

**Note:** `datasets_source/pdf/` directory is created automatically on first use.

---

## Workflow

```bash
# Step 1: Prepare data (with PDFs)
python main.py --prepare-data --aiml --pdf

# Step 2: Build vocabulary
python main.py --train --onlytokenize --aiml --pdf

# Step 3: Train model
python main.py --train --aiml --pdf --epochs 10

# Step 4: Chat
python main.py --chat
```

---

## How It Works

1. **Detect PDFs** - Finds all `.pdf` files in `datasets_source/pdf/` directory
2. **Extract Text** - Reads each PDF file and extracts text
3. **Split Sentences** - Splits by periods into training samples
4. **Filter Short** - Removes sentences < 10 characters
5. **Create Dataset** - Converts to HuggingFace Dataset format
6. **Combine** - Merges with AIML/HF data if specified
7. **Cache** - Saves to `dataset_cache/` for fast future runs

---

## Output Example

```
[3/3] Loading PDF files...
  Reading PDF: research.pdf...
  ✓ Loaded: research.pdf (312 samples)
  Reading PDF: guide.pdf...
  ✓ Loaded: guide.pdf (187 samples)
  Total PDF files processed: 2
  Total PDF samples: 499

📊 Data Sources:
   AIML                     41,569 samples
   PDF                         499 samples
   
📈 Combined Statistics:
   Total Samples:           42,068
   Average Text Length:        2.5 words
```

---

## Statistics

### What You Get

| Metric | What It Means |
|--------|---------------|
| **Total samples** | Number of sentences extracted |
| **Avg text length** | Average words per sentence |
| **Min/Max length** | Shortest/longest sentence |

### Example Stats

- **Single 10-page PDF:** ~100-300 samples
- **Multiple PDFs (50 pages):** ~500-1000 samples
- **Large documents (200 pages):** ~2000-5000 samples

---

## Installation Requirements

### Required
- `PyPDF2` - Installed via: `pip install PyPDF2`

### Already Included
- HuggingFace `datasets` library
- PyTorch
- Python 3.8+

---

## Common Tasks

### Add New PDFs and Train

```bash
# 1. Place PDFs in datasets_source/pdf/ directory
# 2. Refresh cache to include new PDFs
python main.py --prepare-data --aiml --pdf --refresh-cache

# 3. Train with updated data
python main.py --train --aiml --pdf --epochs 10
```

### Use Only PDF Data (No AIML/HF)

```bash
python main.py --prepare-data --pdf
python main.py --train --pdf --epochs 10
```

### Compare PDF vs AIML vs HF

```bash
# Test 1: Only PDFs
python main.py --train --pdf --epochs 5
python main.py --chat  # Test model

# Test 2: Only AIML  
python main.py --train --aiml --epochs 5
python main.py --chat  # Compare results

# Test 3: PDFs + AIML (best combination)
python main.py --train --aiml --pdf --epochs 5
python main.py --chat  # Compare
```

---

## Troubleshooting

### PDFs not found?
```bash
# Check directory exists and has PDFs
dir pdfs  # On Windows

# Verify file extensions are .pdf (lowercase)
```

### PyPDF2 not installed?
```bash
pip install PyPDF2

# Or in project venv:
.\envMyIAModelChat\Scripts\pip install PyPDF2
```

### Not many samples extracted?
- PDFs may be too short
- Consider adding more/longer PDFs
- Some PDFs may have encoding issues

### Cache still showing old data?
```bash
# Refresh cache with new PDFs
python main.py --clear-cache
python main.py --prepare-data --aiml --pdf
```

---

## CLI Flags

| Flag | Purpose |
|------|---------|
| `--pdf` | Include PDF data from `datasets_source/pdf/` directory |
| `--prepare-data` | Prepare and validate datasets |
| `--train` | Train the model |
| `--epochs N` | Number of training epochs |
| `--refresh-cache` | Rebuild cache from scratch |
| `--clear-cache` | Delete cached data |
| `--use-cache` | Load from cache (fast) |

---

## Tips & Tricks

✅ **Best Practices:**
- Place domain-specific PDFs for better model performance
- Mix with AIML for conversational AI
- Use `--refresh-cache` after adding new PDFs
- Start with small number of PDFs to test

❌ **Avoid:**
- Scanned PDFs (images) without OCR
- Corrupted PDF files
- Non-PDF files in `datasets_source/pdf/` directory
- Very large PDFs (>500 pages) on slow systems

---

## Data Sources Comparison

| Option | Type | Count | Best For |
|--------|------|-------|----------|
| `--aiml` | Dialogue patterns | ~40K | Conversational |
| `--hf` | Text corpora | ~1K-10K | General text |
| `--pdf` | Documents | ~100-1K per file | Domain-specific |
| `--aiml --hf --pdf` | All combined | ~50K+ | Hybrid models |

---

## Performance

| Operation | Time |
|-----------|------|
| Load 1 PDF (10 pages) | 1-2 sec |
| Load 5 PDFs (50 pages) | 3-5 sec |
| Train 1 epoch | 30-60 sec |
| Load from cache | <1 sec |

---

## Related Files

- **[USAGE.md](USAGE.md)** - Complete CLI reference
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training and caching details
- **[data_preparer.py](data_preparer.py)** - Implementation code

---

## Quick Start (30 seconds)

```bash
# 1. Download a PDF and place in datasets_source/pdf/ directory
# (or create datasets_source/pdf/ directory manually if it doesn't exist)

# 2. Prepare with PDFs
python main.py --prepare-data --aiml --pdf

# 3. Train
python main.py --train --aiml --pdf --epochs 10

# 4. Chat
python main.py --chat

# Done! Your model now learns from PDF content 📚
```

---

## Summary

**PDF Support Added:**
- ✅ Automatic PDF detection in `datasets_source/pdf/` directory
- ✅ Text extraction from all PDF pages
- ✅ Sentence splitting and filtering
- ✅ Integration with existing AIML and HF data
- ✅ Automatic caching for fast iterations
- ✅ Statistics and validation included

**Ready to use!**
```bash
python main.py --prepare-data --pdf
```
