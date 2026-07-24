# EPUB E-Book Support - Quick Reference

## TL;DR

1. Put EPUB files in `datasets_source/epub/` directory
2. Run: `python main.py --prepare-data --epub`
3. Train: `python main.py --train --aiml --epub --epochs 10`

---

## Quick Commands

```bash
# Load only EPUB data (e-books)
python main.py --prepare-data --epub

# Load EPUBs + AIML dialogue
python main.py --prepare-data --aiml --epub

# Load EPUBs + Hugging Face datasets
python main.py --prepare-data --hf --epub

# Load all sources (recommended)
python main.py --prepare-data --aiml --hf --pdf --epub

# Train with EPUB data
python main.py --train --aiml --epub --epochs 10

# Refresh cache after adding new EPUB files
python main.py --prepare-data --aiml --epub --refresh-cache
```

---

## Directory Structure

```
project/
├── datasets_source/epub/                    ← Put your EPUB e-books here
│   ├── novel1.epub
│   ├── book2.epub
│   └── ebook3.epub
├── main.py
├── data_preparer.py
└── requirements.txt
```

**Note:** `datasets_source/epub/` directory is created automatically on first use.

---

## Workflow

```bash
# Step 1: Prepare data (with EPUBs)
python main.py --prepare-data --aiml --epub

# Step 2: Build vocabulary
python main.py --train --onlytokenize --aiml --epub

# Step 3: Train model
python main.py --train --aiml --epub --epochs 10

# Step 4: Chat
python main.py --chat
```

---

## How It Works

1. **Detect EPUBs** - Finds all `.epub` files in `datasets_source/epub/` directory
2. **Extract Text** - Reads each e-book and extracts chapter text
3. **Remove HTML** - Strips all markup tags for clean text
4. **Split Sentences** - Splits by periods into training samples
5. **Filter Short** - Removes sentences < 10 characters
6. **Create Dataset** - Converts to HuggingFace Dataset format
7. **Combine** - Merges with AIML/PDF/HF data if specified
8. **Cache** - Saves to `dataset_cache/` for fast future runs

---

## Output Example

```
[4/3] Loading EPUB files...
  Reading EPUB: novel.epub...
  ✓ Loaded: novel.epub (1,245 samples)
  Reading EPUB: guide.epub...
  ✓ Loaded: guide.epub (567 samples)
  Total EPUB files processed: 2
  Total EPUB samples: 1,812

📊 Data Sources:
   AIML                     41,569 samples
   EPUB                     1,812 samples
   
📈 Combined Statistics:
   Total Samples:           43,381
   Average Text Length:        2.8 words
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

- **Small e-book (100 pages):** ~500-1000 samples
- **Medium e-book (300 pages):** ~1500-2500 samples
- **Large e-book (600 pages):** ~3000-5000 samples
- **Multiple books (1000 pages):** ~5000-10000 samples

---

## Installation Requirements

### Required
- `ebooklib` - Installed via: `pip install ebooklib`

### Already Included
- HuggingFace `datasets` library
- PyTorch
- Python 3.8+

---

## Common Tasks

### Add New EPUBs and Train

```bash
# 1. Place EPUB files in datasets_source/epub/ directory
# 2. Refresh cache to include new EPUBs
python main.py --prepare-data --aiml --epub --refresh-cache

# 3. Train with updated data
python main.py --train --aiml --epub --epochs 10
```

### Use Only EPUB Data (No AIML/HF/PDF)

```bash
python main.py --prepare-data --epub
python main.py --train --epub --epochs 10
```

### Train on Fiction (Narrative Model)

```bash
# Place novels in datasets_source/epub/ directory
python main.py --prepare-data --epub
python main.py --train --epub --epochs 15
python main.py --chat
```

### Knowledge System (All Sources)

```bash
# Combine all knowledge sources
python main.py --prepare-data --aiml --hf --pdf --epub
python main.py --train --aiml --hf --pdf --epub --epochs 10
```

---

## Troubleshooting

### EPUBs not found?
```bash
# Check directory exists and has EPUBs
dir epub

# Verify file extensions are .epub (lowercase)
```

### ebooklib not installed?
```bash
pip install ebooklib

# Or in project venv:
.\envMyIAModelChat\Scripts\pip install ebooklib
```

### Not many samples extracted?
- E-books may be too short
- Consider adding longer/more books
- Some EPUBs have complex formatting

### DRM-Protected Kindle Books?
- Convert using **Calibre** (free tool)
- Export as EPUB format
- Place converted file in `datasets_source/epub/` directory
- Check local laws before removing DRM

### Cache still showing old data?
```bash
# Refresh cache with new EPUBs
python main.py --clear-cache
python main.py --prepare-data --aiml --epub
```

---

## CLI Flags

| Flag | Purpose |
|------|---------|
| `--epub` | Include EPUB data from `datasets_source/epub/` directory |
| `--prepare-data` | Prepare and validate datasets |
| `--train` | Train the model |
| `--epochs N` | Number of training epochs |
| `--refresh-cache` | Rebuild cache from scratch |
| `--clear-cache` | Delete cached data |
| `--use-cache` | Load from cache (fast) |

---

## Tips & Tricks

✅ **Best Practices:**
- Use diverse e-books for better model performance
- Mix with AIML for conversational systems
- Use `--refresh-cache` after adding new EPUBs
- Start with small number of e-books to test

❌ **Avoid:**
- DRM-protected Kindle books (use Calibre to convert)
- Corrupted EPUB files
- Non-EPUB files in `datasets_source/epub/` directory
- Very large collections (100+ books) on slow systems

---

## Data Sources Comparison

| Option | Type | Count | Best For |
|--------|------|-------|----------|
| `--aiml` | Dialogue patterns | ~40K | Conversational |
| `--hf` | Text corpora | ~1K-10K | General text |
| `--pdf` | Documents | ~100-1K per file | Technical |
| `--epub` | E-books | ~500-2K per book | Narrative |
| All | Combined | ~50K+ | Hybrid |

---

## Performance

| Operation | Time |
|-----------|------|
| Load 1 e-book (100 pages) | 1-3 sec |
| Load 5 e-books (500 pages) | 5-8 sec |
| Load 10 e-books (1000 pages) | 10-15 sec |
| Train 1 epoch | 30-60 sec |
| Load from cache | <1 sec |

---

## Related Files

- **[USAGE.md](USAGE.md)** - Complete CLI reference
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training and caching details
- **[data_preparer.py](data_preparer.py)** - Implementation code

---

## Free E-Book Sources

### Project Gutenberg (70,000+ books)
- https://www.gutenberg.org/
- All public domain
- EPUB download available
- Completely free

### Open Library
- https://openlibrary.org/
- Millions of books
- Many in EPUB format
- Free borrowing

### Standard Ebooks
- https://standardebooks.org/
- High-quality formatting
- Public domain books
- Free downloads

---

## Quick Start (30 seconds)

```bash
# 1. Download free e-book from Project Gutenberg
#    (or place your own EPUB files)

# 2. Place in datasets_source/epub/ directory

# 3. Prepare with EPUBs
python main.py --prepare-data --aiml --epub

# 4. Train
python main.py --train --aiml --epub --epochs 10

# 5. Chat
python main.py --chat

# Done! Your model now learns from e-books 📚
```

---

## Summary

**EPUB Support Added:**
- ✅ Automatic EPUB detection in `datasets_source/epub/` directory
- ✅ Text extraction from all chapters
- ✅ HTML tag removal for clean text
- ✅ Sentence splitting and filtering
- ✅ Integration with existing data sources
- ✅ Automatic caching for fast iterations
- ✅ Statistics and validation included

**Ready to use!**
```bash
python main.py --prepare-data --epub
```

---

## Get Free E-Books

**Project Gutenberg** has 70,000+ free books in EPUB format!

1. Go to https://www.gutenberg.org/
2. Search for your book
3. Download EPUB format
4. Place in `datasets_source/epub/` directory
5. Run: `python main.py --prepare-data --epub`

That's all! Your model now learns from the e-book content. 📖
