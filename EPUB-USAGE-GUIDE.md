# EPUB E-Book Support Guide

## Overview

MyIAModelChat now supports loading text data from **EPUB files** (e-books). EPUB is the standard e-book format used by most devices and platforms. You can now train your model on e-books and digital publications!

---

## Quick Start

### 1. Add EPUB Files

Place your EPUB files in the `epub/` directory (created automatically):

```
project_root/
├── epub/
│   ├── book1.epub
│   ├── book2.epub
│   └── book3.epub
├── main.py
└── ... (other files)
```

### 2. Include EPUBs in Dataset Preparation

```bash
python main.py --prepare-data --epub
```

Or combine with other sources:

```bash
python main.py --prepare-data --aiml --hf --pdf --epub
```

### 3. Train with EPUB Data

```bash
python main.py --train --aiml --epub --epochs 10
```

---

## Commands

### Load Only EPUB Data
```bash
python main.py --prepare-data --epub
```
**Result:** Extracts text from all EPUB files in `epub/` directory

### Combine EPUB with AIML
```bash
python main.py --prepare-data --aiml --epub
```
**Result:** Combines dialogue patterns (AIML) with e-book content (EPUB)

### Combine All Sources (Recommended)
```bash
python main.py --prepare-data --aiml --hf --pdf --epub
```
**Result:** Uses AIML dialogue + Wikipedia/HF + PDFs + E-books

### Train with EPUB Data
```bash
python main.py --train --aiml --epub --epochs 10
```
**Result:** Trains model using AIML dialogue patterns + e-book content

### Train with All Sources
```bash
python main.py --train --aiml --hf --pdf --epub --epochs 10
```
**Result:** Full training with all available data sources

### Refresh Cache After Adding EPUBs
```bash
python main.py --prepare-data --aiml --epub --refresh-cache
```
**Result:** Rebuilds cache including newly added EPUB files

---

## How EPUB Processing Works

### Text Extraction
1. **EPUB Reading** - ebooklib reads the EPUB structure
2. **Chapter Extraction** - Extracts text from all chapters
3. **HTML Removal** - Strips markup tags for clean text
4. **Sentence Splitting** - Splits by periods (.)
5. **Filtering** - Removes sentences shorter than 10 characters
6. **Dataset Creation** - Converts to training samples

### Data Format
Each extracted sentence becomes one training sample:
```python
{
    'input_ids': 'This is text extracted from an e-book file'
}
```

### Sample Count
- A typical **100-page e-book** yields **500-2000 samples**
- Depends on page density and sentence length
- Dense text (novels) → more samples
- Sparse text (technical) → fewer samples

---

## Examples

### Example 1: Load Single E-Book
```bash
# Place a novel in epub/ directory
python main.py --prepare-data --epub

# Output:
# [4/3] Loading EPUB files...
#   Reading EPUB: novel.epub...
#   ✓ Loaded: novel.epub (1,245 samples)
#   Total EPUB files processed: 1
#   Total EPUB samples: 1,245
```

### Example 2: Load Multiple E-Books
```bash
# Place multiple EPUBs in epub/ directory:
# - fiction_book.epub
# - science_guide.epub
# - history_story.epub

python main.py --prepare-data --epub

# Output:
# [4/3] Loading EPUB files...
#   ✓ Loaded: fiction_book.epub (1,245 samples)
#   ✓ Loaded: science_guide.epub (789 samples)
#   ✓ Loaded: history_story.epub (654 samples)
#   Total EPUB files processed: 3
#   Total EPUB samples: 2,688
```

### Example 3: Knowledge System (All Sources)
```bash
# Combine dialogue patterns + web knowledge + PDFs + e-books
python main.py --prepare-data --aiml --hf --pdf --epub

# Output:
# 📊 Data Sources:
#    AIML                     41,569 samples
#    HuggingFace              1,000 samples
#    PDF                        750 samples
#    EPUB                     2,688 samples
#
# 📈 Combined Statistics:
#    Total Samples:           46,007
```

### Example 4: Fiction Writing Assistant
```bash
# Train on fiction e-books for creative writing
python main.py --prepare-data --epub
python main.py --train --epub --epochs 20
python main.py --chat

# Result: Model trained on writing style and narrative patterns
```

---

## EPUB File Types

### ✅ Supported
- Standard EPUB 2.0 format (.epub)
- EPUB 3.0 format (.epub)
- E-books from major publishers
- Books from Project Gutenberg (free)
- Self-published e-books
- Digital magazines preserved as EPUB
- Academic papers in EPUB format

### ❌ Not Supported
- DRM-protected e-books (Amazon Kindle locked)
- Encrypted/password-protected EPUBs
- Corrupted EPUB files
- Pre-release/malformed EPUBs

---

## EPUB Directory Structure

The `epub/` directory is automatically created when you first use `--epub` flag:

```
epub/
├── fiction_novel.epub          # ✅ Will be loaded
├── technical_guide.epub        # ✅ Will be loaded
├── company_report.epub         # ✅ Will be loaded
├── locked_ebook.epub           # ❌ DRM-protected (may fail)
├── notes.txt                   # ❌ Ignored (not EPUB)
└── image.png                   # ❌ Ignored (not EPUB)
```

**All .epub files in `epub/` directory are automatically loaded and processed.**

---

## Statistics Output

When running `--prepare-data --epub`, you see:

```
📊 Data Sources:
   EPUB                       2,688 samples

📈 Combined Statistics:
   Total Samples:             2,688
   Average Text Length:        18.5 words
   Min Text Length:            10 words
   Max Text Length:           512 words
```

### What These Mean

| Metric | Meaning |
|--------|---------|
| **Total Samples** | Number of sentences extracted from e-books |
| **Avg Text Length** | Average words per sentence |
| **Min Text Length** | Shortest sentence extracted |
| **Max Text Length** | Longest sentence extracted |

---

## Performance & Tips

### Processing Speed
```
Small e-book (100 pages):    1-3 seconds
Medium e-book (300 pages):   3-5 seconds
Large e-book (600 pages):    5-10 seconds
Multiple books (1000 pages): 10-15 seconds
```

### Best Practices

✅ **DO:**
- Place well-formatted EPUB files in `epub/` directory
- Use diverse e-books for better model training
- Mix EPUBs with AIML for conversational systems
- Refresh cache after adding new e-books
- Use domain-specific books for specialized knowledge

❌ **DON'T:**
- Use DRM-protected (locked) Kindle e-books
- Place non-EPUB files in `epub/` directory (ignored)
- Use corrupted or malformed EPUB files
- Expect perfect extraction from all e-books

### Quality Tips

| Issue | Solution |
|-------|----------|
| **Low sample count** | E-book may be short, add longer books |
| **Poor text extraction** | Try opening EPUB in reader (Calibre) first |
| **DRM error** | Remove DRM protection using tools like Calibre |
| **Missing chapters** | Some EPUBs have complex structures, try others |

---

## Troubleshooting

### Issue: EPUBs not being loaded

**Symptoms:**
```
[4/3] Loading EPUB files...
⚠ No EPUB files found in 'epub' directory
```

**Solution:**
- Check files are in `epub/` directory
- Verify files have `.epub` extension (lowercase)
- Ensure EPUBs aren't corrupted (open in Calibre first)
- Test with a known good EPUB file

### Issue: "ebooklib not installed"

**Error:**
```
⚠ ebooklib not installed. Install with: pip install ebooklib
```

**Solution:**
```bash
pip install ebooklib

# Or for project virtual environment:
.\envMyIAModelChat\Scripts\pip install ebooklib
```

### Issue: Error extracting chapter from EPUB

**Symptoms:**
```
⚠ Error extracting chapter from book.epub: ...
```

**Solution:**
- EPUB file may have complex formatting
- Try opening in Calibre and re-saving
- Remove any DRM protection
- Use a different EPUB file as test

### Issue: DRM-Protected Kindle E-Books

**Problem:** Amazon Kindle books (.azw, .azw3) won't load

**Solution:**
- Convert using Calibre (free tool; removes DRM)
- Export as EPUB format
- Place converted file in `epub/` directory
- Then use with MyIAModelChat

**Note:** Check local laws regarding DRM circumvention before removing protections.

---

## Sources for Free EPUB Files

### Public Domain Books (Free & Legal)
- **Project Gutenberg** (https://www.gutenberg.org/)
  - 70,000+ classic books in EPUB format
  - All in public domain
  - Completely free

- **Open Library** (https://openlibrary.org/)
  - Millions of books
  - Many available in EPUB format
  - Free borrowing and downloading

- **Standard Ebooks** (https://standardebooks.org/)
  - High-quality formatted public domain books
  - Beautiful EPUB formatting
  - Free to download

### Creative Commons Books
- **Smashwords** - Some free CC-licensed books
- **Librivox** - Audiobooks with text versions
- **Wattpad** - Community-uploaded books (check licensing)

### Academic & Research
- **arXiv** - Research papers (some in EPUB)
- **SciELO** - Open access research
- **JSTOR Daily** - Free academic articles

---

## Advanced Usage

### Custom EPUB Processing

You can modify the EPUB loading behavior in `data_preparer.py`:

```python
def _load_epub_data(self) -> Dataset:
    # Minimum sentence length (currently 10)
    if len(sentence) > 10:  # ← Modify this value
        
    # Split method (currently by periods)
    sentences = text.split('.')  # ← Can use other delimiters
```

Example: Load shorter sentences (for poetry)
```python
if len(sentence) > 5:  # Changed from 10 to 5
```

---

## Data Source Comparison

| Feature | AIML | HF | PDF | EPUB |
|---------|------|-----|-----|------|
| **Type** | Dialogue | Text corpus | Documents | E-books |
| **Best for** | Conversation | General text | Technical | Narrative |
| **Sample count** | ~40K | ~1K | ~100-1K | ~500-2K |
| **Format** | XML patterns | Pre-formatted | PDF text | Structured |
| **Setup** | Auto | Auto download | Manual | Manual |
| **Examples** | "How are you?" | Wiki text | Manuals | Novels |

---

## Workflow Recommendations

### For Narrative Models (Creative Writing)
```bash
python main.py --prepare-data --epub  # Use fiction e-books
python main.py --train --epub --epochs 15
```

### For Knowledge Systems
```bash
python main.py --prepare-data --hf --pdf --epub  # Articles + papers + books
python main.py --train --hf --pdf --epub --epochs 10
```

### For Hybrid Chatbots (Recommended)
```bash
python main.py --prepare-data --aiml --hf --pdf --epub  # All sources
python main.py --train --aiml --hf --pdf --epub --epochs 10
python main.py --chat
```

### For Domain-Specific Models
```bash
# Place domain-specific e-books (medical, legal, technical, etc.)
python main.py --prepare-data --epub
python main.py --train --epub --epochs 20
# Result: Expert model in that domain
```

---

## Summary

| Task | Command |
|------|---------|
| **Load only EPUBs** | `python main.py --prepare-data --epub` |
| **EPUBs + AIML** | `python main.py --prepare-data --aiml --epub` |
| **All sources** | `python main.py --prepare-data --aiml --hf --pdf --epub` |
| **Train with EPUBs** | `python main.py --train --aiml --epub --epochs 10` |
| **Refresh after changes** | `python main.py --prepare-data --epub --refresh-cache` |

---

## File Locations

- **EPUB directory:** `project_root/epub/`
- **Config file:** `project_root/main.py` (--epub flag)
- **Implementation:** `project_root/data_preparer.py` (_load_epub_data method)
- **Dependencies:** ebooklib in `requirements.txt`

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Verify ebooklib installation: `pip show ebooklib`
3. Test with a free Project Gutenberg EPUB first
4. View logs: `python main.py --prepare-data --epub` (detailed logging)

---

## Related Files

- **EPUB-QUICK-REFERENCE.txt** - Quick commands
- **EPUB-SUPPORT-IMPLEMENTATION.md** - Technical details
- **PDF-USAGE-GUIDE.md** - Similar guide for PDFs
- **data_preparer.py** - Implementation code

**Ready to train your model on e-books!** 📚
