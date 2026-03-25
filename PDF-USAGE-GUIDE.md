# PDF Data Loading Guide

## Overview

The MyIAModelChat now supports loading text data from PDF files. PDFs are automatically extracted, split into samples, and added to your training dataset.

---

## Quick Start

### 1. Add PDF Files

Place your PDF files in the `pdfs/` directory (created automatically):

```
project_root/
├── pdfs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
├── main.py
└── ... (other files)
```

### 2. Include PDFs in Dataset Preparation

```bash
python main.py --prepare-data --pdf
```

Or combine with AIML and Hugging Face:

```bash
python main.py --prepare-data --aiml --hf --pdf
```

### 3. Train with PDF Data

```bash
python main.py --train --aiml --hf --pdf --epochs 10
```

---

## Commands

### Load Only PDF Data
```bash
python main.py --prepare-data --pdf
```
**Result:** Extracts text from all PDF files in `pdfs/` directory

### Combine PDF with AIML
```bash
python main.py --prepare-data --aiml --pdf
```
**Result:** Combines dialogue patterns (AIML) with document text (PDF)

### Combine All Three Sources
```bash
python main.py --prepare-data --aiml --hf --pdf
```
**Result:** Uses AIML dialogue patterns + Wikipedia/HF datasets + PDF documents

### Train with PDF Data
```bash
python main.py --train --aiml --pdf --epochs 10
```
**Result:** Trains model using AIML and PDF data

### Train with All Sources
```bash
python main.py --train --aiml --hf --pdf --epochs 10
```
**Result:** Full training with all available data sources

### Refresh Cache After Adding PDFs
```bash
python main.py --prepare-data --aiml --pdf --refresh-cache
```
**Result:** Rebuilds cache including newly added PDF files

---

## How PDF Processing Works

### Text Extraction
1. **PDF Reading** - PyPDF2 reads each page
2. **Text Extraction** - Extracts raw text from all pages
3. **Sentence Splitting** - Splits by periods (.)
4. **Filtering** - Removes sentences shorter than 10 characters
5. **Dataset Creation** - Converts to training samples

### Data Format
Each extracted sentence becomes one training sample:
```python
{
    'input_ids': 'This is text extracted from a PDF file'
}
```

### Sample Count
- A 10-page PDF typically yields **100-500 samples** depending on content
- PDFs with dense text → more samples
- PDFs with sparse text → fewer samples

---

## Examples

### Example 1: Load Single PDF
```bash
# Place research_paper.pdf in pdfs/ directory
python main.py --prepare-data --pdf

# Output:
# [3/3] Loading PDF files...
#   Reading PDF: research_paper.pdf...
#   ✓ Loaded: research_paper.pdf (245 samples)
#   Total PDF files processed: 1
#   Total PDF samples: 245
```

### Example 2: Combine Multiple PDFs
```bash
# Place multiple PDFs in pdfs/ directory:
# - technical_documentation.pdf
# - user_guide.pdf  
# - knowledge_base.pdf

python main.py --prepare-data --pdf

# Output:
# [3/3] Loading PDF files...
#   Reading PDF: technical_documentation.pdf...
#   ✓ Loaded: technical_documentation.pdf (312 samples)
#   Reading PDF: user_guide.pdf...
#   ✓ Loaded: user_guide.pdf (187 samples)
#   Reading PDF: knowledge_base.pdf...
#   ✓ Loaded: knowledge_base.pdf (456 samples)
#   Total PDF files processed: 3
#   Total PDF samples: 955
```

### Example 3: Mixed Source Training
```bash
# PDFs in pdfs/ directory with AIML dialogue patterns
python main.py --prepare-data --aiml --pdf

# Then train:
python main.py --train --aiml --pdf --epochs 10

# Result: Model trained on both dialogue patterns and document text
```

### Example 4: Full Workflow with All Sources
```bash
# 1. Prepare with all data sources
python main.py --prepare-data --aiml --hf --pdf

# 2. View statistics
# Output shows:
#   📊 Data Sources:
#      AIML                     41,569 samples
#      HuggingFace              1,000 samples
#      PDF                        955 samples

# 3. Train
python main.py --train --aiml --hf --pdf --epochs 10

# 4. Chat
python main.py --chat
```

---

## File Formats

### Supported Formats
- **PDF (.pdf)** ✅ Fully supported via PyPDF2
- Text extraction from all standard PDF formats
- Handles both text-based and scanned PDFs*

*Note: Scanned PDFs (images) require OCR - not included in basic PyPDF2

### Unsupported Formats
- Word documents (.docx, .doc) - Not supported yet
- Images (.png, .jpg) - Not supported yet
- Excel (.xlsx) - Not supported yet

---

## PDF Directory Structure

The `pdfs/` directory is automatically created when you first use `--pdf` flag:

```
pdfs/
├── research_paper.pdf          # ✅ Will be loaded
├── documentation.pdf           # ✅ Will be loaded
├── meeting_notes.txt           # ❌ Ignored (not PDF)
├── image_file.png             # ❌ Ignored (not PDF)
└── previous_run_backup.pdf    # ✅ Will be loaded (all PDFs loaded)
```

**All .pdf files in `pdfs/` directory are automatically loaded and processed.**

---

## Statistics Output

When running `--prepare-data --pdf`, you'll see:

```
📊 Data Sources:
   PDF                            955 samples

📈 Combined Statistics:
   Total Samples:                 955
   Average Text Length:           15.3 words
   Min Text Length:               10 words
   Max Text Length:               245 words
```

### What These Mean

| Metric | Meaning |
|--------|---------|
| **Total Samples** | Number of sentences extracted from PDFs |
| **Avg Text Length** | Average words per sentence |
| **Min Text Length** | Shortest sentence extracted |
| **Max Text Length** | Longest sentence extracted |

---

## Performance & Tips

### Performance Impact
```
Single PDF (10 pages):      ~1-2 seconds
Multiple PDFs (50 pages):   ~3-5 seconds
Large PDFs (200+ pages):    ~10+ seconds
```

### Best Practices

✅ **DO:**
- Place well-formatted PDFs in `pdfs/` directory
- Use PDFs with consistent text formatting
- Mix PDF data with AIML for diverse training
- Refresh cache after adding new PDFs
- Use domain-specific PDFs for better results

❌ **DON'T:**
- Use scanned image PDFs without OCR
- Place non-PDF files in `pdfs/` directory (will be ignored)
- Use PDFs with corrupted pages
- Expect perfect text extraction from all PDFs

### Quality Tips

| Issue | Solution |
|-------|----------|
| **Low sample count** | Add more/longer PDFs |
| **Poor text quality** | Use well-formatted PDFs |
| **Duplicates found** | Remove duplicate PDFs |
| **Encoding issues** | Ensure PDFs are UTF-8 compatible |

---

## Troubleshooting

### Issue: PDFs not being loaded

**Symptoms:** 
```
[3/3] Loading PDF files...
⚠ No PDF files found in 'pdfs' directory
```

**Solution:**
- Check files are in `pdfs/` directory
- Verify files have `.pdf` extension (lowercase)
- Ensure PDFs aren't corrupted

### Issue: "PyPDF2 not installed"

**Error:**
```
⚠ PyPDF2 not installed. Install with: pip install PyPDF2
```

**Solution:**
```bash
pip install PyPDF2
# Or for the project virtual environment:
.\envMyIAModelChat\Scripts\pip install PyPDF2
```

### Issue: Error extracting page from PDF

**Symptoms:**
```
⚠ Error extracting page 0 from document.pdf: ...
```

**Solution:**
- PDF may be corrupted
- Try opening PDF in PDF reader (Adobe, etc.)
- Regenerate PDF if possible
- Use a different PDF file

### Issue: Very few samples extracted

**Symptoms:**
```
✓ Loaded: document.pdf (15 samples)
```

**Solution:**
- PDF may have very short text
- Increase minimum sentence length in code
- Add more/longer PDFs
- Consider using different PDFs

---

## Advanced Usage

### Combine PDF with Custom Processing

You can modify the PDF loading behavior in `data_preparer.py`:

```python
def _load_pdf_data(self) -> Dataset:
    # Minimum sentence length (currently 10)
    if len(sentence) > 10:  # ← Modify this value
        
    # Split method (currently by periods)
    sentences = text.split('.')  # ← Can use other delimiters
```

Example: Load shorter sentences
```python
if len(sentence) > 5:  # Changed from 10 to 5
```

---

## Data Source Comparison

| Feature | AIML | HF Datasets | PDF |
|---------|------|------------|-----|
| **Type** | Dialogue patterns | Text corpora | Documents |
| **Best for** | Conversational AI | General text | Domain-specific |
| **Sample count** | ~40K | ~1K-10K | ~100-1K per PDF |
| **Format** | XML patterns | Pre-formatted | Raw documents |
| **Setup** | Automatic | Automatic download | Manual placement |
| **Examples** | "How are you?" | Wikipedia text | Research papers |

---

## Workflow Recommendations

### For Dialogue Systems
```bash
python main.py --prepare-data --aiml --hf  # Best for conversational AI
```

### For Document-Based Models
```bash
python main.py --prepare-data --pdf --hf  # Facts + external knowledge
```

### For Hybrid Models (Recommended)
```bash
python main.py --prepare-data --aiml --hf --pdf  # All advantages combined
```

### Then Train
```bash
python main.py --train --aiml --hf --pdf --epochs 10
```

---

## Summary

| Task | Command |
|------|---------|
| **Load only PDFs** | `python main.py --prepare-data --pdf` |
| **PDFs + AIML** | `python main.py --prepare-data --aiml --pdf` |
| **All sources** | `python main.py --prepare-data --aiml --hf --pdf` |
| **Train with PDFs** | `python main.py --train --aiml --pdf --epochs 10` |
| **Refresh after changes** | `python main.py --prepare-data --pdf --refresh-cache` |
| **Check statistics** | `python main.py --prepare-data --aiml --pdf` |

---

## File Locations

- **PDFs directory:** `project_root/pdfs/`
- **Config file:** `project_root/main.py` (--pdf flag)
- **Implementation:** `project_root/data_preparer.py` (_load_pdf_data method)
- **Requirements:** PyPDF2 in `requirements.txt`

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Verify PyPDF2 installation: `pip show PyPDF2`
3. Test with a simple PDF first
4. View logs: `python main.py --prepare-data --pdf` (shows detailed logging)

**Ready to add PDF content to your model!** 📚
