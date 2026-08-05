================================================================================
                   DATASET PREPARATION - QUICK REFERENCE
================================================================================

NEW FEATURE: `--prepare-data` flag for dataset validation and statistics

================================================================================
                            QUICK COMMANDS
================================================================================

Validate datasets before training:
    python main.py --prepare-data --aiml --hf

Validate only AIML data:
    python main.py --prepare-data --aiml

Validate only Hugging Face data:
    python main.py --prepare-data --hf

================================================================================
                            WHY USE IT?
================================================================================

✅ Catch data problems BEFORE training (saves hours)
✅ See dataset statistics (total samples, text length, source breakdown)
✅ Validate data structure and identify null values
✅ Understand dataset composition
✅ Fast execution (seconds to minutes, no training overhead)
✅ Part of best-practice ML workflow

================================================================================
                        RECOMMENDED WORKFLOW
================================================================================

1. PREPARE DATA (validate + statistics):
    python main.py --prepare-data --aiml --hf

2. BUILD VOCABULARY (no training):
    python main.py --train --onlytokenize --aiml --hf

3. TRAIN MODEL (with confidence):
    python main.py --train --aiml --hf --epochs 20 --num_cores 8

4. TEST CHAT:
    python main.py --chat

================================================================================
                            OUTPUT EXAMPLE
================================================================================

When you run: python main.py --prepare-data --aiml --hf

You'll see:

  [1/7] Loading AIML data...
    Parsed ai.aiml: 375 samples
    Parsed alice.aiml: 8200 samples
    ... loaded 5 AIML files

  [2/7] Loading Hugging Face datasets...
    ✓ wikitext (1000 samples)

  [6/7] Combining datasets...
    Total: 10,310 samples

  [6.42/7] Applying noise filter...
  [6.43/7] Applying quality filter...
  [6.44/7] Applying cross-source deduplication...

  DATASET PREPARATION SUMMARY
  ================================================================================
  Data Sources:
     AIML                     10,310 samples

  Combined Statistics:
     Total Samples:           10,310
     Average Text Length:        9.8 words
     Min Text Length:              2 words
     Max Text Length:            202 words

  [OK] DATASET PREPARATION COMPLETED SUCCESSFULLY

================================================================================
                        WHAT IT VALIDATES
================================================================================

✓ AIML files can be loaded
✓ Hugging Face datasets accessible (requires internet)
✓ Data structure has required 'input_ids' column
✓ No critical null/empty values
✓ Dataset can be combined successfully

================================================================================
                        DATA SOURCES SUPPORTED
================================================================================

AIML Data:
  - Location: datasets_source/aiml/ directory
  - Format: .aiml XML files
  - Contains: ~60 dialogue pattern files

Hugging Face Datasets:
  - wikitext (Wikipedia text - default)
  - Easily extensible to other HF datasets
  - Requires internet connection

Local Data:
  - Can be added by extending data_preparer.py
  - Contact developers for custom data source integration

================================================================================
                        TROUBLESHOOTING
================================================================================

Error: "Directory not found: datasets_source/aiml/"
  → Check working directory is project root
  → Verify datasets_source/aiml/ folder exists

Error: "Error loading wikitext"
  → Check internet connection
  → May need to re-run (network timeout)
  → Run AIML-only if offline: python main.py --prepare-data --aiml

Error: "No .aiml files found in datasets_source/aiml/"
  → Place .aiml XML files in datasets_source/aiml/
  → Or run with --aiml flag to include AIML data

Success but "very few samples"?
  → AIML data may not be preprocessed
  → Run: python main.py --train --onlytokenize --aiml
  → Then re-run --prepare-data

================================================================================
                        NEXT STEPS AFTER PREP
================================================================================

If preparation succeeds:
  1. Review statistics in the terminal output
  2. Check sample count is reasonable (>100 samples recommended)
  3. Verify both data sources loaded (if using --aiml --hf)
  
Then proceed with training:
  python main.py --train --aiml --hf --epochs 20

If preparation fails:
  1. Check error message carefully
  2. Verify AIML files exist and are readable
  3. Test connectivity if using --hf
  4. Try --aiml only first
  5. Consult DEVELOPMENT_GUIDE.md for more help

================================================================================
                        STATISTICS INTERPRETATION
================================================================================

Total Samples:
  - More samples = longer training, but better model coverage
  - Minimum recommended: 500+
  - Typical range: 1,000-100,000

Average Text Length:
  - Higher = more words per sample
  - Affects GPU memory requirements
  - Typical range: 20-100 words

Min/Max Text Length:
  - Identifies extremes in dataset
  - Very short (<5 words) samples may not be useful
  - Very long (>1000 words) samples may need truncation

Source Breakdown:
  - Shows proportion from each data source
  - Helps understand data diversity
  - Unbalanced data may need resampling

================================================================================

For detailed information, see:
  - USAGE.md (full command documentation)
  - TRAINING_GUIDE.md (training and caching details)
  - README.md (project overview)

================================================================================
