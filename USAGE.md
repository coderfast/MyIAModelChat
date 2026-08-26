================================================================================
                    MyIAModelChat - USAGE GUIDE
================================================================================

This file describes all command line parameters for training and inference 
with the MyIAModelChat model. Supports AIML, PDF, EPUB, and Hugging Face datasets.

================================================================================
                            QUICK START
================================================================================

TRAIN MODE (build/train the model):
    python main.py --train --epochs 30

CHAT MODE (run interactive chat):
    python main.py --chat --cpu --num_cores 4 --num_threads 4

PREPARE DATASETS (validate data and create cache):
    python main.py --prepare-data --aiml --pdf --epub

REFRESH CACHE (rebuild cached dataset):
    python main.py --prepare-data --aiml --pdf --epub --refresh-cache

TOKENIZE ONLY (prepare vocabulary without training):
    python main.py --train --onlytokenize

================================================================================
                        COMMAND LINE ARGUMENTS
================================================================================

Primary Mode Arguments (choose one):
================================================================================

--train
    Description: Run training mode (uses cached dataset from --prepare-data)
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --train --epochs 30
    Notes:
        - Uses cached dataset (must run --prepare-data first)
        - Trains the GPT-2 Transformer model
        - Saves checkpoints to checkpoints/ directory
        - Can be interrupted with ESC key

--chat
    Description: Run inference/chat mode (interactive dialogue)
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --chat
    Notes:
        - Loads model from checkpoints/chat_model.pth
        - Loads tokenizer from checkpoints/tokenizer_vocab.json
        - Provides interactive chat interface with dynamic n-gram penalization
        - Uses intent classification and sentiment analysis
        - Requires trained model and tokenizer
        - Supports min-length enforcement and repetition prevention

--prepare-data
    Description: Prepare and validate datasets only (no training)
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --pdf --epub
    Notes:
        - Loads AIML, PDF, EPUB, and/or Hugging Face datasets
        - Validates data structure and content
        - Shows dataset statistics (samples, text length, sources)
        - Automatically saves prepared dataset to cache (dataset_cache/)
        - Does NOT train the model
        - Useful before starting training to verify data quality
        - Helps identify data issues early
        - Supports schema alignment for dataset concatenation

--bpe-vocab-size <N>
    Description: Vocabulary size for SentencePiece BPE tokenizer (triggers BPE training automatically)
    Type: Integer
    Default: 8000
    Example: python main.py --prepare-data --aiml --bpe-vocab-size 8000
    Notes:
        - Requires `sentencepiece` package to be installed (`pip install sentencepiece`)
        - When used, prepared cache will include `token_ids` per sample and a `sentencepiece.model` in `dataset_cache/`
        - Omit this flag to skip BPE and use raw text
        - BPE automatically registers 5 special tokens as user_defined_symbols:
          <thinking>, </thinking>, <|context|>, <|answer|>, <|thinking|>
        - These tokens are never fragmented (always whole tokens)
        - Recommended vocab size: 8000 (default), increase for larger datasets

================================================================================

Text Processing Arguments (advanced data preparation):
================================================================================

--enable-chunking
    Description: Enable text chunking by tokens for PDF/EPUB documents
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --pdf --enable-chunking
    Notes:
        - Splits long texts into overlapping chunks
        - Useful for models with fixed context windows
        - Preserves context between chunks with overlap
        - Only applies to PDF and EPUB data sources

--chunk-max-tokens <N>
    Description: Maximum tokens per chunk when chunking is enabled
    Type: Integer
    Default: 512
    Example: python main.py --prepare-data --pdf --enable-chunking --chunk-max-tokens 256
    Notes:
        - Smaller values = more chunks, less context per chunk
        - Larger values = fewer chunks, more context per chunk
        - Recommended: 256-1024 depending on model

--chunk-overlap <N>
    Description: Number of overlapping tokens between chunks
    Type: Integer
    Default: 50
    Example: python main.py --prepare-data --pdf --enable-chunking --chunk-overlap 100
    Notes:
        - More overlap = better context continuity
        - Less overlap = fewer duplicate tokens
        - Recommended: 10-20% of chunk-max-tokens

--enable-dedup
    Description: Enable deduplication of similar texts using MinHash LSH
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --pdf --enable-dedup
    Notes:
        - Removes duplicate and near-duplicate texts
        - Uses MinHash LSH for efficient similarity detection
        - Improves training quality by reducing redundancy
        - Falls back to exact matching if datasketch not installed

--dedup-threshold <float>
    Description: Similarity threshold for deduplication (0-1)
    Type: Float
    Default: 0.8
    Range: 0.0 to 1.0
    Example: python main.py --prepare-data --pdf --enable-dedup --dedup-threshold 0.9
    Notes:
        - Higher threshold = less aggressive deduplication
        - Lower threshold = more aggressive deduplication
        - 0.8 = remove texts that are 80%+ similar
        - 0.9 = remove texts that are 90%+ similar

--enable-quality-filter
    Description: Enable quality filtering of texts
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --pdf --enable-quality-filter
    Notes:
        - Filters out low-quality texts
        - Checks word count, alpha ratio, spam detection
        - Removes texts with too many numbers/symbols
        - Removes texts with unreasonable sentence structure

--min-words <N>
    Description: Minimum words per text for quality filter
    Type: Integer
    Default: 5
    Example: python main.py --prepare-data --pdf --enable-quality-filter --min-words 10
    Notes:
        - Texts with fewer words are filtered out
        - Higher value = stricter filtering
        - Recommended: 3-10 depending on use case

--max-words <N>
    Description: Maximum words per text for quality filter
    Type: Integer
    Default: 1000
    Example: python main.py --prepare-data --pdf --enable-quality-filter --max-words 500
    Notes:
        - Texts with more words are filtered out
        - Lower value = stricter filtering
        - Recommended: 500-2000 depending on use case

--preserve-metadata
    Description: Preserve document metadata (title, author, etc.) in dataset
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --pdf --epub --preserve-metadata
    Notes:
        - Extracts and preserves document metadata
        - PDF: title, author, page count
        - EPUB: title, author, language, chapter count
        - Metadata stored in 'metadata' field of each sample

--enable-lang-filter
    Description: Enable language filtering of texts
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --pdf --enable-lang-filter
    Notes:
        - Filters texts by detected language
        - Uses langdetect library (if available)
        - Falls back to heuristic-based detection
        - Keeps texts in allowed languages + unknown

--allowed-languages <lang1> <lang2> ...
    Description: Allowed language codes for language filtering
    Type: List of strings
    Default: es en (Spanish and English)
    Example: python main.py --prepare-data --pdf --enable-lang-filter --allowed-languages es en fr
    Notes:
        - Use ISO 639-1 language codes
        - Common codes: es (Spanish), en (English), fr (French), de (German)
        - Unknown language texts are always kept
        - Requires --enable-lang-filter to work

Contamination Filtering Arguments:
================================================================================

--filter-noise
    Description: Apply noise filter (URLs, emails, code, boilerplate)
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --filter-noise
    Notes:
        - Detects and removes noisy texts (navigation, social media, etc.)
        - Uses 12+ heuristics for noise detection

--filter-contamination
    Description: Apply quality-based contamination filter
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --filter-contamination
    Notes:
        - Classifies texts as good/fixable/discardable
        - Removes low-quality contaminated samples

--filter-dedup
    Description: Apply cross-source deduplication
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --filter-dedup
    Notes:
        - Removes duplicate texts across all data sources
        - Uses exact and near-duplicate detection

--dedup-mode <mode>
    Description: Deduplication mode
    Type: String (exact, near, all)
    Default: all
    Example: python main.py --prepare-data --aiml --filter-dedup --dedup-mode near

--filter-balance
    Description: Apply source balance control
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --filter-balance
    Notes:
        - Prevents one source from dominating the dataset
        - Limits each source to max-source-ratio of total

--filter-leakage
    Description: Apply train/test leakage detection
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --filter-leakage
    Notes:
        - Detects overlapping texts between train/test splits
        - Uses similarity threshold to find near-duplicates

--audit-report
    Description: Generate contamination audit report
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --audit-report
    Notes:
        - Saves detailed audit report to file
        - Shows contamination statistics per source

--refresh-cache
    Description: Clear old cache and rebuild from fresh data
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml --hf --refresh-cache
    When to use:
        - After updating AIML files
        - After adding/modifying HF dataset selection
        - When data sources change
    Notes:
        - Removes old dataset_cache/ directory
        - Reloads all data from sources
        - Automatically saves new cache

--clear-cache
    Description: Remove cached datasets and exit (no other operations)
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --clear-cache
    Notes:
        - Removes entire dataset_cache/ directory
        - Exits immediately (ignores other flags)
        - Useful for disk space cleanup
        - Next --prepare-data run will rebuild cache

================================================================================

Dataset Caching Arguments (improves speed on repeated runs):
================================================================================

Note: Dataset caching significantly improves performance for repeated runs.
First run prepares and caches data, subsequent runs load from cache.

Cache Location: dataset_cache/ directory in project root
Cache Size: Typically 100MB - 1GB depending on data sources

Typical Cache Workflow:
    1. python main.py --prepare-data --aiml --hf     (5-30 seconds, creates cache)
    2. python main.py --train --epochs 30             (uses cached data)
    3. python main.py --prepare-data --aiml --hf --refresh-cache  (5-30 seconds, updates)

Quick verification (BPE + cached training smoke test):
    # Prepare data with BPE and build tokenized cache
    python main.py --prepare-data --aiml --hf --bpe-vocab-size 2000 --refresh-cache

    # Run a very short training using the cached token_ids to verify integration
    python main.py --train --epochs 1

================================================================================

Data Source Arguments (use with --train or --prepare-data):
================================================================================

Data Source Arguments (for --prepare-data only):
================================================================================

--aiml
    Description: Include AIML files in dataset
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --aiml
    Notes:
        - Loads AIML files from datasets_source/aiml/ directory
        - Contains ~60 dialogue pattern files
        - Uses XML-based patterns for training
        - Recommended for dialogue variety

--pdf
    Description: Include PDF documents in dataset
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --pdf
    Notes:
        - Loads PDF files from datasets_source/pdf/ directory
        - Uses pypdf for page-by-page text extraction
        - Removes repeated headers/footers and standalone page numbers
        - Samples are whole paragraphs for semantic coherence
        - Place PDF files in datasets_source/pdf/ folder

--epub
    Description: Include EPUB e-books in dataset
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --epub
    Notes:
        - Loads EPUB files from datasets_source/epub/ directory
        - Uses ebooklib for native chapter extraction
        - Structured paragraph extraction with BeautifulSoup
        - Whole chapter as sample when it fits the context window; otherwise split into whole paragraphs
        - Removes repeated headers/footers and standalone page numbers
        - Supports metadata and structure
        - Place EPUB files in datasets_source/epub/ folder

--hf
    Description: Include Hugging Face datasets
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --hf
    Notes:
        - Loads public datasets from Hugging Face
        - Examples: wikitext, bookcorpus, common_voice, opus_100
        - Requires internet connection
        - Larger vocabulary and more examples

--web
    Description: Include web documentation data
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --web
    Notes:
        - Scrapes web pages for training data
        - Reads URLs from --web-url flag or datasets_source/web/urls.txt
        - Requires internet connection

--csv
    Description: Include CSV data files
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --prepare-data --csv
    Notes:
        - Loads CSV files from datasets_source/csv/ directory
        - Expects text column for training data

    Combine multiple data sources:
        python main.py --prepare-data --aiml --pdf --epub --hf

================================================================================

Performance & Hardware Arguments:
================================================================================

--cpu
    Description: Force CPU usage even if GPU is available
    Type: Boolean flag (no value needed)
    Default: False (auto-detect GPU)
    Example: python main.py --train --cpu
    Notes:
        - Useful for testing or when GPU memory is insufficient
        - Prevents CUDA-related errors
        - Slower than GPU but more stable

--gpu [N,N,...]
    Description: Use GPU for training/inference
    Type: Optional string (no value=auto GPU, or comma-separated indices)
    Default: auto-detect best available GPU
    Example: python main.py --train --gpu 0
    Example: python main.py --train --gpu 0,1  (DDP multi-GPU)
    Notes:
        - Without value: uses default CUDA GPU
        - With indices: uses specific GPUs (e.g. 0,1,2)
        - Multiple GPUs automatically use DistributedDataParallel

--vulkan
    Description: Force Vulkan backend (if available)
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --train --vulkan
    Notes:
        - Requires PyTorch with Vulkan support
        - Experimental: not all operations supported

--cpu+gpu N,N,...
    Description: CPU+GPU hybrid mode (split model layers by VRAM)
    Type: String (GPU indices)
    Example: python main.py --train --cpu+gpu 0
    Notes:
        - Loads as many layers as GPU VRAM allows
        - Remaining layers stay on CPU
        - Useful when model doesn't fully fit in GPU memory

--gpu-enum
    Description: Enumerate available GPUs and exit
    Type: Boolean flag (no value needed)
    Example: python main.py --gpu-enum
    Notes:
        - Shows GPU name, VRAM, compute capability
        - Shows AI recommendation level for each GPU

--model-info NAME
    Description: Show detailed layer info for a specific model
    Type: String (model name)
    Example: python main.py --model-info chat_model
    Notes:
        - Shows all layer names, shapes, parameter counts
        - Shows total model size and parameter count

--num_cores <N>
    Description: Number of CPU cores for multiprocessing
    Type: Integer
    Default: 4
    Range: 1 to CPU count on your machine
    Example: python main.py --train --num_cores 8
    Notes:
        - Used in DataLoader for parallel data loading
        - Use machine CPU count for optimal speed
        - Check CPU count: python -c "import os; print(os.cpu_count())"
        - Recommended: Set to 50-75% of available cores

--num_threads <N>
    Description: Number of threads per worker
    Type: Integer
    Default: 4
    Range: 1 to 16 (typically)
    Example: python main.py --train --num_threads 8
    Notes:
        - Controls OpenMP and MKL threading
        - Affects BLAS operations efficiency
        - Usually num_threads = num_cores or slightly less

================================================================================

Training Arguments:
================================================================================

--epochs <N>
    Description: Number of training epochs
    Type: Integer
    Default: 30
    Example: python main.py --train --epochs 50
    Notes:
        - One epoch = one pass through entire dataset
        - Typical range: 10-100 epochs for good convergence
        - Higher epochs = more training time but better quality
        - Early stopping via ESC key (press ESC to stop training)
        - Model saves best checkpoint after each epoch

--onlytokenize
    Description: Build vocabulary only, skip training
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --train --onlytokenize --aiml
    Notes:
        - Useful for vocabulary inspection
        - Saves tokenizer.pth without training model
        - Fast operation (seconds to minutes)
        - Recommended before long training runs

================================================================================
                        THINKING GENERATION
================================================================================

The model supports chain-of-thought reasoning using special tokens:
- Mode tokens: <|thinking|>, <|context|>, <|answer|>
- Content tags: <thinking>, </thinking>

Sample formats:
  THINKING: <|thinking|>question<thinking>reasoning</thinking><|answer|>answer
  CONTEXT:  <|context|>question<|answer|>answer

================================================================================

Thinking Arguments:
================================================================================

--thinking-mode <mode>
    Description: Generate thinking/reasoning data for each sample
    Type: String (nlp, ollama)
    Default: none (no thinking generation)
    Example: python main.py --prepare-data --aiml --hf --thinking-mode nlp
    Notes:
        - nlp: Uses ThinkingEngine (NLP-based, no external dependencies)
        - ollama: Uses Ollama LLM teacher (requires Ollama running locally)
        - Creates TWO samples per entry (CONTEXT and THINKING)
        - Thinking tokens are registered in BPE automatically
        - Requires --bpe-vocab-size to be specified

--show-thinking
    Description: Show thinking/reasoning in chat output
    Type: Boolean flag (no value needed)
    Default: False
    Example: python main.py --chat --show-thinking
    Notes:
        - Without flag: only shows final answer
        - With flag: shows [thinking] section before answer
        - Thinking is hidden by default to keep responses clean

--thinking-model <model>
    Description: Ollama model to use for thinking generation (only with --thinking-mode ollama)
    Type: String
    Default: llama3.2
    Example: python main.py --prepare-data --aiml --thinking-mode ollama --thinking-model llama3.2
    Notes:
        - Requires Ollama running locally on localhost:11434
        - Higher quality thinking but slower
        - nlp mode is recommended for most cases

================================================================================

Training with Thinking:
================================================================================

When training with thinking data, the model learns:
1. When to generate <thinking>...</thinking> (reasoning)
2. When to generate <|answer|> (direct response)
3. Loss weighting: reasoning tokens get 0.5 weight, answer tokens get 1.0

Expected metrics during training:
  thinking_token_accuracy: 0.8+ (delimiter accuracy)
  thinking_coverage: 0.6+ (reasoning vs total tokens)
  response_token_accuracy: 0.7+ (answer accuracy)

================================================================================
                            USAGE EXAMPLES
================================================================================

TRAINING EXAMPLES:
================================================================================

1. Basic training with default settings:
    python main.py --train

2. Train for 30 epochs:
    python main.py --train --epochs 30

3. Train with optimized performance (assuming 16 CPU cores):
    python main.py --train --epochs 30 --num_cores 12 --num_threads 6

4. Just build vocabulary without training:
    python main.py --train --onlytokenize

5. Extensive training with maximum cores and CPU-only:
    python main.py --train --epochs 100 --num_cores 16 --num_threads 8 --cpu

6. Train on specific GPU:
    python main.py --train --gpu 0 --epochs 30

7. Train on multiple GPUs (DDP):
    python main.py --train --gpu 0,1 --epochs 30

8. Train CPU+GPU hybrid:
    python main.py --train --cpu+gpu 0 --epochs 30

9. Prepare data with thinking (NLP-based, no Ollama):
    python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --refresh-cache

10. Prepare data with thinking (Ollama teacher):
    python main.py --prepare-data --aiml --hf --thinking-mode ollama --bpe-vocab-size 8000 --refresh-cache

11. Prepare data with thinking + multilingual support:
    python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --allowed-languages es en fr de --refresh-cache

12. Train with thinking data:
    python main.py --train --epochs 30

13. Chat with thinking visible:
    python main.py --chat --show-thinking

================================================================================

INFERENCE/CHAT EXAMPLES:
================================================================================

1. Start interactive chat (default settings):
    python main.py --chat

2. Chat with specific CPU configuration:
    python main.py --chat --num_cores 8 --num_threads 4

3. Chat with GPU (if available):
    python main.py --chat
    (Automatically detects and uses GPU if available)

================================================================================

DATA PREPARATION & CACHING EXAMPLES:
================================================================================

Data preparation validates datasets and shows statistics before training.
Dataset caching enables fast repeated runs without reloading source data.

1. Prepare AIML data only:
    python main.py --prepare-data --aiml

2. Prepare PDF documents only:
    python main.py --prepare-data --pdf

3. Prepare EPUB e-books only:
    python main.py --prepare-data --epub

4. Prepare multiple sources (recommended):
    python main.py --prepare-data --aiml --pdf --epub

5. Load from cache on second run (automatic):
    python main.py --train --epochs 10

6. Refresh cache after adding new documents:
    python main.py --prepare-data --aiml --pdf --epub --refresh-cache

7. Clear cache to free disk space:
    python main.py --clear-cache

8. Prepare data and build a BPE-tokenized cache:
    python main.py --prepare-data --aiml --pdf --epub --bpe-vocab-size 8000


9. Complete workflow (prepare -> train -> chat):
    python main.py --prepare-data --aiml --pdf --epub
    python main.py --train --epochs 30
    python main.py --chat

10. Workflow for repeated experiments (cache is automatic):
    python main.py --prepare-data --aiml --pdf --epub          (first time: ~30 seconds)
    python main.py --train --epochs 10
    python main.py --train --epochs 20
    python main.py --train --epochs 30

================================================================================

ADVANCED DATA PROCESSING EXAMPLES:
================================================================================

10. Prepare PDF with chunking (for long documents):
    python main.py --prepare-data --pdf --enable-chunking --chunk-max-tokens 256

11. Prepare data with deduplication (remove duplicates):
    python main.py --prepare-data --aiml --pdf --enable-dedup --dedup-threshold 0.9

12. Prepare data with quality filtering:
    python main.py --prepare-data --pdf --epub --enable-quality-filter --min-words 10

13. Prepare data with language filtering (Spanish only):
    python main.py --prepare-data --pdf --enable-lang-filter --allowed-languages es

14. Prepare data with thinking + BPE (recommended workflow):
    python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --refresh-cache
    python main.py --train --epochs 30
    python main.py --chat --show-thinking

15. Prepare data without thinking (context only):
    python main.py --prepare-data --aiml --pdf --epub
    python main.py --train --epochs 30
    python main.py --chat
    python main.py --prepare-data --aiml --pdf --epub \
        --enable-chunking \
        --enable-dedup \
        --enable-quality-filter \
        --preserve-metadata \
        --enable-lang-filter \
        --allowed-languages es en

15. Prepare PDF with metadata preservation:
    python main.py --prepare-data --pdf --preserve-metadata

16. Prepare EPUB with chunking and quality filter:
    python main.py --prepare-data --epub \
        --enable-chunking \
        --chunk-max-tokens 512 \
        --enable-quality-filter \
        --min-words 5 \
        --max-words 1000

================================================================================

BATCH PROCESSING EXAMPLES:
================================================================================

Run multiple training sessions with data preparation:
    python main.py --prepare-data --aiml --hf
    python main.py --train --epochs 5
    python main.py --train --epochs 10

Train, then switch to chat:
    python main.py --train --epochs 5
    python main.py --chat

Complete workflow (prepare -> train -> chat):
    python main.py --prepare-data --aiml --hf
    python main.py --train --epochs 20 --num_cores 8
    python main.py --chat

Workflow with cached data (fast iteration):
    python main.py --prepare-data --aiml --hf          (first run, creates cache)
    python main.py --train --epochs 5
    python main.py --train --epochs 10
    python main.py --train --epochs 15

================================================================================
                        EXPECTED FILE OUTPUTS
================================================================================

Training Mode Creates (in checkpoints/ directory):
    - chat_model.pth              Latest model weights (overwritten every epoch)
    - chat_model_epoch_N_*.pth    Best model checkpoints (only when loss improves)
    - chat_model_metrics.csv      Per-epoch training metrics
    - chat_model_metrics_report.html  HTML report with Chart.js graphs
    - tokenizer_vocab.json        Bilingual tokenizer with vocabulary and mappings

Data Preparation Creates (in dataset_cache/ directory):
    - dataset_cache/
        ├── prepared_dataset/      Prepared and validated datasets
        ├── dataset_stats.pkl      Statistics (samples, lengths, sources)
        └── cache_metadata.pkl     Cache metadata and creation time

These are required for chat mode to work properly.
Dataset cache can be safely deleted with --clear-cache or manually.

================================================================================
                        TROUBLESHOOTING
================================================================================

Error: "No such file or directory: checkpoints/tokenizer_vocab.json"
    → Run data preparation first: python main.py --prepare-data --aiml
    → Then train: python main.py --train

Error: "CUDA out of memory"
    → Use --cpu flag
    → Use --cpu+gpu 0 to split layers across CPU and GPU
    → Reduce num_cores: python main.py --train --num-cores 2 --cpu

Error: Directory "datasets_source/aiml/" not found
    → Current directory must be project root
    → Files should be in: G:\PROJECTS\MyIAModelChat\datasets_source\aiml\

Error: PDF/EPUB loading fails
    → Install dependencies: pip install pypdf ebooklib
    → Check files exist in datasets_source/pdf/ or datasets_source/epub/ directories
    → Verify file formats are valid

Training too slow:
    → Increase num_cores: --num_cores 16
    → Use GPU if available (remove --cpu)
    → Use --gpu 0 for GPU acceleration
    → Reduce epochs for testing: --epochs 5

Training stuck or unresponsive:
    → Press ESC key to stop gracefully
    → Go back to terminal and restart

Chat produces gibberish:
    → Ensure model is trained with sufficient epochs (30+)
    → Check tokenizer matches training data
    → Try different temperature/top_p values in dialogmanager.py

================================================================================
                        ADVANCED CONFIGURATION
================================================================================

Modifying Model Architecture:
    Edit in chatmodel.py:
        embed_size: embedding dimension (default 256)
        num_layers: Transformer layers (default 2)
    Then retrain: python main.py --train --epochs 30

Changing Tokenizer Settings:
    Edit --bpe-vocab-size flag (default: 8000):
        python main.py --prepare-data --aiml --bpe-vocab-size 16000
    Then retrain: python main.py --train --epochs 30

Using Different Data Directories:
    Place files in:
        datasets_source/aiml/ for AIML files
        datasets_source/pdf/ for PDF documents
        datasets_source/epub/ for EPUB e-books
    Then run: python main.py --prepare-data --aiml --pdf --epub

Adjusting Generation Parameters:
    Edit in dialogmanager.py __init__:
        top_k: sampling parameter (default 12)
        top_p: nucleus sampling (default 0.8)
        temperature: randomness (default 0.65)
        min_length: minimum response length (default 5)
        no_repeat_ngram_size: repetition prevention (default 3)

================================================================================
                        SYSTEM REQUIREMENTS
================================================================================

Minimum:
    - Python 3.8+
    - 4 CPU cores
    - 8 GB RAM
    - 5 GB disk space

Recommended:
    - Python 3.10+
    - 8+ CPU cores
    - 16 GB RAM
    - 10 GB disk space

Dependencies:
    - PyTorch 2.0+
    - Transformers (Hugging Face)
    - pypdf (for PDF support)
    - ebooklib (for EPUB support)
    - datasets (for Hugging Face integration)
    - NVIDIA GPU with CUDA support

================================================================================
                        ENVIRONMENT SETUP
================================================================================

Activate virtual environment:
    Windows (PowerShell):
        .\envMyIAModelChat\Scripts\Activate.ps1
    
    Windows (CMD):
        envMyIAModelChat\Scripts\activate.bat
    
    Linux/Mac:
        source envMyIAModelChat/bin/activate

Install dependencies:
    pip install -r requirements.txt

Check GPU availability:
    python -c "import torch; print(torch.cuda.is_available())"

================================================================================
                        TYPICAL WORKFLOW
================================================================================

1. Prepare Environment:
    cd G:\PROJECTS\MyIAModelChat
    .\envMyIAModelChat\Scripts\Activate.ps1

2. Prepare and Validate Datasets (recommended):
    python main.py --prepare-data --aiml --hf
    (Review statistics and check for data issues)
    (Cache is automatically created)

3. Build Vocabulary (first time):
    python main.py --train --onlytokenize --aiml --hf

4. Train Model (uses cached data from step 2):
    python main.py --train --epochs 20 --num_cores 8

5. Test Chat:
    python main.py --chat

6. Iterate (cache is automatic):
    python main.py --train --epochs 30 --num_cores 8
    python main.py --chat
    
7. If source data changes:
    python main.py --prepare-data --aiml --hf --refresh-cache
    (Re-runs data loading and updates cache)

8. To clean up disk space:
    python main.py --clear-cache
    - Retrain: python main.py --train --epochs 10
    - Monitor loss convergence and stop early if plateau reached

================================================================================
                        MONITORING TRAINING
================================================================================

Output Information:
    - Epoch number and loss value (per epoch)
    - Device type: CUDA, MPS, or CPU
    - Number of CPU cores and threads
    - Loaded files and datasets

Optimization Tips:
    - Loss should decrease each epoch
    - If loss plateaus, reduce learning rate or stop
    - Press ESC to stop early if converged
    - Check GPU memory: nvidia-smi (if using CUDA)

================================================================================
                        DATA PREPARATION OUTPUT
================================================================================

When running `python main.py --prepare-data`, you'll see:

1. Data Loading Phase:
    - Lists each AIML file loaded with sample count
    - Shows Hugging Face datasets being loaded
    - Displays any warnings about missing files

2. Dataset Combination:
    - Shows samples from each source
    - Total combined sample count

3. Statistics Summary:
    - Data Sources: Breakdown by source (AIML, HuggingFace)
    - Total Samples: Complete dataset size
    - Average Text Length: Average words per sample
    - Min/Max Text Length: Range of sample lengths

4. Validation Results:
    - Column structure check
    - Null/empty value warnings
    - Data quality indicators

Use these statistics to:
    - Verify data was loaded correctly
    - Detect data quality issues before training
    - Estimate training time (more samples = longer training)
    - Identify imbalanced datasets

================================================================================
                        KNOWN LIMITATIONS
================================================================================

- Device detection is automatic (prioritizes CUDA > MPS > CPU)
- Vocabulary size currently fixed at initialization
- Maximum batch size depends on available memory
- No distributed multi-GPU training
- Chat mode uses single-turn responses (queue-based context)

================================================================================

For more details, see: README.md, MODEL_ARCHITECTURE.md

================================================================================
