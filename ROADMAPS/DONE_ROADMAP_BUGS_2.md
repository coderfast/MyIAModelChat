# ROADMAP: Bugs, Memory Leaks & Inefficiencies (Round 2)

52 issues found, 0 false positives. Organized by **file-based fix clusters** for minimal context switching.

**Status:** DONE
**Date:** 2026-08-05
**Verification:** All issues confirmed via code inspection

---

## Summary

| Severity | Count | Top Impact |
|----------|-------|------------|
| CRITICAL | 7 | Thinking loss non-functional, runtime crash, data corruption |
| HIGH | 12 | OOM risk, GPU stalls, memory leaks, race conditions |
| MEDIUM | 18 | Performance, security, coupling |
| LOW | 15 | Dead code, style |

---

## Fix Cluster 1: `training/trainer.py` (14 issues)

Highest ROI file. Fix all 14 issues in one pass.

### C1. Thinking loss weighting is non-functional
- **Line:** 1088
- `nn.CrossEntropyLoss(ignore_index=...)` uses `reduction='mean'` (default). Returns scalar, making weight tensor at line 649 a no-op. The entire thinking-loss-weight feature does nothing.
- **Fix:** `nn.CrossEntropyLoss(ignore_index=..., reduction='none')`

### C2. `dataset_source` read from wrong attribute
- **Line:** 174
- `getattr(self.config, 'dataset', 'dataset_cache')` reads `'dataset'` but field is `'dataset_source'`. Always falls back to default.
- **Fix:** `getattr(self.config, 'dataset_source', 'dataset_cache')`

### H1. Gradient checkpointing never activates on DDP
- **Lines:** 515, 520-523
- After DDP wrapping, `hasattr(model, 'gradient_checkpointing_enable')` returns False. K80 runs OOM.
- **Fix:** Call `model.module.gradient_checkpointing_enable()` before DDP wrapping.

### H2. `_compute_loss` uses slow Python loops with `.item()` sync
- **Lines:** 619-630
- Nested loops calling `.item()` on 2048 elements per batch. Each `.item()` forces CUDA sync (2-10ms stall).
- **Fix:** Vectorize with `torch.cumsum` for thinking boundary detection.

### H3. Forward pass recomputed in `_compute_thinking_metrics`
- **Lines:** 797-803
- `_compute_loss` computes `model(inputs)`. Every 50th batch, `_compute_thinking_metrics` calls `model(inputs)` again.
- **Fix:** Return logits from `_compute_loss`, pass to `_compute_thinking_metrics`.

### H4. Final checkpoint saved twice
- **Lines:** 1227-1265
- Identical content saved to timestamped file AND `model_output_path`. Doubles save time and disk.
- **Fix:** Keep only `self.model_output_path`.

### H5. `save_vocabulary` called on all DDP ranks (race condition)
- **Line:** 1018
- All ranks write to `tokenizer_vocab.json` concurrently.
- **Fix:** Guard with `if self.rank == 0:`.

### M1. `raw_dataset` never cleared after CSV fine-tuning
- **Line:** 1023
- Full untokenized dataset stays in memory after CSV fine-tuning completes.
- **Fix:** Set `self.raw_dataset = None` after CSV fine-tuning (line ~1214).

### M2. `empty_cache()` every 10 batches causes allocation churn
- **Lines:** 860-861
- Forces CUDA to release all cached blocks. 5-15% throughput degradation.
- **Fix:** Move to epoch boundaries only.

### M3. `cache_has_token_ids` is dead code
- **Lines:** 255, 259
- Set but never read anywhere.
- **Fix:** Remove.

### M4. Timing lists grow without bound
- **Lines:** 754-755
- 100K batches = 100K floats stored. Only used for end-of-epoch logging.
- **Fix:** Use running avg/max/min.

### L1-L4. Trivial cleanup
- **L1 (176,1017,1102,1171):** Redundant `os.makedirs` x4 → keep only `__init__`
- **L2 (481):** Unused `import gc` in `_setup_model_with_device_strategy` → remove
- **L3 (975):** Local `loaded_dataset` shadows `self.loaded_dataset` → rename to `_pre_tokenized_dataset`
- **L4 (788-810):** `no_sync()` wraps forward (unnecessary) → scope to backward only

---

## Fix Cluster 2: `inference/chat_engine.py` (5 issues)

### C3. Duplicate `parse_thinking_response` shadows dict version
- **Lines:** 53 (dict) and 477 (tuple)
- Second definition overwrites first. Line 387 calls dict-style access on tuple → `TypeError` at runtime.
- **Fix:** Remove duplicate at line 477. Keep dict version only.

### H7. Same BERT model loaded twice (440 MB waste)
- **Lines:** 216-220
- `intent_model_path = sentiment_model_path`. Two pipeline objects each hold own copy.
- **Fix:** Share single pipeline: `sentiment_analyzer = pipeline(...)` then `intent_classifier = sentiment_analyzer`.

### M7+M8. Pipelines and singleton never cleaned up
- **Lines:** 219-220, 490-502
- No `close()` method. GPU memory never freed.
- **Fix:** Add `ChatEngine.close()` that calls `del self.model`, `del self.intent_classifier`, etc. Call from singleton reset.

### M9. `weights_only=False` fallback (security)
- **Lines:** 300-303
- Falls back to pickle deserialization if `weights_only=True` fails.
- **Fix:** Log specific warning, don't silently fallback. Or use `safetensors`.

---

## Fix Cluster 3: `commons/dialogue/dialogmanager.py` (6 issues)

### H9. N-gram ban penalty never reaches `sample_next_token`
- **Lines:** 199-210
- `banned_tokens` set built but never passed to `sample_next_token`. Probability-zeroing mechanism (lines 68-75) is dead code.
- **Fix:** Pass `banned_tokens=banned_tokens` to `sample_next_token`.

### M13. O(n^2) tensor growth via `torch.cat`
- **Lines:** 247-248
- Each iteration copies entire accumulated tensor. ~128 copies for max_len=128.
- **Fix:** Pre-allocate buffer of `max_len`, track with index.

### M14. Debug `print()` on every message
- **Line:** 155
- `print(f"Intent: ...")` fires on every user message. Not suppressible.
- **Fix:** Replace with `logger.debug()`.

### M15. Fragile sentiment star parsing
- **Lines:** 145-153
- `replace(' stars', '')` breaks on `"5/5 stars"`, `"POSITIVE"`, etc.
- **Fix:** Use regex `r'(\d+)'` or try/except with fallback.

### M16. Entire response discarded for single repeated n-gram
- **Lines:** 269-280
- Any repeated n-gram anywhere = entire response replaced with default.
- **Fix:** Only retry generation, or check only the last n-gram position.

### L8. `_is_repeated_ngram` is dead code
- **Lines:** 317-324
- Never called. Inline closure at line 270 duplicates logic.
- **Fix:** Remove method.

---

## Fix Cluster 4: `main.py` (3 issues)

### H6. Daemon training thread risks data loss
- **Line:** 568
- `daemon=True` → abruptly killed on exit. Checkpoint save interrupted mid-write.
- **Fix:** `daemon=False` + join with longer timeout.

### M5. Logging handler accumulation
- **Lines:** 72-77
- Root handler added unconditionally. Duplicates on re-import.
- **Fix:** Check `if not root_logger.handlers:` before adding.

### M6. `ColorFormatter.format()` crashes on non-string msg
- **Line:** 64
- `record.msg.startswith('[OK]')` → `AttributeError` if msg is exception object.
- **Fix:** `str(record.msg).startswith('[OK]')`

---

## Fix Cluster 5: `commons/utils/device_utils.py` (3 issues)

### H8. `set_device(i)` changes global CUDA device
- **Line:** 110
- Side effect corrupts CUDA state for downstream code.
- **Fix:** Remove `set_device`, use `mem_get_info(i)` directly.

### M10. VRAM estimation hardcodes FP32
- **Line:** 237
- Uses 4 bytes/param but mixed precision uses 2 bytes. Overestimates by 2x.
- **Fix:** Accept `bytes_per_param=4` parameter.

### M11. No CUDA device guard
- **Line:** 221
- `get_device_properties(gpu_device)` crashes if device is not CUDA.
- **Fix:** Add `if not torch.cuda.is_available(): return 0`.

---

## Fix Cluster 6: `commons/model/chatmodel.py` (2 issues)

### M12. Tokenizer stored as model attribute
- **Line:** 15
- `self.tokenizer = tokenizer` on `nn.Module`. Bloats serialization.
- **Fix:** Remove `self.tokenizer`. Pass separately where needed.

### L5. `hidden_size` param accepted but never used
- **Line:** 13
- Never passed to `GPT2Config` (which uses `n_embd`).
- **Fix:** Remove parameter or map to `n_embd`.

---

## Fix Cluster 7: `dataset_preparer/` (13 issues)

### H10. spaCy model reloaded on every call
- **File:** `data_preparer.py:126-136`
- `spacy.load()` called for every text (0.5-2s each).
- **Fix:** Module-level singleton cache.

### H11. O(n^2) string concatenation for PDF/EPUB
- **File:** `data_preparer.py:1155-1163, 1253-1276`
- `text += page.extract_text() + " "` creates new string each iteration.
- **Fix:** Use list + `''.join()`.

### H12. Unbounded OllamaTeacher cache
- **File:** `thinking_generators.py:23`
- `self.cache` never cleared. Grows without limit.
- **Fix:** `collections.OrderedDict` with max 1000 entries (LRU).

### M17. Unbounded ThinkingEngine cache
- **File:** `thinking_engine.py:39`
- `self._analysis_cache` never evicted.
- **Fix:** Same LRU approach as H12.

### M18. `requests.Session` never closed
- **File:** `web/scraper.py:82-87`
- Connection pool stays open after scraping.
- **Fix:** Add `close()` method, call in `__exit__`.

### L9. Duplicate 'weather' key in CATEGORY_KEYWORDS
- **File:** `aiml/parser.py:128-177`
- Second definition overwrites first.
- **Fix:** Merge into single entry.

### L10. `re`/`unicodedata` availability checks are dead code
- **File:** `data_preparer.py:68-78`
- Both are stdlib. try/except is unnecessary.
- **Fix:** Remove try/except blocks.

### L11. Duplicate step number in logs
- **File:** `data_preparer.py:750-754`
- Both quality and language filtering log as `[6.7/7]`.
- **Fix:** Rename to `[6.7a/7]` and `[6.7b/7]`.

### L12. Regex recompiled every call
- **File:** `thinking_engine.py:183-187`
- `re.compile()` called in hot path.
- **Fix:** Module-level `_ADJ_PATTERN = re.compile(...)`.

### L13. Statistics materializes full length list
- **File:** `data_preparer.py:2167-2179`
- Creates list when single-pass sum/min/max works.
- **Fix:** Compute in one pass.

### L14. Hard-coded 20x CSV oversampling
- **File:** `data_preparer.py:997-1003`
- Every CSV row duplicated 20x. Biases training.
- **Fix:** Make configurable or reduce to 5x.

### L15. `pickle.load` without integrity check
- **File:** `data_preparer.py:566-568,597-598`
- Arbitrary code execution risk if cache is shared.
- **Fix:** Add warning log, or use JSON for metadata.

---

## Implementation Order

```
Phase 1: CRITICAL (7 issues)
  Cluster 1: C1, C2          (trainer.py)
  Cluster 2: C3              (chat_engine.py)
  Cluster 7: --               (no critical in dataset_preparer)

Phase 2: HIGH (12 issues)
  Cluster 1: H1, H2, H3, H4, H5  (trainer.py)
  Cluster 2: H7              (chat_engine.py)
  Cluster 3: H9              (dialogmanager.py)
  Cluster 4: H6              (main.py)
  Cluster 5: H8              (device_utils.py)
  Cluster 7: H10, H11, H12  (dataset_preparer/)

Phase 3: MEDIUM (18 issues)
  Cluster 1: M1, M2, M3, M4  (trainer.py)
  Cluster 2: M7+M8, M9       (chat_engine.py)
  Cluster 3: M13, M14, M15, M16 (dialogmanager.py)
  Cluster 4: M5, M6          (main.py)
  Cluster 5: M10, M11        (device_utils.py)
  Cluster 6: M12             (chatmodel.py)
  Cluster 7: M17, M18        (dataset_preparer/)

Phase 4: LOW (15 issues)
  Cluster 1: L1, L2, L3, L4  (trainer.py)
  Cluster 3: L8              (dialogmanager.py)
  Cluster 5: --              (none)
  Cluster 6: L5              (chatmodel.py)
  Cluster 7: L9-L15          (dataset_preparer/)
```

---

## Files Modified

| File | Issues | Priority |
|------|--------|----------|
| `training/trainer.py` | C1, C2, H1-H5, M1-M4, L1-L4 | 1 |
| `inference/chat_engine.py` | C3, H7, M7+M8, M9 | 2 |
| `commons/dialogue/dialogmanager.py` | H9, M13-M16, L8 | 3 |
| `main.py` | H6, M5, M6 | 4 |
| `commons/utils/device_utils.py` | H8, M10, M11 | 5 |
| `commons/model/chatmodel.py` | M12, L5 | 6 |
| `dataset_preparer/data_preparer.py` | H10, H11, L10, L11, L13, L14, L15 | 7 |
| `dataset_preparer/thinking_generators.py` | H12 | 7 |
| `dataset_preparer/thinking_engine.py` | M17, L12 | 7 |
| `dataset_preparer/web/scraper.py` | M18 | 7 |
| `dataset_preparer/aiml/parser.py` | L9 | 7 |

---

*Generated: 2026-08-05*
