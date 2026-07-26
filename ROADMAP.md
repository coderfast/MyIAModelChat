# ROADMAP - MyIAModelChat Thinking Implementation

## Overview

This roadmap defines the implementation of complete chain-of-thought ("thinking") capabilities for the MyIAModelChat system. The system currently has a partial scaffold (tokens, detection, templates) but lacks loss masking, generation control, streaming support, and LLM-powered data generation.

**Status Legend:** `TODO` → `IN_PROGRESS` → `BLOCKED` → `DONE`

**Priority Legend:**
- **P0 (MVP)** — Blocking: must be done first, system is broken without it
- **P1 (High)** — Core: required for thinking to function correctly
- **P2 (Medium)** — Quality: improves thinking behavior significantly
- **P3 (Low)** — Polish: nice-to-have, can be deferred

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loss masking breaks training | Training produces NaN or diverges | Rollback: revert `_compute_loss()` to original, keep all other changes |
| Stop token causes infinite loop | Generation hangs | Add hard max_tokens limit per phase (thinking: 64, response: 128) |
| LLM models too large for user disk | Phase 6 unusable | Fallback to templates, document min RAM/disk requirements |
| Streaming format incompatible | Clients break | Keep `stream=false` as default, streaming opt-in only |
| Thinking reduces response quality | Model confused by thinking tokens | `thinking_loss_weight` tunable, default 0.5 allows partial learning |

---

## Rollback Strategy

Each phase is independently revertible:

```bash
# After any phase, if something breaks:
git stash           # Save current work
git checkout -- <file>  # Revert specific file
# Or revert entire phase:
git log --oneline   # Find commit before phase
git revert <commit> # Revert phase
```

**Critical rollback points:**
- After Phase 1: System should work as before (bug fixes only)
- After Phase 2: Training should still converge (loss masking is additive)
- After Phase 3: Generation should still work (stop token is additive)
- After Phase 4: API should still respond (streaming is opt-in)

---

## Phase 1: Critical Bug Fixes — P0 (MVP)

**Priority:** P0 — System is broken without these fixes
**Estimated Time:** 1-2 hours
**Risk:** Low (bug fixes only)

### Task 1.1: Fix `parse_thinking_response()` in `main_chat.py`
- **File**: `main_chat.py:410-424`
- **Problem**: Line 421 contains corrupted string literal (`效益` instead of `` `)
- **Fix**: Replace corrupted string with correct `` token
- **Lines**: 421
- **Impact**: `parse_thinking_response()` will correctly separate thinking from response
- **Status**: `TODO`
- **Testing After:** Run `parse_thinking_response()` with sample thinking text

### Task 1.2: Fix `data_preparer.py` cache metadata
- **File**: `data_preparer.py:568-593` (`_save_to_cache`)
- **Problem**: `has_thinking_tokens` flag never written to `cache_metadata`
- **Fix**: Add `has_thinking_tokens` detection and write to metadata during `_save_to_cache()`
- **Lines**: 584-588
- **Impact**: `main_train.py:221` detection will work correctly
- **Status**: `TODO`
- **Testing After:** Prepare data with thinking, verify `cache_metadata.pkl` contains `has_thinking_tokens=True`

### Phase 1 Documentation Update
- [ ] Update `AGENTS.md` if file structure changed
- [ ] Update `THINKING_GUIDE.md` with bug fix notes

---

## Phase 2: Thinking-Weighted Loss Masking — P1 (High)

**Priority:** P1 — Core thinking functionality
**Estimated Time:** 4-6 hours
**Risk:** Medium (modifies training loop, could affect convergence)

### Task 2.1: Add `thinking_loss_weight` to `TRAINING_CONFIG`
- **File**: `main_train.py`
- **Location**: `TRAINING_CONFIG` dict
- **Change**: Add `'thinking_loss_weight': 0.5`
- **Lines**: ~139-150
- **Status**: `TODO`

### Task 2.2: Modify `_compute_loss()` for thinking-aware masking
- **File**: `main_train.py:430-443`
- **Current**: Flat CrossEntropyLoss on all non-pad tokens
- **New**:
  - Identify thinking token positions (between `<think>` and `</think>` token IDs)
  - Apply `thinking_loss_weight` to thinking token losses
  - Apply full weight (1.0) to response token losses
- **Performance**: Vectorized implementation using PyTorch tensors (no Python loops)
- **Implementation**:
  ```python
  def _compute_loss(self, model, inputs, targets, criterion):
      outputs = model(inputs)
      outputs = outputs.contiguous().view(-1, outputs.size(-1))
      targets = targets.contiguous().view(-1)

      # Create weight mask (vectorized)
      weights = torch.ones_like(targets, dtype=torch.float)
      thinking_id = self.tokenizer.get_thinking_index()
      thinking_end_id = self.tokenizer.get_thinking_end_index()

      if thinking_id >= 0 and thinking_end_id >= 0:
          # Vectorized: find positions of thinking open/close tokens
          is_open = targets == thinking_id
          is_close = targets == thinking_end_id

          # Cumulative sum to track in_thinking state per sequence
          # Reset at sequence boundaries (pad tokens)
          pad_mask = targets == self.tokenizer.get_pad_index()

          # Build thinking mask per sequence
          cumsum = torch.cumsum(is_open.long() - is_close.long(), dim=0)
          in_thinking = cumsum > 0

          # Apply weight to thinking positions
          weights[in_thinking] = self.thinking_loss_weight

          # Reset at sequence boundaries to prevent cross-sequence contamination
          seq_starts = torch.where(pad_mask | torch.cat([torch.ones(1, dtype=torch.bool, device=targets.device), targets[1:] == self.tokenizer.get_pad_index()]))[0]
          # Actually, simpler: just use unfold or manual tracking

          # Per-sequence approach (safe for batched data):
          batch_size = inputs.size(0)
          seq_len = inputs.size(1)
          for b in range(batch_size):
              seq_targets = targets[b * seq_len:(b + 1) * seq_len]
              in_think = False
              for i in range(len(seq_targets)):
                  if seq_targets[i].item() == thinking_id:
                      in_think = True
                  if in_think:
                      weights[b * seq_len + i] = self.thinking_loss_weight
                  if seq_targets[i].item() == thinking_end_id:
                      in_think = False

      non_pad_mask = targets.ne(self.tokenizer.get_pad_index())
      outputs = outputs[non_pad_mask]
      targets = targets[non_pad_mask]
      weights = weights[non_pad_mask]

      loss = criterion(outputs, targets)
      return (loss * weights).sum() / weights.sum()
  ```
- **Lines**: 430-443
- **Status**: `TODO`

### Phase 2 Testing
- [ ] Unit test: Verify thinking tokens get weight 0.5, response tokens get 1.0
- [ ] Unit test: Verify cross-sequence contamination does not occur
- [ ] Integration test: Train 1 epoch with thinking data, verify loss decreases
- [ ] Rollback test: Revert `_compute_loss()`, verify training still works

### Phase 2 Documentation Update
- [ ] Update `THINKING_GUIDE.md` with loss masking explanation
- [ ] Update `README.md` Configuration section with `thinking_loss_weight`

---

## Phase 3: Generation with Thinking Stop Token — P1 (High)

**Priority:** P1 — Core thinking functionality
**Estimated Time:** 4-6 hours
**Risk:** Medium (modifies generation loop)

### Task 3.1: Add thinking stop detection in `dialogmanager.py`
- **File**: `dialogmanager.py:186-224` (generation loop)
- **Current**: No stop on `</think>` token
- **New**:
  - Track when `</think>` token is generated
  - After `</think>`, switch to "response mode"
  - Continue generating response until EOS or max_len
  - Return both thinking and response separately
- **Hard limits** (prevent infinite loops):
  - Max thinking tokens: 64
  - Max response tokens: 128
  - If `</think>` never generated, treat all output as response
- **Implementation**:
  - Add `self.thinking_end_id` to `DialogueManager.__init__`
  - In generation loop: after generating a token, check if it's `</think>`
  - If `</think>` generated, set `in_response_mode = True`
  - Continue generating for response tokens
  - Collect thinking tokens and response tokens separately
- **Lines**: 186-224
- **Status**: `TODO`

### Task 3.2: Update `generate_response()` return format
- **File**: `dialogmanager.py:160-224`
- **Current**: Returns `str` (response only)
- **New**: Return `{'thinking': str|None, 'response': str}`
- **Backward compatible**: Check if return value is dict or str
- **Lines**: 160-224
- **Status**: `TODO`

### Phase 3 Testing
- [ ] Unit test: Generate response with thinking, verify `thinking` and `response` keys
- [ ] Unit test: Generate response without thinking, verify `thinking` is None
- [ ] Unit test: Verify max tokens limits are enforced
- [ ] Integration test: Full chat with `--show-thinking`, verify output format

### Phase 3 Documentation Update
- [ ] Update `THINKING_GUIDE.md` with generation behavior
- [ ] Update `README.md` examples with thinking output

---

## Phase 4: Streaming with Thinking/Response Separation — P2 (Medium)

**Priority:** P2 — Quality of life for API consumers
**Estimated Time:** 4-6 hours
**Risk:** Low (streaming is opt-in)

### Task 4.1: Update `stream_chat_text()` for thinking streaming
- **File**: `main_chat.py:437-450`
- **Current**: Character-level chunking of final text
- **New**:
  - Accept pre-parsed `{'thinking': str, 'response': str}`
  - Send thinking chunks as `{"reasoning": "chunk"}` deltas
  - Send response chunks as `{"content": "chunk"}` deltas
  - Use token-level chunking (not character-level)
- **Lines**: 437-450
- **Status**: `TODO`

### Task 4.2: Add `reasoning` field to Pydantic schemas
- **File**: `main_chat.py`
- **Changes**:
  - Add `reasoning: Optional[str]` to `ChatRequest`
  - Add `reasoning: Optional[str]` to response models
- **Lines**: 71-80 (ChatRequest), ~300-350 (response models)
- **Status**: `TODO`

### Task 4.3: Update `/v1/chat/completions` endpoint for streaming
- **File**: `main_chat.py`
- **Change**: When `stream=True` and `include_thinking=True`:
  - Send thinking delta chunks first
  - Then send response delta chunks
  - Format: OpenAI-compatible with `reasoning` field
- **Lines**: ~300-360
- **Status**: `TODO`

### Phase 4 Testing
- [ ] Unit test: Streaming format contains `reasoning` and `content` deltas
- [ ] Integration test: SSE stream with thinking, verify chunk order
- [ ] Compatibility test: Verify `stream=false` still works unchanged

### Phase 4 Documentation Update
- [ ] Update API docs in `README.md` with streaming format

---

## Phase 5: Thinking-Specific Metrics — P2 (Medium)

**Priority:** P2 — Helps debug and tune thinking behavior
**Estimated Time:** 3-4 hours
**Risk:** Low (metrics only, no training impact)

### Task 5.1: Enhance `_compute_thinking_metrics()`
- **File**: `main_train.py:445-486`
- **Current**: Only `thinking_token_accuracy`, `thinking_open_accuracy`, `thinking_close_accuracy`
- **New metrics**:
  - `thinking_loss` — average loss on thinking tokens
  - `response_loss` — average loss on response tokens
  - `thinking_length_avg` — average thinking block length (tokens)
  - `thinking_coverage` — % of output that is thinking
  - `thinking_token_accuracy` (existing, keep)
  - `response_token_accuracy` — accuracy on response tokens only
- **Lines**: 445-486
- **Status**: `TODO`

### Task 5.2: Log thinking metrics separately in training loop
- **File**: `main_train.py:545-585`
- **Change**: Log thinking vs response loss separately
- **Lines**: 545-585
- **Status**: `TODO`

### Phase 5 Testing
- [ ] Unit test: Verify all new metrics are computed correctly
- [ ] Unit test: Verify metrics are logged per epoch

---

## Phase 6: LLM-Powered Thinking Data Generation — P2 (Medium)

**Priority:** P2 — Higher quality thinking data
**Estimated Time:** 6-8 hours
**Risk:** Medium (external dependencies: HF models, Ollama server)

### Disk Space Requirements
- HuggingFace models: 1-4 GB per model (Qwen2.5-1.5B = ~1.5GB)
- Minimum 5 GB free disk space recommended for HF mode
- Ollama mode: model must be pre-pulled (`ollama pull <model>`)

### Task 6.1: Add HuggingFace integration to `generate_thinking_data.py`
- **File**: `generate_thinking_data.py`
- **New function**: `generate_thinking_with_hf(text, answer, model_name)`
- **Implementation**:
  - Load model via `transformers.pipeline("text-generation", model=model_name)`
  - Prompt: Spanish instruction to generate step-by-step reasoning
  - Parse output to extract thinking content
  - Fallback to templates if model unavailable
  - Cache loaded model to avoid re-loading per sample
- **Lines**: New section after line 103
- **Status**: `TODO`

### Task 6.2: Add Ollama integration to `generate_thinking_data.py`
- **File**: `generate_thinking_data.py`
- **New function**: `generate_thinking_with_ollama(text, answer, model_name)`
- **Implementation**:
  - POST to `http://localhost:11434/api/generate`
  - Prompt: Spanish instruction for chain-of-thought
  - Parse JSON response for generated text
  - Fallback to templates if server unavailable
  - Timeout handling (30s default)
- **Lines**: New section after HF integration
- **Status**: `TODO`

### Task 6.3: Update CLI with new modes
- **File**: `generate_thinking_data.py:194-292`
- **New CLI args**:
  - `--mode [template|hf|ollama]` — Generation mode (default: template)
  - `--model <model_id>` — Model name for HF/Ollama mode
  - `--validate` — Validate thinking consistency with answer
  - `--max-samples <int>` — Limit number of samples to process
- **Lines**: 194-292
- **Status**: `TODO`

### Task 6.4: Add English thinking templates
- **File**: `generate_thinking_data.py:29-63`
- **Change**: Add `THINKING_TEMPLATES_EN` with English equivalents
- **See**: [THINKING_DATASETS.md](THINKING_DATASETS.md) for template examples
- **Lines**: 29-63
- **Status**: `TODO`

### Task 6.5: Add thinking validation
- **File**: `generate_thinking_data.py`
- **New function**: `validate_thinking_consistency(text, thinking, answer)`
- **Implementation**:
  - Check if thinking mentions key concepts from the question
  - Check if thinking logically leads to the answer
  - Return `True`/`False`
- **Lines**: New section
- **Status**: `TODO`

### Phase 6 Testing
- [ ] Unit test: Template mode generates valid thinking text
- [ ] Integration test: HF mode with small model (if available)
- [ ] Integration test: Ollama mode (if server running)
- [ ] Unit test: Validation function with consistent/inconsistent examples
- [ ] Manual test: Generate 10 samples, inspect quality

### Phase 6 Documentation Update
- [ ] Update `THINKING_GUIDE.md` with new modes
- [ ] Update `README.md` with `--thinking-mode` examples

---

## Phase 7: CLI and Configuration — P2 (Medium)

**Priority:** P2 — User-facing configuration
**Estimated Time:** 2-3 hours
**Risk:** Low (additive flags)

### Task 7.1: Add thinking-related CLI flags to `main.py`
- **File**: `main.py`
- **New flags**:
  - `--thinking-loss-weight FLOAT` — Weight for thinking token loss (default: 0.5)
  - `--generate-thinking` — Generate thinking data before training
  - `--thinking-mode [template|hf|ollama]` — Mode for thinking generation
  - `--thinking-model <model>` — Model for HF/Ollama thinking generation
- **Lines**: ~620-650 (argparse section)
- **Status**: `TODO`

### Task 7.2: Add thinking config to `TRAINING_CONFIG`
- **File**: `main_train.py`
- **New entries**:
  ```python
  'thinking_loss_weight': 0.5,
  'thinking_enabled': True,
  'thinking_stop_on_end': True,
  'thinking_metrics_interval': 50,
  ```
- **Lines**: ~139-150
- **Status**: `TODO`

### Task 7.3: Add thinking config to `DialogueManager`
- **File**: `dialogmanager.py:33-50`
- **New parameters**:
  - `thinking_end_id` — Token ID for `</think>`
  - `thinking_enabled` — Whether to generate thinking
  - `thinking_max_tokens` — Max tokens for thinking phase (default: 64)
- **Lines**: 33-50
- **Status**: `TODO`

### Phase 7 Testing
- [ ] Unit test: CLI flags parsed correctly
- [ ] Integration test: `--thinking-loss-weight 0.3` changes training behavior
- [ ] Integration test: `--thinking-mode template` generates data

---

## Phase 8: Integration and Testing — P1 (High)

**Priority:** P1 — Everything must work together
**Estimated Time:** 4-6 hours
**Risk:** Low (integration only)

### Task 8.1: Update `main.py` training pipeline
- **File**: `main.py`
- **Change**: If `--generate-thinking` flag is set, call `generate_thinking_data.py` before training
- **Lines**: ~350-370 (training dispatch)
- **Status**: `TODO`

### Task 8.2: Update `main.py` chat pipeline
- **File**: `main.py`
- **Change**: Pass `thinking_enabled` and `thinking_loss_weight` to `MainTrain` and `MainChat`
- **Lines**: ~370-390
- **Status**: `TODO`

### Task 8.3: Write tests for thinking functionality
- **File**: `tests/test_thinking.py`
- **Tests**:
  - `test_loss_masking()` — Verify thinking tokens have lower loss weight
  - `test_thinking_detection()` — Verify `<think>` detection in training data
  - `test_parse_thinking_response()` — Verify thinking/response separation
  - `test_thinking_metrics()` — Verify thinking metrics computation
  - `test_streaming_with_thinking()` — Verify streaming with thinking chunks
  - `test_stop_token_limits()` — Verify max tokens enforced
  - `test_backward_compatibility()` — Verify existing behavior unchanged
- **Status**: `TODO`

### Phase 8 Final Verification
- [ ] Full pipeline: prepare data -> train 1 epoch -> chat with thinking
- [ ] API test: POST to `/v1/chat/completions` with `include_thinking: true`
- [ ] Streaming test: SSE stream with thinking chunks
- [ ] Regression test: All existing tests pass

### Phase 8 Documentation Update
- [ ] Update `AGENTS.md` with thinking capabilities
- [ ] Update `README.md` with complete thinking documentation
- [ ] Update `THINKING_GUIDE.md` with all new features
- [ ] Update `APP_ARCHITECTURE.md` with thinking data flow

---

## Implementation Order

```
Phase 1 (Bug Fixes) — P0 — 1-2h
  -> Task 1.1 (parse_thinking_response fix)
  -> Task 1.2 (cache metadata fix)

Phase 2 (Loss Masking) — P1 — 4-6h
  -> Task 2.1 (config)
  -> Task 2.2 (loss function)
  -> [TEST] Unit + integration tests

Phase 3 (Generation Control) — P1 — 4-6h
  -> Task 3.1 (stop token)
  -> Task 3.2 (return format)
  -> [TEST] Unit + integration tests

Phase 8.1-8.2 (Integration) — P1 — 2-3h
  -> Task 8.1 (training pipeline)
  -> Task 8.2 (chat pipeline)

Phase 5 (Metrics) — P2 — 3-4h
  -> Task 5.1 (enhance metrics)
  -> Task 5.2 (logging)
  -> [TEST] Unit tests

Phase 7 (CLI/Config) — P2 — 2-3h
  -> Task 7.1 (main.py flags)
  -> Task 7.2 (TRAINING_CONFIG)
  -> Task 7.3 (DialogueManager config)
  -> [TEST] Unit + integration tests

Phase 6 (LLM Data Generation) — P2 — 6-8h
  -> Task 6.1 (HF integration)
  -> Task 6.2 (Ollama integration)
  -> Task 6.3 (CLI)
  -> Task 6.4 (English templates)
  -> Task 6.5 (validation)
  -> [TEST] Unit + integration tests

Phase 4 (Streaming) — P2 — 4-6h
  -> Task 4.1 (stream_chat_text)
  -> Task 4.2 (schemas)
  -> Task 4.3 (endpoint)
  -> [TEST] Unit + integration tests

Phase 8.3 (Final Tests) — P1 — 2-3h
  -> Task 8.3 (all tests)
  -> [VERIFY] Full pipeline test
```

**Total estimated time: 30-44 hours**

---

## Key Files Summary

| File | Tasks | Priority | Total Lines Changed |
|------|-------|----------|-------------------|
| `main_train.py` | 2.1, 2.2, 5.1, 5.2, 7.2 | P1 | ~100 lines |
| `main_chat.py` | 1.1, 4.1, 4.2, 4.3, 7.1 | P1-P2 | ~150 lines |
| `dialogmanager.py` | 3.1, 3.2, 7.3 | P1 | ~80 lines |
| `generate_thinking_data.py` | 6.1, 6.2, 6.3, 6.4, 6.5 | P2 | ~300 lines |
| `data_preparer.py` | 1.2 | P0 | ~10 lines |
| `bpe_tokenizer.py` | (minor improvements) | P1 | ~20 lines |
| `main.py` | 7.1, 8.1, 8.2 | P1-P2 | ~50 lines |
| `tests/test_thinking.py` | 8.3 | P1 | ~200 lines (new file) |

---

## Dependencies

| Task | Depends On | Priority |
|------|-----------|----------|
| 1.1 | (none) | P0 |
| 1.2 | (none) | P0 |
| 2.1 | 1.2 | P1 |
| 2.2 | 2.1 | P1 |
| 3.1 | 2.2 | P1 |
| 3.2 | 3.1 | P1 |
| 4.1 | 3.2 | P2 |
| 4.2 | 4.1 | P2 |
| 4.3 | 4.2 | P2 |
| 5.1 | 2.2 | P2 |
| 5.2 | 5.1 | P2 |
| 6.1 | (none) | P2 |
| 6.2 | (none) | P2 |
| 6.3 | 6.1, 6.2 | P2 |
| 6.4 | 6.3 | P2 |
| 6.5 | 6.3 | P2 |
| 7.1 | 6.3 | P2 |
| 7.2 | 2.1 | P2 |
| 7.3 | 3.1 | P2 |
| 8.1 | 7.1, 7.2 | P1 |
| 8.2 | 7.1, 7.3 | P1 |
| 8.3 | All above | P1 |

---

## Testing Strategy

### Per-Phase Testing
- **Phase 1:** Manual verification of bug fixes
- **Phase 2:** Unit test loss masking + integration test training convergence
- **Phase 3:** Unit test stop token + integration test chat output
- **Phase 4:** Unit test streaming format + compatibility test
- **Phase 5:** Unit test metrics computation
- **Phase 6:** Unit test template generation + integration test LLM modes
- **Phase 7:** Unit test CLI parsing + integration test config propagation

### Final Validation Checklist
```bash
# 1. Prepare data with thinking
python main.py --prepare-data --aiml --generate-thinking --thinking-mode template

# 2. Train with thinking
python main.py --train --use-cache --epochs 1 --thinking-loss-weight 0.5

# 3. Chat with thinking
python main.py --chat --show-thinking

# 4. API test
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola"}],"include_thinking":true}'

# 5. Run all tests
pytest tests/test_thinking.py -v
```

---

*Generated from codebase analysis - reflects the current state and planned improvements.*
