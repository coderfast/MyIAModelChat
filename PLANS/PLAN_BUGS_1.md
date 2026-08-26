# PLAN_BUGS_1 — Bug Audit & Remediation Plan

> Scope: Full codebase review (read AGENTS.md + all project Python files).
> Status: 45/45 bugs addressed (44 implemented + 1 deferred to design decision).
> Generated: 2026-08-24

---

## Structural note (also a bug)
`AGENTS.md` and the code's own launcher/tests reference a package named
**`envAIModels/`**, but the directory on disk is **`ServerFastAPI/`** (and no
`envAIModels/` dir exists). `python -m envAIModels.app` and `runserver.bat`
fail with `ModuleNotFoundError`. Fix by renaming the directory OR by rewriting
all references (`ServerFastAPI/__init__.py`, `app.py`, `routers_v1.py`,
`runserver.bat`, `tests/test_envaimodels_smoke.py`, AGENTS.md).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.**
Todas las referencias a `envAIModels` reemplazadas por `ServerFastAPI` en:
`__init__.py`, `app.py`, `routers_v1.py`, `runserver.bat`,
`test_envaimodels_smoke.py`, `AGENTS.md`, `README.md`, `APP_ARCHITECTURE.md`,
`APP_TECHNICALSTACK.md`, `FICHA_HUGGINGFACE.md`, `ServerFastAPI/README.md`,
`ServerFastAPI/ROADMAP.md`, roadmaps. El server ahora arranca correctamente
con `uvicorn ServerFastAPI.app:app`.

---

## CRITICAL (crashes / silent data corruption)

1. **CPU+GPU split → device-mismatch crash**
   - `training/trainer.py:558-625` (`_setup_model_cpu_gpu_split`)
   - With `layers_on_gpu < num_layers`, activations on GPU hit CPU layers →
     `RuntimeError: tensor on device cuda:0 but expected on device cpu`.
   - Fix: custom forward that moves residuals between devices between layer
     groups, or only shift whole-model groups (embeddings/head + all layers),
     falling back to all-CPU / all-GPU for partial splits.

MIMO_V25: **IMPLEMENTADO.** `_install_cpu_gpu_forward_wrapper` añadido en
trainer.py. Inserta `.to('cpu')` en el split boundary y `.to(gpu_device)`
antes de lm_head. Bypass automático si todas las capas están en el mismo
dispositivo.

2. **MoE is entirely dead**
   - `training/trainer.py:1357,1389` always builds plain `ChatModel`; never
     imports/uses `ChatModelMoE`. `--moe-enabled` does nothing; load-balance
     loss always 0.
   - Fix: branch on `config.moe_enabled` and construct
     `ChatModelMoE(tokenizer, embed_size, num_layers, num_experts, top_k,
     load_balance_weight)`; if `moe_freeze_attention`, call
     `model.freeze_attention()` after device setup.

MIMO_V25: **IMPLEMENTADO.** `ChatModelMoE` importado en trainer.py; ambos
caminos de construcción (resume + fresh) ahora bifurcan en `moe_enabled`;
campos moe añadidos a ambos dicts de arquitectura en checkpoints.

3. **MoE checkpoints can't load for inference**
   - `inference/chat_engine.py:188-201` builds plain `ChatModel`; `architecture`
     metadata (`trainer.py:1518-1525,1569-1576`) omits `moe_*` keys →
     `load_state_dict` size mismatch.
   - Fix: persist `moe_enabled/moe_num_experts/moe_top_k` in metadata and
     reconstruct `ChatModelMoE` when the flag is set (mirror
     `model_export.py:285`).

MIMO_V25: **IMPLEMENTADO.** `ChatModelMoE` importado en chat_engine.py;
construcción del modelo lee `arch.get('moe_enabled')` y construye
`ChatModelMoE` con parámetros moe del checkpoint.

4. **Gradient-accumulation double-scaled → ~8× slower convergence**
   - `training/trainer.py:949-959` divides by `accumulation_steps` inside
     `_backward_pass`, called per micro-step, but `train()` does not re-scale →
     net `1/k²` update for `k=8`.
   - Fix: divide loss by `accumulation_steps` exactly once, in `train()`
     (`loss = loss / accumulation_steps`), and have `_backward_pass` just call
     `loss.backward()` (don't multiply back). Log the unscaled `loss.detach()`.

MIMO_V25: **FALSO POSITIVO — VERIFICADO CONTRA EL CÓDIGO REAL.**
`_backward_pass` hace `loss = loss / accumulation_step` ANTES de `.backward()`
(línea 952). `train()` NO divide antes de llamar a `_backward_pass` (líneas
1069, 1071). Cada micro-step contribuye `grad/8` al gradiente acumulado.
Después de 8 micro-steps: `Σ(grad_t/8) = media(grad)`. El optimizador da un
paso con `lr × media(grad)`. Esto es el patrón estándar correcto para gradient
accumulation. El análisis del agente se confundió con su propia matemática.
**El código es correcto. NO TOCAR.**

5. **`filters.py:123` crash**
   - `result['empty'] = ...` on dataclass `FilterResult` → `TypeError`
     (`object does not support item assignment`).
   - Fix: `result.stats['empty'] = result.stats.get('empty', 0) + 1`.

MIMO_V25: **IMPLEMENTADO.** `result.stats['empty']` en filters.py:123.

6. **`dedup.py:157-177` short texts collapse**
   - Texts with <3 words produce empty MinHash (all-max) → identical → distinct
     answers ("YES"/"NO"/"OK") deduplicated to one survivor.
   - Fix: skip MinHash/LSH for `len(words) < 3` (treat as exact-only); only
     `lsh.query` when `minhashes[i] is not None`.

MIMO_V25: **IMPLEMENTADO.** Skip MinHash para `<3` palabras; `minhashes[i]`
puede ser `None`.

7. **`balance.py:92` drops ALL data**
   - `max_per_source = int(total_samples * 0.3)` can be `0` →
     `rng.sample(idxs, 0)` returns `[]`.
   - Fix: `max_per_source = max(1, int(total_samples * self.max_ratio))`.

MIMO_V25: **IMPLEMENTADO.** `max(1, int(...))` en balance.py:92.

8. **`hf/thinking.py:80` typo**
   - `self.teacher.generate(prompt, max_max=max_tokens)` → `TypeError`.
   - Fix: `max_tokens=max_tokens`.

MIMO_V25: **IMPLEMENTADO.** `max_tokens=max_tokens` en hf/thinking.py:80.

9. **GGUF server naming** — see Structural note.

MIMO_V25: **IMPLEMENTADO.** Todas las referencias renombradas. BOM eliminado
de app.py. ServerFastAPI arranca correctamente.

10. **`ServerFastAPI/utils.py:266-272` streaming duplicate text**
    - `yield buffer[:buffer.find(s)]` includes already-sent text → client gets
      duplicated output on any stop sequence.
    - Fix: track `sent` offset and yield `buffer[sent:buffer.find(s)]`, or only
      check/truncate the current `chunk`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.**
`yield buffer[:find(s)]` enviaba buffer completo desde posición 0. Fix: añadido
offset `sent` para rastrear contenido ya enviado; solo se envía `buffer[sent:end]`
cuando se detecta secuencia de stop. Verificación: tracing manual con chunks
"Hello " y "world\nuser:" confirma que el cliente recibe "Hello world\nuser"
sin duplicación. Tests de tool_executor (52) pasan OK.

11. **`models/exported/gguf_quantizer.py` corruption**
    - Q3_K never writes `scales`/`hmask` (zero scales → garbage dequant).
    - Q6_K writes only `qs[:192]` of 256 values (64 weights lost).
    - `IQ2_XS`/`IQ2_S`/`IQ3_XXS` `type_size` mismatch (74/82/98 vs
      actual 70/78/74 bytes) → wrong per-block stride.
    - Fix: write scales/hmask per Q3_K layout; store all 256 Q6_K values; set
      `type_size` to actual bytes written.

MIMO_V25: **IMPLEMENTADO.** Q3_K: scales calculados y empaquetados en 4-bit.
Q6_K: 256 valores 6-bit empaquetados en 192 bytes. IQ2_XS/IQ2_S/IQ3_XXS:
type_size corregido a bytes reales (70/78/74). Syntax OK verificado.
Pendiente: validación funcional con exportación real.

---

## HIGH

### Tools / security / functionality
- `commons/dialogue/dialogmanager.py:396`: calls
  `tool_executor.execute_tool_call(raw, None)`. `ToolExecutor` has no
  `_registry` attr → every agent tool call raises `AttributeError` (caught →
  no-op, no tool ever runs). Fix: `ToolExecutor` must hold a `ToolRegistry`
  reference; pass the real registry. Also `parse_tool_call` result (line 392)
  is assigned but re-parsed from `raw_output` (dead variable).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.**
`ToolExecutor.__init__` no almacena `ToolRegistry`. `dialogmanager.py:396` pasaba
`None` → agente nunca ejecutaba herramientas. Fix: añadido `registry=None` a
`ToolExecutor.__init__` con fallback `reg = registry or self.registry`.
`ChatEngine` pasa `registry=self.tool_registry`. `dialogmanager.py` elimina el
hack `_registry if hasattr...`. Tests tool_executor/tool_registry/permissions
(52) pasan OK.
**NUEVO:** Todas las herramientas (no solo shell) ahora pasan por
`permission_callback` + `dry_run`. `read_file`/`list_directory` marcadas
`requires_permission=True`.

- `commons/tools/tool_executor.py:165-189`: only `category=='shell'` goes
  through permission_callback/dry_run; `read_file`/`list_directory`/`web_search`
  run unconditionally → security bypass. Fix: route every tool through
  permission_callback + dry_run short-circuit.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Todas las herramientas ahora pasan por
`permission_callback` + `dry_run`. `read_file`/`list_directory` marcadas
`requires_permission=True` en `tool_registry.py`.

- `commons/tools/permission_manager.py`: `AUTO_APPROVE` returns before the
  CRITICAL rejection; `classify_risk` never returns `RiskLevel.CRITICAL` →
  destructive commands only classified HIGH and auto-approved.
  Fix: reject CRITICAL regardless of mode; make `classify_risk` return CRITICAL
  for `rm -rf`, `format`, `shutdown`, etc.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `AUTO_APPROVE` ahora verifica CRITICAL
antes de auto-approve. `classify_risk` ahora retorna CRITICAL para `rm -rf`,
`format c:`, `shutdown`, `reboot`, `Remove-Item -Recurse -Force`.

- `commons/tools/tool_registry.py`: `requires_permission` never read; mark
  `read_file`/`list_directory` as requiring permission.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `read_file` y `list_directory` marcadas
`requires_permission=True` en `tool_registry.py`.

### Export
- `commons/registry/model_export.py`:
  - XE1: Modelfile written with quadrupled `{{{{` braces (non-f-string) →
    invalid Ollama Go-template. Fix: use two braces.
  - XE2: `from model_merge import ...` wrong path → `ImportError` on `+` merge
    export. Fix: `from commons.registry.model_merge import ...`.
  - XE3: `export_to_onnx_quantized` ignores `per_channel`/`block_size`.
    Fix: forward them into `quantize_onnx(...)`.
  - XE4: assumes `tokenizer` is a live object, not a path string. Fix:
    `isinstance(tokenizer, str)` → load via `SentencePieceTokenizerWrapper`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** XE1: `{{{{` corregido a `{{`. XE2: ruta
import corregida. XE3: `per_channel`/`block_size` forward. XE4: `isinstance(tokenizer, str)`
con `SentencePieceTokenizerWrapper`.

### Data preparer
- `dataset_preparer/data_preparer.py`:
  - BUG1: dedup/quality/lang filters rebuild dataset from `input_ids` only,
    dropping `thinking/language/source/format_type/has_tool_call/token_ids`.
    Fix: filter by kept indices (`self.combined_data.select(...)`).
  - BUG2: `--validate-sources` re-concatenates raw per-source datasets,
    discarding standardized/migrated/language-tagged data. Fix: run validation
    on `self.combined_data` or re-apply standardization/migration after.
  - BUG3: `_load_hf_data` crashes (`select_columns(['input_ids'])`) when HF
    dataset lacks `text`/`sentence` column. Fix: fallback / clear error.
  - BUG4: `chunk_text_by_tokens` infinite-loops when `overlap>=max_tokens`.
    Fix: `start += max(1, max_tokens - overlap_tokens)`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** BUG1: `select(kept_indices)` preserva
columnas. BUG2: re-apply `_standardize_combined_dataset()` tras validación.
BUG3: skip datasets sin `text`/`sentence`. BUG4: `max(1, max_tokens - overlap)`
evita loop infinito.

### Model / inference
- `commons/model/chatmodel_moe.py`: load-balance loss computed outside autograd
  graph (latent, given #2); `get_expert_utilization` crashes on multi-device
  (`torch.stack` of different-device tensors) → `torch.stack([g.cpu() ...])`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `get_expert_utilization` usa
`.detach()` antes de `stack`. `get_load_balancing_loss` mueve tensores
al device correcto con `.to(device)`.

- `inference/chat_engine.py:236-249`: `intent_classifier` loads the *sentiment*
  model → intent channel is actually sentiment. Fix: load dedicated intent
  model or drop intent pretense.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.**
`intent_classifier` y `sentiment_analyzer` apuntan al mismo modelo BERT de
sentimiento (`nlptown/bert-base-multilingual-uncased-sentiment`). No hay modelo
de intents separado. Fix: añadido comentario explicativo documentando que se usa
el mismo modelo para ambos canales (sentiment-as-intent). Eliminada variable
redundante `intent_model_path`.

- `dataset_preparer/migrator.py:17-39`: double `<|final|>` from legacy
  `</thinking><|answer|>` → `<|problem|>Q<|thinking|>R<|final|><|final|>A`.
  Fix: map `</thinking>`→`` (close block) and only `<|answer|>`→`<|final|>`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.**
`</thinking>` y `<|answer|>` ambos mapeaban a `<|final|>` → doble token. Fix:
añadido patrón combinado `</thinking><|answer|>` → `<|final|>` ANTES de los
standalone. Verificación manual:
- `<|context|>Q<thinking>R</thinking><|answer|>A` → `<|problem|>Q<|thinking|R<|final|>A` ✓
- `<|context|>Q<thinking>R</thinking>A` → `<|problem|>Q<|thinking|R<|final|>A` ✓
Tests chatmodel (22) y mode_tokens pasan OK.

---

## MEDIUM

- `dataset_preparer/thinking_quality.py`:
  - `:543-572` meta-commentary containing a step word gets `+0.3` bonus and is
    marked valid. Fix: apply non-meta bonus only when
    `not is_meta and not is_re_declaration and not is_placeholder`.
  - `validate_thinking_batch` `low_quality` counter never increments (issues
    never contains `low_reasoning`/`low_quality`). Fix: append a
    `low_quality` issue when below threshold, or repurpose counter.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Bonus condicional: solo aplica cuando
`not is_meta and not is_re_declaration and not is_placeholder`. Counter
`low_quality` se incrementa cuando `score < 0.5`.

- `dataset_preparer/agent/thinking.py`:
  - BUG22: reasoning (`result['thinking']`) never inserted into `input_ids`
    training text. Fix: embed `<thinking>{thinking}</thinking>` in assistant
    turn.
  - BUG23: reasoning hardcoded Spanish regardless of sample language. Fix:
    route through `LANGUAGE_CONFIG`/`ThinkingEngine`.
  - BUG24: `_simulate_tool_result` returns first number from answer → wrong
    simulated observations. Fix: compute arithmetic for `calculator`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** BUG22: thinking incluido en `input_ids`
para samples con tool calls (Formato 2). BUG23: thinking multilingual — detecta
idioma y genera en ES/EN. BUG24: `_simulate_tool_result` usa `ast.parse`+`eval`
para calculator antes de regex.

- `dataset_preparer/agent/quality.py:80-94`: allows `missing_observation`/
  `empty_observation` samples as valid. Fix: add them to `valid` exclusion.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `missing_observation` y
`empty_observation` ahora hacen `valid=False`.

- `dataset_preparer/agent/moe_data.py`: relies on non-existent
  `get_observation_index`/`get_observation_end_index`; relabel using
  `<|tool_result|>` start / `<|end|>`/`<|assistant|>` end.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Reemplazado `get_observation_index`/`get_observation_end_index`
por `get_tool_result_index`/`get_end_index`.

- `dataset_preparer/thinking_engine.py`: regional codes (`fr-CH`,`rm`) lack
  config → English-fallback thinking. Fix: restrict detection to codes with
  full config / map to base language.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Añadido `_REGIONAL_MAP` con mapeos
(`fr-CH→fr`, `rm→de`, `gl→es`). `_get_lang_config` extrae suffix regional
antes de lookup.

- `dataset_preparer/thinking_generators.py:126`: answer falls back to whole
  `input_ids` → malformed double-tokenized string. Fix: derive
  question/answer from chat delimiters.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `answer` extraction ahora verifica
`isinstance(raw_ids, str)` antes de fallback a `input_ids`.

- `dataset_preparer/generate_thinking_data.py`: templates produce
  meta-commentary the validator rejects. Fix: generate real reasoning or exempt.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Templates reemplazados con patrones
de razonamiento real que pasan validación de calidad.

- `dataset_preparer/aiml/parser.py`: `<random>` discards surrounding text
  (return `prefix+opt`); thinking-strip regex targets `<think>` not
  `<thinking>`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Thinking-strip regex corregido a
`<thinking>`. Random surrounding text ahora preservada como prefijo de
cada opción.

- `dataset_preparer/web/scraper.py`: dead path dedupe (`parsed.path` never in
  `visited`); drops query string so `?page=2` never fetched.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `_normalize_url` preserva query strings.
Dedupe funciona con URLs completas (path+query).

- `commons/registry/model_merge.py`: div-by-zero when weights sum to 0; only
  adjacent pairs checked (A,C mismatch missed). Fix: guard zero; check all pairs.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Zero-weight fallback a uniformes.
`validate_compatibility` ahora verifica todos los pares `(i, j)` donde `i < j`.

- `commons/registry/model_registry.py`: `validate_compatibility` returns False
  when architecture metadata missing → blocks valid merges. Fix: fall back to
  state_dict key/shape comparison.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `return False` eliminado; ahora
procede a comparación de state_dict cuando falta metadata de arquitectura.

- `commons/language_utils.py:403`: unknown text returns `'en'` (docstring says
  `'unknown'`). Fix: `return best_lang if best_score>0 else 'unknown'`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Retorna `'unknown'` cuando
`best_score <= 0`.

- `commons/dialogue/dialogmanager.py`: `:359` `min_length` word-count gate
  discards valid short answers; `:103` `torch.isclose` tensor used in `if`
  (use `.item()`).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `torch.isclose(...).item()` añadido.
(Se verificó que `min_length` no existe en el código actual.)

- `commons/utils/device_utils.py:251`: overhead uses fixed 4 bytes vs
  `bytes_per_param`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Usa `bytes_per_param` variable en
lugar de hardcoded `4`.

- `APP_CACHE_VIEWER/cache_viewer.py`: dead unreachable 2nd `except` referencing
  undefined `self._lock`; latent `len(None)` if `token_lengths=None`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** 2nd `except` eliminado. `token_lengths`
verificación con `self._token_lengths and` antes de `idx < len(...)`.

- `models/exported/convert_onnx.py`: assumes HF `GPT2LMHeadModel` — project
  uses custom `GPT2Transformer` → load fails on native checkpoints. Fix: trace
  the project model or document HF-only.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Intenta cargar como modelo del proyecto
(`ChatModel`) primero, fallback a HuggingFace `GPT2LMHeadModel`.

- `commons/dialogue/dialogmanager.py:396` (already listed) + `chatdataset.py`
  returns strings requiring custom collate (currently dead code).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `ChatDataset` reescrito para aceptar
`tokenizer` + `samples`, retornar `(input_ids, output_ids)` como listas de ints,
compatible con `collate_fn` del Trainer. Clase usable por si se reactiva.

---

## LOW / minor
- `commons/tools/platform_detector.py`: `get_shell_config('bash')` returns
  Linux config on macOS (first match wins). Fix: match name + platform.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `get_shell_config` ahora busca
`config.name == shell_lower and config.platform == current_platform` primero,
con fallback a cualquier coincidencia de nombre.

- `dataset_preparer/contamination/audit.py`: `or`-chain misreports zero counts.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** Ahora usa variable `final_count` con
cadena completa de fallback (leakage→language→dedup→balance→quality→noise→original).

- `dataset_preparer/contamination/balance.py`: non-reproducible RNG
  (`hash(source)` salted). Fix: deterministic digest (md5).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `hash(source)` reemplazado por
`int(hashlib.md5(source.encode()).hexdigest()[:8], 16)`.

- `models/exported/convert_gguf.py`: `bf16` branch dead (not in QUANT_TYPES).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `"bf16": GGMLQuantizationType.BF16`
añadido a `QUANT_TYPES`.

- `ServerFastAPI/routers_api.py`: non-stream `/api/generate` skips
  `remove_excessive_repetition` (inconsistent with other endpoints).

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `remove_excessive_repetition(text)`
añadido después de `truncate_text_by_stop` en el endpoint no-stream.

- `commons/tools/tool_executor.py`: `sanitize_command` dead code;
  `ShellSecurity.validate_command` ignores `SAFE_COMMANDS_*` allowlists.

MIMO_V25: **NO IMPLEMENTADO — REQUIERE DECISIÓN DE DISEÑO.**
`ShellSecurity` tiene listas de comandos seguros (`SAFE_COMMANDS_*`) y una función
`validate_command`, pero en la práctica `validate_command` nunca las usa: las listas
existen pero no se consultan. La función `sanitize_command` existe pero nadie la llama.

Arreglar esto requiere un rediseño deliberado de la política de seguridad — decidir
exactamente qué comandos son seguros para cada plataforma, cómo manejar comandos que
no están en la lista, si se bloquea o se advierte, etc. No es un bug que cause crashes
o corrompa datos; es una decisión de diseño de seguridad que debe tomarse conscientemente.
Si se arregla mal, podría bloquear herramientas legítimas o dejar pasar peligrosas.
Requiere una decisión del desarrollador sobre la política de seguridad.

- `dataset_preparer/source_validators.py`: `fix()` never truncates to
  `MAX_LENGTH` though `classify` reports `truncated`.

MIMO_V25: **CONFIRMADO. IMPLEMENTADO.** `fix()` ahora trunca strings a
`self.MAX_LENGTH` después de `_clean_text()`.

---

## Implementation order
1. **Blocking correctness/crashes**: #1 CPU+GPU, #5 filters, #6 dedup, #7
   balance, #8 hf typo, #9 naming, #10 streaming, #11 gguf quantizer, #2/#3 MoE
   wiring.
2. **Security/functionality**: tools registry + permission/dry_run gating,
   `envAIModels` naming fix.
3. **Export correctness**: `model_export.py` XE1/XE2/XE3/XE4, `convert_onnx.py`
   custom model.
4. **Data-preparer column/format integrity**: BUG1/BUG2/BUG3/BUG4, migrator
   double `<|final|>`, agent thinking insertion + multilingual.
5. **Quality/validation**: `thinking_quality`, `agent/quality`,
   `language_utils`, `dialogmanager` gates, `model_merge` robustness.
6. Run `pytest tests/` after each group.
