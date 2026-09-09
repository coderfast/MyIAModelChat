# PLAN: Revisión de Código — Bugs, Código Muerto y Mejoras

**Fecha:** 2026-09-09
**Proyecto:** MyIAModelChat
**Alcance:** Revisión completa del codebase (main.py, commons/, dataset_preparer/, training/, inference/, ServerFastAPI/, tests/)
**Estado:** COMPLETADO

---

## Resumen General

| Severidad | Ronda 1-3 | Ronda 4 (Nueva) | Total Implementados | Pendientes |
|-----------|-----------|-----------------|---------------------|------------|
| **CRÍTICO** | 4 | 0 | 4 | 0 |
| **ALTO** | 8 | 8 | 16 | 0 |
| **MEDIO** | 25+ | 11 | 36+ | 0 |
| **BAJO** | 5 | 4 | 9 | 0 |
| **REFACTOR** | 4 | 0 | 4 | 0 |
| **TOTAL** | **42** | **23** | **65** | **0** |

---

## 1. BUGS CRÍTICOS — COMPLETADOS

| # | Estado | Archivo | Fix |
|---|--------|---------|-----|
| 1.1 | ✅ | `trainer.py:1052` | `in_observation` → `in_tool_result` (NameError) |
| 1.2 | ✅ | `main.py:523-528` | Eliminado override incondicional de `agent_ratio` |
| 1.3 | ✅ | `main.py:753` | Comparación `thinking_loss_weight` corregida a `!= 0.5` |
| 1.4 | ✅ | `agent/thinking.py:182` | `eval()` protegido con whitelist AST + `__builtins__={}` |

---

## 2. BUGS ALTOS — COMPLETADOS

| # | Estado | Archivo | Fix |
|---|--------|---------|-----|
| 2.1 | ✅ | `trainer.py:3511` | Validación movida antes de `scheduler.step()` |
| 2.2 | ✅ | `data_preparer.py:2670` | Index mapping corregido en `_apply_noise_filter` |
| 2.3 | ✅ | `leakage.py:84-96` | Sampling de 5000 para datasets grandes |
| 2.4 | ✅ | `routers_v1.py:243` | `run_in_executor()` para carga de modelo |
| 2.5 | ✅ | `chat_engine.py:356`, `trainer.py:3228` | `weights_only=True` primero |
| 2.6 | ✅ | `routers_v1.py:56-91` | API key + validación de path |
| 2.7 | ✅ | `ServerFastAPI/utils.py:291` | `build_prompt_from_messages` unificado con tokens GPT-2 |
| 2.8 | ✅ | `data_preparer.py` | Text processing functions kept (used internally) |

---

## 3. CÓDIGO MUERTO / INALCANZABLE — COMPLETADOS

| # | Estado | Archivo | Fix |
|---|--------|---------|-----|
| 3.1 | ✅ | `main.py:684-685` | Eliminado `else: count = 0` inalcanzable |
| 3.2 | ✅ | `main.py:892-894` | Agregado `sys.exit(0)` tras `--chat` |
| 3.3 | ✅ | `main.py:730,739` | Eliminados re-imports redundantes de `TrainingConfig` |
| 3.4 | ✅ | `permission_manager.py:285-293` | Eliminado check `CRITICAL` inalcanzable |
| 3.5 | ✅ | `dialogmanager.py:38-49` | Eliminadas asignaciones duplicadas y atributos no usados |
| 3.6 | ✅ | `thinking_engine.py:1400-1411` | Eliminados 4 métodos legacy |
| 3.7 | ✅ | `model_export.py:355` | Eliminada re-asignación redundante de `stem` |
| 3.8 | ✅ | `model_export.py:506` | Corregidos strings de tokens (`<\|token\|>` en vez de `<token>`) |
| 3.9 | ✅ | `epub/web/pdf_to_md.py` | Eliminados branches `fmt == 'chat'` idénticos |
| 3.10 | ✅ | `main.py:18-19` | Eliminado set redundante de `CUDA_VISIBLE_DEVICES` |
| 3.11 | ✅ | `dialogmanager.py:181` | `print()` → `logger.debug()` |

---

## 4. IMPORTS NO USADOS — COMPLETADOS

| # | Estado | Archivo | Import eliminado |
|---|--------|---------|------------------|
| 4.1 | ✅ | `trainer.py` | `contextlib`, `shutil` |
| 4.2 | ✅ | `text_utils.py` | `os`, `Dict`, `Tuple`, `Optional` |
| 4.3 | ✅ | `language_utils.py` | `re`, `Optional` |
| 4.4 | ✅ | `tool_executor.py` | `json` |
| 4.5 | ✅ | `tool_registry.py` | `field` + `import os` redundante en `_list_directory()` |
| 4.6 | ✅ | `permission_manager.py` | `field` |
| 4.7 | ✅ | `chatdataset.py` | `torch` |
| 4.8 | ✅ | `model_registry.py` | `List`, `Optional` |
| 4.9 | ✅ | `ServerFastAPI/utils.py` | `inspect` |
| 4.10 | ✅ | `ServerFastAPI/model.py` | `onnxruntime as ort` |
| 4.11 | ✅ | `routers_v1.py` | `build_text_completion_response` |
| 4.12 | ✅ | `routers_api.py` | `build_text_completion_response` |
| 4.13 | ✅ | `tests/test_aiml_parser.py` | `parse_aiml_file` |
| 4.14 | ✅ | `tests/test_chatmodel.py` | `MagicMock` |
| 4.15 | ✅ | `tests/test_thinking_comprehensive.py` | `torch.nn as nn` |
| 4.16 | ✅ | `tests/test_mode_tokens.py` | `Trainer` (2 instancias) |

> **Nota:** `shutil` y `multiprocessing` en `data_preparer.py`, y `unicodedata` en `source_validators.py` fueron verificados como USADOS — no se eliminaron.

---

## 5. SEGURIDAD (ServerFastAPI) — COMPLETADOS

| # | Estado | Problema | Fix |
|---|--------|----------|-----|
| 5.1 | ✅ | Sin CORS | Agregado `CORSMiddleware` en `app.py` |
| 5.2 | ✅ | Pickle inseguro | `weights_only=True` primero en ambos archivos |
| 5.3 | ✅ | Recarga sin auth | API key + validación de path en `/models/reload` |
| 5.4 | ✅ | Error messages filtran paths | Mensajes genéricos en responses de error |
| 5.5 | ✅ | Sin validación temperature/top-p | `field_validator` en `schemas.py` |

> **Nota:** Rate limiting no implementado (requiere dependencia adicional `slowapi`). Se recomienda para producción.

---

## 6. TESTS — COMPLETADOS

| # | Estado | Archivo | Fix |
|---|--------|---------|-----|
| 6.1 | ✅ | `test_chatmodel.py:97` | Assertion tautológica corregida |
| 6.2 | ✅ | `test_chatmodel_mtp.py:154` | Assertion tautológica corregida |
| 6.3 | ✅ | `test_markdown_generation.py:242` | Assertion corregida para verificar directorios en disco |

---

## 7. MISCELLANEOUS — COMPLETADOS

| # | Estado | Archivo | Fix |
|---|--------|---------|-----|
| 7.1 | ✅ | `data_preparer.py:2055` | Source label `'hf'` → `'HuggingFace'` |
| 7.2 | ✅ | `language_utils.py:329,334,349` | Datos duplicados eliminados (lt, lv, et) |
| 7.3 | ✅ | `ServerFastAPI/schemas.py` | Validación de temperature (0-2) y top_p (0-1] |

---

## 8. REFACTORS DE DISEÑO (completados)

| # | Descripción | Estado |
|---|-------------|--------|
| 8.1 | Eliminar funciones duplicadas de `data_preparer.py` (321 líneas → importadas desde `commons/utils/text_utils.py`) | ✅ |
| 8.2 | Consolidar `TRAINING_CONFIG` dict y `TrainingConfig` dataclass (12 campos movidos al dataclass, dict eliminado) | ✅ |
| 8.3 | Extraer config de idiomas de `thinking_engine.py` a JSON (`thinking_engine_config.json`, 30 idiomas) | ✅ |
| 8.4 | Refactorizar construcción de `TrainingConfig` en `main.py` (función helper `_build_training_config`) | ✅ |

---

## 9. RONDA 4 — Bugs y Código Muerto (Nuevos)

### 9.1 Bugs HIGH — Completados

| # | Estado | Archivo | Línea(s) | Fix |
|---|--------|---------|----------|-----|
| 9.1.1 | ✅ | `training/loss.py` | 30-36 | MoE `gate_scores` se asignaban incorrectamente a `mtp_logits_list`. Corregido unpacking para distinguir `(logits, gate_scores)` de `(logits, mtp_logits)` |
| 9.1.2 | ✅ | `training/loss.py` / `trainer.py` | 348 / 1207 | Key mismatch: `agent_tool_result_accuracy` vs `agent_observation_accuracy`. Unificado a `'agent_observation_accuracy'` |
| 9.1.3 | ✅ | `dataset_preparer/data_preparer.py` | 2398 | NameError: variable `sources` indefinida. Corregido a `original_sources` |
| 9.1.4 | ✅ | `dataset_preparer/data_preparer.py` | 2550 | Index misalignment en `_apply_language_filter_contamination`: `zip(filtered_texts, sources)` emparejaba fuentes incorrectas tras filtrado |
| 9.1.5 | ✅ | `dataset_preparer/data_preparer.py` | 2376-2550 | Todos los filtros de contaminación perdían columnas del dataset (`language`, `thinking`, `has_tool_call`, `format_type`, `bpe_text`, `token_ids`). Corregido para preservar todas las columnas |
| 9.1.6 | ✅ | `commons/model/chatmodel_moe_mtp.py` | 61-88 | Return type mismatch: retornaba `(primary_logits, mtp_logits)` pero el contrato de `ChatModelMoE` es `(logits, gate_scores)`. Corregido para retornar `(logits, gate_scores)` con gate_scores como side-channel |
| 9.1.7 | ✅ | `commons/model/chatmodel_moe.py` | 209 | Zero-tensor fallback sin device: `torch.tensor(0.0)` causaba device mismatch. Corregido a `torch.tensor(0.0, device=...)` |
| 9.1.8 | ✅ | `ServerFastAPI/routers_v1.py` | 299-304 | Llamada síncrona bloqueante en handler async: `generate_response()` bloqueaba el event loop. Envuelto en `run_in_executor` |

### 9.2 Bugs MEDIUM — Completados

| # | Estado | Archivo | Línea(s) | Fix |
|---|--------|---------|----------|-----|
| 9.2.1 | ✅ | `training/trainer.py` | 444-450 | DDP `no_sync()` branch no acumulaba `total_loss`, produciendo avg_loss incorrecto. Agregado `total_loss += original_loss.item()` en rama `no_sync` |
| 9.2.2 | ✅ | `training/trainer.py` | 852-870 | `num_layers` hardcodeado a 4 en vez de leer `self.config.num_layers`. Reemplazado por `self.config.num_layers` |
| 9.2.3 | ✅ | `training/reporting.py` | 40-65 | `epoch_offset` calculado pero nunca usado (dead code). Eliminado |
| 9.2.4 | ✅ | `training/datasets.py` | 110 | `collate_fn` crash en batch vacío (`max()` en secuencia vacía). Agregado guard `if not batch: return ...` |
| 9.2.5 | ✅ | `main.py` | 217-218 | Override JSON de `num_cores`/`num_threads` roto (`default=None`). Corregido `default` a `0` con lógica de comparación |
| 9.2.6 | ✅ | `commons/tokenizer/bpe_tokenizer.py` | 242-249 | Export HF tokenizer con `merges` list vacía. Poblado `merges` desde SentencePiece model |
| 9.2.7 | ✅ | `dataset_preparer/thinking_engine.py` | 69 | `FALLBACK_CONFIG` crash si `'en'` falta de `LANGUAGE_CONFIG`. Agregado fallback seguro |
| 9.2.8 | ✅ | `ServerFastAPI/routers_v1.py` | 78 | Comparación de API key no era timing-safe. Reemplazado por `hmac.compare_digest()` |
| 9.2.9 | ✅ | `ServerFastAPI/routers_v1.py` | 92 | Path traversal validation débil. Reemplazado `startswith` por `Path.is_relative_to()` |
| 9.2.10 | ✅ | `commons/registry/model_merge.py` | 96-100 | Cada checkpoint cargado 2-3 veces. Reutilizado `state_dict` ya cargado |
| 9.2.11 | ✅ | `commons/model/chatmodel_mtp.py` | 101-120 | `get_mtp_head_accuracies` solo computaba primary accuracy. Agregado cálculo de MTP head accuracies |

### 9.3 Dead Code / Unused Imports — Completados

| # | Estado | Archivo | Fix |
|---|--------|---------|-----|
| 9.3.1 | ✅ | `training/trainer.py` | Eliminados imports no usados: `csv`, `IterableDataset`, `Optional`, `List`. Eliminado `KEYBOARD_AVAILABLE` (dead). Eliminado `LATEST_MODEL_FILE` (no referenciado) |
| 9.3.2 | ✅ | `commons/model/chatmodel.py` | Eliminado `import torch` no usado |
| 9.3.3 | ✅ | `commons/model/chatmodel_moe.py` | Eliminado `MoEConfig` dataclass (nunca instanciada). Eliminado `routing_stats` dict (nunca usado) |
| 9.3.4 | ✅ | `commons/dialogue/dialogmanager.py` | Eliminado parámetro `persona` (almacenado pero nunca usado). Eliminadas inicializaciones redundantes de `thinking_text`/`response_text` |
| 9.3.5 | ✅ | `dataset_preparer/aiml/parser.py` | Eliminados imports no usados: `import random`, `import unicodedata` |
| 9.3.6 | ✅ | `dataset_preparer/aiml/loader.py` | Eliminado `from xml.etree import ElementTree` no usado |
| 9.3.7 | ✅ | `ServerFastAPI/schemas.py` | Eliminado campo `ChatRequest.model` (nunca leído) |
| 9.3.8 | ✅ | `commons/tools/tool_executor.py` | Eliminados `SAFE_COMMANDS_*` lists y `sanitize_command()` (nunca usados). Eliminados branches dead `powershell`/`bash` |
| 9.3.9 | ✅ | `commons/registry/model_export.py` | Eliminados getters de tokens ficticios (`observation`, `action`, `context`, `answer`) que nunca existen |
| 9.3.10 | ✅ | `inference/chat_engine.py` | Eliminado `import time` no usado. Eliminado bloque dead `MAX_RAM_GB`. Eliminados config fields muertos (`thinking_enabled`, `thinking_max_tokens`, `draft_model_name`). Eliminados imports redundantes (`hashlib`, `json as _json`) |

---

## Archivos Modificados (42 total)

### Primera ronda (bugs críticos + altos)
1. `training/trainer.py` — in_observation fix, val_loss scheduler, pickle security
2. `main.py` — agent_ratio, thinking_loss_weight, dead code, imports, CUDA
3. `dataset_preparer/agent/thinking.py` — eval() security
4. `dataset_preparer/data_preparer.py` — index mismatch, source label
5. `dataset_preparer/contamination/leakage.py` — O(n²) memory guard
6. `ServerFastAPI/routers_v1.py` — event loop, model reload auth, imports
7. `inference/chat_engine.py` — pickle security
8. `ServerFastAPI/app.py` — CORS middleware
9. `ServerFastAPI/utils.py` — build_prompt_from_messages unification, import
10. `ServerFastAPI/routers_api.py` — import cleanup

### Segunda ronda (código muerto + imports + tests)
11. `commons/dialogue/dialogmanager.py` — dead attributes, print→debug
12. `commons/tools/permission_manager.py` — unreachable code, import
13. `commons/utils/text_utils.py` — imports
14. `commons/language_utils.py` — imports, duplicate data
15. `commons/tools/tool_executor.py` — import
16. `commons/tools/tool_registry.py` — imports
17. `commons/dataset/chatdataset.py` — import
18. `commons/registry/model_registry.py` — imports
19. `tests/test_chatmodel.py` — import, assertion
20. `tests/test_chatmodel_mtp.py` — assertion
21. `tests/test_markdown_generation.py` — assertion
22. `tests/test_thinking_comprehensive.py` — import
23. `tests/test_aiml_parser.py` — import

### Tercera ronda (refactors de diseño)
24. `ServerFastAPI/model.py` — onnxruntime import
25. `tests/test_mode_tokens.py` — Trainer imports
26. `dataset_preparer/thinking_engine.py` — eliminados ~1100 líneas de config inline, ahora importa desde JSON
27. `dataset_preparer/thinking_engine_config.json` — generado: 30 idiomas, 450 campos
28. `commons/registry/model_export.py` — redundant stem, token strings
29. `dataset_preparer/epub/epub_to_md.py` — identical branches
30. `dataset_preparer/web/web_to_md.py` — identical branches
31. `dataset_preparer/pdf/pdf_to_md.py` — identical branches
32. `ServerFastAPI/schemas.py` — temperature/top-p validation
33. `main.py` — función helper `_build_training_config()` reemplaza 100 líneas duplicadas
34. `dataset_preparer/data_preparer.py` — eliminadas 321 líneas de funciones duplicadas (importadas desde `commons/utils/text_utils.py`)
35. `training/trainer.py` — consolidación TRAINING_CONFIG dict → TrainingConfig dataclass (12 campos movidos, dict eliminado)
36. `tests/test_thinking_fixes.py` — eliminada referencia a TRAINING_CONFIG dict

### Cuarta ronda (bugs + dead code)
37. `training/loss.py` — MoE gate_scores unpacking fix, agent observation key fix
38. `training/datasets.py` — collate_fn empty batch guard
39. `training/reporting.py` — eliminado epoch_offset dead code
40. `commons/model/chatmodel.py` — eliminado import torch no usado
41. `commons/model/chatmodel_moe.py` — eliminado MoEConfig dead code, routing_stats dead code, zero-tensor device fix
42. `commons/model/chatmodel_mtp.py` — MTP head accuracies fix, hardcoded pad_token_id fix
43. `commons/model/chatmodel_moe_mtp.py` — return type mismatch fix
44. `commons/tokenizer/bpe_tokenizer.py` — HF export merges list fix
45. `dataset_preparer/data_preparer.py` — NameError fix, index misalignment fix, column preservation in filters
46. `dataset_preparer/aiml/parser.py` — eliminados imports no usados
47. `dataset_preparer/aiml/loader.py` — eliminado import no usado
48. `dataset_preparer/thinking_engine.py` — FALLBACK_CONFIG crash fix
49. `commons/tools/tool_executor.py` — eliminados SAFE_COMMANDS dead code, sanitize_command dead code
50. `commons/registry/model_export.py` — eliminados token getters ficticios
51. `commons/registry/model_merge.py` — checkpoint loading dedup
52. `ServerFastAPI/routers_v1.py` — async blocking fix, timing-safe key comparison, path traversal fix
53. `ServerFastAPI/schemas.py` — eliminado ChatRequest.model dead field
54. `inference/chat_engine.py` — eliminados imports y dead code
55. `training/config.py` — agregados campos num_layers, n_head, n_positions
56. `training/trainer.py` — eliminados unused imports, num_layers config fix, DDP loss accumulation fix

---

*Ronda 1-3 completada el 2026-09-09. Ronda 4 completada 2026-09-09.*
