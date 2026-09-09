# PLAN: Revisión de Código — Bugs, Código Muerto y Mejoras

**Fecha:** 2026-09-09
**Proyecto:** MyIAModelChat
**Alcance:** Revisión completa del codebase (main.py, commons/, dataset_preparer/, training/, inference/, ServerFastAPI/, tests/)

---

## Resumen General

| Severidad | Cantidad |
|-----------|----------|
| **CRÍTICO** | 4 |
| **ALTO** | 8 |
| **MEDIO** | 25+ |
| **BAJO** | 30+ |

---

## 1. BUGS CRÍTICOS

### 1.1 `NameError` en `trainer.py:1052` — Variable `in_observation` no definida

**Archivo:** `training/trainer.py`
**Línea:** 1052

En `_compute_loss()`, se usa `in_observation` pero nunca se define. Las variables locales solo definen `in_tool_call` (línea 1026) y `in_tool_result` (línea 1027). Causará `NameError` en tiempo de ejecución cada vez que un sample agentic carezca de tokens `tool_call`/`tool_result`.

**Fix probable:** La variable debería ser `in_tool_result`, ya que el comment dice "observation content" y `in_tool_result` es el estado que rastrea el bloque `<|tool_result|>`.

---

### 1.2 `agent_ratio` forzado a 0.0 incondicionalmente — `main.py:523-528`

**Archivo:** `main.py`
**Líneas:** 523-528

```python
if hasattr(args, 'agent_ratio') and args.agent_ratio > 0:
    args.agent_ratio = 0.0
    args.agent_enabled = False
    logger.info("Agent ratio forced to 0.0 (model too small for agentic patterns)")
```

El comment dice "model too small" pero **no hay chequeo del tamaño del modelo** (`embed_size`). Los flags `--generate-agent-data` y `--agent-enabled` siempre se desactivan silenciosamente. El usuario no puede habilitarlos.

**Fix:** Hacer el chequeo condicional al tamaño real del modelo, o eliminar el override y dejar que el usuario decida.

---

### 1.3 Comparación incorrecta de `thinking_loss_weight` — `main.py:753`

**Archivo:** `main.py`
**Línea:** 753

```python
thinking_loss_weight=args.thinking_loss_weight if args.thinking_loss_weight != 1.0 else json_config.thinking_loss_weight,
```

El default de argparse es `0.5`, pero la comparación es contra `1.0`. Si el usuario pasa `--thinking-loss-weight 1.0`, su valor se descarta silenciosamente y se usa el del JSON.

**Fix:** Cambiar la comparación a `!= 0.5` (el default de argparse), o usar un sentinel `None`.

---

### 1.4 Vulnerabilidad de seguridad: `eval()` en `generate_thinking_data.py:182`

**Archivo:** `dataset_preparer/generate_thinking_data.py`
**Línea:** 182

```python
eval(compile(ast.parse(expr, mode='eval'), '<expr>', 'eval'))
```

Ejecución arbitraria de código via `eval()` en argumentos de calculadora. Aunque `ast.parse` ofrece algo de protección, es un vector de ataque potencial.

**Fix:** Reemplazar con `ast.literal_eval()` o un evaluator seguro que solo soporte operaciones aritméticas básicas.

---

## 2. BUGS ALTOS

### 2.1 `val_loss` usado antes de calcularse — `trainer.py:3511`

**Archivo:** `training/trainer.py`
**Línea:** 3511

```python
val_metric = val_loss if val_dataloader is not None else train_loss
scheduler.step(val_metric)
```

El scheduler de plateau opera con datos stale porque `val_loss` se calcula después (línea 3520). El primer epoch usa `0.0` y los siguientes usan el valor del epoch anterior.

**Fix:** Mover `scheduler.step()` después del bloque de validación.

---

### 2.2 Index mismatch en `_apply_noise_filter` — `data_preparer.py:2670-2701`

**Archivo:** `dataset_preparer/data_preparer.py`
**Líneas:** 2670-2701

Cuando se mapean resultados filtrados de vuelta a sources, `sources[i]` usa el índice incorrecto porque `result.kept` solo contiene items que pasaron el filtro, desalineando los índices.

**Fix:** Mantener un mapping de índices originales o iterar con `zip` sobre `result.kept` y sus sources correspondientes.

---

### 2.3 O(n²) en `leakage.py:84-96` — Sin guardia de memoria

**Archivo:** `dataset_preparer/contamination/leakage.py`
**Líneas:** 84-96

La comparación pairwise de todos los texts para n-gram overlap puede agotar la memoria con datasets grandes (10K+ texts). Crea ~50M tuples para 100K texts de 500 palabras.

**Fix:** Agregar sampling, batching o límite de tamaño del dataset.

---

### 2.4 Event loop bloqueado en `routers_v1.py:243`

**Archivo:** `ServerFastAPI/routers_v1.py`
**Línea:** 243

`get_chat_engine_instance()` carga un modelo PyTorch de forma síncrona dentro de un `async def` generator, bloqueando todo el event loop de FastAPI.

**Fix:** Usar `asyncio.to_thread()` o `run_in_executor()` para la carga del modelo.

---

### 2.5 `weights_only=False` — Carga insegura de pickle

**Archivos:** `inference/chat_engine.py:365-367`, `training/trainer.py:3228`

Caen en `weights_only=False`, permitiendo ejecución arbitraria de código vía checkpoint files maliciosos.

**Fix:** Agregar validación de integridad (hash/checksum) antes de la carga, o eliminar el fallback.

---

### 2.6 Endpoint de recarga sin autenticación — `routers_v1.py:56-91`

**Archivo:** `ServerFastAPI/routers_v1.py`
**Líneas:** 56-91

`/v1/models/reload` permite a cualquiera recargar el modelo desde una ruta arbitraria del sistema.

**Fix:** Agregar autenticación (API key, token) al endpoint.

---

### 2.7 `build_prompt_from_messages` incompatible — `ServerFastAPI/utils.py:291` vs `chat_engine.py:76`

**Archivos:** `ServerFastAPI/utils.py:291`, `inference/chat_engine.py:76`

Dos implementaciones con el mismo nombre pero formatos completamente diferentes. La del servidor usa `"role: content"` mientras el modelo espera tokens GPT-2 (`<|user|>`, `<|assistant|>`). Resultado: output degradado en los endpoints `/api/chat`.

**Fix:** Unificar en una sola implementación, usar la de `chat_engine.py`.

---

### 2.8 Duplicación de funciones de text processing — `data_preparer.py:199-246`

**Archivo:** `dataset_preparer/data_preparer.py`
**Líneas:** 199-246

`clean_text`, `split_sentences`, `split_paragraphs`, `chunk_text_by_tokens` están duplicadas desde `commons/utils/text_utils.py`.

**Fix:** Eliminar las definiciones duplicadas e importar desde `commons/utils/text_utils.py`.

---

## 3. CÓDIGO MUERTO / INALCANZABLE

| Archivo | Línea(s) | Descripción |
|---------|----------|-------------|
| `main.py` | 684-685 | `else: count = 0` nunca se ejecuta (todos los cases del `sources` tienen `if/elif`) |
| `main.py` | 892-894 | "Program completed successfully" tras `--chat` nunca se alcanza (loop infinito) |
| `main.py` | 730, 739 | Re-import redundante de `TrainingConfig` (ya importado en línea 35) |
| `permission_manager.py` | 285-293 | Check `RiskLevel.CRITICAL` inalcanzable (ya retornado en líneas 228-236) |
| `dialogmanager.py` | 38-39 | Asignaciones de `problem_id`/`final_id` sobreescritas en líneas 47-48 |
| `thinking_engine.py` | 1400-1411 | Métodos legacy `_build_reasoning_es`, `_build_reasoning_en`, `_format_steps_es`, `_format_steps_en` nunca llamados |
| `model_export.py` | 355 | Re-asignación redundante de `stem` (ya calculado en línea 318) |
| `tool_executor.py` | 113-116 | `ShellSecurity.sanitize_command()` solo usado en tests, no en producción |
| `dialogmanager.py` | 40-46, 49 | 7 atributos de instancia nunca leídos: `user_id`, `assistant_id`, `tool_result_id`, `system_id`, `sep_id`, `thinking_id`, `thinking_mode_id` |
| `model_export.py` | 506 | Tokens generados incorrectamente: `<thinking >` en vez de `<|thinking|>` |
| `epub/epub_to_md.py` | 156-159, 165-168 | Branches `fmt == 'chat'` producen output idéntico al else |
| `web/web_to_md.py` | 127-129, 136-138 | Mismo problema: branches idénticos |
| `pdf/pdf_to_md.py` | 112-115, 121-124 | Mismo problema: branches idénticos |
| `data_preparer.py` | 199 | `split_sentences` definida pero nunca llamada dentro de `data_preparer.py` |
| Tests tautológicos | test_chatmodel.py:97, test_chatmodel_mtp.py:154 | `assert param.grad is not None or param.grad is None` — siempre True |
| Tests tautológicos | test_markdown_generation.py:242 | `assert d in expected_dirs` — siempre True |
| `test_think.py` | 240-241 | `importlib.util` crea module spec pero nunca lo ejecuta |
| `main.py` | 18-19 | Set redundante de `CUDA_VISIBLE_DEVICES` (duplicado en línea 622-624) |

---

## 4. IMPORTS NO USADOS

| Archivo | Línea(s) | Import |
|---------|----------|--------|
| `training/trainer.py` | 9 | `contextlib` |
| `training/trainer.py` | 23 | `shutil` |
| `dataset_preparer/data_preparer.py` | 15 | `shutil` (solo usado en 1 lugar, reemplazable con `os.replace`) |
| `dataset_preparer/data_preparer.py` | 22 | `multiprocessing as mp` |
| `dataset_preparer/source_validators.py` | 8 | `unicodedata` |
| `commons/dataset/chatdataset.py` | 11 | `torch` |
| `commons/utils/text_utils.py` | 9 | `os` |
| `commons/utils/text_utils.py` | 13 | `Dict`, `Tuple`, `Optional` de typing |
| `commons/language_utils.py` | 7 | `re` |
| `commons/language_utils.py` | 9 | `Optional` de typing |
| `commons/tools/tool_executor.py` | 4 | `json` |
| `commons/tools/tool_registry.py` | 7 | `field` de dataclasses |
| `commons/tools/tool_registry.py` | 132 | `os` (redundante, ya importado a nivel módulo) |
| `commons/tools/permission_manager.py` | 11 | `field` de dataclasses |
| `commons/registry/model_registry.py` | 8 | `List`, `Optional` de typing |
| `ServerFastAPI/utils.py` | 4 | `inspect` |
| `ServerFastAPI/model.py` | 13-15 | `onnxruntime as ort` |
| `ServerFastAPI/routers_v1.py` | 15 | `build_text_completion_response` |
| `ServerFastAPI/routers_v1.py` | 22 | `EXPORTED_DIR` |
| `ServerFastAPI/routers_api.py` | 16 | `build_text_completion_response` |
| `tests/test_aiml_parser.py` | 6 | `parse_aiml_file` |
| `tests/test_chatmodel.py` | 5 | `MagicMock` |
| `tests/test_thinking_comprehensive.py` | 7 | `torch.nn as nn` |
| `tests/test_mode_tokens.py` | 85, 104 | `Trainer` (en ambos tests) |

---

## 5. PROBLEMAS DE SEGURIDAD (ServerFastAPI)

| # | Problema | Archivo | Línea(s) |
|---|----------|---------|----------|
| 1 | **Sin CORS configurado** — cualquier origen puede llamar a la API | `app.py` | — |
| 2 | **Sin rate limiting** — vulnerable a DoS | `app.py` | — |
| 3 | **Pickle inseguro** — `weights_only=False` permite ejecución de código | `chat_engine.py` | 365-367 |
| 4 | **Recarga de modelo sin auth** — `/v1/models/reload` accesible públicamente | `routers_v1.py` | 56-91 |
| 5 | **Error messages filtran paths internos** — `str(e)` expone rutas del sistema | `routers_v1.py` | 88 |
| 6 | **Respuesta filtra metadata interna** — campo `raw` en respuestas API | `routers_api.py` | 95 |
| 7 | **`.env` sin validación** — puede sobreescribir variables críticas (`PATH`, `PYTHONPATH`) | `model.py` | 20-27 |
| 8 | **Sin validación de temperature/top-p** — valores extremos causan errores del modelo | `schemas.py` | — |

---

## 6. PROBLEMAS DE DISEÑO / CALIDAD

### 6.1 Archivos gigantes
- `data_preparer.py` tiene **3053 líneas** — debería dividirse en módulos más pequeños (`data_preparer_core.py`, `data_preparer_contamination.py`, `data_preparer_sources.py`)
- `trainer.py` tiene **4147 líneas** — considerar extraer helpers
- `thinking_engine.py` tiene un dict de 36 idiomas inline (~800 líneas) — mover a archivo JSON de configuración

### 6.2 Configuración dual
- `trainer.py` tiene tanto un dict `TRAINING_CONFIG` (constantes hardcodeadas) como un dataclass `TrainingConfig`. Ambos se usan para diferentes campos, lo cual es confuso y propenso a errores.

### 6.3 Código de debug en producción
- `dialogmanager.py:181` — `print()` debería ser `logger.debug()`

### 6.4 Configuración duplicada en main.py
- Las líneas 742-841 tienen dos branches (~50 líneas cada uno) de construcción de `TrainingConfig` casi idénticas. Debería extraerse a una función helper.

### 6.5 Default mismatches
- `early_stopping_patience`: argparse=`0`, TrainingConfig=`5` — confuso para el usuario

### 6.6 Strings de tokens incorrectos
- `model_export.py:506` genera `<thinking >` en vez de `<|thinking|>` debido a `token_name.replace('_', ' ')`

### 6.7 Atributos duplicados en bpe_tokenizer
- `self._thinking_id` y `self._thinking_mode_id` apuntan al mismo token ID (líneas 100-102)

### 6.8 Datos duplicados en language_utils
- Línea 329: Lithuanian indicators tiene `' yra '` duplicado
- Línea 336: Latvian indicators tiene `' ir '` duplicado
- Línea 351: Estonian indicators tiene `' selle '` duplicado

### 6.9 Valores de source inconsistentes
- `data_preparer.py:2055` usa `'hf'` (minúscula) mientras todas las demás fuentes usan Title case (`'AIML'`, `'PDF'`, `'EPUB'`, `'Web'`, `'CSV'`)

---

## 7. TESTS CON PROBLEMAS

| Archivo | Línea | Problema |
|---------|-------|----------|
| `test_chatmodel.py` | 97 | Assertion tautológica: `x is not None or x is None` siempre True |
| `test_chatmodel_mtp.py` | 154 | Mismo problema |
| `test_markdown_generation.py` | 242 | Assertion tautológica: `d in expected_dirs` siempre True |
| `test_thinking_comprehensive.py` | 390 | Pasa `None` como `self` a método de instancia |
| `test_thinking_comprehensive.py` | múltiples | `return` en vez de `pytest.skip()` — tests se reportan como passed sin ejecutarse |
| `test_mode_tokens.py` | 85, 104 | Import de `Trainer` nunca usado |
| `test_think.py` | 240-241 | Crea module spec pero nunca lo ejecuta |
| Varios tests | — | Runners standalone `run_tests()` duplican funcionalidad de pytest |

---

## 8. PLAN DE CORRECCIONES RECOMENDADAS

### Prioridad Inmediata (Críticos)

| # | Acción | Archivo | Bug |
|---|--------|---------|-----|
| 1 | Cambiar `in_observation` → `in_tool_result` | `trainer.py:1052` | #1.1 |
| 2 | Hacer condicional el override de `agent_ratio` | `main.py:523-528` | #1.2 |
| 3 | Corregir comparación a `!= 0.5` o usar sentinel | `main.py:753` | #1.3 |
| 4 | Reemplazar `eval()` con evaluator seguro | `generate_thinking_data.py:182` | #1.4 |

### Prioridad Alta

| # | Acción | Archivo | Bug |
|---|--------|---------|-----|
| 5 | Mover `scheduler.step()` después de validación | `trainer.py:3511` | #2.1 |
| 6 | Corregir index mapping en `_apply_noise_filter` | `data_preparer.py:2670` | #2.2 |
| 7 | Agregar guardia de tamaño en `leakage.py` | `leakage.py:84-96` | #2.3 |
| 8 | Usar `asyncio.to_thread()` para carga de modelo | `routers_v1.py:243` | #2.4 |
| 9 | Eliminar `weights_only=False` o agregar validación | `chat_engine.py:365`, `trainer.py:3228` | #2.5 |
| 10 | Agregar autenticación al endpoint de recarga | `routers_v1.py:56-91` | #2.6 |
| 11 | Unificar `build_prompt_from_messages` | `ServerFastAPI/utils.py` | #2.7 |
| 12 | Eliminar funciones duplicadas de text processing | `data_preparer.py:199-246` | #2.8 |

### Prioridad Media

| # | Acción | Archivo(s) |
|---|--------|------------|
| 13 | Eliminar imports no usados (~24 instancias) | Ver sección 4 |
| 14 | Eliminar código muerto (~18 instancias) | Ver sección 3 |
| 15 | Corregir assertions tautológicos en tests | Ver sección 7 |
| 16 | Eliminar `print()` de debug | `dialogmanager.py:181` |
| 17 | Agregar CORS y rate limiting | `app.py` |
| 18 | Agregar validación de temperature/top-p | `schemas.py` |
| 19 | Corregir strings de tokens en exportación | `model_export.py:506` |
| 20 | Eliminar datos duplicados en language_utils | `language_utils.py:329,336,351` |

### Prioridad Baja (Mejoras de diseño)

| # | Acción |
|---|--------|
| 21 | Dividir `data_preparer.py` en módulos más pequeños |
| 22 | Consolidar `TRAINING_CONFIG` dict y `TrainingConfig` dataclass |
| 23 | Extraer configuración de idiomas de `thinking_engine.py` a JSON |
| 24 | Refactorizar construcción de `TrainingConfig` en `main.py` (dos branches idénticos) |
| 25 | Corregir source labels inconsistentes (`'hf'` → `'HuggingFace'`) |
| 26 | Eliminar atributos no usados en `dialogmanager.py` (7 atributos) |

---

*Revisión generada automáticamente el 2026-09-09*
