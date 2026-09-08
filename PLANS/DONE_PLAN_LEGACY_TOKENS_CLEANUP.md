# PLAN: Limpieza de Tokens Legacy

## Objetivo
Eliminar los 6 tokens legacy del proyecto de forma segura, manteniendo backward compat
durante la transicion y finalmente eliminandolos completamente.

## Tokens Legacy Identificados

| Legacy | Canonical | Donde se usa |
|--------|-----------|-------------|
| `<thinking>` | `<|thinking|>` | migrator, tokenizer, cache_viewer, check_vocab, docs |
| `</thinking>` | `<|final|>` | migrator, tokenizer, cache_viewer, check_vocab, docs |
| `<\|context\|>` | `<\|problem\|>` | migrator, tokenizer, trainer, dialogmanager, tests |
| `<\|answer\|>` | `<\|final\|>` | migrator, tokenizer, trainer, dialogmanager, tests |
| `<observation>` | `<\|tool_result\|>` | migrator, tokenizer, trainer, dialogmanager, tests |
| `</observation>` | `<\|tool_result\|>` | migrator, tokenizer, trainer, dialogmanager, tests |

---

## FASE 1: Renombrar getters (tokenizer + callers)

### 1.1 En `commons/tokenizer/bpe_tokenizer.py`

**Cambiar nombres de metodos** (mantener backward compat con alias):

```python
# Nuevos nombres ( principales )
def get_problem_index(self) -> int:
    return self._problem_id

def get_final_index(self) -> int:
    return self._final_id

def get_tool_result_index(self) -> int:
    return self._tool_result_id

# Aliases legacy (deprecated, para backward compat)
get_context_index = get_problem_index
get_answer_index = get_final_index
get_observation_index = get_tool_result_index
get_observation_end_index = get_tool_result_id
```

**Eliminar del tokenizer:**
- `_legacy_thinking_id`, `_legacy_thinking_end_id`
- `_legacy_context_id`, `_legacy_answer_id`
- `_legacy_observation_id`, `_legacy_observation_end_id`
- `_context_id`, `_answer_id`, `_observation_id`, `_observation_end_id`
- `_resolve_id('<thinking>')`, `_resolve_id('</thinking>')`, etc.
- Los logs de `<thinking>`, `</thinking>` en `_initialize_special_token_ids()`

**Mantener en `skip_special_tokens`:**
- Solo los IDs de tokens GPT-2 estandar (problem, thinking, final, user, assistant, etc.)

### 1.2 En `training/trainer.py`

Cambiar todas las llamadas:
- `get_context_index()` → `get_problem_index()`
- `get_answer_index()` → `get_final_index()`
- `get_observation_index()` → `get_tool_result_index()`
- `get_observation_end_index()` → `get_tool_result_index()`
- Eliminar string checks de `<thinking>`, `<|context|>` en la deteccion (linea 530)

### 1.3 En `commons/dialogue/dialogmanager.py`

Cambiar:
- `get_context_index()` → `get_problem_index()`
- `get_answer_index()` → `get_final_index()`
- `get_observation_index()` → `get_tool_result_index()`
- `get_observation_end_index()` → `get_tool_result_index()`
- Renombrar variables: `context_id` → `problem_id`, `answer_id` → `final_id`, etc.

### 1.4 En `check_vocab.py`

- Eliminar `<thinking>`, `</thinking>`, `<|context|>`, `<|answer|>`, `<observation>`, `</observation>` de la lista de tokens a verificar
- Cambiar getters: `get_context_index()` → `get_problem_index()`, etc.

### 1.5 En `APP_CACHE_VIEWER/cache_viewer.py`

- Eliminar `'<thinking>'`, `'</thinking>'` del set `special_tokens` (linea 1459)

### 1.6 En `tests/test_mode_tokens.py` y `tests/test_chatmodel_mtp.py`

- Actualizar todos los tests que usan getters legacy
- Cambiar `get_context_index()` → `get_problem_index()`, etc.

---

## FASE 2: Eliminar migrator (o simplificar)

### 2.1 Decision
El `migrator.py` es necesario SOLO si hay datasets cacheados con tokens legacy.
Una vez que todos los datasets se regeneren con el nuevo formato, el migrator puede eliminarse.

### 2.2 Estrategia
1. **Mantener migrator** durante la fase de transicion (3-6 meses)
2. **Agregar flag** `--skip-migration` para datasets nuevos
3. **Eliminar migrator** cuando se confirme que no hay datasets legacy

### 2.3 En `data_preparer.py`
- Mantener `migrate_dataset_texts()` por ahora
- Agregar log cuando se detecten tokens legacy en datos

---

## FASE 3: Limpiar docs y archivos no-code

### 3.1 Archivos a actualizar
- `AGENTS.md` - Eliminar referencias a tokens legacy
- `README.md` - Actualizar tabla de tokens
- `USAGE.md` - Actualizar
- `DOCS/DOC_AIML_21.md` - Mantener (es documentacion AIML, no nuestro formato)
- `EPUB-QUICK-REFERENCE.md` - Actualizar
- `FICHA_HUGGINGFACE.md` - Actualizar
- `THINKING_DATASETS.md` - Actualizar
- `PLAN_SPECIAL_TOKENS.md` - Marcar como DONE
- `PLAN_BUGS_1.md` - Mantener (historial)

### 3.2 NO tocar
- `DOCS/DOC_AIML_21.md` - El `<thinking>` de AIML es un elemento XML estandar, NO es nuestro token

---

## FASE 4: Tests y verificacion

### 4.1 Tests a ejecutar
```bash
pytest tests/test_mode_tokens.py -v
pytest tests/test_chatmodel_mtp.py -v
pytest tests/test_thinking.py -v
pytest tests/test_bpe_tokenizer.py -v
pytest tests/test_tool_executor.py -v
pytest tests/test_permission_system.py -v
```

### 4.2 Verificacion manual
1. Entrenar un modelo pequeno (1-2 epochs) y verificar que los stats de thinking/agent funcionan
2. Hacer chat y verificar que el decode funciona correctamente
3. Verificar que el cache_viewer muestra los tokens correctamente

---

## FASE 5: Eliminacion final (futuro)

Despues de 3-6 meses con todo funcionando:
1. Eliminar aliases legacy del tokenizer
2. Eliminar `migrator.py` completo
3. Eliminar `_resolve_id()` para tokens legacy
4. Eliminar todos los checks de fallback
5. Actualizar AGENTS.md con el estado final

---

## Archivos a modificar (resumen)

| Archivo | Cambio | FASE |
|---------|--------|------|
| `commons/tokenizer/bpe_tokenizer.py` | Renombrar getters, eliminar legacy IDs | 1 |
| `training/trainer.py` | Cambiar llamadas a getters | 1 |
| `commons/dialogue/dialogmanager.py` | Cambiar llamadas a getters | 1 |
| `check_vocab.py` | Eliminar legacy tokens de la lista | 1 |
| `APP_CACHE_VIEWER/cache_viewer.py` | Eliminar legacy del display | 1 |
| `tests/test_mode_tokens.py` | Actualizar tests | 1 |
| `tests/test_chatmodel_mtp.py` | Actualizar tests | 1 |
| `dataset_preparer/migrator.py` | Mantener por ahora (futuro: eliminar) | 2 |
| `AGENTS.md` | Actualizar docs | 3 |
| `README.md` | Actualizar docs | 3 |
| `USAGE.md` | Actualizar docs | 3 |
| `PLANS/PLAN_SPECIAL_TOKENS.md` | Marcar como DONE | 3 |

---

## Orden de ejecucion recomendado

1. FASE 1.1: tokenizer (base)
2. FASE 1.2: trainer.py
3. FASE 1.3: dialogmanager.py
4. FASE 1.4: check_vocab.py
5. FASE 1.5: cache_viewer.py
6. FASE 1.6: tests
7. FASE 4: Ejecutar tests y verificar
8. FASE 3: Limpiar docs
9. FASE 2: Simplificar migrator (futuro)
10. FASE 5: Eliminacion final (futuro)

---

*Plan creado: 2026-09-08*
*Estado: PENDIENTE DE EJECUCION*
