# ROADMAP BUGS - MyIAModelChat

## Estado: TODOS CORREGIDOS

---

## v1 - Bugs corregidos (primera revisión)

| Bug | Severidad | Descripción | Fix | Archivos |
|-----|-----------|-------------|-----|----------|
| BUG-01 | Alta | `_validate_and_clean_sources()` no sincroniza atributos de fuente | Sync de atributos después de validar | `data_preparer.py` |
| BUG-02 | Media | `thinking_loss_weight=1.0` debería ser 0.5 | Cambiado default a 0.5 | `main_train.py`, `main.py` |
| BUG-03 | Baja | `_validate_and_clean_sources()` pierde datos si atributos son None | Guard si `cleaned_datasets` vacío | `data_preparer.py` |
| BUG-04 | Baja | `ThinkingGenerator.validate_thinking()` código muerto | Eliminado | `thinking_generators.py` |
| BUG-05 | Baja | Import de `validate_thinking` dentro del loop | Movido al inicio de la función | `data_preparer.py` |
| BUG-06 | Baja | `self.min_length` inicializado dos veces | Eliminada segunda asignación | `dialogmanager.py` |
| BUG-07 | Baja | Thinking inválido no se limpia del campo `thinking` | Limpieza de `thinking` y `thinking_text` | `data_preparer.py` |
| BUG-08 | Baja | `--thinking-depth` definido pero nunca usado | Conectado a generadores via `DEPTH_CONFIG` | `data_preparer.py`, `thinking_generators.py`, `*_thinking.py` |
| BUG-09 | Baja | `--thinking-model` no valida contra Ollama | `is_model_available()` verifica modelo específico | `thinking_generators.py`, `data_preparer.py` |

## v2 - Bugs corregidos (segunda revisión)

| Bug | Severidad | Descripción | Fix | Archivos |
|-----|-----------|-------------|-----|----------|
| BUG-10 | Media | `OllamaTeacher` cachea por prompt sin considerar max_tokens/temperature | Cache key: `(prompt, max_tokens, temperature)` | `thinking_generators.py` |
| BUG-11 | Media | `_compute_loss` aplica weight bajo a tokens `<think>` y `` | Weight solo a tokens ENTRE delimitadores | `main_train.py` |
| BUG-12 | Media | `dialogmanager` permite EOS dentro de fase thinking | EOS rechazado durante `in_thinking_phase` | `dialogmanager.py` |
| BUG-13 | Baja | `_validate_and_clean_sources` no genera reporte en error | `QualityReport` de error para fuentes con excepción | `data_preparer.py` |
| BUG-14 | Baja | `OllamaTeacher.is_model_available()` hace doble HTTP request | Un solo request, `is_available()` delega | `thinking_generators.py` |
| BUG-15 | Baja | `GenerateRequest` no tiene campo `include_thinking` | Campo agregado con default `False` | `main_chat.py` |

---

## Resumen Final

- **Total bugs encontrados**: 15
- **Total bugs corregidos**: 15
- **Tests**: 35/35 pasando
- **Estado**: Limpio
