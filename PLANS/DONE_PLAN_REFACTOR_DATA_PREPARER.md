# PLAN: Refactorizacion de Fuentes de Datos a Scripts Independientes de Markdown

## Estado: COMPLETADO ✅

## Objetivo
Convertir cada fuente de datos en un script independiente que genera archivos `.md` en `dataset_preparer/markdowns/`. El `DataPreparer` solo lee markdowns de esa carpeta como input para la cache de entrenamiento.

## Pipeline Implementado

**FASE 1 (scripts independientes):**
```
aiml_to_md.py  → dataset_preparer/markdowns/aiml/*.md
pdf_to_md.py   → dataset_preparer/markdowns/pdf/*.md
epub_to_md.py  → dataset_preparer/markdowns/epub/*.md
web_to_md.py   → dataset_preparer/markdowns/web/*.md
hf_to_md.py    → dataset_preparer/markdowns/hf/*.md
csv_to_md.py   → dataset_preparer/markdowns/csv/*.md
```

**FASE 2 (DataPreparer refactorizado):**
```
DataPreparer.prepare() → Lee TODOS los .md → Combine → Tags → Contamination → Thinking → Agent → BPE → Cache
```

## Archivos Modificados/Creados

### Creados
| # | Archivo | Estado |
|---|---------|--------|
| 1 | `dataset_preparer/markdowns/` | ✅ CREADO |
| 2 | `dataset_preparer/aiml/aiml_config.json` | ✅ CREADO |
| 3 | `dataset_preparer/pdf/pdf_config.json` | ✅ CREADO |
| 4 | `dataset_preparer/epub/epub_config.json` | ✅ CREADO |
| 5 | `dataset_preparer/web/web_config.json` | ✅ CREADO |
| 6 | `dataset_preparer/hf/hf_config.json` | ✅ CREADO |
| 7 | `dataset_preparer/csv/csv_config.json` | ✅ CREADO |
| 8 | `tests/test_markdown_generation.py` | ✅ CREADO |

### Modificados
| # | Archivo | Cambios |
|---|---------|---------|
| 9 | `dataset_preparer/aiml/aiml_to_md.py` | Tokens GPT-2 standard |
| 10 | `dataset_preparer/pdf/pdf_to_md.py` | Tokens GPT-2 standard |
| 11 | `dataset_preparer/epub/epub_to_md.py` | Tokens GPT-2 standard |
| 12 | `dataset_preparer/web/web_to_md.py` | Tokens GPT-2 standard |
| 13 | `dataset_preparer/hf/hf_to_md.py` | Tokens GPT-2 standard |
| 14 | `dataset_preparer/csv/csv_to_md.py` | Tokens GPT-2 standard |
| 15 | `main.py` | Flag `--generate-md` agregado |
| 16 | `dataset_preparer/data_preparer.py` | Metodo `_load_markdowns()` agregado |
| 17 | `AGENTS.md` | Documentacion actualizada |

### Eliminados
| # | Archivo | Razon |
|---|---------|-------|
| 18 | `dataset_preparer/migrator.py` | No necesario (sin tokens legacy) |

## Tokens GPT-2 Estandar

| Token | Uso |
|-------|-----|
| `<\|problem\|>` | Prefijo de pregunta/problema |
| `<\|thinking\|>` | Prefijo de razonamiento |
| `<\|final\|>` | Prefijo de respuesta final |
| `<\|user\|>` | Turno del usuario (chat) |
| `<\|assistant\|>` | Turno del asistente (chat) |
| `<\|end\|>` | Fin de mensaje/turno |
| `<\|system\|>` | Instrucciones de sistema |
| `<\|sep\|>` | Separador intra-mensaje |
| `<\|tool_result\|>` | Resultado de herramienta |
| `<tool_call>` | Inicio de tool call |
| `</tool_call>` | Fin de tool call |

## Formato por Fuente

| Fuente | Formato | Ejemplo |
|--------|---------|---------|
| AIML | chat | `<\|user\|>{pattern}<\|end\|><\|assistant\|>{template}<\|end\|>` |
| PDF | text completion | `<\|problem\|>{paragraph}<\|final\|>` |
| EPUB | text completion | `<\|problem\|>{chapter_text}<\|final\|>` |
| Web | text completion | `<\|problem\|>{content}<\|final\|>` |
| HF (chat) | chat | `<\|user\|>{human}<\|end\|><\|assistant\|>{gpt}<\|end\|>` |
| HF (plain) | text completion | `<\|problem\|>{text}<\|final\|>` |
| CSV | chat | `<\|user\|>{input}<\|end\|><\|assistant\|>{output}<\|end\|>` |

## Uso CLI

```bash
# Paso 1: Generar markdowns desde fuentes crudas
python main.py --generate-md --aiml
python main.py --generate-md --pdf --epub
python main.py --generate-md --hf --csv
python main.py --generate-md --aiml --hf --pdf --epub --web --csv  # todos

# Paso 2: Preparar cache desde markdowns
python main.py --prepare-data

# O combinado:
python main.py --generate-md --prepare-data --aiml --hf --pdf --epub

# Paso 3: Entrenar
python main.py --train --epochs 30
```

## Tests

```bash
# Tests de generacion de markdown
pytest tests/test_markdown_generation.py -v

# Tests de tokens
pytest tests/test_mode_tokens.py -v
```

---
*Plan completado: 2026-09-08*
*Estado: COMPLETADO*
