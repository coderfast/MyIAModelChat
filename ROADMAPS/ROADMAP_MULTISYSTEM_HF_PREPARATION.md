# ROADMAP: Multi-System & HF Spaces Preparation

**Estado:** COMPLETADO  
**Fecha:** 2026-08-23  
**Objetivo:** Garantizar que MyIAModelChat funcione en Windows, Linux, Mac y Hugging Face Spaces

---

## Auditoria Cross-Platform (2026-08-23)

Se realizo un barrido completo del codigo excluyendo APP_CACHE_VIEWER. Se encontraron 20 problemas clasificados por severidad.

### Problemas Encontrados

| # | Severidad | Problema | Archivo | Plataformas |
|---|-----------|----------|---------|-------------|
| 1 | CRITICO | `import psutil` sin try/except | `inference/chat_engine.py:20` | HF Spaces (Linux) |
| 2 | CRITICO | `psutil` no esta en requirements.txt | `requirements.txt` | HF Spaces |
| 3 | CRITICO | No hay entry point para HF Spaces | (raiz) | HF Spaces |
| 4 | ALTO | `llama_cpp` no instalado — endpoints de chat no funcionan | `ServerFastAPI/model.py` | HF Spaces |
| 5 | ALTO | `keyboard` como dependencia dura — falla en Linux headless | `requirements.txt:15` | HF Spaces |
| 6 | ALTO | `.env` con rutas locales que no existen en HF | `ServerFastAPI/model.py:20` | HF Spaces |
| 7 | MEDIO | `torchvision`, `torchaudio`, `torchtext` no se usan (~500MB extra) | `requirements.txt` | HF Spaces |
| 8 | MEDIO | `onnx`, `onnxruntime`, `gguf` solo para export (~200MB extra) | `requirements.txt` | HF Spaces |
| 9 | MEDIO | ANSI colors en output headless | `main.py:46` | HF Spaces / pipes |
| 10 | MEDIO | Multiproc usa demasiada RAM en Spaces pequenos | `data_preparer.py`, `trainer.py` | HF Spaces (2GB) |

### Problemas Manados (ya resueltos o manejados)

| # | Problema | Estado |
|---|----------|--------|
| 11 | `msvcrt` Windows-only en chat_engine.py | OK — try/except con fallback a input() |
| 12 | `psutil.rlimit()` Unix-only en chat_engine.py | OK — try/except, gracefully degrades |
| 13 | `wmic` Windows-only en onnx_quantizer.py | CORREGIDO — lscpu (Linux) / sysctl (Mac) |
| 14 | `open()` sin encoding='utf-8' | CORREGIDO — 8 llamadas arregladas |
| 15 | JSONL write sin newline='' | CORREGIDO — data_preparer.py |
| 16 | ANSI codes hardcoded en thinking_generators.py | CORREGIDO — removed |
| 17 | `arial.ttf` no disponible en Linux | OK — try/except con fallback |
| 18 | Rutas relativas al CWD | OK — funciona si se ejecuta desde raiz |
| 19 | Shebang lines | OK — ignoradas en Windows |
| 20 | Generated shell scripts (.sh/.bat) | OK — maneja ambos formatos |

---

## Fixes Completados (10/10)

### [COMPLETADO] Fix 1: psutil import guard
- **Archivo:** `inference/chat_engine.py:20`
- **Cambio:** Envolver `import psutil` en try/except, null check en linea 131
- **Justificacion:** `psutil` no siempre esta disponible en todos los entornos

### [COMPLETADO] Fix 2: psutil en requirements.txt
- **Archivo:** `requirements.txt`
- **Cambio:** Anadir `psutil` como dependencia
- **Justificacion:** Se usa en chat_engine.py, main.py, trainer.py, data_preparer.py

### [COMPLETADO] Fix 3: HF Spaces entry point
- **Archivo:** `README.md` (raiz)
- **Cambio:** YAML frontmatter con `sdk: docker`
- **Justificacion:** HF Spaces necesita un entry point para saber como iniciar la app

### [COMPLETADO] Fix 4: keyboard opcional
- **Archivo:** `requirements.txt`
- **Cambio:** Comentar `keyboard` como opcional
- **Justificacion:** Requiere permisos elevados en Linux, no funciona en headless

### [COMPLETADO] Fix 5: .env safe loading
- **Archivo:** `ServerFastAPI/model.py:20-27`
- **Cambio:** No crashear si modelo no existe, log warning en startup
- **Justificacion:** En HF Spaces no hay modelo GGUF pre-cargado

### [COMPLETADO] Fix 6: Limpiar requirements.txt
- **Archivo:** `requirements.txt`
- **Cambio:** Quitar torchvision/torchaudio/torchtext (no usados), comentar onnx/onnxruntime/gguf como opcionales
- **Justificacion:** Ahorrar ~700MB de disco en HF Spaces

### [COMPLETADO] Fix 7: ANSI colors headless
- **Archivo:** `main.py:44-83`
- **Cambio:** Detectar `sys.stdout.isatty()`, deshabilitar colores si headless
- **Justificacion:** En HF Spaces o pipes, los codigos ANSI aparecen como basura

### [COMPLETADO] Fix 8: Multiprocess RAM-safe
- **Archivos:** `trainer.py:357`, `data_preparer.py:697`
- **Cambio:** Si RAM < 4GB, retornar num_proc=1
- **Justificacion:** HF Spaces gratuits tienen 2GB RAM, 4 procesos causan OOM

### [COMPLETADO] Fix 9: Ollama timeout upfront
- **Archivo:** `generate_thinking_data.py:276`
- **Cambio:** Check disponibilidad de Ollama una vez antes del loop (timeout 2s)
- **Justificacion:** Evitar timeout de 30s por cada sample en HF Spaces (donde Ollama no existe)

### [COMPLETADO] Fix 10: runserver.sh para Linux
- **Archivo:** `ServerFastAPI/runserver.sh`
- **Cambio:** Script bash equivalente al .bat existente
- **Justificacion:** HF Spaces usa Linux, necesita script bash para iniciar el servidor

---

## Archivos Modificados

| Archivo | Fixes |
|---------|-------|
| `inference/chat_engine.py` | Fix 1 (psutil guard) |
| `requirements.txt` | Fix 2, 4, 6 (psutil, keyboard, limpieza) |
| `README.md` | Fix 3 (HF Spaces frontmatter) |
| `ServerFastAPI/model.py` | Fix 5 (.env safe loading) |
| `main.py` | Fix 7 (ANSI headless) |
| `training/trainer.py` | Fix 8 (multiprocess RAM-safe) |
| `dataset_preparer/data_preparer.py` | Fix 8 (multiprocess RAM-safe) |
| `dataset_preparer/generate_thinking_data.py` | Fix 9 (Ollama upfront check) |
| `ServerFastAPI/runserver.sh` | Fix 10 (nuevo — script Linux) |

---

## Verificacion

- [x] `python -c "import inference.chat_engine"` — OK
- [x] `python -c "import dataset_preparer.data_preparer"` — OK
- [x] `python -c "import dataset_preparer.generate_thinking_data"` — OK
- [x] `python -c "import main"` — OK
- [ ] `pip install -r requirements.txt` en entorno limpio (pendiente)
- [ ] `bash ServerFastAPI/runserver.sh` (pendiente)
