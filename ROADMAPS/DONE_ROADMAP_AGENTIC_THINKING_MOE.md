# ROADMAP_AGENTIC - MyIAModelChat: Modelo End-to-End con Capacidades Agenticas

## Overview

Este roadmap define la implementación completa de capacidades **agenticas end-to-end** para MyIAModelChat. El modelo GPT-2 aprenderá a generar `<tool_call>`, `<tool_call>`, y `<tool_call>` como parte de su generación, permitiendo planificación, ejecución de herramientas y razonamiento con observaciones — todo entrenado via el dataset, sin dependencia de un orquestador externo.

**Sistema de Tokens Actual (ya implementado):**
- Mode tokens: `<|thinking|>`, `<|context|>`, `<|answer|>`
- Content tags: `<thinking>`, `</thinking>`

**Nuevos Tokens Agentic (a implementar):**
- `<tool_call>`, `</tool_call>` — tool call JSON
- `<tool_call>`, `</tool_call>` — action block
- `<tool_call>`, `</tool_call>` — observation result

**Nota sobre permisos de ejecución — Modelo de Seguridad en 3 Capas:**

La seguridad NO depende del modelo. El modelo SOLO genera `<tool_call>` con el JSON del tool call. La ejecución real de comandos es responsabilidad del **agente** que usa el modelo. La seguridad se implementa en 3 capas independientes:

**Capa 1 — PermissionManager (agente):**
- El agente DEBE pedir **permiso al usuario** antes de ejecutar cualquier comando que involucre shell, escritura de archivos, o red
- Herramientas de solo lectura (calculator, current_date, word_count) pueden ejecutarse sin permiso
- El usuario ve el comando completo y responde sí/no/dry-run
- `--auto-approve` para testing, `--dry-run` para validación

**Capa 2 — ShellSecurity (código):**
- Whitelist de comandos seguros por plataforma (solo lectura por defecto)
- Blacklist de patrones destructivos (rm -rf, sudo, Format-Volume, etc.)
- Timeout de 30 segundos por defecto
- Logging de todos los comandos ejecutados

**Capa 3 — Modelo (entrenamiento):**
- El modelo puede aprender a generar advertencias en `<thinking>` cuando detecta que un comando es potencialmente peligroso (ej: "Este comando eliminaría archivos, debería advertir al usuario")
- Esto es **informativo** — el thinking se muestra al usuario pero NO es un gate de seguridad
- El modelo NO tiene capacidad de bloquear o aprobar ejecuciones — solo el agente y el PermissionManager tienen esa autoridad

**Flujo completo:**
```
Modelo genera 
                                                    
```

Esto permite que el modelo sea agnóstico a la plataforma — el agente decide qué shell usar (PowerShell, Bash, zsh) según el SO detectado, y la seguridad siempre está en el agente, nunca en el modelo.

**Leyenda de Estado:** `TODO` → `IN_PROGRESS` → `BLOCKED` → `DONE`

**Leyenda de Prioridad:**
- **P0 (MVP)** — Bloqueante: debe hacerse primero, el agente no funciona sin esto
- **P1 (High)** — Core: requerido para que el agente genere tool calls
- **P2 (Medium)** — Calidad: mejora significativamente el comportamiento
- **P3 (Low)** — Pulido: nice-to-have

---

## Arquitectura Objetivo

```
User Input → Tokenizer → GPT-2 Model → Tokens generados
                                            ↓
                                    ¿Generó ?
                                   /              \
                                  NO               SÍ
                                   ↓                ↓
                              Respuesta        Parsear tool_call
                              final           ┌─────────────┐
                                              │ tool_name    │
                                              │ arguments    │
                                              └──────┬──────┘
                                                     ↓
                                              PermissionManager
                                              (¿permiso del usuario?)
                                                /          \
                                              SÍ             NO
                                               ↓              ↓
                                         ToolExecutor    Rechazado
                                         ShellSecurity   (error al modelo)
                                         (¿comando seguro?)
                                           /       \
                                         SÍ         NO
                                          ↓           ↓
                                     subprocess   Bloqueado
                                          ↓        (error al modelo)
                                     <observation>
                                     resultado   </observation>
                                          ↓
                                     Feed back al modelo
                                          ↓
                                     Genera respuesta final
```

---

## Rol del Modelo vs Seguridad Real

**El modelo NO tiene autoridad de seguridad.** Su único rol es generar tokens. Sin embargo, puede contribuir a la seguridad de forma **informativa**:

| Capa | Responsable | ¿Puede bloquear ejecución? | Mecanismo |
|------|-------------|---------------------------|-----------|
| PermissionManager | Agente (código) | SÍ — pide permiso al usuario | `request_permission()` antes de ejecutar |
| ShellSecurity | ToolExecutor (código) | SÍ — valida comandos contra whitelist/blacklist | `validate_command()` antes de subprocess |
| Modelo thinking | GPT-2 (entrenado) | NO — solo genera texto informativo | Advertencias en `<thinking>` (ej: "Este comando es peligroso") |

**Entrenamiento del modelo para advertencias:**
- El dataset puede incluir ejemplos donde el thinking contiene advertencias antes de tool calls peligrosos
- Ejemplo: `<thinking>El usuario pide eliminar todos los archivos. Debo generar un tool_call, pero debo advertir que esto es destructivo y el usuario debe confirmar.</thinking>`
- Esto es **solo natural language** — no tiene efecto en la ejecución real
- La ventaja: el usuario ve la advertencia del modelo **antes** de que el PermissionManager pida confirmación

**Conclusión:** La seguridad es responsabilidad del agente (PermissionManager + ShellSecurity). El modelo puede ayudar a comunicar riesgos al usuario, pero nunca a bloquear o aprobar ejecuciones.

---

## Comparación: Estado Actual vs Objetivo

| Aspecto | Estado Actual | Objetivo Agente |
|---------|--------------|-----------------|
| **Tokens** | `` / `` | + `<tool_call>`, `</tool_call>`, `<tool_call>`, `</tool_call>`, `<tool_call>`, `</tool_call>` |
| **Generación** | thinking → respuesta | thinking → plan → tool_call → observation → respuesta |
| **Tool use** | No existe | El modelo decide qué herramienta usar |
| **Dataset** | `{thinking}{response}` | `{thinking}{plan}{tool_call}{observation}{response}` |
| **Inference** | Single-pass | Multi-turn con ejecución de herramientas |
| **Loop** | Sin loop | Loop plan→act→observe hasta completar tarea |

---

## Fase 0: Tokens + Fix Thinking — P0 (MVP)

**Prioridad:** P0 — Tokens de tool use + thinking real son base obligatoria
**Tiempo estimado:** 8-11 horas
**Riesgo:** Medio (modifica tokenizer, pipeline de datos y quality validator)

### Task 0.1: Añadir tokens de tool use al tokenizer
- **Archivo**: `dataset_preparer/data_preparer.py`
- **Ubicación**: Línea 1411, donde se define `user_symbols` en `_prepare_bpe_tokenizer_and_tokenize()`
- **Nota**: La ubicación original del roadmap decía `bpe_tokenizer.py:40`, pero los `user_symbols` se definen en `data_preparer.py:1411`
- **Cambio**: Añadir los 6 nuevos tokens a la cadena de `user_symbols`:
  ```python
  # ACTUAL (línea 1411):
  user_symbols = '--user_defined_symbols=<thinking>,</thinking>,<|context|>,<|answer|>,<|thinking|>'

  # NUEVO:
  user_symbols = '--user_defined_symbols=<thinking>,</thinking>,<|context|>,<|answer|>,<|thinking|>,<tool_call>,</tool_call>,<tool_call>,</tool_call>,<tool_call>,</tool_call>'
  ```
- **Métodos nuevos en `bpe_tokenizer.py`** (añadir después de los métodos de thinking existentes, ~línea 227):
  ```python
  def get_tool_call_index(self) -> int: ...
  def get_tool_call_end_index(self) -> int: ...
  def get_action_index(self) -> int: ...
  def get_action_end_index(self) -> int: ...
  def get_observation_index(self) -> int: ...
  def get_observation_end_index(self) -> int: ...
  def has_tool_call(self, text: str) -> bool: ...
  def split_tool_call(self, text: str) -> tuple: ...
  def extract_tool_response(self, text: str) -> str: ...
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.2: Actualizar ChatModel para soportar los nuevos tokens
- **Archivo**: `commons/model/chatmodel.py`
- **Cambio**: Ninguno en arquitectura — `vocab_size` ya viene del tokenizer
- Solo verificar que `GPT2Config(vocab_size=tokenizer.vocab_size)` se ajusta automáticamente
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.3: Pre-entrenar BPE con nuevos tokens
- **Archivo**: `dataset_preparer/data_preparer.py`
- **Ubicación**: `_prepare_bpe_tokenizer_and_tokenize()`
- **Cambio**: Verificar que los nuevos tokens `<tool_call>` se registran como `user_defined_symbols`
- **Verificación**:
  ```bash
  python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
  # Verificar que los 6 tokens aparecen en sentencepiece.vocab
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Problemas detectados en Thinking

| # | Problema | Severidad | Ubicación |
|---|---------|-----------|-----------|
| 1 | Thinking rule-based es meta-comentario inútil | ALTO | `dataset_preparer/*/thinking.py` |
| 2 | Quality validator demasiado permisivo | ALTO | `thinking_quality.py:55-101` |
| 3 | `thinking_loss_weight` = 0.5 debería ser 1.0 | MEDIO | `training/trainer.py:121` |
| 4 | Doble forward pass cada 50 batches | MEDIO | `training/trainer.py:533` |
| 5 | Columna `source` eliminada del dataset | ALTO | `training/trainer.py:1436` |
| 6 | `encode_with_thinking()` nunca se usa | BAJO | `bpe_tokenizer.py:227` |

### Task 0.4: Reescribir rule-based thinking con reasoning real
- **Archivos**: `dataset_preparer/aiml/thinking.py`, `csv/thinking.py`, `pdf/thinking.py`, `epub/thinking.py`, `web/thinking.py`, `hf/thinking.py`
- **Estado actual**: Los 6 generadores YA usan `ThinkingEngine` (NLP-based) como método primario, Ollama teacher como secundario, y fallbacks estructurados como terciario. El thinking rule-based original ya fue modernizado.
- **Cambio**: Verificar calidad del reasoning generado. Si el fallback rule-based sigue produciendo meta-comentarios, refinar las plantillas de fallback para incluir conectores causales y detalles específicos
- **Nota**: El `ThinkingEngine` en `thinking_engine.py` ya implementa análisis NLP con sentence tokenization, keyword extraction, y causal connectors
- **Ejemplo AIML** (antes vs después):
  ```python
  # ANTES (meta-comentario inútil):
  "El usuario saluda 'hello'. Debo responder amigable."

  # DESPUÉS (reasoning real):
  "La entrada 'hello' es un saludo en inglés. La respuesta apropiada es un saludo recíproco
  que establece el tono de la conversación. Respondo con un saludo amigable para mantener la interacción."
  ```
- **Ejemplo CSV** (antes vs después):
  ```python
  # ANTES:
  "La pregunta es: ¿Capital de Francia? La respuesta correcta es: París."

  # DESPUÉS:
  "La pregunta pide la capital de Francia. Francia es un país europeo cuya capital es París,
  ubicada en el norte sobre el río Sena. La respuesta es París."
  ```
- **Cada fuente** tiene su patrón de reasoning adaptado:
  | Fuente | Estrategia de reasoning |
  |--------|----------------------|
  | AIML | Analizar categoría del patrón → explicar por qué esa respuesta |
  | CSV | Conectar pregunta con respuesta usando conocimiento general |
  | PDF | Resumir contenido del chunk + explicar relevancia |
  | EPUB | Contextualizar capítulo + extraer información clave |
  | Web | Analizar contenido de página + relacionar con consulta |
  | HF | Si QA: derivar respuesta de pregunta. Si text: resumir+razonar |
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.5: Fortalecer quality validator
- **Archivo**: `dataset_preparer/thinking_quality.py`
- **Estado actual**: El archivo ya tiene 60+ META_PATTERNS en 20+ idiomas (líneas 49-156), con `has_steps` (STEP_INDICATORS), `has_vocabulary_diversity`, `has_answer_derivation`, `has_logical_connectors` (líneas 521-563). Score mínimo actual: 0.5 en `validate_thinking` (línea 567), 0.4 en `filter_low_quality` (línea 608)
- **Cambios necesarios**:
  - Subir score mínimo de 0.5 a 0.7 en `validate_thinking` (línea 567)
  - Subir `min_score` de 0.4 a 0.5 en `filter_low_quality` (línea 608)
  - Verificar que los patrones existentes cubren los casos del roadmap (ya cubiertos en su mayoría)
  - Reforzar `has_logical_connectors` con más conectores causales si es necesario
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.6: Cambiar `thinking_loss_weight` default a 1.0
- **Archivo**: `training/trainer.py`
- **Ubicación**: `TrainingConfig` línea 121 (también en `TRAINING_CONFIG` dict línea 72)
- **Cambio**: `thinking_loss_weight: float = 0.5` → `thinking_loss_weight: float = 1.0`
- **Razón**: Con 0.5 el modelo aprende MENOS thinking que respuesta. Con 1.0 aprende ambos por igual.
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.7: Eliminar doble forward pass en métricas
- **Archivo**: `training/trainer.py`
- **Cambio**: Reutilizar `outputs` del forward pass principal
  ```python
  # ANTES:
  def _compute_thinking_metrics(self, model, inputs, targets, device):
      with torch.no_grad():
          outputs = model(inputs)  # FORWARD PASS DUPLICADO

  # DESPUÉS:
  def _compute_thinking_metrics(self, model, inputs, targets, device, outputs=None):
      if outputs is None:
          with torch.no_grad():
              outputs = model(inputs)
  ```
- **Caller en training loop**: Pasar `outputs` del forward pass principal
- **Estado**: `DONE` — `_compute_thinking_metrics` ya acepta parámetro `logits=` (línea 722) y el caller lo pasa en línea 888

### Task 0.8: Preservar columna `source` en el dataset
- **Archivo**: `training/trainer.py`
- **Ubicación**: `_get_tokenized_dataset()` línea 1436
- **Cambio**:
  ```python
  # ACTUAL (línea 1436):
  PRESERVED_COLUMNS = ('input_ids', 'token_ids', 'question', 'answer', 'type', 'thinking')

  # NUEVO (añadir 'source' y 'has_tool_call'):
  PRESERVED_COLUMNS = ('input_ids', 'token_ids', 'question', 'answer', 'type', 'thinking', 'source', 'has_tool_call')
  ```
- **Razón**: Necesitamos `source` para estadísticas por fuente, `has_tool_call` para datos agentic
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.9: Limpiar código muerto
- **Archivo**: `commons/tokenizer/bpe_tokenizer.py` — marcar `encode_with_thinking()` con TODO o eliminar (línea 227, nunca se usa)
- **Nota**: `q_lower` en `csv/thinking.py:74` NO es código muerto — se usa en condicionales de fallback (líneas 76-84). No eliminar.
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 0.10: Tests de thinking real
- **Archivo**: `tests/test_thinking_fixes.py` (nuevo)
- **Tests**:
  - `test_rule_based_thinking_has_causal_link()` — AIML thinking contiene "porque" o "ya que"
  - `test_rule_based_thinking_connects_qa()` — CSV thinking conecta pregunta con respuesta
  - `test_quality_validator_rejects_meta()` — Validator rechaza meta-comentario
  - `test_quality_validator_accepts_real_thinking()` — Validator acepta reasoning genuino
  - `test_thinking_loss_weight_default()` — Default es 1.0
  - `test_source_column_preserved()` — Columna `source` sobrevive tokenización
  - `test_no_double_forward_pass()` — Solo 1 forward pass por batch
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Verificación Fase 0 (Fix Thinking)
```bash
# 1. Preparar datos con thinking (sin Ollama — rule-based mejorado)
python main.py --prepare-data --aiml --csv --generate-thinking

# 2. Verificar reasoning real en el dataset
python -c "
from datasets import Dataset
ds = Dataset.load_from_disk('dataset_cache/prepared_dataset')
for i in range(5):
    print(f'Sample {i}: {ds[i][\"input_ids\"][:200]}')
"

# 3. Entrenar 1 epoch
python main.py --train --use-cache --epochs 1

# 4. Tests
pytest tests/test_thinking_fixes.py -v
```

---

## Fase 1: Tool Registry — P0 (MVP)

**Prioridad:** P0 — Sin herramientas registradas no hay tool use
**Tiempo estimado:** 4-6 horas
**Riesgo:** Bajo

### Task 1.1: Crear `commons/tools/tool_registry.py`
- **Archivo**: `commons/tools/tool_registry.py` (nuevo)
- **Responsabilidad**: Registro centralizado de herramientas con detección automática de plataforma
- **Arquitectura**:
  ```python
  import platform
  from dataclasses import dataclass
  from typing import Callable

  @dataclass
  class ToolDef:
      name: str
      description: str
      parameters: dict       # JSON Schema
      func: Callable         # Python function to execute
      category: str          # 'search', 'computation', 'file', 'shell', 'api'
      platform: str | None   # None = cross-platform, 'windows'/'linux'/'darwin'

  class ToolRegistry:
      def __init__(self):
          self.tools: dict[str, ToolDef] = {}
          self.current_platform = platform.system().lower()  # 'windows', 'linux', 'darwin'

      def register(self, tool: ToolDef): ...
      def get(self, name: str) -> ToolDef: ...
      def list_tools(self) -> list[ToolDef]: ...
      def list_platform_tools(self) -> list[ToolDef]: ...
      def get_descriptions(self, include_platform: bool = False) -> str: ...
      def call(self, name: str, arguments: dict) -> str: ...
  ```
- **Herramientas iniciales**:
  | Tool | Descripción | Parámetros | Categoría | Plataforma |
  |------|-------------|------------|-----------|------------|
  | `calculator` | Operaciones matemáticas seguras | `expression: str` | computation | cross |
  | `web_search` | Búsqueda web (trafilatura) | `query: str` | search | cross |
  | `read_file` | Leer contenido de archivo | `path: str` | file | cross |
  | `list_directory` | Listar archivos en directorio | `path: str` | file | cross |
  | `current_date` | Obtener fecha/hora actual | (sin params) | computation | cross |
  | `word_count` | Contar palabras de texto | `text: str` | computation | cross |
  | `shell` | Ejecutar comando shell (auto-detecta plataforma) | `command: str` | shell | cross |
  | `powershell` | Ejecutar comando Windows PowerShell | `command: str` | shell | windows |
  | `bash` | Ejecutar comando Bash | `command: str` | shell | linux/darwin |
  | `get_platform` | Detectar sistema operativo actual | (sin params) | computation | cross |

- **Nota sobre permisos**: El modelo genera el `<tool_call>`, pero el **agente** es responsable de pedir permiso al usuario antes de ejecutar. Ver sección "Seguridad y Permisos" más abajo.
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 1.2: Crear `commons/tools/tool_executor.py`
- **Archivo**: `commons/tools/tool_executor.py` (nuevo)
- **Responsabilidad**: Parsear `<tool_call>` del output del modelo, ejecutar en plataforma correcta, generar `<tool_call>`
- **Shell Cross-Platform**:
  ```python
  import subprocess
  import platform
  import logging

  logger = logging.getLogger(__name__)

  def get_platform() -> str:
      """Retorna 'windows', 'linux', o 'darwin'."""

  def get_default_shell() -> str:
      """Retorna shell por defecto: 'powershell' en Windows, 'bash' en Linux/Mac."""

  def execute_shell(command: str, timeout: int = 30) -> str:
      """
      Ejecuta comando en la plataforma correcta:
      - Windows: powershell.exe -Command
      - Linux/Mac: /bin/bash -c
      Retorna stdout o stderr.
      """

  def execute_powershell(command: str, timeout: int = 30) -> str:
      """Ejecuta comando PowerShell (Windows)."""

  def execute_bash(command: str, timeout: int = 30) -> str:
      """Ejecuta comando Bash (Linux/Mac)."""

  def has_tool_call(text: str) -> bool:
      """Detecta si el texto contiene <thinking>."""

  def parse_tool_call(text: str) -> tuple[str, dict]:
      """Extrae (tool_name, arguments) de un tool_call."""

  def execute_tool_call(tool_call_text: str, registry: ToolRegistry) -> str:
      """Ejecuta un tool_call y retorna <observation>result</observation>."""

  def format_observation(result: str, max_length: int = 500) -> str:
      """Formatea el resultado como <observation>result</observation>, trunca si es largo."""
  ```
- **Seguridad Cross-Platform**:
  ```python
  class ShellSecurity:
      # ============================================
      # WINDOWS POWERSHELL - Comandos seguros (solo lectura)
      # ============================================
      SAFE_COMMANDS_WINDOWS = [
          # FileSystem - lectura
          'Get-ChildItem', 'Get-Content', 'Get-Item', 'Get-ItemProperty',
          'Get-ChildItem -Recurse', 'Test-Path', 'Resolve-Path',
          'Get-FileHash', 'Get-AuthenticodeSignature',
          # Procesos - lectura
          'Get-Process', 'Get-Service', 'Get-WmiObject', 'Get-CimInstance',
          # Sistema - lectura
          'Get-Host', 'Get-Date', 'Get-History', 'Get-Command', 'Get-Alias',
          'Get-Help', 'Get-Variable', 'Get-Option', 'Get-Debug',
          'Get-Error', 'Get-Warning', 'Get-Information',
          # Red - lectura
          'Test-Connection', 'Test-NetConnection', 'Get-NetAdapter',
          'Get-NetIPAddress', 'Get-DnsClientServerAddress',
          # Seguridad - lectura
          'Get-Acl', 'Get-Certificate', 'Get-AuthenticodeSignature',
          # Variables de entorno
          'Get-ChildItem Env:', 'Get-Content Env:PATH',
          # Utilidades
          'Select-String', 'Measure-Object', 'Sort-Object', 'Where-Object',
          'Format-Table', 'Format-List', 'Out-String', 'Out-GridView',
          'ConvertTo-Json', 'ConvertFrom-Json', 'ConvertTo-Csv',
          # Compatibilidad Unix (PowerShell Core en Windows)
          'cat', 'ls', 'dir', 'echo', 'pwd', 'whoami', 'date',
      ]

      # ============================================
      # LINUX/MAC - Comandos seguros (solo lectura)
      # ============================================
      SAFE_COMMANDS_UNIX = [
          # FileSystem - lectura
          'ls', 'll', 'la', 'l', 'tree', 'find', 'locate', 'which', 'whereis',
          'cat', 'head', 'tail', 'less', 'more', 'file', 'stat', 'du', 'df',
          'touch', # crear archivo vacío (inocuo)
          # Contenido - lectura
          'grep', 'egrep', 'fgrep', 'ag', 'rg', 'rgrep',
          'wc', 'diff', 'comm', 'cmp', 'md5sum', 'sha256sum',
          # Procesos - lectura
          'ps', 'top', 'htop', 'atop', 'pstree', 'pgrep',
          'uptime', 'w', 'who', 'last', 'lastb', 'lastlog',
          # Sistema - lectura
          'uname', 'hostname', 'id', 'groups', 'whoami', 'date', 'cal',
          'env', 'printenv', 'set', 'export', # solo lectura
          'free', 'vmstat', 'iostat', 'sar', 'lscpu', 'lscpu',
          'cat /proc/cpuinfo', 'cat /proc/meminfo', 'cat /proc/version',
          # Red - lectura
          'ifconfig', 'ip', 'ip addr', 'ip route', 'netstat', 'ss',
          'ping', 'dig', 'nslookup', 'host', 'traceroute', 'tracepath',
          'curl', 'wget', # solo lectura por defecto
          # Disk - lectura
          'mount', 'lsblk', 'fdisk -l', 'blkid', 'lsusb',
          # Utilidades
          'echo', 'printf', 'date', 'cal', 'bc', # calculadora
          'seq', 'yes', 'true', 'false',
          'base64', 'xxd', 'od', # hex dump
          'jq', # JSON parser
          'xargs', # con -n1 es seguro
      ]

      # ============================================
      # PATRONES BLOQUEADOS (destructivos/escritura)
      # ============================================
      BLOCKED_PATTERNS = [
          # Windows - destructivos
          r'Remove-Item\s+-Recurse', r'Remove-Item\s+-Force',
          r'del\s+/[sSqQfF]', r'del\s+/[aA]',
          r'Format-Volume', r'Initialize-Disk', r'Clear-Disk',
          r'Set-Content', r'Out-File', r'Tee-Object',
          r'Start-Process\s+-Verb\s+RunAs', # UAC bypass
          r'Invoke-WebRequest.*-OutFile', r'Invoke-RestMethod.*-OutFile',
          r'Set-ItemProperty', r'New-ItemProperty',
          
          # Linux/Mac - destructivos
          r'rm\s+-rf', r'rm\s+-r\s+-f', r'rmdir',
          r'mkfs', r'fdisk.*-w', r'parted.*mklabel',
          r'chmod\s+777', r'chmod\s+-R\s+777', r'chown',
          r'>\s*/dev/', r'dd\s+.*of=/dev/',
          r'sudo', r'su\s+-', r'su\s+root',
          r'curl.*\|\s*(ba)?sh', r'wget.*\|\s*(ba)?sh', # pipe to shell
          r'eval\s+', r'exec\s+',
          r'iptables', r'ufw', r'firewall-cmd',
          r'systemctl\s+(stop|disable|mask)',
          r'service\s+.*stop',
          
          # Cross-platform
          r'shutdown', r'reboot', r'poweroff', r'init\s+[06]',
          r'dd\s+', r'mkfs', r'format\s+',
      ]

      @classmethod
      def validate_command(cls, command: str, sys_platform: str) -> tuple[bool, str]:
          """Valida si un comando es seguro. Retorna (es_seguro, razon)."""

      @classmethod
      def sanitize_command(cls, command: str) -> str:
          """Limpia caracteres peligrosos del comando."""
  ```
- **Restricciones de seguridad**:
  - **Timeout**: 30 segundos por defecto, configurable
  - **Whitelist**: Solo comandos de lectura por defecto (configurable)
  - **Blacklist**: Patrones destructivos bloqueados (ver arriba)
  - **Logging**: Todos los comandos ejecutados se registran
  - **Output limit**: Máximo 500 caracteres, truncar si es más largo
  - **Permisos**: ToolExecutor consulta a PermissionManager antes de ejecutar shell/file writes
  - **Dry-run**: Opción `--dry-run` para mostrar qué se ejecutaría sin ejecutar
  - **Flujo**: `execute_tool_call() → PermissionManager.request_permission() → ShellSecurity.validate_command() → subprocess`
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 1.3: Integrar ToolRegistry en ChatEngine
- **Archivo**: `inference/chat_engine.py`
- **Ubicación**: Constructor de `ChatEngine`
- **Cambio**:
  ```python
  self.tool_registry = ToolRegistry()
  self.tool_registry.register(calculator_tool)
  self.tool_registry.register(web_search_tool)
  # ... etc
  self.tool_executor = ToolExecutor(self.tool_registry)
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

---

## Fase 2: Dataset de Entrenamiento Agente — P1 (High)

**Prioridad:** P1 — Sin datos de tool use, el modelo no aprende
**Tiempo estimado:** 8-12 horas
**Riesgo:** Medio (calidad del teacher model importa)

### Task 2.1: Crear `dataset_preparer/agent/thinking.py` — Generador de datos agentic
- **Archivo**: `dataset_preparer/agent/thinking.py` (nuevo)
- **Clase**: `AgentThinkingGenerator(ThinkingGenerator)`
- **Estrategia**: Usar OllamaTeacher para generar datos con tool calls
- **Prompt template para el teacher**:
  ```
  Analiza esta pregunta y decide si necesita usar herramientas.

  Pregunta: {question}
  Respuesta conocida: {answer}

  Si necesita herramientas, genera:
  <thinking>
  {reasoning about what tool is needed and why}
  <tool_call>
  {"name": "tool_name", "arguments": {"param": "value"}}
  </tool_call>
  <observation>{simulated result}</observation>
  </tool_call>
   {final answer}
  </observation>

  Si NO necesita herramientas, genera:
  <thinking>
  {reasoning}
  </tool_call>
  {answer}

  Responde SOLO con el formato anterior, sin explicaciones adicionales:
  ```
- **Categorías de datos**:
  | Categoría | Ejemplo | Tools necesarios |
  |-----------|---------|-----------------|
  | Matemáticas | "¿Cuánto es 234 * 567?" | calculator |
  | Búsqueda | "¿Qué es la capital de Francia?" | web_search |
  | Archivos | "¿Qué hay en mi carpeta?" | list_directory |
  | Feas | "¿Qué día es hoy?" | current_date |
  | Sin tools | "Hola, ¿cómo estás?" | (ninguno) |
  | Multi-step | "Busca info sobre Python y cuenta las palabras" | web_search + word_count |
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 2.2: Crear generadores por fuente para datos agentic
- **Archivos** (nuevos):
  - `dataset_preparer/csv/agent_thinking.py` — CSV con columnas de tool use
  - `dataset_preparer/aiml/agent_thinking.py` — AIML patrones que requieren tools
  - `dataset_preparer/hf/agent_thinking.py` — HF datasets con tool use
- **Cada generador**:
  1. Lee el dataset existente
  2. Clasifica samples: necesitan tool? (sí/no/multi)
  3. Para samples que necesitan tools: genera tool_call + observation sintética
  4. Para samples que no necesitan: genera solo thinking + response
- **Formato de salida por sample**:
  ```
  <|thinking|>El usuario pregunta una multiplicación. Necesito usar la calculadora.<thinking>
  <tool_call>
  {"name": "calculator", "arguments": {"expression": "15 * 37"}}
  </tool_call>
  <observation>555</observation>
  </tool_call>
  <|answer|>La multiplicación de 15 * 37 es 555.
  ```
- **Formato sin tools** (respuesta directa):
  ```
  <|thinking|>El usuario me saluda. Debo responder con un saludo amigable.</thinking><|answer|>¡Hola! ¿En qué puedo ayudarte?
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 2.3: Crear dataset mixto (agente + chat normal)
- **Archivo**: `dataset_preparer/data_preparer.py`
- **Ubicación**: Nuevo paso en `prepare()` después de thinking generation
- **Nuevo flag CLI**: `--generate-agent-data`
- **Nuevo método**: `_generate_agent_data()`
- **Flujo**:
  1. Generar datos agentic (con tool calls) — 30% del dataset
  2. Mantener datos chat normales (thinking + response) — 70% del dataset
  3. Combinar para que el modelo aprenda CUÁNDO usar tools y cuándo no
  4. Balance: si el modelo solo ve tool calls, nunca aprenderá a responder directamente
- **Ratio configurable**: `--agent-ratio 0.3` (default 30% agentic)
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 2.4: Validación de datos agentic
- **Archivo**: `dataset_preparer/agent/quality.py` (nuevo)
- **Validaciones**:
  - `<tool_call>` tiene JSON válido
  - `tool_name` existe en el registry
  - `arguments` matchean el schema de la herramienta
  - `<tool_call>` contiene resultado no vacío
  - `<tool_call>` contiene respuesta no vacía
  - Balance tool_calls vs no-tool-calls en el batch
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

---

## Fase 3: Training Adaptado — P1 (High)

**Prioridad:** P1 — El modelo debe aprender el patrón tool_call → observation → response
**Tiempo estimado:** 4-6 horas
**Riesgo:** Medio

### Task 3.1: Actualizar loss masking para tokens agentic
- **Archivo**: `training/trainer.py`
- **Ubicación**: `_compute_loss()` — donde se calcula `thinking_mask`
- **Nuevo comportamiento**:
  ```python
  # Tokens que reciben peso completo (1.0):
  # - Tokens de respuesta normales
  # - Tokens dentro de <tool_call>  (el modelo debe aprender a generar tool calls correctos)
  # - Tokens dentro de <observation> (el modelo aprende a interpretar resultados)

  # Tokens que reciben peso reducido (thinking_loss_weight):
  # - Tokens dentro de <thinking> (razonamiento interno)

  # Tokens que NO contribuyen al loss:
  # - Tokens de usuario (<|user|>...<|end|>)
  ```
- **Nuevo método**: `_compute_agent_mask(token_ids)` → retorna weight tensor
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 3.2: Métricas de entrenamiento agentic
- **Archivo**: `training/trainer.py`
- **Ubicación**: `_compute_thinking_metrics()`
- **Nuevas métricas**:
  | Métrica | Descripción |
  |---------|-------------|
  | `agent_tool_call_accuracy` | % de veces que genera `<tool_call>` válido |
  | `agent_tool_name_accuracy` | % de veces que elige la herramienta correcta |
  | `agent_args_accuracy` | % de veces que genera argumentos válidos |
  | `agent_observation_accuracy` | % de veces que procesa correctamente `<observation>` |
  | `agent_response_accuracy` | % de veces que genera `<tool_call>` después de observación |
  | `agent_tool_ratio` | Ratio de muestras con tool call vs sin tool call |
- **Intervalo**: Loggear cada 50 batches (igual que thinking metrics)
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 3.3: Actualizar TrainingConfig
- **Archivo**: `training/trainer.py`
- **Nuevos campos**:
  ```python
  @dataclass
  class TrainingConfig:
      # ... existente ...
      agent_loss_weight: float = 1.0       # Peso para tokens de tool_call
      agent_enabled: bool = False           # Activar training agentic
      agent_ratio: float = 0.3             # Ratio de datos agentic en dataset
      agent_max_tool_calls: int = 3        # Max tool calls por secuencia
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

---

## Fase 4: Inference Agente — P1 (High)

**Prioridad:** P1 — El loop de ejecución es lo que hace al agente funcional
**Tiempo estimado:** 6-8 horas
**Riesgo:** Medio

### Task 4.1: Actualizar DialogManager con agentic loop
- **Archivo**: `commons/dialogue/dialogmanager.py`
- **Nuevo método**: `generate_response_agent(user_text)`
- **Flujo del loop**:
  ```python
  def generate_response_agent(self, user_text):
      context = self._build_initial_prompt(user_text)
      max_iterations = 5  # max tool calls antes de forzar respuesta
      observations = []

      for iteration in range(max_iterations):
          # 1. Generar con el modelo
          raw_output = self._generate_tokens(context)

          # 2. ¿Generó tool_call?
          if not has_tool_call(raw_output):
              # Respuesta final sin tool call
              return self._parse_final_response(raw_output)

          # 3. Parsear tool_call
          tool_name, arguments = parse_tool_call(raw_output)

          # 4. Ejecutar herramienta
          try:
              result = self.tool_executor.execute(tool_name, arguments)
          except Exception as e:
              result = f"Error ejecutando {tool_name}: {str(e)}"

          # 5. Formatear observation
          observation = format_observation(result)
          observations.append(observation)

          # 6. Añadir al contexto para siguiente iteración
          context += raw_output + observation

      # Si se agotaron iteraciones, forzar respuesta
      return self._force_final_response(context)
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 4.2: Actualizar prompt de inferencia
- **Archivo**: `commons/dialogue/dialogmanager.py`
- **Ubicación**: `generate_response()` línea 158 (construcción del prompt)
- **Cambio del prompt**:
  ```python
  # PROMPT ACTUAL:
  prompt_text = f"Pregunta: {user_text}\nPiensa paso a paso antes de responder.\n\nRespuesta:"

  # PROMPT NUEVO (cuando agente está habilitado):
  prompt_text = f"""Pregunta: {user_text}
  Piensa paso a paso. Si necesitas buscar información o hacer cálculos,
  usa las herramientas disponibles con el formato:
  <tool_call>
  {{"name": "tool_name", "arguments": {{"param": "value"}}}}
  </tool_call>

  Si no necesitas herramientas, responde directamente.

  Respuesta:"""
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 4.3: Integrar agente en ChatEngine
- **Archivo**: `inference/chat_engine.py`
- **Cambio**:
  ```python
  class ChatEngine:
      def __init__(self, config):
          # ... existente ...
          if self.config.agent_enabled:
              self.dialog_manager = DialogueManager(
                  # ... params existentes ...
                  tool_registry=self.tool_registry,
                  agent_enabled=True,
              )
  ```
- **Nuevo campo en ChatConfig**:
  ```python
  @dataclass
  class ChatConfig:
      # ... existente ...
      agent_enabled: bool = False
      agent_max_iterations: int = 5
      agent_show_tool_calls: bool = True
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

---

## Fase 5: CLI y Configuración — P2 (Medium)

**Prioridad:** P2 — User-facing flags
**Tiempo estimado:** 2-3 horas
**Riesgo:** Bajo

### Task 5.1: Flags CLI para agente
- **Archivo**: `main.py`
- **Nuevos flags**:
  ```bash
  # Preparación de datos con tool use
  python main.py --prepare-data --aiml --csv \
      --generate-thinking --generate-agent-data \
      --agent-ratio 0.3 --thinking-model Qwen2.5-1.5B-Instruct-Q4_0:latest

  # Entrenamiento con agente
  python main.py --train --use-cache --epochs 50 \
      --agent-enabled --agent-loss-weight 1.0

  # Chat con agente
  python main.py --chat --model agente_v1 \
      --agent-enabled --agent-show-tool-calls

  # Chat sin agente (solo thinking)
  python main.py --chat --model chat_model
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

### Task 5.2: TrainingConfig agente
- **Archivo**: `training/trainer.py`
- **Nuevo default**:
  ```python
  TRAINING_CONFIG = {
      # ... existente ...
      'agent_enabled': False,
      'agent_loss_weight': 1.0,
      'agent_ratio': 0.3,
  }
  ```
- **Estado**: `DONE` — Tokens añadidos a `data_preparer.py:1411` user_symbols + 6 token IDs + 9 métodos en `bpe_tokenizer.py`

---

## Fase 6: Testing — P2 (Medium)

**Prioridad:** P2 — Todo debe estar testeado
**Tiempo estimado:** 6-8 horas
**Riesgo:** Bajo

### Task 6.1: Tests unitarios — Tool Registry
- **Archivo**: `tests/test_tool_registry.py` (nuevo)
- **Tests**:
  - `test_register_tool()` — herramienta se registra correctamente
  - `test_call_tool()` — ejecuta herramienta y retorna resultado
  - `test_unknown_tool()` — error al llamar herramienta inexistente
  - `test_tool_descriptions()` — genera descripción para el prompt
  - `test_calculator_tool()` — 15 * 37 = 555
  - `test_current_date_tool()` — retorna fecha válida
  - `test_get_platform_tool()` — retorna 'windows', 'linux' o 'darwin'
  - `test_shell_tool_windows()` — ejecuta Get-Date en Windows
  - `test_shell_tool_linux()` — ejecuta date en Linux/Mac
  - `test_shell_tool_mac()` — ejecuta date en Mac
  - `test_powershell_tool()` — ejecuta Get-Date y retorna resultado
  - `test_bash_tool()` — ejecuta date y retorna resultado
  - `test_shell_security_block()` — bloquea comandos destructivos
  - `test_shell_timeout()` — timeout después de N segundos
  - `test_shell_output_truncate()` — trunca output largo a 500 chars
  - `test_platform_detection()` — detecta windows/linux/darwin
  - `test_safe_commands_windows()` — whitelist PowerShell funciona
  - `test_safe_commands_linux()` — whitelist Bash funciona
  - `test_safe_commands_mac()` — whitelist zsh/Bash funciona

### Task 6.2: Tests unitarios — Tool Executor
- **Archivo**: `tests/test_tool_executor.py` (nuevo)
- **Tests**:
  - `test_parse_tool_call()` — parsea JSON válido
  - `test_parse_invalid_json()` — maneja JSON malformado
  - `test_format_observation()` — formatea `<observation>...</observation>`
  - `test_has_tool_call()` — detecta presencia de tool_call
  - `test_execute_tool_call()` — ejecuta end-to-end

### Task 6.3: Tests unitarios — Agent Data Generation
- **Archivo**: `tests/test_agent_data.py` (nuevo)
- **Tests**:
  - `test_agent_thinking_generator()` — genera tool_call válido
  - `test_agent_quality_validation()` — valida tool_call + observation
  - `test_agent_ratio()` — respeta ratio configurado
  - `test_mixed_dataset()` — dataset tiene ambos tipos

### Task 6.4: Tests de integración
- **Archivo**: `tests/test_agent_integration.py` (nuevo)
- **Tests**:
  - `test_prepare_data_with_agent()` — pipeline completo
  - `test_train_with_agent()` — training genera métricas agentic
  - `test_chat_agent_mode()` — chat con tool calls
  - `test_multi_step_tool_call()` — modelo ejecuta 2+ tools

### Task 6.5: Test manual end-to-end
```bash
# 1. Preparar datos con agente
python main.py --prepare-data --aiml --csv --generate-thinking \
    --generate-agent-data --agent-ratio 0.3 \
    --thinking-model qwen2.5:1.5b

# 2. Entrenar con agente
python main.py --train --use-cache --epochs 50 \
    --agent-enabled --agent-loss-weight 1.0

# 3. Chat con herramientas
python main.py --chat --model agente_v1 --agent-enabled --agent-show-tool-calls

# Casos de prueba:
# > ¿Cuánto es 123 * 456?
# → [tool_call: calculator {"expression": "123 * 456"}]
# → [observation: 56088]
# → La respuesta es 56,088.

# > ¿Qué es Python?
# → (sin tool call, respuesta directa)
# → Python es un lenguaje de programación...

# > Busca info sobre PyTorch y dime cuántas palabras tiene
# → [tool_call: web_search {"query": "PyTorch"}]
# → [observation: PyTorch es un framework de deep learning...]
# → [tool_call: word_count {"text": "PyTorch es un framework..."}]
# → [observation: 42]
# → La información encontrada tiene 42 palabras.
```

---

## Fase 7: Pulido y Optimización — P3 (Low)

**Prioridad:** P3 — Nice-to-have
**Tiempo estimado:** 6-8 horas
**Riesgo:** Bajo

### Task 7.1: Permission System
- **Archivo**: `commons/tools/permission_manager.py` (nuevo)
- **Responsabilidad**: Gestionar permisos de ejecución del usuario — Capa 1 de seguridad
- **Clasificación de herramientas por riesgo**:
  | Riesgo | Tools | Permiso requerido |
  |--------|-------|-------------------|
  | Ninguno | calculator, current_date, word_count, get_platform | Auto-aprueba |
  | Bajo (lectura) | read_file, list_directory, web_search | Auto-aprueba (whitelist) |
  | Medio (shell lectura) | shell/powershell/bash con comandos de whitelist | Auto-aprueba (whitelist) |
  | Alto (shell escritura) | shell/powershell/bash con comandos fuera de whitelist | **Requiere permiso** |
  | Crítico | Cualquier comando bloqueado por ShellSecurity | **Bloqueado siempre** |
- **Arquitectura**:
  ```python
  class PermissionManager:
      """Gestiona permisos para ejecución de comandos.
      
      El modelo SOLO genera tool_call. La ejecución real es responsabilidad
      del agente, que DEBE pedir permiso al usuario antes de ejecutar.
      """
      
      def __init__(self, auto_approve: bool = False, dry_run: bool = False):
          self.auto_approve = auto_approve  # Para testing
          self.dry_run = dry_run            # Solo mostrar, no ejecutar
          self.permission_log = []          # Historial de permisos
      
      def request_permission(self, tool_name: str, command: str, platform: str) -> bool:
          """Pide permiso al usuario antes de ejecutar.
          
          Muestra:
          - Herramienta a usar
          - Comando completo
          - Plataforma detectada
          - Timeout configurado
          
          Retorna True si el usuario aprueba, False si rechaza.
          """
          
      def format_permission_request(self, tool_name: str, command: str, platform: str) -> str:
          """Formatea el request de permiso para mostrar al usuario.
          
          Ejemplo:
          ┌─────────────────────────────────────────────┐
          │ Tool Call Request                           │
          ├─────────────────────────────────────────────┤
          │ Tool: powershell                            │
          │ Platform: windows                           │
          │ Command: Get-ChildItem -Path . -Filter *.py │
          │ Timeout: 30s                                │
          ├─────────────────────────────────────────────┤
          │ ¿Ejecutar? (sí/no/dry-run):                │
          └─────────────────────────────────────────────┘
          """
  ```
- **Modos de operación**:
  | Modo | Descripción | Uso |
  |------|-------------|-----|
  | `interactive` | Pide permiso por cada tool call | Default, producción |
  | `auto_approve` | Aprobación automática | Testing, desarrollo |
  | `dry_run` | Muestra comando sin ejecutar | Validación, debugging |
  | `whitelist` | Auto-aprueba comandos de lectura | Productividad |
- **Estado**: `DONE` - PermissionManager implementado en `commons/tools/permission_manager.py`

### Task 7.2: Platform Auto-Detection
- **Archivo**: `commons/tools/platform_detector.py` (nuevo)
- **Responsabilidad**: Detectar SO y configurar shell apropiado
- **Arquitectura**:
  ```python
  import platform
  import shutil
  
  class PlatformDetector:
      """Detecta plataforma y configura shell apropiado."""
      
      def detect_platform() -> str:
          """Retorna 'windows', 'linux', o 'darwin'."""
          
      def get_default_shell() -> str:
          """Retorna shell por defecto:
          - Windows: 'powershell' (powershell.exe o pwsh)
          - Linux: 'bash' (/bin/bash)
          - Mac: 'zsh' (/bin/zsh) o 'bash' (/bin/bash)
          """
          
      def get_shell_executable(shell: str) -> str:
          """Retorna path completo del ejecutable del shell."""
          
      def get_shell_args(shell: str) -> list[str]:
          """Retorna argumentos para ejecutar comando."""
          # PowerShell: ['-Command', command]
          # Bash/zsh: ['-c', command]
          
      def is_command_available(command: str) -> bool:
          """Verifica si un comando está disponible en el sistema."""
  ```
- **Estado**: `DONE` - PlatformDetector implementado en `commons/tools/platform_detector.py`

### Task 7.3: Streaming con tool calls
- **Archivo**: `inference/chat_engine.py`
- **Cambio**: Soportar streaming SSE para cada paso del agente
  ```json
  {"type": "thinking", "content": "Necesito calcular..."}
  {"type": "tool_call", "name": "calculator", "arguments": {"expression": "15*37"}}
  {"type": "observation", "content": "555"}
  {"type": "response", "content": "La respuesta es 555."}
  ```

### Task 7.4: API endpoints agentic
- **Archivo**: `ServerFastAPI/routers_v1.py`
- **Nuevo campo en request**: `"agent_enabled": true`
- **Nuevo campo en response**: `"tool_calls": [...]`, `"observations": [...]`

### Task 7.5: Exportar modelo agentic
- **Archivo**: `commons/registry/model_export.py`
- **Cambio**: Asegurar que los tokens de agente se exportan correctamente a GGUF/ONNX

### Task 7.6: Documentación
- **Archivos**:
  - `AGENTIC_GUIDE.md` — Guía de uso del agente
  - `README.md` — Añadir sección de capacidades agenticas
  - `AGENTS.md` — Actualizar con nuevas herramientas

### Task 7.7: Tests de Permission System
- **Archivo**: `tests/test_permission_system.py` (nuevo)
- **Tests**:
  - `test_request_permission_interactive()` — pide permiso correctamente
  - `test_request_permission_auto_approve()` — auto-aprueba en modo testing
  - `test_request_permission_dry_run()` — no ejecuta en dry-run
  - `test_permission_denied()` — maneja rechazo del usuario
  - `test_platform_detection_windows()` — detecta Windows
  - `test_platform_detection_linux()` — detecta Linux
  - `test_platform_detection_mac()` — detecta Mac
  - `test_shell_selection()` — selecciona shell correcto por plataforma
  - `test_audit_log()` — registra tool calls correctamente
- **Estado**: `DONE` - 19 tests implementados en `tests/test_permission_system.py`

---

## Fase 8: Mixture of Experts (MoE) — P3 (Low)

**Prioridad:** P3 — Evolución natural del agente end-to-end
**Depende de:** Fases 0-4 completadas (tokens + tool registry + dataset agente + training + inference)
**Tiempo estimado:** 20-30 horas
**Riesgo:** Alto (cambio arquitectónico significativo)

### Por qué MoE después del agente

El plan agente end-to-end ya crea routing implícito — el modelo aprende a decidir qué herramienta usar via `<tool_call>`. MoE formaliza esto a nivel arquitectónico: en vez de tools externos, el **propio modelo** tiene expertos internos con un gating network.

| Agente (Fases 0-7) | MoE (Fase 8) |
|---------------------|-------------|
| `<tool_call>` → ToolRegistry → ToolExecutor | Gating network → Expert K → Output |
| Router en código Python (DialogManager) | Router en pesos del modelo (gating network) |
| Tools externos (calculator, web_search) | Expertos internos (FFN experts) |
| Observación inyectada como texto | Resultado combinado ponderado por gating |

### Arquitectura MoE objetivo

```
ChatModelMoE (nn.Module)
├── Token Embedding + Positional Embedding
├── MoE Transformer Blocks (x num_layers)
│   ├── Multi-Head Self-Attention (compartido)
│   ├── MoE Feed-Forward Layer
│   │   ├── Expert 1 (FFN) — Lenguaje general / conversación
│   │   ├── Expert 2 (FFN) — Razonamiento / matemáticas
│   │   ├── Expert 3 (FFN) — Tool use / extracción de datos
│   │   ├── Expert 4 (FFN) — Multilingüe / traducción
│   │   └── Gating Network (Linear → Softmax → Top-K)
│   └── Layer Normalization + Residual
└── Language Model Head → Logits
```

### Detalle del Gating Network

```python
class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        gate_scores = F.softmax(self.gate(x), dim=-1)  # (batch, seq, num_experts)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Normalizar pesos de los K experts seleccionados
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Calcular output combinado
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (batch, seq)
            expert_weight = top_k_scores[:, :, k]  # (batch, seq)

            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_out

        return output, gate_scores  # gate_scores para load balancing loss
```

### Task 8.1: Crear `commons/model/chatmodel_moe.py`
- **Archivo**: `commons/model/chatmodel_moe.py` (nuevo)
- **Clase**: `ChatModelMoE(ChatModel)`
- **Cambios respecto a ChatModel actual**:
  ```python
  class ChatModelMoE(ChatModel):
      def __init__(self, tokenizer, embed_size, num_layers=2,
                   num_experts=4, top_k=2):
          # GPT2Config con MoE — usa n_embd (no hidden_size)
          config = GPT2Config(
              vocab_size=tokenizer.vocab_size,
              n_embd=embed_size,
              n_head=4,
              n_layer=num_layers,
              n_positions=512
          )
          self.model = GPT2LMHeadModel(config)
          self.num_experts = num_experts

          # Reemplazar FFN de cada capa con MoE
          for layer in self.model.h:
              original_ffn = layer.mlp
              layer.mlp = MoELayer(embed_size, num_experts, top_k)

          # Estadísticas de routing
          self.routing_stats = {}
      ```
- **Métodos nuevos**:
  - `get_expert_utilization()` → dict con uso por expert
  - `get_load_balancing_loss(gate_scores)` → loss para entrenamiento
  - `freeze_attention()` → congelar self-attention, solo entrenar experts
- **Estado**: `DONE` - MoELayer + ChatModelMoE implementados en `commons/model/chatmodel_moe.py`

### Task 8.2: Mapeo de experts con dominios del agente
- **Archivo**: `commons/model/chatmodel_moe.py`
- **Cada expert se especializa en un dominio** basado en los datos del agente:
  | Expert | Dominio | Datos de entrenamiento |
  |--------|---------|----------------------|
  | Expert 0 | Conversación general / chat | Datos AIML + HF sin tool_call |
  | Expert 1 | Razonamiento / thinking | Datos con `<thinking>` sin tool_call |
  | Expert 2 | Tool use / acciones | Datos con `<tool_call>` |
  | Expert 3 | Observaciones / síntesis | Datos con `<observation>` |
- **Inicialización**: Los experts se inicializan aleatoriamente, pero el training data los fuerza a especializarse
- **Estado**: `DONE` - Mapeo implementado en `commons/model/chatmodel_moe.py`

### Task 8.3: Load Balancing Loss
- **Archivo**: `commons/model/chatmodel_moe.py`
- **Problema**: Sin regularización, el gating network colapsa a usar solo 1 expert (rich-get-richer)
- **Solución**: Auxiliary loss que penaliza routing desbalanceado
  ```python
  def load_balancing_loss(gate_scores, num_experts):
      """
      gate_scores: (batch, seq, num_experts) — softmax probabilities
      Penaliza si un expert recibe demasiado tráfico.
      """
      # Frecuencia de routing por expert
      routing_freq = gate_scores.mean(dim=[0, 1])  # (num_experts,)
      # Target: uniforme = 1/num_experts
      target = torch.ones(num_experts) / num_experts
      # KL divergence
      loss = F.kl_div(routing_freq.log(), target, reduction='sum')
      return loss
  ```
- **Integración en training**: `total_loss = lm_loss + 0.01 * load_balancing_loss`
- **Estado**: `DONE` - Load balancing loss implementado en `commons/model/chatmodel_moe.py`

### Task 8.4: Adapter el Trainer para MoE
- **Archivo**: `training/trainer.py`
- **Cambios**:
  - Detectar si modelo es `ChatModelMoE` → añadir load balancing loss
  - Añadir métricas: `expert_utilization`, `load_balance_loss`, `routing_entropy`
  - Soportar `freeze_attention()` para fine-tuning eficiente
  - Nuevo flag `--moe` en TrainingConfig
- **Estado**: `DONE` - Trainer actualizado con soporte MoE en `training/trainer.py`

### Task 8.5: Dataset con labels de expert por token
- **Archivo**: `dataset_preparer/agent/moe_data.py` (nuevo)
- **Cada token en el dataset tiene un label de expert**:
  ```python
  # Tokens de conversación → expert 0
  # Tokens de thinking → expert 1
  # Tokens de tool_call → expert 2
  # Tokens de observation → expert 3
  ```
- **Formato de entrenamiento**:
  ```
  input_ids:  [token1, token2, ..., tokenN]
  target_ids: [token2, token3, ..., tokenN+1]
  expert_ids: [0,     0,     ..., 2      ]  ← expert label por token
  ```
- **Ventaja**: El gating network aprende a routear cada token a su expert natural
- **Integración en data_preparer**: Nuevo paso `_assign_expert_labels()` después de tokenización
- **Estado**: `DONE` - MoEDataProcessor implementado en `dataset_preparer/agent/moe_data.py`

### Task 8.6: Inference con expert selection
- **Archivo**: `commons/dialogue/dialogmanager.py`
- **Nuevo flujo**:
  ```python
  # Generación con visibilidad de experts
  if isinstance(self.model, ChatModelMoE):
      logits, gate_scores = self.model(src)
      # Loggear qué expert se usa por token
      top_experts = gate_scores.argmax(dim=-1)
      # Inferencia normal con logits
  ```
- **Opcional**: Forzar expert por sección del output
  - Expert 0 para saludos/farewells
  - Expert 1 para thinking
  - Expert 2 para tool_call
- **Estado**: `DONE` - DialogManager actualizado con soporte MoE en `commons/dialogue/dialogmanager.py`

### Task 8.7: Exportar modelo MoE
- **Archivo**: `commons/registry/model_export.py`
- **Estrategia**: Pesos planos + MoE como código externo (Strategy D)
- **Problema**: Los exportadores GGUF/ONNX del proyecto están hardcodeados para GPT-2. El gating network MoE no tiene representación nativa en GGUF (solo arquitecturas registradas como Mixtral/DeepSeek) ni en ONNX (no hay operador MoE nativo)
- **Solución para GGUF**:
  - Exportar solo la parte GPT-2 base (atención + embeddings + LM head)
  - Los experts MoE se guardan como checkpoints PyTorch separados
  - El routing (gating network) se ejecuta en Python durante inference
  - Ventaja: El converter custom (`convert_gguf.py`) funciona sin cambios para la parte GPT-2
- **Solución para ONNX**:
  - Exportar el forward pass completo con routing simplificado
  - Reemplazar `top-k` por multiplicación ponderada de TODOS los experts (compute redundante pero ONNX-compatible)
  - El gating network se incluye en el grafo ONNX como operaciones estándar (MatMul + Softmax)
  - Usar `dynamic_axes` para batch y seq_len (ya implementado)
- **Solución para production**:
  - Para un modelo pequeño (~20M params), usar PyTorch directamente para inference MoE
  - La cuantización no es crítica — el modelo cabe en RAM cómodamente
  - GGUF/ONNX son alternativas, no requisitos
- **Archivos a modificar**:
  - `model_export.py`: Detectar `ChatModelMoE` → exportar parte GPT-2 + checkpoint MoE separado
  - `convert_gguf.py`: Sin cambios (solo exporta la parte GPT-2)
  - `convert_onnx.py`: Añadir modo MoE con routing simplificado
- **Riesgo**: Bajo — el modelo es pequeño, la exportación es opcional
- **Estado**: `DONE` - Exportación MoE implementada en `commons/registry/model_export.py`

### Task 8.8: Configuración y CLI
- **Archivo**: `main.py`
- **Nuevos flags**:
  ```bash
  # Entrenar modelo MoE
  python main.py --train --use-cache --epochs 50 \
      --agent-enabled --moe --moe-experts 4 --moe-top-k 2

  # Chat con MoE
  python main.py --chat --model agente_moe_v1 --agent-enabled --moe
  ```
- **TrainingConfig nuevos campos**:
  ```python
  @dataclass
  class TrainingConfig:
      # ... existente ...
      moe_enabled: bool = False
      moe_num_experts: int = 4
      moe_top_k: int = 2
      moe_load_balance_weight: float = 0.01
      moe_freeze_attention: bool = False
  ```
- **Estado**: `DONE` - CLI flags añadidos en `main.py`

### Task 8.9: Tests MoE
- **Archivo**: `tests/test_moe.py` (nuevo)
- **Tests**:
  - `test_moe_layer_forward()` — output shape correcto
  - `test_gating_network_routing()` — gating produce top-k experts
  - `test_load_balancing_loss()` — loss baja con routing uniforme
  - `test_expert_utilization()` — métricas de uso por expert
  - `test_freeze_attention()` — solo experts se actualizan
  - `test_moe_chatmodel()` — modelo MoE genera respuesta
  - `test_expert_labels_dataset()` — dataset tiene expert_ids correctos
  - `test_moe_training()` — training converge con MoE
- **Estado**: `DONE` - 10 tests implementados en `tests/test_moe.py`

---

## Orden de Implementación

```
Fase 0 (Tokens + Fix Thinking) — P0 — 8-11h
  → Task 0.1 (nuevos tokens en BPE)
  → Task 0.2 (verificar ChatModel)
  → Task 0.3 (pre-entrenar BPE)
  → Task 0.4 (reescribir rule-based thinking)
  → Task 0.5 (fortalecer validator)
  → Task 0.6 (loss weight 1.0)
  → Task 0.7 (eliminar doble forward pass)
  → Task 0.8 (preservar columna source)
  → Task 0.9 (limpiar código muerto)
  → Task 0.10 (tests)
  → [VERIFY] Tokens en vocabulario, thinking tiene reasoning real

Fase 1 (Tool Registry) — P0 — 4-6h
  → Task 1.1 (tool_registry.py)
  → Task 1.2 (tool_executor.py)
  → Task 1.3 (integrar en ChatEngine)
  → [TEST] Herramientas ejecutan correctamente

Fase 2 (Dataset) — P1 — 8-12h
  → Task 2.1 (generador agentic)
  → Task 2.2 (generadores por fuente)
  → Task 2.3 (dataset mixto)
  → Task 2.4 (validación)
  → [VERIFY] Datos generados tienen formato correcto

Fase 3 (Training) — P1 — 4-6h
  → Task 3.1 (loss masking agentic)
  → Task 3.2 (métricas)
  → Task 3.3 (TrainingConfig)
  → [VERIFY] Training converges, métricas mejoran

Fase 4 (Inference) — P1 — 6-8h
  → Task 4.1 (agentic loop en DialogManager)
  → Task 4.2 (prompt de inferencia)
  → Task 4.3 (integrar en ChatEngine)
  → [VERIFY] Chat con tool calls funciona

Fase 5 (CLI) — P2 — 2-3h
  → Task 5.1 (flags CLI)
  → Task 5.2 (TrainingConfig)
  → [VERIFY] Flags propagados correctamente

Fase 6 (Testing) — P2 — 6-8h
  → Task 6.1-6.4 (tests)
  → Task 6.5 (test manual)
  → [VERIFY] Todos los tests pasan

Fase 7 (Pulido) — P3 — 6-8h
  → Task 7.1 (permission_manager)
  → Task 7.2 (platform_detector)
  → Task 7.3 (streaming)
  → Task 7.4 (API endpoints)
  → Task 7.5 (export)
  → Task 7.6 (docs)
  → Task 7.7 (tests)
  → [VERIFY] Funcionalidad completa

Fase 8 (MoE) — P3 — 20-30h
  → Task 8.1 (ChatModelMoE + MoELayer)
  → Task 8.2 (mapeo experts → dominios)
  → Task 8.3 (load balancing loss)
  → Task 8.4 (adapter Trainer)
  → Task 8.5 (dataset con expert labels)
  → Task 8.6 (inference con expert selection)
  → Task 8.7 (export MoE)
  → Task 8.8 (CLI + config)
  → Task 8.9 (tests MoE)
  → [VERIFY] Experts se especializan, routing balanceado
```

**Tiempo total estimado: 66-98 horas**

---

## Key Files Summary

| Archivo | Tareas | Estado |
|---------|--------|--------|
| `commons/tokenizer/bpe_tokenizer.py` | 0.1 | Actual — añadir 6 tokens |
| `commons/model/chatmodel.py` | 0.2 | Sin cambios (auto-ajusta vocab_size) |
| `dataset_preparer/data_preparer.py` | 0.3, 2.3 | Modificado |
| `dataset_preparer/csv/thinking.py` | 0.4, 0.9 | Modificado — reasoning real + limpiar muerto |
| `dataset_preparer/thinking_quality.py` | 0.5 | Modificado — validator más estricto |
| `training/trainer.py` | 0.6, 0.7, 0.8, 3.1, 3.2, 3.3, 8.4 | Modificado — loss weight, no double pass, preservar source, agentic, MoE |
| `tests/test_thinking_fixes.py` | 0.10 | Nuevo |
| `commons/tools/tool_registry.py` | 1.1 | Nuevo — registro de herramientas |
| `commons/tools/tool_executor.py` | 1.2 | Nuevo — ejecución cross-platform |
| `commons/tools/permission_manager.py` | 7.1 | Nuevo — permisos de usuario |
| `commons/tools/platform_detector.py` | 7.2 | Nuevo — detección de SO |
| `commons/tools/audit_log.py` | 7.3 | Nuevo — logging de tool calls |
| `inference/chat_engine.py` | 1.3, 4.3 | Modificado |
| `dataset_preparer/agent/thinking.py` | 2.1 | Nuevo — generador principal |
| `dataset_preparer/agent/quality.py` | 2.4 | Nuevo — validación |
| `dataset_preparer/csv/agent_thinking.py` | 2.2 | Nuevo |
| `dataset_preparer/aiml/agent_thinking.py` | 2.2 | Nuevo |
| `dataset_preparer/hf/agent_thinking.py` | 2.2 | Nuevo |
| `commons/dialogue/dialogmanager.py` | 4.1, 4.2 | Modificado |
| `main.py` | 5.1, 8.8 | Modificado |
| `ServerFastAPI/routers_v1.py` | 7.4 | Modificado |
| `commons/model/chatmodel_moe.py` | 8.1, 8.2, 8.3 | Nuevo |
| `dataset_preparer/agent/moe_data.py` | 8.5 | Nuevo |
| `tests/test_moe.py` | 8.9 | Nuevo |
| `tests/test_permission_system.py` | 7.7 | Nuevo |

---

## Dependencies

| Tarea | Depende De | Prioridad |
|-------|-----------|-----------|
| 0.1 | (ninguna) | P0 |
| 0.2 | 0.1 | P0 |
| 0.3 | 0.1 | P0 |
| 0.4 | 0.1 (tokens thinking en BPE) | P0 |
| 0.5 | 0.4 | P0 |
| 0.6 | (ninguna) | P0 |
| 0.7 | (ninguna) | P0 |
| 0.8 | 0.4 | P0 |
| 0.9 | (ninguna) | P0 |
| 0.10 | 0.4-0.9 | P0 |
| 1.1 | (ninguna) | P0 |
| 1.2 | 1.1 | P0 |
| 1.3 | 1.1, 1.2 | P0 |
| 2.1 | 1.1 (necesitaToolRegistry para descripciones), 0.4 (thinking real) | P1 |
| 2.2 | 2.1 | P1 |
| 2.3 | 2.1, 2.2 | P1 |
| 2.4 | 2.1 | P1 |
| 3.1 | 0.1, 0.6 | P1 |
| 3.2 | 3.1, 0.7 | P1 |
| 3.3 | 3.1 | P1 |
| 4.1 | 1.2, 3.1 | P1 |
| 4.2 | 0.1 | P1 |
| 4.3 | 4.1, 4.2, 1.3 | P1 |
| 5.1 | 4.3 | P2 |
| 5.2 | 3.3 | P2 |
| 6.1-6.5 | Todas | P2 |
| 7.1 | 4.3 | P3 |
| 7.2 | 7.1 | P3 |
| 7.3 | 4.3 | P3 |
| 7.4 | 4.3 | P3 |
| 7.5 | 4.3 | P3 |
| 7.6 | 7.1-7.5 | P3 |
| 7.7 | 7.1, 7.2 | P3 |
| 8.1 | 0.1, 0.2 | P3 |
| 8.2 | 8.1 | P3 |
| 8.3 | 8.1 | P3 |
| 8.4 | 8.1, 8.3, 3.1 | P3 |
| 8.5 | 8.1, 2.3, 0.8 | P3 |
| 8.6 | 8.1, 4.1 | P3 |
| 8.7 | 8.1, 7.3 | P3 |
| 8.8 | 8.4, 5.1 | P3 |
| 8.9 | 8.1-8.8 | P3 |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Modelo pequeño (9.2M) no aprende tool calls | Alto | Empezar con 1-2 tools simples (calculator, current_date), aumentar gradualmente |
| Rule-based thinking sigue siendo meta-comentario tras fix | Medio | Test manuales de samples + teacher model como gold standard |
| Quality validator sigue siendo permisivo | Medio | Añadir test suite con casos conocidos de meta vs real |
| `thinking_loss_weight` 1.0 causa training inestable | Bajo | Monitorear loss curves, fallback a 0.5 si diverge |
| Preservar columnas `source` increase memory | Bajo | Solo 1 columna string extra, impacto mínimo |
| Teacher model genera tool calls inválidos | Alto | Validación estricta de JSON + schema + filtrado |
| Dataset demasiado agentic, modelo nunca responde directo | Medio | Ratio configurable (default 30%), validación de balance |
| Tool call parsing falla en inference | Medio | Fallback a respuesta sin tool, logging de errores |
| Tokens de agente fragmentados por BPE | Bajo | `user_defined_symbols` garantiza tokens indivisibles |
| Observation demasiado larga agota contexto | Medio | Truncar observation a 200 tokens, `agent_max_iterations=5` |
| PowerShell ejecuta comandos destructivos | ALTO | Whitelist de comandos seguros (solo lectura), timeout 30s, logging |
| Modelo genera comandos maliciosos | Alto | Validación de sintaxis + whitelist + sandbox |
| Ejecución sin permiso del usuario | CRÍTICO | Agente DEBE pedir confirmación antes de ejecutar; `--dry-run` para testing |
| Comandos cross-platform incompatibles | Medio | Auto-detectar SO, usar shell nativo (PowerShell/Bash/zsh) |
| MoE gating colapsa a 1 expert | Alto | Load balancing loss + monitoreo de expert utilization |
| Modelo MoE demasiado grande para hardware | Medio | Empezar con 2 experts, escalar gradualmente; `freeze_attention` para fine-tuning |
| Experts no se especializan | Medio | Dataset con expert_labels por token; monitorear routing entropy |
| Export MoE incompatible con GGUF/ONNX | Bajo | Serializar todos los pesos como modelo plano; gating como código |

---

## Rollback Strategy

Cada fase es independientemente revertible:

```bash
# Si el agente no funciona tras una fase:
# 1. Desactivar agente
python main.py --chat --no-agent-enabled

# 2. Revertir cambios
git stash
git checkout -- <archivo>

# 3. Entrenar sin agente (solo thinking)
python main.py --train --use-cache --epochs 30
```

El modelo de thinking sigue funcionando sin el componente agentic — son capas independientes.

---

*Generado desde análisis del código base — refleja el estado actual y las mejoras planificadas para capacidades agenticas end-to-end.*
