# ROADMAP_AGENTIC - MyIAModelChat: Modelo End-to-End con Capacidades Agenticas

## Overview

Este roadmap define la implementación completa de capacidades **agenticas end-to-end** para MyIAModelChat. El modelo GPT-2 aprenderá a generar `<tool_call>`, `<tool_call>`, y `<tool_call>` como parte de su generación, permitiendo planificación, ejecución de herramientas y razonamiento con observaciones — todo entrenado via el dataset, sin dependencia de un orquestador externo.

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
                                              ToolExecutor
                                                     ↓
                                              <observation>
                                              resultado     </observation>
                                                     ↓
                                              Feed back al modelo
                                                     ↓
                                              Genera respuesta final
```

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
- **Archivo**: `commons/tokenizer/bpe_tokenizer.py`
- **Ubicación**: Línea ~40, donde se definen `user_symbols`
- **Cambio**: Añadir los 6 nuevos tokens:
  ```python
  # Thinking tokens existentes
  thinking_tokens = ['<think>', '</think>']

  # NUEVOS: Agentic tokens
  agentic_tokens = [
      '<tool_call>', '</tool_call>',
      '<tool_call>', '</tool_call>',
      '<tool_call>', '</tool_call>',
  ]

  user_symbols = '--user_defined_symbols=' + ','.join(thinking_tokens + agentic_tokens)
  ```
- **Métodos nuevos**:
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
- **Estado**: `TODO`

### Task 0.2: Actualizar ChatModel para soportar los nuevos tokens
- **Archivo**: `commons/model/chatmodel.py`
- **Cambio**: Ninguno en arquitectura — `vocab_size` ya viene del tokenizer
- Solo verificar que `GPT2Config(vocab_size=tokenizer.vocab_size)` se ajusta automáticamente
- **Estado**: `TODO`

### Task 0.3: Pre-entrenar BPE con nuevos tokens
- **Archivo**: `dataset_preparer/data_preparer.py`
- **Ubicación**: `_prepare_bpe_tokenizer_and_tokenize()`
- **Cambio**: Verificar que los nuevos tokens `<tool_call>` se registran como `user_defined_symbols`
- **Verificación**:
  ```bash
  python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
  # Verificar que los 6 tokens aparecen en sentencepiece.vocab
  ```
- **Estado**: `TODO`

### Problemas detectados en Thinking

| # | Problema | Severidad | Ubicación |
|---|---------|-----------|-----------|
| 1 | Thinking rule-based es meta-comentario inútil | ALTO | `dataset_preparer/*/thinking.py` |
| 2 | Quality validator demasiado permisivo | ALTO | `thinking_quality.py:55-101` |
| 3 | `thinking_loss_weight` = 0.5 debería ser 1.0 | MEDIO | `training/trainer.py:108` |
| 4 | Doble forward pass cada 50 batches | MEDIO | `training/trainer.py:533` |
| 5 | Columna `source` eliminada del dataset | ALTO | `training/trainer.py:1108` |
| 6 | `encode_with_thinking()` nunca se usa | BAJO | `bpe_tokenizer.py:188` |
| 7 | Variable muerta `q_lower` | BAJO | `csv/thinking.py:41` |

### Task 0.4: Reescribir rule-based thinking con reasoning real
- **Archivos**: `dataset_preparer/aiml/thinking.py`, `csv/thinking.py`, `pdf/thinking.py`, `epub/thinking.py`, `web/thinking.py`, `hf/thinking.py`
- **Cambio**: Reemplazar meta-comentario por reasoning que conecta pregunta → respuesta
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
- **Estado**: `TODO`

### Task 0.5: Fortalecer quality validator
- **Archivo**: `dataset_preparer/thinking_quality.py`
- **Cambios**:
  ```python
  # ANTES (demasiado permisivo):
  META_PATTERNS = [
      re.compile(r'^(el usuario|the user|el humano|the human)\s*(me\s+)?(saluda|despide|pregunta|pide)'),
      # ... solo 4 patrones, anclados con ^
  ]
  # Score mínimo: 0.5 — un meta-comentario pasa con 0.9

  # DESPUÉS (más estricto):
  META_PATTERNS = [
      re.compile(r'(el usuario|the user|el humano|the human)\s*(me\s+)?(saluda|despide|pregunta|pide|quiere|necesita)'),
      re.compile(r'(debo|i should|debería)\s*(responder|contestar|decir|reply|answer)'),
      re.compile(r'(respondo|contestando|i respond|answering)\s*(con|with|de|de forma)'),
      re.compile(r'(el bot|the bot|asistente|assistant)\s*(responde|answer|debe)'),
      re.compile(r'(esto indica|this indicates|esto sugiere|this suggests)\s*(que|that)'),
      re.compile(r'(la información|the information)\s*(es |está )?\s*(consistente|correcta|valida|suficiente)'),
  ]
  ```
- **Nuevos checks**:
  - `has_causal_link`: Verificar conectores causales ("porque", "por lo tanto", "ya que", "because", "therefore")
  - `has_specific_detail`: Verificar detalles específicos de la pregunta/respuesta
  - `derivation_strength`: Medir conexión pregunta→respuesta
- **Score mínimo sube de 0.5 a 0.7**
- **Estado**: `TODO`

### Task 0.6: Cambiar `thinking_loss_weight` default a 1.0
- **Archivo**: `training/trainer.py`
- **Ubicación**: Línea 108
- **Cambio**: `thinking_loss_weight: float = 0.5` → `thinking_loss_weight: float = 1.0`
- **Razón**: Con 0.5 el modelo aprende MENOS thinking que respuesta. Con 1.0 aprende ambos por igual.
- **Estado**: `TODO`

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
- **Estado**: `TODO`

### Task 0.8: Preservar columna `source` en el dataset
- **Archivo**: `training/trainer.py`
- **Ubicación**: `_get_tokenized_dataset()` línea 1108
- **Cambio**:
  ```python
  # ANTES:
  columns_to_remove = [c for c in self.loaded_dataset.column_names
                       if c not in ('input_ids', 'token_ids')]

  # DESPUÉS:
  PRESERVED_COLUMNS = ('input_ids', 'token_ids', 'source', 'has_tool_call', 'thinking')
  columns_to_remove = [c for c in self.loaded_dataset.column_names
                       if c not in PRESERVED_COLUMNS]
  ```
- **Razón**: Necesitamos `source` para estadísticas por fuente, `has_tool_call` para datos agentic, y `thinking` para debugging
- **Estado**: `TODO`

### Task 0.9: Limpiar código muerto
- **Archivo**: `dataset_preparer/csv/thinking.py` — eliminar `q_lower` (línea 41)
- **Archivo**: `commons/tokenizer/bpe_tokenizer.py` — marcar `encode_with_thinking()` con TODO o eliminar
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
      """Detecta si el texto contiene <think>."""

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
      SAFE_COMMANDS_WINDOWS = [
          'Get-ChildItem', 'Get-Content', 'Get-Date', 'Get-Help',
          'Select-String', 'Measure-Object', 'Get-Process', 'Get-Service',
          'Get-Command', 'Get-Alias', 'Get-Host', 'Get-Item',
          'cat', 'ls', 'dir', 'echo', 'pwd', 'whoami', 'date'
      ]

      SAFE_COMMANDS_UNIX = [
          'ls', 'cat', 'grep', 'find', 'wc', 'head', 'tail', 'echo',
          'pwd', 'whoami', 'date', 'uname', 'df', 'du', 'file', 'stat',
          'ps', 'top', 'uptime', 'which', 'env', 'printenv', 'curl'
      ]

      BLOCKED_PATTERNS = [
          r'rm\s+-rf', r'del\s+/[sSqQfF]', r'Remove-Item\s+-Recurse',
          r'>\s*/dev/', r'Format-Volume', r'Initialize-Disk',
          r'Set-Content', r'Out-File', r'Tee-Object',
          r'sudo', r'su\s+-', r'chmod\s+777', r'chown'
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
  - **Blacklist**: Patrones destructivos bloqueados (`rm -rf`, `Remove-Item -Recurse`, etc.)
  - **Logging**: Todos los comandos ejecutados se registran
  - **Output limit**: Máximo 500 caracteres, truncar si es más largo
  - **Modo seguro**: Por defecto solo comandos de lectura; modo `--unsafe` para escritura
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
  <think>
  {reasoning about what tool is needed and why}
  <tool_call>
  {"name": "tool_name", "arguments": {"param": "value"}}
  </tool_call>
  <observation>{simulated result}</observation>
  </tool_call>
   {final answer}
  </observation>

  Si NO necesita herramientas, genera:
  <think>
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
- **Estado**: `TODO`

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
  <|user|>¿Cuánto es 15 * 37?<|end|>
  <think>El usuario pregunta una multiplicación. Necesito usar la calculadora.</think>
  <tool_call>
  {"name": "calculator", "arguments": {"expression": "15 * 37"}}
  </tool_call>
  <observation>555</observation>
  </tool_call>
  La multiplicación de 15 * 37 es 555.<|end|>
  ```
- **Estado**: `TODO`

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
- **Estado**: `TODO`

### Task 2.4: Validación de datos agentic
- **Archivo**: `dataset_preparer/agent/quality.py` (nuevo)
- **Validaciones**:
  - `<tool_call>` tiene JSON válido
  - `tool_name` existe en el registry
  - `arguments` matchean el schema de la herramienta
  - `<tool_call>` contiene resultado no vacío
  - `<tool_call>` contiene respuesta no vacía
  - Balance tool_calls vs no-tool-calls en el batch
- **Estado**: `TODO`

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
  # - Tokens dentro de <think> (razonamiento interno)

  # Tokens que NO contribuyen al loss:
  # - Tokens de usuario (<|user|>...<|end|>)
  ```
- **Nuevo método**: `_compute_agent_mask(token_ids)` → retorna weight tensor
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
- **Estado**: `TODO`

### Task 4.2: Actualizar prompt de inferencia
- **Archivo**: `commons/dialogue/dialogmanager.py`
- **Ubicación**: `generate_response()` línea ~173
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
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
      --agent-ratio 0.3 --thinking-model qwen2.5:1.5b

  # Entrenamiento con agente
  python main.py --train --use-cache --epochs 50 \
      --agent-enabled --agent-loss-weight 1.0

  # Chat con agente
  python main.py --chat --model agente_v1 \
      --agent-enabled --agent-show-tool-calls

  # Chat sin agente (solo thinking)
  python main.py --chat --model chat_model
  ```
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
  - `test_powershell_tool()` — ejecuta Get-Date y retorna resultado
  - `test_bash_tool()` — ejecuta date y retorna resultado
  - `test_shell_security_block()` — bloquea comandos destructivos
  - `test_shell_timeout()` — timeout después de N segundos
  - `test_shell_output_truncate()` — trunca output largo a 500 chars

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
**Tiempo estimado:** 4-6 horas
**Riesgo:** Bajo

### Task 7.1: Streaming con tool calls
- **Archivo**: `inference/chat_engine.py`
- **Cambio**: Soportar streaming SSE para cada paso del agente
  ```json
  {"type": "thinking", "content": "Necesito calcular..."}
  {"type": "tool_call", "name": "calculator", "arguments": {"expression": "15*37"}}
  {"type": "observation", "content": "555"}
  {"type": "response", "content": "La respuesta es 555."}
  ```

### Task 7.2: API endpoints agentic
- **Archivo**: `envAIModels/routers_v1.py`
- **Nuevo campo en request**: `"agent_enabled": true`
- **Nuevo campo en response**: `"tool_calls": [...]`, `"observations": [...]`

### Task 7.3: Exportar modelo agentic
- **Archivo**: `commons/registry/model_export.py`
- **Cambio**: Asegurar que los tokens de agente se exportan correctamente a GGUF/ONNX

### Task 7.4: Documentación
- **Archivos**:
  - `AGENTIC_GUIDE.md` — Guía de uso del agente
  - `README.md` — Añadir sección de capacidades agenticas
  - `AGENTS.md` — Actualizar con nuevas herramientas

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
      def __init__(self, tokenizer, embed_size, hidden_size, num_layers=2,
                   num_experts=4, top_k=2):
          # GPT2Config con MoE
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
- **Estado**: `TODO`

### Task 8.2: Mapeo de experts con dominios del agente
- **Archivo**: `commons/model/chatmodel_moe.py`
- **Cada expert se especializa en un dominio** basado en los datos del agente:
  | Expert | Dominio | Datos de entrenamiento |
  |--------|---------|----------------------|
  | Expert 0 | Conversación general / chat | Datos AIML + HF sin tool_call |
  | Expert 1 | Razonamiento / thinking | Datos con `<think>` sin tool_call |
  | Expert 2 | Tool use / acciones | Datos con `<tool_call>` |
  | Expert 3 | Observaciones / síntesis | Datos con `<observation>` |
- **Inicialización**: Los experts se inicializan aleatoriamente, pero el training data los fuerza a especializarse
- **Estado**: `TODO`

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
- **Estado**: `TODO`

### Task 8.4: Adapter el Trainer para MoE
- **Archivo**: `training/trainer.py`
- **Cambios**:
  - Detectar si modelo es `ChatModelMoE` → añadir load balancing loss
  - Añadir métricas: `expert_utilization`, `load_balance_loss`, `routing_entropy`
  - Soportar `freeze_attention()` para fine-tuning eficiente
  - Nuevo flag `--moe` en TrainingConfig
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
- **Estado**: `TODO`

### Task 8.7: Exportar modelo MoE
- **Archivo**: `commons/registry/model_export.py`
- **Cambio**: Verificar compatibilidad de MoE con exportación
  - GGUF: Los experts se serializan como pesos normales
  - ONNX: Exportar con todos los experts (aunque solo K se activen por forward)
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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
- **Estado**: `TODO`

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

Fase 7 (Pulido) — P3 — 4-6h
  → Task 7.1-7.4
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

**Tiempo total estimado: 62-90 horas**

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
| `commons/tools/tool_registry.py` | 1.1 | Nuevo |
| `commons/tools/tool_executor.py` | 1.2 | Nuevo |
| `inference/chat_engine.py` | 1.3, 4.3 | Modificado |
| `dataset_preparer/agent/thinking.py` | 2.1 | Nuevo — generador principal |
| `dataset_preparer/agent/quality.py` | 2.4 | Nuevo — validación |
| `dataset_preparer/csv/agent_thinking.py` | 2.2 | Nuevo |
| `dataset_preparer/aiml/agent_thinking.py` | 2.2 | Nuevo |
| `dataset_preparer/hf/agent_thinking.py` | 2.2 | Nuevo |
| `commons/dialogue/dialogmanager.py` | 4.1, 4.2 | Modificado |
| `main.py` | 5.1, 8.8 | Modificado |
| `envAIModels/routers_v1.py` | 7.2 | Modificado |
| `commons/model/chatmodel_moe.py` | 8.1, 8.2, 8.3 | Nuevo |
| `dataset_preparer/agent/moe_data.py` | 8.5 | Nuevo |
| `tests/test_moe.py` | 8.9 | Nuevo |

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
| 7.1-7.4 | 4.3 | P3 |
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
| Modelo genera comandos PowerShell maliciosos | Alto | Validación de sintaxis + whitelist + sandbox |
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
