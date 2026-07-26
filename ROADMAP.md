# ROADMAP - MyIAModelChat: Thinking Real + Validación por Fuente

## Overview

Este roadmap define la implementación completa de chain-of-thought ("thinking") real para MyIAModelChat. El sistema actual tiene scaffold parcial (tokens, detección, templates) pero el thinking es **falso** (meta-comentarios sin razonamiento). Este plan reemplaza el thinking fake por thinking real con generadores por fuente, más validación/limpieza de datos por fuente para evitar contaminación.

**Leyenda de Estado:** `TODO` → `IN_PROGRESS` → `BLOCKED` → `DONE`

**Leyenda de Prioridad:**
- **P0 (MVP)** — Bloqueante: debe hacerse primero, el sistema está roto sin esto
- **P1 (High)** — Core: requerido para que el thinking funcione correctamente
- **P2 (Medium)** — Calidad: mejora significativamente el comportamiento del thinking
- **P3 (Low)** — Pulido: nice-to-have, puede diferirse

---

## Diagnóstico Actual

| Componente | Estado | Problema |
|-----------|--------|----------|
| Tokens `<think>`/`</think>` | Registrados correctamente en SentencePiece | Ninguno |
| Loss masking | Funcional | Peso 0.5 **reduce** aprendizaje de thinking cuando debería mantenerlo |
| Datos thinking | Templates genéricos, no razonamiento | **CRÍTICO** - el modelo no aprende a razonar |
| Pipeline thinking | Desconectado de data_preparer | **CRÍTICO** - generate_thinking_data.py nunca se ejecuta |
| Prompt inferencia | Sin instrucción de thinking | El modelo no sabe que debe "pensar" |
| Evaluación | Solo accuracy de tokens | No mide calidad del razonamiento |
| Validación datos | No existe por fuente | Datos contaminados se mezclan |

---

## Flujo Completo del Pipeline (Objetivo)

```
Raw Sources (AIML, CSV, PDF, EPUB, Web, HF)
    │
    ▼
[1] Per-Source Validator/Cleaner
    │  Clasifica: bueno / arreglable / descartable
    │  Arregla lo que se pueda (encoding, HTML, AIML tags)
    │  Descarta lo que no se pueda arreglar
    │  Reporta estadísticas de calidad
    │
    ▼
[2] Per-Source Thinking Generator
    │  Cada fuente genera thinking apropiado:
    │  - AIML: Rule-based + Teacher model
    │  - CSV: Teacher model (qwen2.5:1.5b vía Ollama)
    │  - PDF: Teacher con contexto de sección
    │  - EPUB: Teacher con contexto de capítulo
    │  - Web: Teacher con contexto de página
    │  - HF: Teacher o passthrough
    │
    ▼
[3] Combined Dataset with Thinking
    │  Formato: <think>reasoning</think>response
    │  BPE tokenizer entrena con thinking tokens en contexto
    │
    ▼
[4] Training con Loss Masking Real
    │  thinking_loss_weight = 1.0 (mismo peso que respuesta)
    │  Metrics: thinking_loss, response_loss, thinking_coverage
    │
    ▼
[5] Inference con Thinking Real
    │  Prompt: "Pregunta: {q}\nPiensa paso a paso.\n\n"
    │  Genera: <think>reasoning</think>answer
    │  Streaming: reasoning + content separados
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ollama no disponible | Generadores por fuente fallan | Fallback a rule-based, modo sin thinking |
| Teacher model genera thinking bajo calidad | Modelo aprende basura | Validación de calidad + filtrado |
| Loss masking alto (1.0) confunde al modelo | Training inestable | Monitorear metrics, ajustar si necesario |
| Datos contaminados pasan validación | Calidad de training baja | Validadores por fuente con reglas específicas |
| Parsing XML falla en AIML | Se pierden datos AIML | Validador AIML con encoding detection + fallback |

---

## Rollback Strategy

Cada fase es independientemente revertible:

```bash
# Después de cualquier fase, si algo falla:
git stash
git checkout -- <archivo>
# O revertir fase completa:
git log --oneline
git revert <commit>
```

---

## Fase 0: Fix Crítico — Loss Weight + Prompt Inferencia — P0 (MVP)

**Prioridad:** P0 — El thinking no funciona sin esto
**Tiempo estimado:** 1-2 horas
**Riesgo:** Bajo

### Task 0.1: Cambiar `thinking_loss_weight` de 0.5 a 1.0
- **Archivo**: `main_train.py`
- **Ubicación**: `TRAINING_CONFIG` dict
- **Cambio**: `'thinking_loss_weight': 0.5` → `'thinking_loss_weight': 1.0`
- **Razón**: Con 0.5 el modelo aprende MENOS thinking que respuesta. Con 1.0 aprende ambos por igual.
- **Líneas**: ~139-150
- **Estado**: `DONE` ✅

### Task 0.2: Añadir instrucción de thinking al prompt de inferencia
- **Archivo**: `dialogmanager.py`
- **Ubicación**: `generate_response()` línea ~205
- **Cambio actual**: `prompt_text = f"Pregunta: {user_text}\nRespuesta:"`
- **Cambio nuevo**: `prompt_text = f"Pregunta: {user_text}\nPiensa paso a paso antes de responder.\n\n"`
- **Razón**: Sin esta instrucción, el modelo nunca genera `<think>` aunque esté entrenado para ello.
- **Líneas**: ~205
- **Estado**: `DONE` ✅

### Verificación Fase 0
- [ ] Entrenar 1 epoch con thinking loss weight 1.0, verificar que thinking_loss ≈ response_loss
- [ ] Chat con `--show-thinking`, verificar que el modelo intenta generar `<think>`

---

## Fase 1: Validadores por Fuente — P1 (High)

**Prioridad:** P1 — Sin validación, los datos contaminados arruinan el training
**Tiempo estimado:** 8-10 horas
**Riesgo:** Medio (modifica flujo de datos)

### Arquitectura de Validadores

```
┌─────────────────────────────────────────────────────────┐
│                   SourceValidator (Base)                 │
│  - classify(sample) → good / fixable / discardable      │
│  - fix(sample) → fixed_sample                           │
│  - report() → QualityReport                             │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────────┐
    │                  │                      │
    ▼                  ▼                      ▼
┌─────────┐    ┌─────────────┐    ┌─────────────────┐
│Generic  │    │Dedicated    │    │Dedicated        │
│Validator│    │Validators   │    │Validators       │
│(AIML,   │    │(PDF, Web,   │    │(EPUB)           │
│ CSV)    │    │ EPUB)       │    │                 │
└─────────┘    └─────────────┘    └─────────────────┘
```

### Task 1.1: Crear `source_validators.py` — Clase base + Validador Genérico
- **Archivo**: `source_validators.py` (nuevo)
- **Clase base**: `SourceValidator`
  - `classify(sample: dict) → Literal['good', 'fixable', 'discardable']`
  - `fix(sample: dict) → dict`
  - `report() → QualityReport`
  - `validate_batch(samples: list) → list[dict]`
- **Validador genérico**: `GenericValidator(SourceValidator)`
  - Validación de campos vacíos
  - Normalización de encoding (UTF-8)
  - Detección de duplicados
  - Filtrado de muestras demasiado cortas (< 5 chars)
  - Filtrado de muestras demasiado largas (> 2000 chars)
- **Campos del reporte**:
  ```python
  @dataclass
  class QualityReport:
      source: str
      total_samples: int
      good: int
      fixable: int
      discardable: int
      fixes_applied: dict  # {fix_type: count}
      discarded_reasons: dict  # {reason: count}
  ```
- **Estado**: `DONE` ✅

### Task 1.2: Crear `aiml_validator.py` — Validador AIML
- **Archivo**: `aiml_validator.py` (nuevo)
- **Hereda de**: `GenericValidator`
- **Problemas específicos AIML**:
  - XML malformado → detectar con `ElementTree.parse()` try/except
  - Encoding mixto (ISO-8859-1 vs UTF-8) → detectar con `chardet` o fallback
  - AIML tags en template text (`<random>`, `<srai>`, `<condition>`, etc.) → limpiar con regex
  - HTML tags en template (`<br/>`, `<a>`, `<b>`) → limpiar con regex
  - Patrones wildcard-only (`_`, `_ _`) → descartar
  - Templates vacíos después de limpiar → descartar
  - Stale `.datasets` pickle → detectar y regenerar
- **Regex de limpieza**:
  ```python
  # Limpiar tags AIML funcionales
  re.sub(r'<(random|srai|condition|set|get|star|bot|topic|that|person|li|learn|eval)[^>]*>.*?</\1>', '', text, flags=re.DOTALL)
  # Limpiar self-closing AIML tags
  re.sub(r'<(star|bot|person|br)\s*/?\s*>', '', text)
  # Limpiar HTML tags
  re.sub(r'<[^>]+>', '', text)
  # Filtrar wildcard-only
  re.match(r'^[_*]+$', pattern_text.strip())
  ```
- **Estado**: `DONE` ✅

### Task 1.3: Crear `pdf_validator.py` — Validador PDF
- **Archivo**: `pdf_validator.py` (nuevo)
- **Hereda de**: `SourceValidator` (dedicado, no genérico)
- **Problemas específicos PDF**:
  - Texto garbled/mojibake → detectar con ratio de caracteres de reemplazo (`\ufffd`)
  - Headers/footers de página → detectar con patrones repetidos cortos
  - Saltos de línea con guión (`Artifi-\ncial`) → reconstruir con regex
  - Páginas vacías (PDFs escaneados) → detectar y descartar
  - Tablas como texto jumbled → detectar con patrones de alineación
  - Artefactos de salto de página (`\x0c`) → limpiar
  - Números de página sueltos → detectar con regex `^\s*\d{1,4}\s*$`
- **Regex de limpieza**:
  ```python
  # Reconstruir guiones partidos
  re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
  # Limpiar page breaks
  re.sub(r'\x0c', ' ', text)
  # Detectar page numbers sueltos
  re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
  # Detectar headers/footers repetidos
  re.sub(r'^(.{1,30})\n\1\n', '', text, flags=re.MULTILINE)
  # Detectar mojibake
  replacement_ratio = text.count('\ufffd') / max(len(text), 1)
  ```
- **Estado**: `DONE` ✅

### Task 1.4: Crear `epub_validator.py` — Validador EPUB
- **Archivo**: `epub_validator.py` (nuevo)
- **Hereda de**: `SourceValidator` (dedicado)
- **Problemas específicos EPUB**:
  - Tags HTML residuales → limpiar con `html.parser` o `BeautifulSoup` lite
  - Entidades HTML (`&amp;`, `&lt;`, `&nbsp;`, `&#8217;`) → decodificar con `html.unescape()`
  - Bloques `<style>` y `<script>` → detectar y eliminar contenido
  - Navegación/TOC inyectado → detectar por tipo de item EPUB
  - Items vacíos → detectar y descartar
  - UTF-8 con `errors='ignore'` → detectar bytes perdidos
- **Estado**: `DONE` ✅

### Task 1.5: Crear `web_validator.py` — Validador Web
- **Archivo**: `web_validator.py` (nuevo)
- **Hereda de**: `SourceValidator` (dedicado)
- **Problemas específicos Web**:
  - Texto boilerplate (menús, headers, footers) → detectar con patrones de navegación
  - Contenido de anuncios → detectar con keywords (`ad`, `sponsored`, `click here`)
  - Texto de cookie notices → detectar con keywords (`cookie`, `privacy`, `accept`)
  - Contenido duplicado (sidebars, related articles) → deduplicar
  - URLs rotas / contenido vacío → detectar y descartar
  - JavaScript residual → limpiar
- **Estado**: `DONE` ✅

### Task 1.6: Crear `csv_validator.py` — Validador CSV
- **Archivo**: `csv_validator.py` (nuevo)
- **Hereda de**: `GenericValidator`
- **Problemas específicos CSV**:
  - Encoding incorrecto → detectar con `chardet` o intentar UTF-8/latin-1
  - Delimitador incorrecto → detectar con `csv.Sniffer`
  - Campos faltantes (input o output vacío) → descartar
  - Filas duplicadas → deduplicar
  - Headers como datos → detectar primera fila
  - Newlines dentro de campos → normalizar
- **Estado**: `DONE` ✅

### Task 1.7: Integrar validadores en `data_preparer.py`
- **Archivo**: `data_preparer.py`
- **Ubicación**: Después de `_standardize_combined_dataset()` (línea 680)
- **Nuevo paso**: `_validate_and_clean_sources()` 
- **Flujo**:
  ```python
  def _validate_and_clean_sources(self):
      validators = {
          'aiml': AIMLValidator(),
          'csv': CSVValidator(),
          'pdf': PDFValidator(),
          'epub': EPUBValidator(),
          'web': WebValidator(),
      }
      reports = {}
      for source_name, dataset in self._get_source_datasets():
          if source_name in validators and dataset:
              validator = validators[source_name]
              dataset, report = validator.validate_batch(dataset)
              reports[source_name] = report
      self._log_quality_reports(reports)
  ```
- **Estado**: `DONE` ✅

### Verificación Fase 1
- [ ] Unit test: Validador AIML detecta tags HTML y los limpia
- [ ] Unit test: Validador PDF detecta mojibake y descarta
- [ ] Unit test: Validador EPUB decodifica entidades HTML
- [ ] Unit test: Validador Web elimina boilerplate
- [ ] Unit test: Validador CSV detecta encoding incorrecto
- [ ] Integration test: Pipeline completo con validación, verificar estadísticas
- [ ] Manual test: Revisar muestras arregladas vs descartadas

---

## Fase 2: Generadores Thinking por Fuente — P1 (High)

**Prioridad:** P1 — Core del thinking real
**Tiempo estimado:** 12-15 horas
**Riesgo:** Medio (depende de Ollama para teacher model)

### Arquitectura de Generadores

```
┌─────────────────────────────────────────────────────────┐
│               ThinkingGenerator (Base)                   │
│  - generate(sample) → SampleWithThinking                 │
│  - validate_thinking(thinking, answer) → bool            │
│  - set_teacher_model(model: OllamaTeacher)               │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────────┐
    │                  │                      │
    ▼                  ▼                      ▼
┌─────────┐    ┌─────────────┐    ┌─────────────────┐
│Rule-based│    │Teacher-only │    │Hybrid           │
│Generators│    │Generators   │    │Generators       │
│(AIML     │    │(CSV, PDF,   │    │(AIML complejas) │
│ simple)  │    │ Web)        │    │                 │
└─────────┘    └─────────────┘    └─────────────────┘
```

### Task 2.1: Crear `thinking_generators.py` — Clase base + OllamaTeacher
- **Archivo**: `thinking_generators.py` (nuevo)
- **Clase base**: `ThinkingGenerator`
  ```python
  class ThinkingGenerator:
      def __init__(self, teacher_model: OllamaTeacher = None):
          self.teacher = teacher_model
      
      def generate(self, sample: dict) -> dict:
          """Genera thinking para un sample. Retorna dict con 'thinking' key."""
          raise NotImplementedError
      
      def validate_thinking(self, thinking: str, answer: str, question: str) -> bool:
          """Valida que el thinking sea razonamiento real."""
          checks = {
              'length': len(thinking) > 20,
              'no_meta': not thinking.startswith('El usuario'),
              'derivation': any(w in thinking.lower() for w in answer.lower().split()[:3]),
              'steps': any(ind in thinking.lower() for ind in ['porque', 'por lo tanto', 'primero', 'paso', 'análisis', 'because', 'therefore', 'first']),
          }
          return sum(checks.values()) >= 3
  ```
- **Clase**: `OllamaTeacher`
  ```python
  class OllamaTeacher:
      def __init__(self, model: str = 'qwen2.5:1.5b', url: str = 'http://localhost:11434'):
          self.model = model
          self.url = url
          self.cache = {}  # Evitar re-generar el mismo prompt
      
      def generate(self, prompt: str, max_tokens: int = 150) -> str:
          if prompt in self.cache:
              return self.cache[prompt]
          # POST a Ollama API
          # Fallback a None si no disponible
  ```
- **Estado**: `DONE` ✅

### Task 2.2: Crear `aiml_thinking.py` — Generador AIML
- **Archivo**: `aiml_thinking.py` (nuevo)
- **Hereda de**: `ThinkingGenerator`
- **Estrategia híbrida**:
  ```python
  class AIMLThinkingGenerator(ThinkingGenerator):
      # Categorías con rule-based
      SIMPLE_CATEGORIES = {
          'greeting': ['hello', 'hi', 'hey', 'hola', 'buenos'],
          'farewell': ['bye', 'goodbye', 'chao', 'adiós', 'hasta'],
          'identity': ['who are you', 'what is your name', 'qué eres'],
          'thanks': ['thank', 'gracias'],
          'yes': ['yes', 'sí', 'ok'],
          'no': ['no', 'nah'],
      }
      
      def generate(self, sample):
          pattern = sample['input'].lower()
          template = sample['output']
          
          # Rule-based para categorías simples
          for category, keywords in self.SIMPLE_CATEGORIES.items():
              if any(kw in pattern for kw in keywords):
                  return {'thinking': self._rule_based_thinking(category, template)}
          
          # Teacher model para complejas
          if self.teacher:
              thinking = self.teacher.generate(
                  f"Analiza esta pregunta AIML y genera un razonamiento paso a paso.\n"
                  f"Pregunta: {sample['input']}\n"
                  f"Respuesta del bot: {template}\n"
                  f"Razonamiento:"
              )
              return {'thinking': thinking}
          
          return {'thinking': None}  # Sin thinking si no hay teacher
      
      def _rule_based_thinking(self, category, template):
          if category == 'greeting':
              return f"El usuario hace un saludo. Respondo con un saludo amigable: {template[:50]}..."
          elif category == 'farewell':
              return f"El usuario se despide. Respondo con una despedida apropiada."
          # etc.
  ```
- **Estado**: `DONE` ✅

### Task 2.3: Crear `csv_thinking.py` — Generador CSV
- **Archivo**: `csv_thinking.py` (nuevo)
- **Hereda de**: `ThinkingGenerator`
- **Estrategia**: Teacher model siempre
  ```python
  class CSVThinkingGenerator(ThinkingGenerator):
      def generate(self, sample):
          question = sample.get('input', sample.get('question', ''))
          answer = sample.get('output', sample.get('answer', ''))
          
          prompt = f"""Analiza esta pregunta y respuesta. Genera un razonamiento paso a paso
que lleve lógicamente de la pregunta a la respuesta.

Pregunta: {question}
Respuesta correcta: {answer}

Razonamiento (1-3 oraciones, en español):"""
          
          thinking = self.teacher.generate(prompt)
          return {'thinking': thinking}
  ```
- **Estado**: `DONE` ✅

### Task 2.4: Crear `pdf_thinking.py` — Generador PDF
- **Archivo**: `pdf_thinking.py` (nuevo)
- **Hereda de**: `ThinkingGenerator`
- **Estrategia**: Teacher con contexto de chunk
  ```python
  class PDFThinkingGenerator(ThinkingGenerator):
      def generate(self, sample):
          text = sample.get('input_ids', sample.get('text', ''))
          # Extraer title del chunk si existe
          title = self._extract_title(text)
          
          prompt = f"""Basado en el siguiente texto, resume y razona sobre la información clave.

Texto: {text[:500]}{'...' if len(text) > 500 else ''}
{f'Título: {title}' if title else ''}

Razonamiento paso a paso:"""
          
          thinking = self.teacher.generate(prompt)
          return {'thinking': thinking}
  ```
- **Estado**: `DONE` ✅

### Task 2.5: Crear `epub_thinking.py` — Generador EPUB
- **Archivo**: `epub_thinking.py` (nuevo)
- **Hereda de**: `ThinkingGenerator`
- **Estrategia**: Teacher con contexto de capítulo
  ```python
  class EPUBThinkingGenerator(ThinkingGenerator):
      def generate(self, sample):
          text = sample.get('input_ids', sample.get('text', ''))
          chapter = sample.get('chapter', 'Unknown')
          
          prompt = f"""Contexto del capítulo: {chapter}
Contenido: {text[:500]}{'...' if len(text) > 500 else ''}

¿Qué información clave contiene este texto? Razona paso a paso:"""
          
          thinking = self.teacher.generate(prompt)
          return {'thinking': thinking}
  ```
- **Estado**: `DONE` ✅

### Task 2.6: Crear `web_thinking.py` — Generador Web
- **Archivo**: `web_thinking.py` (nuevo)
- **Hereda de**: `ThinkingGenerator`
- **Estrategia**: Teacher con contexto de página
  ```python
  class WebThinkingGenerator(ThinkingGenerator):
      def generate(self, sample):
          text = sample.get('input_ids', sample.get('text', ''))
          title = sample.get('title', 'Unknown')
          url = sample.get('url', '')
          
          prompt = f"""Documento: {title}
URL: {url}
Contenido: {text[:800]}

Resume y razona sobre el contenido principal:"""
          
          thinking = self.teacher.generate(prompt)
          return {'thinking': thinking}
  ```
- **Estado**: `DONE` ✅

### Task 2.7: Crear `hf_thinking.py` — Generador HuggingFace
- **Archivo**: `hf_thinking.py` (nuevo)
- **Hereda de**: `ThinkingGenerator`
- **Estrategia**: Teacher o passthrough
  ```python
  class HFThinkingGenerator(ThinkingGenerator):
      def generate(self, sample):
          # Si ya tiene estructura QA, generar thinking para la pregunta
          if 'question' in sample and 'answer' in sample:
              prompt = f"""Pregunta: {sample['question']}
Respuesta: {sample['answer']}
Genera un razonamiento paso a paso que lleve de la pregunta a la respuesta:"""
              thinking = self.teacher.generate(prompt)
              return {'thinking': thinking}
          
          # Texto continuo: generar resumen razonado
          text = sample.get('text', sample.get('input_ids', ''))
          prompt = f"""Texto: {text[:500]}
Resume y razona sobre la información clave:"""
          thinking = self.teacher.generate(prompt)
          return {'thinking': thinking}
  ```
- **Estado**: `DONE` ✅

### Task 2.8: Integrar generadores en `data_preparer.py`
- **Archivo**: `data_preparer.py`
- **Ubicación**: Después de validación (Fase 1), antes de BPE tokenization
- **Nuevo paso**: `_generate_thinking_for_sources()`
- **Flujo**:
  ```python
  def _generate_thinking_for_sources(self):
      teacher = OllamaTeacher(model=self.args.thinking_model)
      generators = {
          'aiml': AIMLThinkingGenerator(teacher),
          'csv': CSVThinkingGenerator(teacher),
          'pdf': PDFThinkingGenerator(teacher),
          'epub': EPUBThinkingGenerator(teacher),
          'web': WebThinkingGenerator(teacher),
          'hf': HFThinkingGenerator(teacher),
      }
      
      for source_name, dataset in self._get_source_datasets():
          if source_name in generators and dataset:
              generator = generators[source_name]
              dataset = dataset.map(generator.generate)
      
      # Formatear: <think>thinking</think>response
      self.combined_data = self._format_thinking_data(self.combined_data)
  ```
- **Estado**: `DONE` ✅

### Verificación Fase 2
- [ ] Unit test: AIML generator produce thinking para greeting
- [ ] Unit test: CSV generator produce thinking con Ollama
- [ ] Unit test: PDF generator incluye contexto del chunk
- [ ] Unit test: EPUB generator incluye título del capítulo
- [ ] Unit test: Web generator incluye URL y título
- [ ] Unit test: HF generator detecta formato QA
- [ ] Unit test: validate_thinking() clasifica correctly
- [ ] Integration test: Generar thinking para cada fuente, verificar formato
- [ ] Manual test: Revisar 10 muestras por fuente, verificar calidad del reasoning

---

## Fase 3: Integración Pipeline — P1 (High)

**Prioridad:** P1 — Todo debe conectarse correctamente
**Tiempo estimado:** 4-6 horas
**Riesgo:** Bajo (integración)

### Task 3.1: Actualizar `data_preparer.py` — Pipeline completo
- **Archivo**: `data_preparer.py`
- **Ubicación**: Método `prepare()` (líneas 605-737)
- **Nuevo flujo**:
  ```python
  def prepare(self):
      # ... existing steps 1-6 ...
      
      # NUEVO: Validación por fuente
      if hasattr(self.args, 'validate_sources') and self.args.validate_sources:
          logger.info("\n[6.5/7] Validating and cleaning sources...")
          self._validate_and_clean_sources()
      
      # NUEVO: Generación thinking por fuente
      if hasattr(self.args, 'generate_thinking') and self.args.generate_thinking:
          logger.info("\n[6.6/7] Generating real thinking data...")
          self._generate_thinking_for_sources()
      
      # ... existing steps 7+ ...
  ```
- **Estado**: `DONE` ✅

### Task 3.2: Actualizar `main.py` — Nuevos flags CLI
- **Archivo**: `main.py`
- **Nuevos flags**:
  ```python
  # Validación por fuente
  '--validate-sources'   # Activar validación/limpieza por fuente
  
  # Thinking real
  '--generate-thinking'  # Generar thinking real durante preparación
  '--thinking-model'     # Modelo teacher para thinking (default: qwen2.5:1.5b)
  '--thinking-depth'     # Profundidad: basic | adaptive | detailed
  ```
- **Líneas**: ~620-650
- **Estado**: `DONE` ✅

### Task 3.3: Actualizar `main_train.py` — Métricas thinking separadas
- **Archivo**: `main_train.py`
- **Cambio**: `_compute_thinking_metrics()` separar thinking_loss de response_loss
- **Métricas nuevas**:
  - `thinking_loss` — loss promedio en tokens de thinking
  - `response_loss` — loss promedio en tokens de respuesta
  - `thinking_length_avg` — largo promedio del bloque thinking (tokens)
  - `thinking_coverage` — % del output que es thinking
  - `response_token_accuracy` — accuracy solo en tokens de respuesta
- **Líneas**: 445-486
- **Estado**: `DONE` ✅

### Task 3.4: Actualizar `dialogmanager.py` — Prompt de inferencia
- **Archivo**: `dialogmanager.py`
- **Cambio**: Añadir instrucción de thinking al prompt
- **Línea**: ~205
- **Estado**: `DONE` ✅

### Verificación Fase 3
- [ ] Integration test: Pipeline completo `--prepare-data --aiml --csv --generate-thinking --validate-sources`
- [ ] Integration test: Training con thinking real, verificar loss convergence
- [ ] Integration test: Chat con thinking real, verificar output
- [ ] API test: POST con `include_thinking: true`

---

## Fase 4: Validación de Calidad — P2 (Medium)

**Prioridad:** P2 — Asegurar calidad del thinking generado
**Tiempo estimado:** 3-4 horas
**Riesgo:** Bajo

### Task 4.1: Crear `thinking_quality.py` — Validador de calidad
- **Archivo**: `thinking_quality.py` (nuevo)
- **Funciones**:
  ```python
  def validate_thinking_batch(samples: list) -> dict:
      """Valida un batch de samples con thinking."""
      results = {
          'valid': 0,
          'too_short': 0,
          'meta_commentary': 0,
          'no_derivation': 0,
          'low_quality': 0,
      }
      for sample in samples:
          thinking = sample.get('thinking', '')
          answer = sample.get('output', '')
          question = sample.get('input', '')
          
          if len(thinking) < 20:
              results['too_short'] += 1
          elif thinking.startswith('El usuario') or thinking.startswith('The user'):
              results['meta_commentary'] += 1
          elif not any(w in thinking.lower() for w in answer.lower().split()[:3]):
              results['no_derivation'] += 1
          else:
              results['valid'] += 1
      
      return results
  ```
- **Estado**: `DONE` ✅

### Task 4.2: Integrar validación en pipeline
- **Archivo**: `data_preparer.py`
- **Ubicación**: Después de generación thinking
- **Flujo**: Generar → Validar → Filtrar bajas calidades → Reportar
- **Estado**: `DONE` ✅

### Verificación Fase 4
- [ ] Unit test: validate_thinking_batch clasifica correctly
- [ ] Integration test: Pipeline con validación, verificar filtrado

---

## Fase 5: CLI y Configuración — P2 (Medium)

**Prioridad:** P2 — User-facing configuration
**Tiempo estimado:** 2-3 horas
**Riesgo:** Bajo

### Task 5.1: Flags CLI completos
- **Archivo**: `main.py`
- **Flags**:
  ```bash
  # Preparación con thinking
  python main.py --prepare-data --aiml --csv --pdf \
      --generate-thinking \
      --validate-sources \
      --thinking-model qwen2.5:1.5b \
      --thinking-depth adaptive
  
  # Solo validación (sin thinking)
  python main.py --prepare-data --aiml --csv --validate-sources
  
  # Training
  python main.py --train --use-cache --epochs 30 \
      --thinking-loss-weight 1.0
  
  # Chat
  python main.py --chat --model mymodel --show-thinking
  ```
- **Estado**: `DONE` ✅

### Task 5.2: Configuración en TRAINING_CONFIG
- **Archivo**: `main_train.py`
- **Entradas nuevas**:
  ```python
  TRAINING_CONFIG = {
      # ... existente ...
      'thinking_loss_weight': 1.0,  # Cambiado de 0.5
      'thinking_enabled': True,
      'thinking_stop_on_end': True,
      'thinking_metrics_interval': 50,
  }
  ```
- **Estado**: `DONE` ✅

### Verificación Fase 5
- [ ] Unit test: CLI flags parsed correctly
- [ ] Integration test: Config propagada correctamente

---

## Fase 6: Testing y Documentación — P2 (Medium)

**Prioridad:** P2 — Todo debe estar documentado y testeado
**Tiempo estimado:** 4-6 horas
**Riesgo:** Bajo

### Task 6.1: Tests unitarios
- **Archivo**: `tests/test_source_validators.py` (nuevo)
- **Tests**:
  - `test_aiml_validator_cleans_html_tags()`
  - `test_pdf_validator_detects_mojibake()`
  - `test_epub_validator_decodes_entities()`
  - `test_web_validator_removes_boilerplate()`
  - `test_csv_validator_detects_encoding()`
  - `test_generic_validator_classifies_correctly()`

- **Archivo**: `tests/test_thinking_generators.py` (nuevo)
- **Tests**:
  - `test_aiml_generator_rule_based()`
  - `test_csv_generator_with_teacher()`
  - `test_pdf_generator_includes_context()`
  - `test_epub_generator_includes_chapter()`
  - `test_web_generator_includes_url()`
  - `test_hf_generator_detects_qa()`
  - `test_validate_thinking_quality()`

- **Archivo**: `tests/test_thinking.py` (actualizar)
- **Tests**:
  - `test_loss_masking_weight_1_0()`
  - `test_thinking_generation_with_real_data()`
  - `test_thinking_in_inference_prompt()`

- **Estado**: `DONE` ✅

### Task 6.2: Documentación
- **Archivos a actualizar**:
  - `README.md` — Sección de thinking con ejemplos de uso
  - `THINKING_GUIDE.md` — Guía completa de thinking real
  - `APP_ARCHITECTURE.md` — Diagrama de flujo actualizado
  - `AGENTS.md` — Nuevas herramientas por fuente

- **Estado**: `TODO`

### Verificación Fase 6
- [ ] `pytest tests/` — Todos los tests pasan
- [ ] `pytest tests/test_source_validators.py -v` — Validadores OK
- [ ] `pytest tests/test_thinking_generators.py -v` — Generadores OK
- [ ] Revisar documentación completeness

---

## Orden de Implementación

```
Fase 0 (Fix Crítico) — P0 — 1-2h
  → Task 0.1 (loss weight 1.0)
  → Task 0.2 (prompt inference)
  → [VERIFY] Training + chat

Fase 1 (Validadores) — P1 — 8-10h
  → Task 1.1 (clase base + genérico)
  → Task 1.2 (AIML validator)
  → Task 1.3 (PDF validator)
  → Task 1.4 (EPUB validator)
  → Task 1.5 (Web validator)
  → Task 1.6 (CSV validator)
  → Task 1.7 (integrar en data_preparer)
  → [TEST] Unit + integration

Fase 2 (Generadores Thinking) — P1 — 12-15h
  → Task 2.1 (clase base + OllamaTeacher)
  → Task 2.2 (AIML generator)
  → Task 2.3 (CSV generator)
  → Task 2.4 (PDF generator)
  → Task 2.5 (EPUB generator)
  → Task 2.6 (Web generator)
  → Task 2.7 (HF generator)
  → Task 2.8 (integrar en data_preparer)
  → [TEST] Unit + integration

Fase 3 (Integración Pipeline) — P1 — 4-6h
  → Task 3.1 (data_preparer pipeline)
  → Task 3.2 (CLI flags)
  → Task 3.3 (métricas)
  → Task 3.4 (prompt inference)
  → [TEST] Full pipeline

Fase 4 (Validación Calidad) — P2 — 3-4h
  → Task 4.1 (thinking_quality.py)
  → Task 4.2 (integrar validación)
  → [TEST] Quality checks

Fase 5 (CLI/Config) — P2 — 2-3h
  → Task 5.1 (flags completos)
  → Task 5.2 (TRAINING_CONFIG)
  → [TEST] Config propagation

Fase 6 (Testing/Docs) — P2 — 4-6h
  → Task 6.1 (tests unitarios)
  → Task 6.2 (documentación)
  → [TEST] All tests pass
```

**Tiempo total estimado: 34-46 horas**

---

## Key Files Summary

| Archivo | Tareas | Prioridad | Líneas Cambiadas |
|---------|--------|-----------|-----------------|
| `source_validators.py` | 1.1 | P1 | ~300 (nuevo) |
| `aiml_validator.py` | 1.2 | P1 | ~150 (nuevo) |
| `pdf_validator.py` | 1.3 | P1 | ~120 (nuevo) |
| `epub_validator.py` | 1.4 | P1 | ~100 (nuevo) |
| `web_validator.py` | 1.5 | P1 | ~100 (nuevo) |
| `csv_validator.py` | 1.6 | P1 | ~80 (nuevo) |
| `thinking_generators.py` | 2.1 | P1 | ~200 (nuevo) |
| `aiml_thinking.py` | 2.2 | P1 | ~120 (nuevo) |
| `csv_thinking.py` | 2.3 | P1 | ~80 (nuevo) |
| `pdf_thinking.py` | 2.4 | P1 | ~80 (nuevo) |
| `epub_thinking.py` | 2.5 | P1 | ~80 (nuevo) |
| `web_thinking.py` | 2.6 | P1 | ~80 (nuevo) |
| `hf_thinking.py` | 2.7 | P1 | ~80 (nuevo) |
| `thinking_quality.py` | 4.1 | P2 | ~100 (nuevo) |
| `data_preparer.py` | 1.7, 2.8, 3.1 | P1 | ~100 |
| `main.py` | 3.2, 5.1 | P1-P2 | ~50 |
| `main_train.py` | 0.1, 3.3, 5.2 | P0-P2 | ~50 |
| `dialogmanager.py` | 0.2, 3.4 | P0-P1 | ~10 |
| `tests/test_source_validators.py` | 6.1 | P2 | ~200 (nuevo) |
| `tests/test_thinking_generators.py` | 6.1 | P2 | ~200 (nuevo) |
| `tests/test_thinking.py` | 6.1 | P2 | ~100 (actualizar) |

---

## Dependencies

| Tarea | Depende De | Prioridad |
|-------|-----------|-----------|
| 0.1 | (ninguna) | P0 |
| 0.2 | (ninguna) | P0 |
| 1.1 | (ninguna) | P1 |
| 1.2 | 1.1 | P1 |
| 1.3 | 1.1 | P1 |
| 1.4 | 1.1 | P1 |
| 1.5 | 1.1 | P1 |
| 1.6 | 1.1 | P1 |
| 1.7 | 1.2-1.6 | P1 |
| 2.1 | (ninguna) | P1 |
| 2.2 | 2.1 | P1 |
| 2.3 | 2.1 | P1 |
| 2.4 | 2.1 | P1 |
| 2.5 | 2.1 | P1 |
| 2.6 | 2.1 | P1 |
| 2.7 | 2.1 | P1 |
| 2.8 | 2.2-2.7, 1.7 | P1 |
| 3.1 | 1.7, 2.8 | P1 |
| 3.2 | 3.1 | P1 |
| 3.3 | 3.1 | P1 |
| 3.4 | 3.1 | P1 |
| 4.1 | (ninguna) | P2 |
| 4.2 | 4.1, 3.1 | P2 |
| 5.1 | 3.2 | P2 |
| 5.2 | 0.1 | P2 |
| 6.1 | Todas | P2 |
| 6.2 | Todas | P2 |

---

## Testing Strategy

### Por Fase
- **Fase 0:** Verificación manual de fix crítico
- **Fase 1:** Unit tests por validador + integration test pipeline
- **Fase 2:** Unit tests por generador + integration test generación
- **Fase 3:** Integration test pipeline completo
- **Fase 4:** Unit test quality validation
- **Fase 5:** Unit test CLI parsing
- **Fase 6:** All tests + documentación

### Checklist Validación Final
```bash
# 1. Preparar datos con validación + thinking
python main.py --prepare-data --aiml --csv --pdf \
    --validate-sources --generate-thinking \
    --thinking-model qwen2.5:1.5b

# 2. Entrenar con thinking real
python main.py --train --use-cache --epochs 30 \
    --thinking-loss-weight 1.0

# 3. Chat con thinking real
python main.py --chat --model mymodel --show-thinking

# 4. API test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"¿Qué es Python?"}],"include_thinking":true}'

# 5. Ejecutar todos los tests
pytest tests/ -v
```

---

## Comparación: Thinking Fake vs Thinking Real

| Aspecto | Thinking Fake (Actual) | Thinking Real (Objetivo) |
|---------|----------------------|------------------------|
| **Generador** | Templates genéricos | Teacher model por fuente |
| **Contenido** | Meta-comentarios ("El usuario saluda") | Razonamiento real ("La pregunta es sobre... porque... por lo tanto...") |
| **Pérdida** | 0.5 (reduce aprendizaje) | 1.0 (mantiene peso) |
| **Prompt** | Sin instrucción de thinking | "Piensa paso a paso antes de responder" |
| **Validación** | No existe | Por fuente + calidad |
| **Largo** | 5-15 tokens | 30-100 tokens |
| **Calidad** | Baja (no razona) | Alta (razonamiento auténtico) |

---

*Generado desde análisis del código base — refleja el estado actual y las mejoras planificadas.*
