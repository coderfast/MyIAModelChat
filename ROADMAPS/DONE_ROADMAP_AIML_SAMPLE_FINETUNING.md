# ROADMAP: AIML Sample Fine-Tuning Parser

## Estado: TODO

## Objetivo

Crear un parser AIML inteligente que resuelva los elementos del estandar AIML 2.0 y genere samples de calidad para el dataset, previo a contamination filtering.

## Problemas Identificados

| # | Problema | Impacto | Ubicacion |
|---|----------|---------|-----------|
| 1 | `ElementTree.find('template').text` solo extrae texto antes del primer hijo XML | Pierde 70-80% del contenido de templates | `loader.py:99-101` |
| 2 | Wildcards (`*`, `_`, `**`, `^`) pasan como texto literal en patterns | Samples sin valor para training | `loader.py:98` |
| 3 | `<srai>` no se resuelve | Redirects no procesados | `loader.py:100` |
| 4 | `<random><li>` solo toma el primero | Pierde variedad de respuestas | `loader.py:101` |
| 5 | `<set>/<get>/<bot>` no se resuelven | Variables sin valor concreto | `loader.py:100-101` |
| 6 | `<thinking>` de AIML choca con `<thinking>` de training | Contaminacion de formato | `loader.py:100` |
| 7 | `<that>/<topic>` no se eliminan | Elementos runtime en datos de training | `loader.py:100-101` |
| 8 | `<person>/<gender>` no se procesan | Pronombres sin convertir | `loader.py:100` |
| 9 | HTML tags (`<br>`, `<a>`, `<em>`) no se eliminan | Contenido HTML residual | `loader.py:100` |
| 10 | Codificacion ISO-8859-1 sin normalizar | Mojibake posible | `loader.py:93` |

## Decisiones Tomadas

| Pregunta | Decision |
|----------|----------|
| `<srai>`: resolver o descartar? | **Resolver** - seguir cadena de redirects (max 5 niveles) |
| Wildcards: generar muestras o descartar? | **Generar multiples samples** con ejemplos por wildcard |
| Formato de salida? | **Normalizar** a `{"input_ids": "PATTERN: RESPONSE", "source": "AIML"}` |
| `<random><li>`: 1 sample o N? | **N samples** - 1 por cada `<li>` |

## Arquitectura Propuesta

```
dataset_preparer/aiml/
├── __init__.py
├── loader.py              # MODIFICAR: usar nuevo parser
├── thinking.py            # SIN CAMBIO
└── parser.py              # NUEVO: parser AIML inteligente
```

### Pipeline del Parser

```
Category AIML (XML)
    │
    ▼
[1] parse_category(category) -> raw_sample
    │  Extrae pattern + template XML completo (NO usa .text)
    │
    ▼
[2] resolve_template(template_element) -> str | List[str]
    │  Resuelve recursivamente:
    │  ├── <thinking>     → ELIMINAR
    │  ├── <that>/<topic> → ELIMINAR
    │  ├── <set>          → extraer valor literal
    │  ├── <get>          → [VAR:nombre]
    │  ├── <bot>          → "Alice" (config)
    │  ├── <star>         → [X], [X1], [X2]
    │  ├── <srai>         → resolver cadena (max 5)
    │  ├── <random><li>   → List[str] (1 por <li>)
    │  ├── <condition>    → primer <li> valido
    │  ├── <person>       → conversion basica I->You
    │  ├── <input>        → [HISTORY]
    │  ├── HTML tags      → eliminar
    │  └── text           → preservar
    │
    ▼
[3] clean_pattern(pattern_text) -> str
    │  * → [X], _ → [PHRASE], ** → [TEXT], ^ → [OPTIONAL]
    │
    ▼
[4] expand_sample(raw_sample) -> List[sample]
    │  ├── <random><li> → N samples
    │  ├── Wildcards *  → K samples (2-3 por wildcard)
    │  ├── <srai> chain → sample resuelto
    │  └── Simple       → 1 sample
    │
    ▼
[5] normalize(sample) -> {"input_ids": "...", "source": "AIML", ...}
    │
    ▼
[6] classify_quality(sample) -> good/fixable/discardable
```

## Transformaciones por Elemento AIML

| Elemento | Transformacion | Ejemplo |
|----------|---------------|---------|
| `<star/>` | `[X]` | `YOU ARE * ME` → `YOU ARE [X] ME` |
| `<star index="N"/>` | `[XN]` | `MY * IS *` → `MY [X1] IS [X2]` |
| `_` | `[PHRASE]` | `_ IS GOOD` → `[PHRASE] IS GOOD` |
| `**` | `[TEXT]` | `TELL ME ABOUT **` → `TELL ME ABOUT [TEXT]` |
| `^` | `[OPTIONAL]` | `WHAT IS ^ NAME` → `WHAT IS [OPTIONAL] NAME` |
| `<set name="X">V</set>` | `V` | Solo el valor literal |
| `<get name="X"/>` | `[VAR:X]` | Placeholder de variable |
| `<bot name="name"/>` | `Alice` | Valor configurable |
| `<srai>PATRON</srai>` | Resolver cadena | Seguir redirect |
| `<random><li>A</li><li>B</li>` | `["A", "B"]` | Lista de opciones |
| `<person>I → You</person>` | Conversion basica | Pronombres |
| `<thinking>...</thinking>` | **Eliminar** | AIML thinking, no training |
| `<that>...</that>` | **Eliminar** | Runtime-only |
| `<topic>...</topic>` | **Eliminar** | Runtime-only |
| `<condition>` | Primer `<li>` | Simplificar |
| `<input index="N"/>` | `[HISTORY]` | Placeholder |
| `<br/>`, `<a>`, `<em>` | **Eliminar** | HTML residual |

## Diccionario de Wildcards

```python
WILDCARD_EXAMPLES = {
    'greeting': {
        '*': ['HELLO', 'HI', 'GOOD MORNING', 'HOW ARE YOU'],
    },
    'farewell': {
        '*': ['GOODBYE', 'BYE', 'SEE YOU LATER'],
    },
    'identity': {
        '*': ['A ROBOT', 'AN AI', 'A CHATBOT', 'A PROGRAM'],
    },
    'generic': {
        '*': ['SOMETHING', 'ANYTHING', 'THAT', 'IT'],
        '_': ['SOMETHING NICE', 'A GOOD THING', 'MY FRIEND'],
        '**': ['SOMETHING INTERESTING', 'A TOPIC'],
    },
    'action': {
        '*': ['LIKE', 'LOVE', 'HATE', 'WANT'],
    },
    'adjective': {
        '*': ['FUNNY', 'SMART', 'NICE', 'WEIRD', 'COOL'],
    },
}

CATEGORY_KEYWORDS = {
    'greeting': ['HELLO', 'HI', 'HEY', 'GOOD MORNING'],
    'farewell': ['GOODBYE', 'BYE', 'SEE YOU'],
    'identity': ['WHO ARE YOU', 'WHAT ARE YOU', 'YOUR NAME'],
    'question': ['WHAT', 'WHY', 'HOW', 'WHEN', 'WHERE'],
    'opinion': ['DO YOU LIKE', 'DO YOU THINK', 'FAVORITE'],
}
```

## Ejemplos de Expansion

### Wildcards en Pattern
```
Pattern: "YOU ARE *"
Template: "Thank you, I try my best."

→ "YOU ARE A ROBOT"     → "Thank you, I try my best."
→ "YOU ARE FUNNY"       → "Thank you, I try my best."
→ "YOU ARE SMART"       → "Thank you, I try my best."
```

### Resolucion de `<srai>`
```
Pattern: "WHAT ARE YOU CALLED"
Template: <srai>what is your name</srai>

→ Resuelve patron "WHAT IS YOUR NAME"
→ Template: "My <bot name='name'/> is Alice."
→ Resultado: "WHAT IS YOUR NAME" → "My name is Alice."
```

### Expansion de `<random>`
```
Pattern: "DO YOU LIKE *"
Template: <random><li>Yes, I do.</li><li>Sometimes.</li><li>Not really.</li></random>

→ "DO YOU LIKE CATS" → "Yes, I do."
→ "DO YOU LIKE CATS" → "Sometimes."
→ "DO YOU LIKE CATS" → "Not really."
```

## Formato de Salida Normalizado

```python
{
    "input_ids": "WHAT IS YOUR NAME: My name is Alice.",
    "source": "AIML",
    "original_pattern": "WHAT IS YOUR NAME",
    "original_template": "My name is <bot name='name'/>.",
    "wildcards_resolved": False,
    "srai_resolved": False,
    "random_expanded": False,
    "quality": "good"
}
```

## Filtros de Calidad Pre-Contamination

| Criterio | Accion |
|----------|--------|
| Pattern es solo wildcards (`* * *`) | Discard |
| Template < 3 chars despues de resolver | Discard |
| Template era solo `<thinking>` (AIML) | Discard |
| Template tiene `<srai>` no resuelto | Discard |
| Template tiene `<thinking>` + texto real | Fix: eliminar `<thinking>`, conservar texto |
| Template tiene HTML tags | Fix: limpiar tags |
| Pattern tiene wildcards + template sustantivo | Good: se expande |

## Orden de Implementacion

| Fase | Archivo | Descripcion | Estado |
|------|---------|-------------|--------|
| 1 | `aiml/parser.py` | Crear parser: resolve_template, clean_pattern | TODO |
| 2 | `aiml/parser.py` | Agregar expand_wildcards con diccionario | TODO |
| 3 | `aiml/parser.py` | Agregar resolve_srai con index_categories | TODO |
| 4 | `aiml/parser.py` | Agregar expand_random para `<random><li>` | TODO |
| 5 | `aiml/parser.py` | Agregar classify_quality y normalize | TODO |
| 6 | `aiml/loader.py` | Integrar nuevo parser en create_hf_dataset | TODO |
| 7 | Tests | Verificar con archivos reales | TODO |

## Archivos a Modificar

| Archivo | Cambio |
|---------|--------|
| `dataset_preparer/aiml/parser.py` | **NUEVO** - Parser AIML inteligente (~400 lineas) |
| `dataset_preparer/aiml/loader.py` | Integrar parser, reemplazar Extraccion manual (~50 lineas cambia) |
| `dataset_preparer/source_validators.py` | AIMLValidator: ajustar checks post-parser |
| `main.py` | Sin cambio (flags existentes suficientes) |

## Metricas de Exito

| Metrica | Objetivo |
|---------|----------|
| Samples extraidos vs actual | >= 3x mas samples |
| Templates con contenido real | >= 80% (vs ~20% actual) |
| Wildcards resueltos correctamente | >= 90% |
| `<srai>` chains resueltas | >= 70% |
| `<random>` expandidos correctamente | 100% |
| Tiempo de procesamiento | < 5x del actual |
| Samples discardados por calidad | < 15% |

## Riesgos

| Riesgo | Impacto | Mitigation |
|--------|---------|------------|
| Loops infinitos en `<srai>` | Alto | MAX_SRAI_DEPTH = 5 |
| Wildcards generan samples nonsensicos | Medio | Diccionario contextual por categoria |
| `<srai>` apunta a patron inexistente | Bajo | Descartar sample si target no encontrado |
| `<random>` con muchos `<li>` | Bajo | Limitar a max 10 `<li>` por category |
| Performance con archivos grandes | Medio | Procesar por chunks, cachear categories indexadas |

---

*Last updated: 2026-07-30*
