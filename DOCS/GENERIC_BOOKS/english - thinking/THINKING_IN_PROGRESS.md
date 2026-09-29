# THINKING_IN_PROGRESS.md

**Carpeta**: `G:\PROJECTS\AI\PROJECTS\MyIAModelChat\DOCS\GENERIC_BOOKS\english - thinking\`
**Fecha de inicio**: 2026-09-23
**Ultima actualizacion**: 2026-09-29
**Estado**: Fases 0-3 COMPLETADAS (2026-09-29) — corpus con thinking listo (121 archivos); pendiente Fase 4 (matematicas) y fusion BPE/entrenamiento

---

## Objetivo

Generar cadenas de razonamiento (`<|thinking|>...<|final|>`) en ingles, parrafo a parrafo, para
entrenar al modelo de IA MyIAModelChat en razonamiento encadenado (CoT), sobre los libros MD
de esta carpeta.

---

## Resumen de cambios (v1 → v2)

| Aspecto | v1 (2026-09-23) | v2 (2026-09-29) |
|---------|-----------------|-----------------|
| Ruta del proyecto | `G:\PROJECTS\MyIAModelChat\...` (incorrecta) | `G:\PROJECTS\AI\PROJECTS\MyIAModelChat\...` (corregida) |
| Alcance | Los 151 MD sin filtro | **121 MD limpios** (122 inicialmente + revisión: `125` descubierto 100% plantilla); 30 excluidos |
| Calidad asumida | "151 archivos buenos" | Auditoria real: 29 son boilerplate, 14,4% bloques duplicados |
| Thinking | ThinkingEngine tal cual | **ThinkingEngine arreglado (solo NLP; Ollama prohibido)** |
| ThinkingEngine | Asumido correcto | **Bugs verificados en vivo** (ver Fase 0) — requiere arreglo antes de usar |
| Matemáticas | Dentro del alcance | **Excluidas** (141-144 son filler puro); dataset real aparte en fase posterior |

---

## Auditoria de datos (2026-09-29, evidencia real)

### 1. 30 archivos son relleno de plantilla — NO aptos para entrenar

Detectados por la frase magica *"represents an extremely broad and fascinating area of study"*:
**~82% de sus parrafos son el mismo texto con solo el titulo de seccion sustituido**
(`addition` → `subtraction` → `triangles`, etc.), sin ningun contenido real del tema.

- **Excluidos**: `121`-`150` (30 archivos, ~4,7M palabras de paja) — incluye `125`,
  detectado despues al verificar las 6 variantes del template (`125` tenia 16,5% T1
  pero **100% de bloques** con la familia completa T0-T6)
- **Incluyen todos los de matemáticas**: `141_arithmetic_numbers`, `142_fractions_decimals`,
  `143_basic_geometry`, `144_elementary_algebra` — cero formulas, cero ejemplos, cero
  ejercicios resueltos. Ejemplo real de `143`: *"triangles is a dynamic and constantly
  evolving field"* (ni siquiera define que es un triangulo).
- Señales: textos gramaticalmente rotos, parrafos de hasta 3.270 palabras, secciones
  idénticas repetidas 5+ veces dentro del mismo archivo.

### 2. 121 archivos limpios — aptos ✅ (procesados en Fases 1-2)

- `001`-`120` (UE 001-040 + General 041-120) + `151_molecular_biology`
- **24.076 parrafos, ~2,42M palabras** (recuento real con `parse_blocks`;
  la auditoria inicial de 24.968 fusionaba headers con texto por falta de linea en blanco)
- Contenido factual real verificado: `001` (Historia UE), `075` (Sistema Solar), `100` (IA) — buenos ejemplos.
- `125` quedo descartado tras revision profunda (ver seccion 1).

### 3. Duplicados

- **14,4%** de los bloques son exactamente identicos entre archivos (se deduplica con cache MD5 + `contamination/dedup.py` del proyecto).
- `topics.json` tenia `total_topics: 150` con **151 entradas** — ✅ corregido a 151.
- Titulos repetidos: `072`/`104` "History of Science" y `100`/`107` "Artificial Intelligence".
  **Contenido verificado: 0 bloques compartidos** → se conservan ambos, solo se desambigua el
  titulo en `topics.json` (p. ej. "(Advanced)") para que el `context={'title': ...}` no repita texto.

### 4. ThinkingEngine — bugs verificados (hay que arreglar ANTES de generar)

Prueba en vivo con parrafo real de `001_history_european_union.md`:

```
Analyzing about 'History of the European Union'. First,the text contains 5 sentences
with 132 words in total. Additionally,the text presents a narrative or description of
events. On the other hand,the key concepts identified are: : europe was devastated after
the second world war, and the need for reconstruction was combined with...
```

Problemas:
1. **Carga spaCy `es` sobre texto en ingles** → NER y noun-chunks incorrectos; los "concepts"
   son frases enteras de 30 palabras y las "entities" tambien (`the success of the ecsc was
   due to several factors.`).
2. **Formato roto**: `First,the` (falta espacio), `are: : europe` (doble dos puntos),
   conceptos con coma final.
3. **Conceptual**: genera **metadatos** (*"contains 5 sentences with 132 words"*), no
   razonamiento. Como supervision CoT esto ensena al modelo a emitir estadisticas, no a pensar.

### 5. Infraestructura

- **Ollama PROHIBIDO** (decision 2026-09-29) — no se usa en ningun paso del plan.
  Solo spaCy `en_core_web_sm` (ya instalado) + regex fallback.

---

## Decisiones confirmadas

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| **Idioma del thinking** | Ingles (en) | Carpeta "english - thinking"; config inglesa en `thinking_engine_config.json` |
| **Origen del thinking** | **Solo NLP (ThinkingEngine arreglado)** | **Ollama PROHIBIDO** (decision 2026-09-29). Razonamiento via spaCy `en_core_web_sm` + plantillas arregladas. Sin infraestructura adicional |
| **Alcance** | **121 MD limpios en 2 fases** ✅ | Piloto 20 + resto 101. Los 30 de relleno excluidos (125 incluido). El problema era calidad, no cantidad |
| **Matematicas** | **Excluir 141-144; dataset real aparte (fase posterior)** | Imposible generar thinking honesto sobre filler; luego se añade CoT matematico real (formato `<|problem|><|thinking|><|final|>`) |
| **Ambito** | Solo parrafos de cuerpo | Encabezados `#`, `##`, `###` intactos |
| **Profundidad** | `adaptive` | `ThinkingEngine.DEPTH_CONFIG` |
| **Formato** | `<|thinking|>razonamiento<|final|>` | Token estandar GPT-2 del proyecto |
| **Sin LLM externo** | Confirmado | **Solo NLP local (spaCy/regex). Ollama y APIs de pago prohibidos** |

---

## Alcance explicito

**IN (121 archivos)**: `001`–`120`, `151`
*(revision 2026-09-29: `125` excluido — verificado **100% plantilla** (1.813/1.813
bloques con las 6 variantes del template; el 16,5% inicial solo contaba T1 de 6)*

**OUT (30 archivos — relleno de plantilla)**:
`121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150`

**Piloto (Fase 1, 20 archivos)** — muestra representativa por categoria, tamano y complejidad:

| # | Archivo | Motivo de inclusion |
|---|---------|---------------------|
| 1 | 001_history_european_union | UE, referencia de calidad |
| 2 | 020_legislative_process_eu | UE, texto institucional |
| 3 | 031_european_green_deal | UE, ciencia+politica |
| 4 | 041_greetings_introductions | General, texto corto/coloquial |
| 5 | 045_climate_weather | General, ciencia divulgativa |
| 6 | 054_capitals_cities | Muchos bloques (309) — lista/datos |
| 7 | 066_ancient_civilizations | Historia narrativa |
| 8 | 075_solar_system | Ciencia con cifras (numeros escritos) |
| 9 | 081_human_body | Ciencia, 344 bloques |
| 10 | 087_force_motion | Fisica basica (lo mas cercano a razonamiento STEM) |
| 11 | 092_chemical_elements | Quimica, terminos tecnicos |
| 12 | 100_artificial_intelligence | Tecnologia |
| 13 | 102_basic_programming | 421 bloques, terminologia tecnica |
| 14 | 106_contemporary_philosophy | 454 bloques, texto argumentativo |
| 15 | 111_meat_fish | 319 bloques, tematica cotidiana |
| 16 | 116_football | Deporte |
| 17 | 058_autonomous_communities | Geografia/Espana |
| 18 | 097_computers | Tecnologia basica |
| 19 | 103_immune_system | Biomedicina, 352 bloques |
| 20 | 105_tropical_ecosystems | Ecologia, 414 bloques |

**Fase 2**: los 101 restantes (`002-019, 021-030, 032-040, 042-044, 046-053, 055-057,
059-065, 067-074, 076-080, 082-086, 088-091, 093-096, 098-099, 101, 104, 107-110,
112-115, 117-120, 151`) — ✅ ejecutada 2026-09-29.

---

## Especificaciones tecnicas

### Formato de insercion

De:

    El texto del parrafo comienza aqui...

A:

    <|thinking|>Analyzing about 'Titulo del tema'. This paragraph explains [idea central]. It connects [concepto A] with [concepto B] because [razon del texto]. Key concepts: c1, c2, c3.<|final|>

    El texto del parrafo comienza aqui...

### Logica de segmentacion

1. Leer archivo MD completo (UTF-8)
2. Separar encabezados (`#`/`##`/`###`) de bloques de parrafo (separados por linea en blanco)
3. **Normalizar**: parrafos >400 palabras se parten en fronteras de oracion (5,7% del corpus;
   parrafo de 3.270 palabras es inutil como muestra de entrenamiento)
4. **Omitir thinking** si:
   - parrafo <10 palabras
   - es boilerplate de intro (*"This book presents a complete and detailed study"* — presente en 28 archivos)
   - ya tiene `<|thinking|>` (idempotente / re-ejecutable)
5. Para el resto: generar thinking (NLP) y anteponer `<|thinking|>...<|final|>\n\n`
6. Encabezados: nunca se modifican

### Generacion NLP (Ollama prohibido)

```
1. ThinkingEngine.generate_thinking(paragraph, context={'title', 'lang'})
   - modelo spaCy por idioma (en_core_web_sm para EN) con cache perezosa
   - thinking grounded: 1a afirmacion del parrafo + conceptos + entidades
   - conectores con espacio, mayusculas preservadas
2. Validar formato (sin meta-conteos rotos, sin doble puntuacion)
3. Cache MD5 interna del engine → parrafos repetidos = re-ejecuciones gratis
```

### Arreglos al ThinkingEngine (Fase 0) — ✅ REALIZADOS 2026-09-29

1. Seleccionar modelo spaCy en ingles/multilingue para texto EN (`en_core_web_sm` o
   `xx_ent_wiki_sm`) — hoy carga `es` y produce basura
2. Arreglar `_format_steps`: espacio tras conector (`First, `), sin doble dos puntos
3. Acotar concepts/entities: max ~8 palabras, sin puntuacion final, no frases enteras
4. Añadir grounding de razonamiento: usar la primera afirmacion del parrafo + conectores
   causales presentes en el texto (`because`, `therefore`, `as a result`, `led to`) para
   producir "idea central + relacion", no solo conteos de oraciones/palabras
5. Tests: añadir caso de regresion con el parrafo real de `001` (falla hoy)

---

## Plan de implementacion

### Fase 0 — Preparacion (sin tocar los MD todavia) — ✅ COMPLETADA 2026-09-29

- [x] Arreglar ThinkingEngine (5 puntos anteriores) + test de regresion
      (`tests/test_thinking_regression_en.py`)
- [x] Crear `add_thinking_to_md.py` (en esta carpeta)
      - Lee `topics.json`, filtra alcance IN, logging de progreso, `--dry-run`, `--limit N`
      - Backup automatico a `backup/` antes de sobrescribir (obligatorio)
      - Idempotente: detecta `<|thinking|>` existente y lo respeta
      - Solo NLP con cache (Ollama prohibido)
- [x] Actualizar metadata de `topics.json` (`total_topics: 151`, titulos 104/107 con "(Comprehensive)")
- [x] Sin dependencias externas: solo spaCy `en_core_web_sm` (ya instalado)

### Fase 1 — Piloto (20 archivos) — ✅ COMPLETADA 2026-09-29

- [x] Dry-run con 1 archivo (`001`) — 209/209 thinking, 5,5 s (2026-09-29)
- [x] Revision humana del thinking — samples validados (razonamiento en ingles,
      claim grounded, sin meta-conteos rotos, sin bugs de formato)
- [x] Ejecutar los 20 del piloto con backup — **4.720 thinking en 84,6 s, 20/20 archivos**
- [x] Validar: formato `<|thinking|>...<|final|>` consistente (pares balanceados),
      encabezados intactos, **0 perdida de contenido** (comparacion bloque a bloque
      contra `backup/`), thinking nunca precede a encabezado — **ALL OK**
- [x] Decidir: aprobar → Fase 2 — aprobado y ejecutado (2026-09-29)

### Fase 2 — Resto (101 archivos) — ✅ COMPLETADA 2026-09-29

- [x] `125` excluido del alcance (no normalizar): verificado **100% plantilla**
      (1.813/1.813 bloques; las 6 variantes del template, 0 hits en el corpus IN)
- [x] Ejecutado en un solo lote con logging — **121/121 archivos** (101 nuevos +
      20 piloto saltados por idempotencia: `skipped existing = 4.720`)
- [x] Ritmo NLP medido: 404 s para 101 archivos (~4 s/archivo) — total Fase 1+2: ~8 min
- [x] Checkpoint/idempotencia verificados en produccion (piloto intacto, 0 regeneracion)

### Fase 3 — Validacion global — ✅ COMPLETADA 2026-09-29

- [x] **121/121 archivos procesados** (contra `topics.json` alcance IN)
- [x] Conteos: **23.160 thinking insertados**, 916 omitidos (<10 palabras),
      0 boilerplate, 0 fallidos, 2 parrafos partidos (>400 palabras)
- [x] 0 roturas de encabezados y 0 perdida de contenido (comparacion bloque a bloque
      contra los **121 backups**; `032` verificado con contenido normalizado = identico,
      la diferencia de 2 bloques eran los splits de 451 y 411 palabras)
- [x] Muestra aleatoria de 10 samples: ingles correcto, claim grounded en el contenido,
      entidades/concepts saneados — calidad uniforme
- [x] Estadisticas finales registradas abajo (estado: Fases 0-3 Completadas)

### Fase 4 — Dataset matematico real (FUERA de esta pasada)

- [ ] Descartados `141-144` (filler) — documentado en este plan
- [ ] Añadir matemáticas con CoT real en formato `<|problem|>...<|thinking|>...<|final|>...`:
  fuente sintetica (ejercicios resueltos generados) o dataset publico tipo GSM8K
- [ ] Integrar con `main.py --prepare-data` como source independiente

---

## Estadisticas

| Metrica | v1 (estimada) | v2 (auditada) | Real (2026-09-29) |
|---------|---------------|---------------|--------------------|
| Archivos objetivo | 151 | 122 (+29 excl.) | **121** (+30 excl.; 125 descubierto 100% plantilla) |
| Parrafos con thinking | ~7.500-15.000 | ~24.900 | **23.160** |
| Parrafos sin thinking | — | — | **916** (<10 palabras) + 0 boilerplate |
| Parrafos totales IN | — | ~24.900 | **24.076** (29.078 bloques con headers) |
| Palabras del corpus | ~22-45 MB | 2,37M palabras | **~2,42M palabras** (~14,5 MB) |
| Backups primitivos | — | — | **121** en `backup/` |
| Tiempo total | horas (Ollama) | — | **~8 min** (NLP puro: 85 s piloto + 404 s resto) |
| ThinkingEngine | asumido OK | 5 arreglos (Fase 0) | **hechos + 9 tests regresion** |
| Ollama | no considerado | hibrido | **PROHIBIDO (2026-09-29) — 0 uso** |

---

## Consideraciones tecnicas

1. **Cache**: `ThinkingEngine._analysis_cache` (MD5 texto+contexto) evita recalcular
   parrafos repetidos — con 14,4% de duplicados ahorra tiempo en cada re-ejecucion.
2. **Idioma**: conectores y etiquetas en ingles via `thinking_engine_config.json`.
3. **Edge cases**: UTF-8, parrafos muy cortos (omitir), parrafos gigantes (partir >400 palabras),
   re-ejecucion (idempotente).
4. **Backup obligatorio**: `backup/` antes de escribir; los originales no se pierden jamas.
5. **Separacion de concerns**: esta pasada solo inserta thinking en MD. La fusion a cache BPE
   (`main.py --prepare-data --markdown`) y el entrenamiento son pasos posteriores.
6. **Nada de APIs externas**: solo NLP local (spaCy/regex). Ollama y APIs de pago prohibidos.

---

## Proximos pasos

- [ ] Aprobar este plan v2
- [ ] Fase 0: arreglar ThinkingEngine + crear `add_thinking_to_md.py` + metadata `topics.json`
- [ ] Fase 1: piloto de 20 archivos + revision humana
- [ ] Fase 2: 102 restantes
- [ ] Fase 3: validacion global + actualizar este archivo
- [ ] Fase 4 (posterior): dataset matematico real

---

## Por donde empezar: Fase 0, en este orden (ORDEN DE EJECUCION)

> **Decision final: Ollama PROHIBIDO en este plan.** Thinking exclusivamente con
> `ThinkingEngine` (NLP) arreglado. Los pasos siguientes sustituyen al enfoque hibrido
> descrito arriba (pendiente de limpiar esas referencias en el Paso 6).
> Dato verificado: spaCy tiene **`en_core_web_sm` instalado** (hoy el engine carga `es`
> por orden de preferencia — ese es el origen de los conceptos basura).

**Paso 1 — Arreglar `dataset_preparer/thinking_engine_config.json` (seccion `en`)**
Conectores sin espacio final (`"First,"` → `"First, "`, igual el resto).
Causa raiz del `First,the`.

**Paso 2 — Arreglar `dataset_preparer/thinking_engine.py`** (4 bugs):
1. `_load_nlp_model()`: seleccion de modelo **por idioma del texto** con cache perezosa
   (ver diseno en Paso 2.1-bis abajo) — hoy carga `es_core_news_sm` sobre texto ingles
   (orden fijo de la lista, ignorante del idioma) → entities/concepts son frases enteras.
2. Sanear concepts/entities: <=8 palabras, sin puntuacion sobrante (`:` inicial causa
   el `are: : europe`), sin frases completas, dedupe.
3. `_format_steps()`: quitar el `.lower()` global — hoy escribe
   `the treaty of paris` en minusculas.
4. Grounding: anadir paso con la **primera afirmacion del parrafo** (recortada) —
   que el thinking diga algo del contenido, no solo *"contains 5 sentences"*.

#### Paso 2.1-bis — Diseno: modelo spaCy por idioma (soporte multilingue completo)

**Cascada de resolucion de idioma** (mismo orden de prioridad que
`get_language_for_file()` de `language_utils.py`):

```
1. context['lang'] explicito      → add_thinking_to_md.py lo pasa desde topics.json
                                     ("language": "en") o language_manifest.json
2. _detect_language(text)         → heuristica de indicadores (38 idiomas)
3. manifest['default']            → fallback final

mapa lang → modelo spaCy:
  en → en_core_web_sm (INSTALADO ✓)     es → es_core_news_sm (INSTALADO ✓)
  fr/de/it/pt/nl/pl/ro/... → *_core_news_sm / *_core_web_sm (si descargados)
  sin modelo (bg/sk/sr o cualquier otro) → xx_ent_wiki_sm (multilingue, si instalado)
  sin nada                            → _analyze_text_regex (funciona SIEMPRE)

cache: self._nlp_cache[lang]  — un modelo cargado por idioma, carga perezosa
```

**Matriz de soporte multilingue (verificada 2026-09-29):**

| Nivel | Cobertura |
|-------|-----------|
| Deteccion de idioma (`LANGUAGE_INDICATORS`) | **38 idiomas** — todos los europeos del proyecto ✓ |
| Plantillas thinking (`thinking_engine_config.json`) | **30/36** — `be fo is lb nn rm uk` sin plantilla → salen en ingles (fallback `FALLBACK_CONFIG`) |
| Modelos spaCy oficiales en 3.8 (NER/concepts) | **21 europeos**: `ca da de el en es fi fr hr it lt mk nb nl pl pt ro sl sv uk ru` + `xx_ent_wiki_sm` |
| Sin modelo oficial en spaCy 3.8 | `bg sk sr` (eliminados de la compatibilidad 3.8) + `be bs fo ga is lb nn rm sq` (nunca hubo) → cascada `xx_ent_wiki_sm` → regex |
| Instalados hoy | **solo `en_core_web_sm` + `es_core_news_sm`** — el resto: `python -m spacy download <modelo>` solo si se procesa ese idioma (~120 MB c/u) |
| **Esta carpeta (100% ingles)** | **Soporte completo con lo ya instalado** ✓ |

**Opcional (backlog, no bloqueante)**: anadir las 7 plantillas que faltan
(`be fo is lb nn rm uk`) a `thinking_engine_config.json` para thinking 100% nativo
en todos los tokens del proyecto.

**Paso 3 — Test de regresion** con el parrafo real de `001` (hoy falla):
sin `First,` suelto, sin `: :`, conceptos cortos, mayusculas preservadas.

**Paso 4 — Suite existente**: `pytest tests/test_thinking*.py -v`
(raiz del proyecto) para no romper los 8 tests de thinking que ya hay.

**Paso 5 — `topics.json`**: `total_topics: 151` + desambiguar titulos 072/104 y 100/107.

**Paso 6 — Actualizar este archivo (`THINKING_IN_PROGRESS.md`)**: eliminar Ollama
(prohibido), decision final = solo NLP arreglado, tiempos (minutos, no horas).

**Paso 7 — Crear `add_thinking_to_md.py`**: `--dry-run`, backup a `backup/`,
idempotente, cero dependencia de Ollama.

**Paso 8 — Dry-run con `001`** → revision humana → piloto de 20 archivos (Fase 1).

**Nota de blast radius**: arreglar el engine compartido beneficia a todas las fuentes
(pdf, epub, hf…) pero toca codigo comun — por eso van los tests antes que el script.

- [x] Paso 1
- [x] Paso 2 (incluye Paso 2.1-bis: modelo spaCy por idioma + cache)
- [x] Paso 3 (regresion en `tests/test_thinking_regression_en.py`)
- [x] Paso 4 (101 pass / 2 fail preexistentes en `dialogmanager.py:348`, ajenos al plan)
- [x] Paso 5
- [x] Paso 6
- [x] Paso 7
- [ ] Paso 8 — dry-run OK (209/209 thinking en 001); falta revision humana → piloto
- [ ] (Opcional, backlog) Anadir 7 plantillas faltantes `be fo is lb nn rm uk`

---

## Registro de ejecucion — Fase 0 (2026-09-29)

| Paso | Resultado |
|------|-----------|
| 1. `thinking_engine_config.json` | Conectores `en` y `es` con espacio final + `claim_prefix` anadido |
| 2. `thinking_engine.py` | 4 bugs arreglados + diseno multilingue: modelo por idioma (`_get_nlp`, cache `_nlp_cache`), `_sanitize_span` (<=8 palabras, sin puntuacion sobrante), sin `.lower()` global, `_first_claim` grounding, dedupe de entidades, filtro ORDINAL/CARDINAL/stopwords, conectores blindados contra config sin espacio |
| 3. Test de regresion | `tests/test_thinking_regression_en.py` — **9/9 pass** (parrafo real de `001`) |
| 4. Suite existente | **101 pass / 2 fail** — los 2 son `UnboundLocalError` preexistente en `dialogmanager.py:348` (ajeno al plan). Antes: 11 fail / 64 pass. Instalado `huggingface_hub 1.33.0` (faltaba en el entorno; lo requiere `datasets`) |
| 5. `topics.json` | `total_topics: 151`; `104` y `107` con "(Comprehensive)" |
| 6. Este archivo | Ollama eliminado del plan (prohibido); decision = solo NLP |
| 7. `add_thinking_to_md.py` | Creado en esta carpeta: `--dry-run --show --limit --files --depth`, backup `backup/` (copia primitiva solo la 1a vez), idempotente, skip <10 palabras + boilerplate, split >400 palabras |
| 8. Dry-run `001` | **209/209 thinking en 5,5 s** — sin `First,the`, sin `: :`, claim grounded, entidades unicas, mayusculas OK |

**Samples de thinking generados (dry-run 001):**

    Analyzing about 'History of the European Union'. First, the paragraph states:
    "The European Coal and Steel Community (ECSC) was established in 1951 through
    the Treaty of Paris, signed by six countries..." Additionally, the text
    contains 5 sentences with 132 words in total. On the other hand, the text
    presents a narrative or description of events. Likewise, the key concepts
    identified are: The European Coal and Steel Community, This treaty, European
    integration. Finally, the mentioned entities are: The European Coal and Steel
    Community, ECSC, 1951.

**Siguiente accion**: revision humana de samples → ejecutar piloto (Fase 1, 20 archivos).

---

## Registro de ejecucion — Fase 1 (2026-09-29)

- **Piloto ejecutado**: 20/20 archivos, **4.720 thinking insertados**, 79 parrafos
  cortos omitidos (<10 palabras), 0 boilerplate, 0 splits, **84,6 s**
- **Validacion ALL OK**: pares `<|thinking|>`/`<|final|>` balanceados, encabezados
  identicos a backup, contenido bloque a bloque identico tras strip, 0 thinkings
  antes de encabezados, 20 backups primitivos en `backup/`
- **Ritmo real**: ~4,2 s/archivo → Fase 2 (102 archivos) estimada en **~7-8 min**
- **Samples verificados** en 054/001/097/092: claims grounded en contenido real
  (Prague WWII, European Research Area, supercomputer components, Miller-Urey)

---

## Registro de ejecucion — Fase 2 + Fase 3 (2026-09-29)

**Fase 2**:
- Exclusion de `125` detectada ANTES de procesar: verificado que el 100% de sus
  bloques (1.813/1.813) son las 6 variantes de la plantilla T0-T6 (el filtro T1
  inicial solo capturaba 1 de cada 6 → 16,5% engañoso). Red de seguridad: las 8
  frases de plantilla anadidas a `FILLER_PHRASES` (0 hits en corpus IN = sin efectos
  colaterales).
- Ejecucion unica: **121/121 archivos**, 18.440 thinking nuevos en 404 s,
  4.720 del piloto saltados por idempotencia, 916 cortos omitidos, 2 splits.

**Fase 3 (validacion global)**:
- **121/121 OK**: pares `<|thinking|>`/`<|final|>` balanceados, encabezados identicos
  a backup, contenido identico bloque a bloque (0 perdida), thinking nunca precede
  a encabezado, 121 backups pristinos.
- Unico caso aparente (`032_eu_digitalization`, 156 vs 154 bloques) verificado:
  eran los 2 splits >400 palabras (451w y 411w) — contenido normalizado **identico**.
- Muestra aleatoria (seed=7, 10 samples de temas variados): claims grounded
  (Peruvian arts, brain/skull, Fusion cuisine, Meaning theory...), ingles correcto,
  formato uniforme.

**Totales finales del corpus (esta carpeta)**:
- **121 archivos** con thinking | **23.160 thinking** | 24.076 parrafos
- 916 parrafos cortos sin thinking (<10 palabras, por diseno)
- 30 archivos excluidos (121-150) sin tocar — originales intactos
- Tiempo total Fases 1+2: **~8 min** | Fallidos: 0

**Siguiente fase**: Fase 4 (dataset matematico real, fuera de esta pasada) →
fusion a cache BPE (`main.py --prepare-data --markdown --generate-thinking` NO
requerido, thinking ya insertado en MD) → entrenamiento.

---

*Ultima actualizacion: 2026-09-29 — Fases 0-3 completadas: 121 archivos con thinking (23.160), validacion global ALL OK; pendiente Fase 4 (matematicas)*
