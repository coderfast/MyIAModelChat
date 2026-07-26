# ROADMAP BUGS - MyIAModelChat

## Resumen

Bugs encontrados en revisión completa del pipeline de thinking + validación de fuentes.

---

## BUG-01: `_validate_and_clean_sources()` no sincroniza atributos de fuente

**Severidad**: Alta  
**Archivo**: `data_preparer.py:1574-1622`  
**Problema**:  
`_validate_and_clean_sources()` limpia `self.combined_data`, pero **no actualiza** `self.aiml_data`, `self.hf_data`, `self.pdf_data`, etc. Luego `_generate_thinking_for_sources()` (línea 1659) lee de esos atributos **originales sin validar**, por lo que el thinking se genera con datos sucios.

**Flujo actual (ROTO)**:
```
1. _validate_and_clean_sources() → limpia combined_data
2. _generate_thinking_for_sources() → lee de aiml_data, hf_data (SUCIOS)
```

**Flujo esperado**:
```
1. _validate_and_clean_sources() → limpia Y actualiza atributos de fuente
2. _generate_thinking_for_sources() → lee de aiml_data, hf_data (LIMPIOS)
```

**Fix**: En `_validate_and_clean_sources()`, después de limpiar cada fuente, actualizar el atributo correspondiente:
```python
# Dentro del loop for source_name, dataset in source_datasets.items():
# Después de cleaned_datasets.append(Dataset.from_list(cleaned_samples)):
if source_name == 'aiml':
    self.aiml_data = Dataset.from_list(cleaned_samples)
elif source_name == 'hf':
    self.hf_data = Dataset.from_list(cleaned_samples)
# ... etc para cada fuente
```

---

## BUG-02: `_compute_loss()` no aplica thinking_loss_weight correctamente

**Severidad**: Media  
**Archivo**: `main_train.py:433-473`  
**Problema**:  
El comentario dice "Thinking tokens receive a **lower** loss weight" pero `thinking_loss_weight = 1.0` significa que thinking y respuesta tienen el **mismo peso**. Si la intención es dar MENOR peso al thinking (para que el modelo se enfoque en la respuesta), el valor debería ser < 1.0.

**Código actual** (línea 455):
```python
if in_thinking:
    weights[b, s] = self.thinking_loss_weight  # = 1.0 (igual que respuesta)
```

**Fix**: Cambiar `thinking_loss_weight` default a `0.5` en `TRAINING_CONFIG` (línea 67) para que el thinking tenga menor peso que la respuesta. O actualizar el comentario si se desea peso igual.

---

## BUG-03: `_validate_and_clean_sources()` falla si `combined_data` tiene datos pero atributos de fuente son None

**Severidad**: Baja  
**Archivo**: `data_preparer.py:1574-1622`  
**Problema**:  
La función verifica `if self.combined_data is None or len(self.combined_data) == 0` al inicio, pero luego itera sobre `source_datasets` que puede tener todos los valores en None. Si `combined_data` tiene datos (de una ejecución previa en cache), pero los atributos de fuente no se cargaron, el loop produce `cleaned_datasets = []` y `self.combined_data` se pierde.

**Fix**: Guardar el conteo original de `self.combined_data` antes del loop y restaurarlo si `cleaned_datasets` queda vacío:
```python
if not cleaned_datasets:
    logger.warning("  ⚠ No source datasets to validate, keeping combined_data as-is")
    return
```

---

## BUG-04: `ThinkingGenerator.validate_thinking()` nunca se usa

**Severidad**: Baja  
**Archivo**: `thinking_generators.py:80-124`  
**Problema**:  
`ThinkingGenerator` tiene un método `validate_thinking()` (línea 80) que **nunca se llama**. En `data_preparer.py:1689` se importa y usa `validate_thinking` de `thinking_quality.py`, que es una función independiente con lógica similar pero no idéntica.

**Impacto**: Duplicación de lógica de validación. El método de la clase base es código muerto.

**Fix**: Eliminar `ThinkingGenerator.validate_thinking()` de `thinking_generators.py` ya que `thinking_quality.validate_thinking()` es la función autoritativa usada en el pipeline.

---

## BUG-05: `thinking_quality.validate_thinking()` importa dentro del loop

**Severidad**: Baja  
**Archivo**: `data_preparer.py:1689`  
**Problema**:  
`from thinking_quality import validate_thinking` se ejecuta **dentro del loop** `for sample in samples`, para cada sample. El import debería estar fuera del loop.

**Fix**: Mover el import al inicio de `_generate_thinking_for_sources()`, junto a los otros imports:
```python
def _generate_thinking_for_sources(self):
    ...
    from thinking_quality import validate_thinking  # Aquí, no dentro del loop
    ...
```

---

## BUG-06: `dialogmanager.py` inicializa `self.min_length` dos veces

**Severidad**: Baja  
**Archivo**: `dialogmanager.py:26,33`  
**Problema**:
```python
self.min_length = min_length  # línea 26
...
self.min_length = min_length  # línea 33 (duplicado)
```

**Fix**: Eliminar la segunda asignación en la línea 33.

---

## BUG-07: `_generate_thinking_for_sources()` no actualiza `total_thinking` si thinking es valid

**Severidad**: Baja  
**Archivo**: `data_preparer.py:1691-1696`  
**Problema**:  
Cuando `validate_thinking` retorna `valid=False`, se resetea `result['input_ids']` al output original, pero el sample **sigue teniendo el campo `thinking`** con el thinking inválido. Esto puede causar que `_detect_thinking_data()` en `main_train.py` detecte `<think>` tags en samples que no deberían tener thinking.

**Fix**: Cuando la validación falla, también eliminar el campo `thinking`:
```python
else:
    result['input_ids'] = result.get('output', '')
    result['thinking'] = ''  # Limpiar thinking inválido
    result.pop('thinking_text', None)  # Limpiar thinking_text también
```

---

## BUG-08: `--thinking-depth` no se usa en ningún generador

**Severidad**: Baja  
**Archivo**: `main.py:261`, `data_preparer.py:1641`  
**Problema**:  
El flag `--thinking-depth` se define en CLI pero **nunca se pasa** a los generadores. Todos los generadores usan profundidad fija (2-4 oraciones).

**Fix**: Pasar `thinking_depth` a los generadores y usarlo para ajustar la cantidad de razonamiento:
```python
thinking_depth = getattr(self.args, 'thinking_depth', 'adaptive')
# Pasar a cada generador: PDFThinkingGenerator(teacher, depth=thinking_depth)
```

---

## BUG-09: `--thinking-model` no se valida contra modelos disponibles en Ollama

**Severidad**: Baja  
**Archivo**: `thinking_generators.py:18-33`  
**Problema**:  
`OllamaTeacher` recibe el modelo pero no verifica si está disponible antes de intentar generarlo. Si el modelo no existe, falla silenciosamente y cae en fallback rule-based sin aviso claro.

**Fix**: En `OllamaTeacher.is_available()`, además de verificar el endpoint, verificar que el modelo esté en la lista de modelos disponibles:
```python
def is_available(self) -> bool:
    ...
    # Verificar que el modelo específico esté disponible
    available_models = [m['name'] for m in data.get('models', [])]
    if self.model not in available_models:
        logger.warning(f"Model {self.model} not found in Ollama. Available: {available_models}")
        return False
```

---

## Resumen de Prioridad

| Bug | Severidad | Esfuerzo | Prioridad |
|-----|-----------|----------|-----------|
| BUG-01 | Alta | Medio | P0 |
| BUG-07 | Baja | Bajo | P1 |
| BUG-02 | Media | Bajo | P1 |
| BUG-05 | Baja | Bajo | P2 |
| BUG-06 | Baja | Bajo | P2 |
| BUG-04 | Baja | Bajo | P2 |
| BUG-03 | Baja | Bajo | P2 |
| BUG-08 | Baja | Medio | P3 |
| BUG-09 | Baja | Bajo | P3 |
