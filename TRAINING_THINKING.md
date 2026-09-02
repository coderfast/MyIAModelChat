# Guia de Entrenamiento con Thinking (Razonamiento)

## Que es thinking?

Thinking (razonamiento) le ensena al modelo a **pensar antes de responder**. En vez de dar una respuesta directa, el modelo:

1. Analiza la pregunta
2. Razona paso a paso (como un humano)
3. Llega a una respuesta fundamentada

## Tokens especiales

| Token | Funcion |
|-------|---------|
| `<\|problem\|>` | Indica que empieza una pregunta/problema |
| `<\|thinking\|>` | Indica que empieza el razonamiento |
| `<\|final\|>` | Indica que termina el razonamiento y empieza la respuesta |
| `<\|lang\|>` | Token de idioma (prefijo: `<\|es\|>`, `<\|en\|>`, etc.) |

### Tokens de idioma (36 idiomas europeos)

| Familia | Codigos |
|---------|---------|
| Romances | `es`, `fr`, `it`, `pt`, `ro`, `ca`, `gl`, `rm` |
| Germanicas | `en`, `de`, `nl`, `sv`, `da`, `nb`, `nn`, `is`, `lb`, `fo` |
| Eslavas | `pl`, `cs`, `sk`, `bg`, `hr`, `sr`, `sl`, `bs`, `mk`, `uk`, `be` |
| Balticas | `lt`, `lv` |
| Fino-ugricas | `fi`, `et`, `hu` |
| Celticas/Hellenicas | `ga`, `el`, `sq` |

**Formato:** `<|lang|><|problem|>pregunta<|thinking|>razonamiento<|final|>respuesta`

El token de idioma se añade automáticamente al prepocesar datos. La función `_create_bpe_text_column()` en `data_preparer.py` usa un regex para detectar si una muestra ya tiene token de idioma (por ejemplo, de formatos agentic) y no lo añade para evitar duplicación.

## Formato basico

### Con thinking (el modelo razona)

```
<|es|><|problem|>¿Cual es la capital de Francia?<|thinking|>Francia es un pais de Europa. Su capital es Paris.<|final|>La capital de Francia es Paris.
```

### Sin thinking (respuesta directa)

```
<|en|><|problem|>What is the capital of France?<|final|>The capital of France is Paris.
```

## Formato con usuario/asistente

```
<|user|>¿Cual es la capital de Francia?<|end|><|assistant|><|thinking|>Francia es un pais de Europa. Su capital es Paris.<|final|>La capital de Francia es Paris.<|end|>
```

## Niveles de thinking

### 1. Basico (basic)

Pensamiento corto, 1-2 oraciones.

**Ejemplo:**
```
<|problem|>¿Cuanto es 2 + 2?<|thinking|>Es una suma simple. 2 mas 2 es 4.<|final|>2 + 2 = 4
```

**Para que sirve:** Preguntas simples, datos rapidos.

---

### 2. Adaptativo (adaptive)

Pensamiento medio, 2-3 oraciones.

**Ejemplo:**
```
<|problem|>¿Por que el cielo es azul?<|thinking|>La luz del sol viaja por el aire. Las moleculas del aire esparcen la luz azul mas que la roja. Por eso vemos el cielo de color azul.<|final|>El cielo es azul porque las moleculas del aire esparcen la luz azul del sol mas que otras colores.
```

**Para que sirve:** La mayoria de preguntas, buen balance.

---

### 3. Detallado (detailed)

Pensamiento largo, 3-5 oraciones con pasos claros.

**Ejemplo:**
```
<|problem|>Explica la fotosintesis<|thinking|>La fotosintesis es como una fabrica dentro de las plantas. Primero, la planta absorbe agua por las raices. Luego, las hojas capturan luz del sol. Con esa energia, la planta convierte el agua y el aire en azucar. El oxigeno es un subproducto que la planta suelta al aire.<|final|>La fotosintesis es el proceso por el cual las plantas convierten la luz solar en energia. Usan agua del suelo y dioxide de carbono del aire para producir azucar y oxigeno.
```

**Para que sirve:** Preguntas complejas que necesitan explicacion larga.

## Ejemplos por tipo de pregunta

### Preguntas de conocimiento

**Pregunta simple:**
```
<|problem|>¿Que es Python?<|thinking|>Python es un lenguaje de programacion. Es popular porque es facil de aprender. Se usa para web, ciencia de datos e inteligencia artificial.<|final|>Python es un lenguaje de programacion de alto nivel, conocido por su simplicidad y versatilidad. Se usa en desarrollo web, ciencia de datos e IA.
```

**Pregunta con contexto:**
```
<|problem|>¿Cual es la diferencia entre Python y JavaScript?<|thinking|>Python se usa mas para ciencia de datos y scripts. JavaScript se usa para paginas web. Python es mas facil de leer. JavaScript corre en el navegador.<|final|>Python es ideal para ciencia de datos y backend, mientras que JavaScript es el rey del frontend web. Python tiende a ser mas legible, y JavaScript es mas versatil en el navegador.
```

### Preguntas de matematicas

**Suma:**
```
<|problem|>¿Cuanto es 15 + 27?<|thinking|>Necesito sumar 15 y 27. Primero sumo 5 + 7 = 12. Luego 1 + 2 = 3. El resultado es 42.<|final|>15 + 27 = 42
```

**Multiplicacion:**
```
<|problem|>¿Cuanto es 12 * 8?<|thinking|>Multiplico 12 por 8. Puedo pensarlo como 10 * 8 = 80 y 2 * 8 = 16. Luego sumo 80 + 16 = 96.<|final|>12 * 8 = 96
```

### Preguntas de programacion

**Como hacer algo:**
```
<|problem|>Como leer un archivo en Python?<|thinking|>En Python se usa la funcion open() para abrir archivos. Despues se puede leer con read() o iterar linea por linea. Es importante cerrar el archivo despues de usarlo.<|final|>Para leer un archivo en Python: `with open('archivo.txt', 'r') as f: contenido = f.read()`. El bloque `with` cierra el archivo automaticamente.
```

**Explicar codigo:**
```
<|problem|>Que hace este codigo: `print([x**2 for x in range(5)])`?<|thinking|>Es una list comprehension. Range(5) genera 0,1,2,3,4. Cada numero se eleva al cuadrado con **2. El resultado es [0,1,4,9,16].<|final|>El codigo imprime la lista [0, 1, 4, 9, 16], que son los cuadrados de los numeros del 0 al 4.
```

### Preguntas de ciencia

**Fisica:**
```
<|problem|>Que es la gravedad?<|thinking|>La gravedad es una fuerza que atrae los objetos entre si. Mientras mas masa tiene un objeto, mas gravedad tiene. La Tierra tiene gravedad suficiente para mantenernos en el suelo.<|final|>La gravedad es la fuerza de atraccion entre dos objetos con masa. En la Tierra, nos mantiene pegados al suelo y hace que los objetos caigan.
```

**Biologia:**
```
<|problem|>Que es el ADN?<|thinking|>ADN son las siglas de Acido Desoxirribonucleico. Es una molecula que contiene las instrucciones para construir y mantener un organismo. Se parece a una escalera retorcida.<|final|>El ADN es la molecula que contiene las instrucciones geneticas para el desarrollo y funcionamiento de todos los seres vivos. Tiene forma de escalera retorcida (doble helice).
```

### Preguntas de cotidiano

**Cocina:**
```
<|problem|>Como hacer arroz blanco?<|thinking|>Primero se lava el arroz. Luego se pone en agua hirviendo con sal. Se cocina a fuego bajo por 15-20 minutos hasta que el agua se absorba.<|final|>Para hacer arroz blanco: lava 1 taza de arroz, ponla en 2 tazas de agua hirviendo con sal, cocina a fuego bajo 15-20 minutos tapado.
```

**Tecnologia:**
```
<|problem|>Como conectar auriculares bluetooth?<|thinking|>Primero hay que activar el bluetooth en el celular. Luego poner los auriculares en modo de emparejamiento. Buscarlos en la lista de dispositivos del celular y conectar.<|final|>Para conectar auriculares bluetooth: activa bluetooth en tu celular, pon los auriculares en modo emparejamiento, buscalos en la lista de dispositivos y selecciona conectar.
```

## Como generar datos de thinking

### Opcion 1: Automatico con el generador

```bash
# Generar datos con thinking basico
python main.py --prepare-data --aiml --generate-thinking --thinking-mode nlp

# Generar datos con thinking detallado
python main.py --prepare-data --aiml --generate-thinking --thinking-mode ollama --thinking-model llama3.2
```

### Opcion 2: Manual (crear tus propios datos)

Crea un archivo JSON con preguntas y respuestas:

```json
{"input": "¿Que es Python?", "output": "Python es un lenguaje de programacion de alto nivel."}
{"input": "¿Cuanto es 5 + 3?", "output": "5 + 3 = 8"}
```

Luego ejecuta:

```bash
python main.py --prepare-data --aiml --generate-thinking
```

El generador automaticamente creara el thinking para cada ejemplo.

### Opcion 3: Con modelo externo (Ollama)

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar un modelo
ollama pull llama3.2

# Generar thinking con Ollama
python main.py --prepare-data --aiml --generate-thinking --thinking-mode ollama --thinking-model llama3.2
```

## Configuracion

En `training_config.json`:

```json
{
  "thinking": {
    "enabled": true,
    "loss_weight": 1.0,
    "max_tokens": 64
  }
}
```

| Parametro | Que hace | Default |
|-----------|----------|---------|
| `enabled` | Activa el thinking | `true` |
| `loss_weight` | Peso del loss para thinking (0-1) | `1.0` |
| `max_tokens` | Maximo de tokens en thinking | `64` |

## CLI Args

```bash
# Habilitar thinking
python main.py --train --thinking-enabled

# Cambiar modo de generacion
python main.py --prepare-data --aiml --generate-thinking --thinking-mode nlp

# Usar modelo especifico
python main.py --prepare-data --aiml --generate-thinking --thinking-mode ollama --thinking-model llama3.2

# Ver thinking en chat
python main.py --chat --model mi_modelo --show-thinking
```

## Consejos para datos de thinking

1. **Paso a paso:** El thinking debe mostrar el razonamiento, no solo la respuesta.
   - BIEN: `Para sumar 5 + 3, primero sumo 5 y 3 = 8`
   - MAL: `La respuesta es 8`

2. **Natural:** El thinking debe sonar como pensamiento humano.
   - BIEN: `Primero veo que es una suma. Luego calculo...`
   - MAL: `El algoritmo de suma es: a + b = c`

3. **Completo:** Incluye todos los pasos importantes.
   - BIEN: `Multiplico 12 * 8 = 96. El resultado es 96.`
   - MAL: `Es 96.`

4. **Idioma:** Escribe thinking en el mismo idioma que la pregunta.

5. **Longitud:** Adapta la longitud al nivel de complejidad:
   - Preguntas simples: thinking corto (1-2 oraciones)
   - Preguntas complejas: thinking largo (3-5 oraciones)

## Tokens en el dataset

El dataset puede contener estos tokens:

| Token | Descripcion |
|-------|-------------|
| `<\|problem\|>` | Inicio del problema/pregunta |
| `<\|thinking\|>` | Inicio del razonamiento |
| `<\|final\|>` | Fin del razonamiento, inicio de respuesta |
| `<\|user\|>` | Usuario (formato conversacional) |
| `<\|assistant\|>` | Asistente (formato conversacional) |
| `<\|end\|>` | Fin del mensaje |
| `<\|system\|>` | Instrucciones del sistema |

## Estructura del dataset

```
dataset_cache/
├── prepared_dataset/        # Dataset principal
├── sentencepiece.model      # Tokenizer BPE
└── dataset_stats.pkl        # Estadisticas
```

## Preguntas frecuentes

**P: El thinking hace el modelo mas lento?**
R: Slightly, pero la calidad de las respuestas mejora mucho. El thinking se genera durante el entrenamiento, no durante la inferencia.

**P: Puedo mezclar thinking con herramientas?**
R: Si. El formato agentic combina ambas cosas:
```
<|user|>¿Cuanto es 5 + 3?<|end|><|assistant|><tool_call>calculator(5 + 3)</tool_call><|tool_result|>8<|end|><|assistant|><|thinking|>Use la calculadora para sumar 5 + 3. El resultado es 8.<|final|>5 + 3 = 8<|end|>
```

**P: Que pasa si el thinking es muy largo?**
R: El modelo puede quedarse sin espacio. Usa `max_tokens` para limitar la longitud del thinking.

**P: El thinking es obligatorio?**
R: No. Puedes entrenar sin thinking usando el formato `<|problem|>pregunta<|final|>respuesta`. Pero el thinking mejora la calidad de las respuestas.

**P: Como se si el modelo esta usando bien el thinking?**
R: Usa `--show-thinking` en el chat para ver el razonamiento del modelo.
