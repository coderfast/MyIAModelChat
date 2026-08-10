# THINKING_DATASETS.md - Thinking Dataset Format & Examples

This document defines the format and examples for chain-of-thought thinking datasets used to train MyIAModelChat's thinking capabilities.

---

## 1. Training Format

Each training sample uses a dual-token system with mode tokens and content tokens:

### THINKING Sample

```
<|thinking|>question<thinking>reasoning</thinking><|answer|>answer
```

### CONTEXT Sample (no reasoning)

```
<|context|>question<|answer|>answer
```

**Token Roles:**

| Token | Type | Purpose |
|-------|------|---------|
| `<\|thinking\|>` | Mode token | Marks sample as THINKING |
| `<\|context\|>` | Mode token | Marks sample as CONTEXT |
| `<\|answer\|>` | Mode token | Delimiter before final answer |
| `<thinking>` | Content tag | Opens reasoning block |
| `</thinking>` | Content tag | Closes reasoning block |

---

## 2. Dataset Columns

Each dataset row contains:

| Column | Type | Description |
|--------|------|-------------|
| `input_ids` | list[int] | Token sequence (model input) |
| `token_ids` | list[int] | Token sequence (teacher-forced target) |
| `question` | string | Raw question text |
| `answer` | string | Raw answer text |
| `type` | string | `CONTEXT` or `THINKING` |
| `thinking` | string | Reasoning text (empty for CONTEXT) |

---

## 3. Template Examples (Spanish)

### Category: identity

```csv
question,answer,type,thinking
quién eres,Soy Eduardo Piñera Aznárez,THINKING,El usuario pregunta sobre mi identidad. Soy Eduardo Piñera Aznárez, un asistente de IA de 52 años especializado en tecnologías de la información.
cómo te llamas,Me llamo Eduardo Piñera Aznárez,THINKING,El usuario quiere saber mi nombre. Mi nombre es Eduardo Piñera Aznárez.
```

### Category: greeting

```csv
question,answer,type,thinking
hola,¡Hola! ¿Cómo estás?,THINKING,El usuario me saluda. Debo responder con un saludo amigable y preguntar cómo está.
buenos días,Buenos días! Espero que tengas un gran día,THINKING,Es un saludo matutino. Respondo con un saludo apropiado para la hora del día.
```

### Category: question

```csv
question,answer,type,thinking
qué es la inteligencia artificial,La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana,THINKING,El usuario pregunta sobre un concepto técnico. La IA es un campo de la informática que incluye aprendizaje automático, procesamiento de lenguaje natural y visión por computadora. La respuesta debe ser clara y concisa.
cómo funciona un transformador,Un transformador es una arquitectura de red neuronal que utiliza mecanismos de atención para procesar secuencias de datos de forma paralela,THINKING,El usuario pregunta sobre arquitecturas de deep learning. Los transformadores usan self-attention y feed-forward networks. Fueron introducidos en el paper "Attention Is All You Need".
```

### Category: farewell

```csv
question,answer,type,thinking
adiós,¡Adiós! Que tengas un buen día,THINKING,El usuario se despide. Debo despedirme de forma amigable.
hasta luego,Hasta luego! Fue un placer ayudarte,THINKING,Es una despedida. Respondo con cortesía y cierro la conversación.
```

### Context (no thinking)

```csv
question,answer,type,thinking
cuéntame algo interesante,Los agujeros negros pueden tener una fuerza gravitacional tan extrema que ni siquiera la luz puede escapar de ellos,CONTEXT,
cuánto es 2+2,4,CONTEXT,
```

---

## 4. Template Examples (English)

### Category: identity

```csv
question,answer,type,thinking
who are you,I am Eduardo Piñera Aznárez,THINKING,The user is asking about my identity. I am Eduardo Piñera Aznárez, a 52-year-old AI assistant specialized in information technology.
what is your name,My name is Eduardo Piñera Aznárez,THINKING,The user wants to know my name. I should respond with my name directly.
```

### Category: greeting

```csv
question,answer,type,thinking
hello,Hello! How are you today?,THINKING,The user is greeting me. I should respond with a friendly greeting and ask how they are.
good morning,Good morning! I hope you have a great day,THINKING,It's a morning greeting. I should respond with an appropriate morning greeting.
```

### Category: question

```csv
question,answer,type,thinking
what is machine learning,Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed,THINKING,The user is asking about a technical concept. Machine learning is a branch of AI that focuses on algorithms that can learn from data. I should give a clear and concise definition.
how does a neural network work,A neural network processes data through layers of interconnected nodes that apply weights and biases, transforming inputs into outputs through activation functions,THINKING,The user is asking about neural networks. They consist of input, hidden, and output layers. Each connection has a weight that is adjusted during training.
```

### Category: farewell

```csv
question,answer,type,thinking
goodbye,Goodbye! It was nice talking to you,THINKING,The user is saying goodbye. I should respond politely and close the conversation.
bye,Bye! Have a great day!,THINKING,The user is leaving. I should give a friendly farewell.
```

### Context (no thinking)

```csv
question,answer,type,thinking
tell me something interesting,Black holes can have gravitational force so extreme that not even light can escape them,CONTEXT,
what do you think about technology,Technology has transformed how we live, work, and communicate, making information accessible to billions of people worldwide,CONTEXT,
```

---

## 5. Complex Thinking Examples

### Multi-step Reasoning

```
<|thinking|>El usuario pregunta sobre machine learning.

Paso 1: Definir qué es ML
- ML es un subconjunto de la IA
- Permite a los sistemas aprender de datos

Paso 2: Explicar cómo funciona
- Se entrenan con datos históricos
- Los algoritmos encuentran patrones
- Hacen predicciones sin programación explícita

Paso 3: Dar ejemplos prácticos
- Reconocimiento de voz
- Sistemas de recomendación
- Diagnóstico médico

Paso 4: Resumir en una respuesta clara
</thinking><|answer|>El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender de los datos y mejorar su rendimiento sin ser programados explícitamente. Se utiliza en reconocimiento de voz, sistemas de recomendación y diagnóstico médico.
```

### Mathematical Reasoning

```
<|thinking|>El usuario pregunta cuánto es 17 * 23.

Paso 1: Descomponer el problema
- 17 * 23 = 17 * (20 + 3)
- 17 * 20 = 340
- 17 * 3 = 51

Paso 2: Sumar los resultados
- 340 + 51 = 391

Paso 3: Verificar
- 17 * 23 = 391 ✓
</thinking><|answer|>391
```

### Code Explanation

```
<|thinking|>El usuario pregunta qué hace esta función Python: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)

Paso 1: Identificar el patrón
- Es una función recursiva
- Caso base: n <= 1 retorna 1
- Caso recursivo: n * factorial(n-1)

Paso 2: Ejecutar mentalmente con n=5
- factorial(5) = 5 * factorial(4)
- factorial(4) = 4 * factorial(3)
- factorial(3) = 3 * factorial(2)
- factorial(2) = 2 * factorial(1)
- factorial(1) = 1
- Resultado: 5*4*3*2*1 = 120

Paso 3: Explicar el propósito
- Calcula el factorial de un número
- factorial(n) = n * (n-1) * (n-2) * ... * 1
</thinking><|answer|>Esta función calcula el factorial de un número de forma recursiva. El factorial de n (escrito como n!) es el producto de todos los números enteros desde 1 hasta n. Por ejemplo, factorial(5) = 5 * 4 * 3 * 2 * 1 = 120.
```

---

## 6. Loss Weighting

The trainer applies differentiated loss based on token position:

### THINKING samples

| Segment | Weight | Rationale |
|---------|--------|-----------|
| `<\|thinking\|>` mode token | 0.0 | Prefix learning |
| Question tokens | 0.0 | Don't penalize |
| `<thinking>` delimiter | 1.0 | Must learn |
| Reasoning content | 0.5 | Learn structure |
| `</thinking>` delimiter | 1.0 | Must learn |
| `<\|answer\|>` delimiter | 1.0 | Must learn |
| Answer tokens | 1.0 | Full weight |

### CONTEXT samples

| Segment | Weight | Rationale |
|---------|--------|-----------|
| `<\|context\|>` mode token | 0.0 | Prefix learning |
| Question tokens | 0.0 | Don't penalize |
| `<\|answer\|>` delimiter | 1.0 | Must learn |
| Answer tokens | 1.0 | Full weight |

---

## 7. Language-Specific Considerations

### Spanish
- Use natural, conversational Spanish
- Thinking can be more formal/technical than response
- Common patterns: "El usuario pregunta...", "Analizo la consulta..."
- Greeting responses should include warmth

### English
- Use clear, professional English
- Thinking should be structured and logical
- Common patterns: "The user is asking...", "I need to analyze..."
- Responses should be concise and direct

### Multilingual (30 Languages)

The thinking engine supports: English, Spanish, French, German, Italian, Portuguese, Catalan, Galician, Basque, Irish, Dutch, Danish, Swedish, Finnish, Polish, Czech, Slovak, Hungarian, Romanian, Bulgarian, Croatian, Slovenian, Greek, Estonian, Latvian, Lithuanian, Maltese, Serbian, Bosnian, Macedonian, Albanian.

---

## 8. Quality Guidelines

### Good Thinking
- Shows step-by-step reasoning
- References the user's question
- Explains WHY the answer is what it is
- Is internally consistent
- Stays on topic

### Bad Thinking
- Repeats the question without reasoning
- Contains irrelevant information
- Contradicts the final response
- Is too long (more than 2x the response length)
- Is too short (less than one sentence)

### Validation Checklist
- [ ] `<|thinking|>` prefix is present at start
- [ ] `<thinking>` and `</thinking>` delimiters are present
- [ ] `<|answer|>` delimiter separates reasoning from answer
- [ ] Reasoning logically leads to the response
- [ ] No thinking/response contradictions
- [ ] Appropriate length ratio (1:1 to 3:1 thinking:response)
- [ ] Correct language matching

---

## 9. Using Generated Datasets

### Step 1: Prepare Data

```bash
# NLP-based thinking (no external dependencies)
python main.py --prepare-data --aiml --hf --thinking-mode nlp --refresh-cache

# With Ollama teacher
python main.py --prepare-data --aiml --hf --thinking-mode ollama --refresh-cache

# Multilingual
python main.py --prepare-data --aiml --hf --thinking-mode nlp --allowed-languages es,en,fr,de --refresh-cache
```

### Step 2: Train

```bash
python main.py --train --epochs 30
```

### Step 3: Infer

```bash
python main.py --chat --show-thinking
```

---

*Updated: 2026-08-09 - Reflects new mode token system and dataset format*
