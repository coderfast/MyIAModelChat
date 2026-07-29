# THINKING_DATASETS.md - Thinking Dataset Format & Examples

This document defines the format and examples for chain-of-thought thinking datasets used to train MyIAModelChat's thinking capabilities.

---

## 1. Training Format

Each training sample follows this structure:

```
</thinking>Final clean response here
```

**Rules:**
- `<thinking>` and `</thinking>` are special tokens, NOT text
- Thinking content goes between the tags
- Response goes after `</thinking>`
- Model learns to generate thinking first, then response
- At inference, thinking can be shown or hidden

---

## 2. Template Examples (Spanish)

### Category: identity (Identidad)

```csv
input,output,thinking
quien eres,yo soy Eduardo Piñera Aznárez,El usuario pregunta sobre mi identidad. Soy Eduardo Piñera Aznárez, un asistente de IA de 52 años especializado en tecnologías de la información.
cómo te llamas,me llamo Eduardo Piñera Aznárez,El usuario quiere saber mi nombre. Mi nombre es Eduardo Piñera Aznárez.
```

### Category: greeting (Saludo)

```csv
input,output,thinking
hola,hola! cómo estás?,El usuario me saluda. Debo responder con un saludo amigable y preguntar cómo está.
buenos días,buenos días! espero que tengas un gran día,Es un saludo matutino. Respondo con un saludo apropiado para la hora del día.
```

### Category: question (Pregunta)

```csv
input,output,thinking
qué es la inteligencia artificial,la inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana,El usuario pregunta sobre un concepto técnico. La IA es un campo de la informática que incluye aprendizaje automático, procesamiento de lenguaje natural y visión por computadora. La respuesta debe ser clara y concisa.
cómo funciona un transformador,un transformador es una arquitectura de red neuronal que utiliza mecanismos de atención para procesar secuencias de datos de forma paralela,El usuario pregunta sobre arquitecturas de deep learning. Los transformadores usan self-attention y feed-forward networks. Fueron introducidos en el paper "Attention Is All You Need".
```

### Category: farewell (Despedida)

```csv
input,output,thinking
adiós,adiós! que tengas un buen día,El usuario se despide. Debo despedirme de forma amigable.
hasta luego,hasta luego! fue un placer ayudarte,Es una despedida. Respondo con cortesía y cierro la conversación.
```

### Category: default (Default)

```csv
input,output,thinking
cuéntame algo interesante,los agujeros negros pueden tener una fuerza gravitacional tan extrema que ni siquiera la luz puede escapar de ellos,El usuario pide un dato interesante. Selecciono un tema curioso de ciencia que sea fácil de entender y mantener el interés.
```

---

## 3. Template Examples (English)

### Category: identity

```csv
input,output,thinking
who are you,I am Eduardo Piñera Aznárez,The user is asking about my identity. I am Eduardo Piñera Aznárez, a 52-year-old AI assistant specialized in information technology.
what is your name,My name is Eduardo Piñera Aznárez,The user wants to know my name. I should respond with my name directly.
```

### Category: greeting

```csv
input,output,thinking
hello,Hello! How are you today?,The user is greeting me. I should respond with a friendly greeting and ask how they are.
good morning,Good morning! I hope you have a great day,It's a morning greeting. I should respond with an appropriate morning greeting.
```

### Category: question

```csv
input,output,thinking
what is machine learning,Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed,The user is asking about a technical concept. Machine learning is a branch of AI that focuses on algorithms that can learn from data. I should give a clear and concise definition.
how does a neural network work,A neural network processes data through layers of interconnected nodes that apply weights and biases, transforming inputs into outputs through activation functions,The user is asking about neural networks. They consist of input, hidden, and output layers. Each connection has a weight that is adjusted during training.
```

### Category: farewell

```csv
input,output,thinking
goodbye,Goodbye! It was nice talking to you,The user is saying goodbye. I should respond politely and close the conversation.
bye,Bye! Have a great day!,The user is leaving. I should give a friendly farewell.
```

### Category: default

```csv
input,output,thinking
tell me something interesting,Black holes can have gravitational force so extreme that not even light can escape them,The user wants an interesting fact. I should pick an engaging science topic that is easy to understand.
what do you think about technology,Technology has transformed how we live, work, and communicate, making information accessible to billions of people worldwide,The user is asking for my opinion on technology. I should give a balanced and thoughtful response about the impact of technology.
```

---

## 4. Complex Thinking Examples

### Multi-step Reasoning

```
<thinking>
El usuario pregunta sobre machine learning.

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
</thinking>
El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender de los datos y mejorar su rendimiento sin ser programados explícitamente. Se utiliza en reconocimiento de voz, sistemas de recomendación y diagnóstico médico.
```

### Mathematical Reasoning

```
<thinking>
El usuario pregunta cuánto es 17 * 23.

Paso 1: Descomponer el problema
- 17 * 23 = 17 * (20 + 3)
- 17 * 20 = 340
- 17 * 3 = 51

Paso 2: Sumar los resultados
- 340 + 51 = 391

Paso 3: Verificar
- 17 * 23 = 391 ✓
</thinking>
391
```

### Code Explanation

```
<thinking>
El usuario pregunta qué hace esta función Python: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)

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
</thinking>
Esta función calcula el factorial de un número de forma recursiva. El factorial de n (escrito como n!) es el producto de todos los números enteros desde 1 hasta n. Por ejemplo, factorial(5) = 5 * 4 * 3 * 2 * 1 = 120.
```

---

## 5. Dataset CSV Format

The final CSV file should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `input` | string | User question/prompt |
| `output` | string | Expected final response (after `</thinking>`) |
| `thinking` | string | Reasoning content (between tags) |
| `thinking_text` | string | Full text with thinking tags: `<thinking>{thinking}</thinking>{output}` |
| `category` | string | Category: identity, greeting, question, farewell, default |

### Example CSV

```csv
input,output,thinking,thinking_text,category
quien eres,yo soy Eduardo Piñera Aznárez,El usuario pregunta sobre mi identidad. Soy Eduardo Piñera Aznárez.,"<thinking>El usuario pregunta sobre mi identidad. Soy Eduardo Piñera Aznárez.</thinking>yo soy Eduardo Piñera Aznárez",identity
qué es la IA,la IA es una rama de la informática,La pregunta es sobre inteligencia artificial. Es un campo de la informática que crea sistemas inteligentes.,"<thinking>La pregunta es sobre inteligencia artificial. Es un campo de la informática que crea sistemas inteligentes.</thinking>la IA es una rama de la informática",question
```

---

## 6. Language-Specific Considerations

### Spanish
- Use natural, conversational Spanish
- Thinking can be more formal/technical than response
- Common patterns: "El usuario pregunta...", "Analizo la consulta..."
- Greeting responses should include warmth and emoji-like expressions

### English
- Use clear, professional English
- Thinking should be structured and logical
- Common patterns: "The user is asking...", "I need to analyze..."
- Responses should be concise and direct

### Bilingual
- When training on mixed data, ensure equal representation
- Thinking language should match the input language
- Response language should match the input language

---

## 7. Quality Guidelines

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
- [ ] Thinking starts after `<thinking>` token
- [ ] Response starts after `</thinking>` token
- [ ] Thinking logically leads to the response
- [ ] No thinking/response contradictions
- [ ] Appropriate length ratio (1:1 to 3:1 thinking:response)
- [ ] Correct language matching

---

## 8. Using Generated Datasets

### Step 1: Generate Thinking Data
```bash
# From CSV sources with templates
python generate_thinking_data.py --source csv --mode template

# From AIML sources with Ollama
python generate_thinking_data.py --source aiml --mode ollama --model llama3.2

# From all sources with HuggingFace
python generate_thinking_data.py --source all --mode hf --model Qwen/Qwen2.5-1.5B-Instruct
```

### Step 2: Validate Output
```bash
# Check output format
head -5 datasets/thinking/thinking_data.csv

# Validate consistency
python generate_thinking_data.py --source csv --validate
```

### Step 3: Add to Training Pipeline
```bash
# Copy to datasets_source/csv/
cp datasets/thinking/thinking_data.csv datasets_source/csv/

# Prepare and train
python main.py --prepare-data --aiml --hf --bpe-vocab-size 8000 --refresh-cache
python main.py --train --use-cache --epochs 30
```

---

## 9. Template Source Code Reference

The templates used by `generate_thinking_data.py` are defined in:

```python
# Spanish templates (generate_thinking_data.py:29-54)
THINKING_TEMPLATES = {
    'identity': [
        "El usuario pregunta sobre mi identidad.",
        "Quiere saber quién soy.",
    ],
    'greeting': [
        "El usuario me saluda. Debo responder con un saludo amigable.",
        "Es un saludo. Respondo con cortesía.",
    ],
    'question': [
        "El usuario hace una pregunta técnica.",
        "Analizo la consulta del usuario.",
    ],
    'farewell': [
        "El usuario se despide.",
        "Es una despedida. Respondo con amabilidad.",
    ],
    'default': [
        "El usuario me escribe. Analizo el mensaje.",
        "Proceso la solicitud del usuario.",
    ],
}
```

To add new templates, edit `generate_thinking_data.py` and add to the `THINKING_TEMPLATES` dict.

---

*See [ROADMAP.md](ROADMAP.md) for the implementation plan of thinking capabilities.*
