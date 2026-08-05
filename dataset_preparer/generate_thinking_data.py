"""
Generate training data with <thinking> tags for chain-of-thought reasoning.

Reads existing QA pairs and generates synthetic reasoning inside
<thinking>...</thinking> tags before the final answer.

Usage:
    python generate_thinking_data.py --source csv --input datasets_source/csv/special_facts.csv --output datasets/thinking/thinking_data.csv
    python generate_thinking_data.py --source aiml --input datasets_source/aiml --output datasets/thinking/thinking_data.csv
    python generate_thinking_data.py --source all --output datasets/thinking/thinking_data.csv
"""

import os
import csv
import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Setup logging (configured by main.py)
logger = logging.getLogger(__name__)

# Thinking templates by category
THINKING_TEMPLATES = {
    'identity': [
        "El usuario pregunta sobre mi identidad. Debo responder con la información correcta sobre mi creador.",
        "Esta es una pregunta sobre quién me creó. Voy a dar la respuesta directa.",
        "Pregunta de identidad. Respondo de forma clara y directa.",
    ],
    'greeting': [
        "El usuario me saluda. Debo responder de forma amigable y preguntar cómo puedo ayudar.",
        "Saludo del usuario. Respondo de manera cálida y ofrezco mi ayuda.",
        "El usuario inicia la conversación con un saludo. Mantengo un tono amigable.",
    ],
    'question': [
        "El usuario hace una pregunta. Voy a analizar la información disponible y dar una respuesta clara.",
        "Pregunta del usuario. Debo dar una respuesta completa y precisa.",
        "Esta es una pregunta directa. Proporciono la información más relevante.",
    ],
    'farewell': [
        "El usuario se despide. Debo despedirme de forma amigable.",
        "Despedida del usuario. Respondo con amabilidad y cierro la conversación.",
    ],
    'default': [
        "El usuario me escribe. Analizo el mensaje y preparo una respuesta apropiada.",
        "Proceso el mensaje del usuario. Genero una respuesta relevante y útil.",
        "Recibo el mensaje. Evalúo el contenido y formulo mi respuesta.",
    ],
}

# Spanish thinking patterns for variety
THINKING_PATTERNS_ES = [
    "Voy a analizar esto: {context}. Mi respuesta será: {answer}",
    "El usuario pregunta sobre {topic}. {context} Respondo directamente.",
    "{context} Esto es lo que sé: {answer}",
    "Analizo la consulta: {context}. Proporciono la respuesta correcta.",
]

# English thinking templates
THINKING_TEMPLATES_EN = {
    'identity': [
        "The user is asking about my identity. I should respond with information about my creator.",
        "This is a question about who created me. I'll give a direct answer.",
        "Identity question. I respond clearly and directly.",
    ],
    'greeting': [
        "The user is greeting me. I should respond in a friendly manner and ask how I can help.",
        "User greeting. I respond warmly and offer my assistance.",
        "The user starts the conversation with a greeting. I maintain a friendly tone.",
    ],
    'question': [
        "The user is asking a question. I'll analyze the available information and give a clear answer.",
        "User question. I should give a complete and accurate response.",
        "This is a direct question. I provide the most relevant information.",
    ],
    'farewell': [
        "The user is saying goodbye. I should respond politely.",
        "User farewell. I respond with kindness and close the conversation.",
    ],
    'default': [
        "The user is writing to me. I'll analyze the message and prepare an appropriate response.",
        "Processing the user's message. I'll generate a relevant and helpful response.",
        "I received the message. I'll evaluate the content and formulate my response.",
    ],
}

THINKING_PATTERNS_EN = [
    "Let me analyze this: {context}. My answer will be: {answer}",
    "The user is asking about {topic}. {context} I'll respond directly.",
    "{context} Here's what I know: {answer}",
    "Analyzing the query: {context}. Providing the correct answer.",
]


def detect_category(text: str) -> str:
    """Detect the category of a text for thinking generation."""
    text_lower = text.lower().strip()

    identity_keywords = ['quién es tu creador', 'quién te creó', 'quien te creo', 'quién eres', 'quien eres']
    greeting_keywords = ['hola', 'hello', 'buenos días', 'buenas tardes', 'hey', 'saludos']
    farewell_keywords = ['adiós', 'adios', 'bye', 'hasta luego', 'nos vemos', 'chao']

    for kw in identity_keywords:
        if kw in text_lower:
            return 'identity'
    for kw in greeting_keywords:
        if kw in text_lower:
            return 'greeting'
    for kw in farewell_keywords:
        if kw in text_lower:
            return 'farewell'

    if '?' in text or 'cuál' in text_lower or 'qué' in text_lower or 'cómo' in text_lower:
        return 'question'

    return 'default'


def generate_thinking(text: str, answer: str, category: str, lang: str = 'es') -> str:
    """Generate a thinking block for a given text and answer."""
    if lang == 'en':
        templates = THINKING_TEMPLATES_EN.get(category, THINKING_TEMPLATES_EN.get('default', []))
        patterns = THINKING_PATTERNS_EN
    else:
        templates = THINKING_TEMPLATES.get(category, THINKING_TEMPLATES['default'])
        patterns = THINKING_PATTERNS_ES

    if not templates:
        return "Analyzing the message..."

    template = random.choice(templates)

    # Add variety with patterns
    if random.random() > 0.5 and category == 'question':
        context = text[:80] + ('...' if len(text) > 80 else '')
        topic = text[:40]
        pattern = random.choice(patterns)
        thinking = pattern.format(context=context, topic=topic, answer=answer[:60])
    else:
        thinking = template

    return thinking


def generate_thinking_with_hf(text: str, answer: str, pipeline_instance) -> str:
    """Generate thinking using a HuggingFace text-generation model."""
    prompt = (
        f"Genera un breve razonamiento interno (1-3 oraciones) para esta pregunta y respuesta. "
        f"Formato: solo el razonamiento, sin etiquetas.\n"
        f"Pregunta: {text}\nRespuesta: {answer}\nRazonamiento:"
    )
    try:
        result = pipeline_instance(prompt, max_new_tokens=100, num_return_sequences=1,
                                   do_sample=True, temperature=0.7, top_p=0.9)
        generated = result[0].get('generated_text', '')
        # Extract only the reasoning part after the prompt
        if 'Razonamiento:' in generated:
            reasoning = generated.split('Razonamiento:')[-1].strip()
        else:
            reasoning = generated[len(prompt):].strip()
        return reasoning if reasoning else "Analizo la consulta del usuario."
    except Exception as e:
        logger.warning(f"  HF generation failed: {e}. Falling back to template.")
        return generate_thinking(text, answer, detect_category(text))


def generate_thinking_with_ollama(text: str, answer: str, model: str = 'llama3.2') -> str:
    """Generate thinking using Ollama API."""
    import requests as req_lib
    prompt = (
        f"Genera un breve razonamiento interno (1-3 oraciones) para esta pregunta y respuesta. "
        f"Formato: solo el razonamiento, sin etiquetas.\n"
        f"Pregunta: {text}\nRespuesta: {answer}\nRazonamiento:"
    )
    try:
        response = req_lib.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            reasoning = data.get('response', '').strip()
            return reasoning if reasoning else "Analizo la consulta del usuario."
        else:
            logger.warning(f"  Ollama returned status {response.status_code}")
            return generate_thinking(text, answer, detect_category(text))
    except Exception as e:
        logger.warning(f"  Ollama generation failed: {e}. Falling back to template.")
        return generate_thinking(text, answer, detect_category(text))


def validate_thinking_consistency(text: str, thinking: str, answer: str) -> bool:
    """Validate that thinking is logically consistent with the answer."""
    if not thinking or not answer:
        return False
    # Basic validation: thinking should mention some words from the answer
    import re
    answer_words = set(re.findall(r'\w+', answer.lower()))
    thinking_words = set(re.findall(r'\w+', thinking.lower()))
    overlap = answer_words & thinking_words
    # At least some overlap between answer and thinking
    if len(answer_words) > 0 and len(overlap) / len(answer_words) < 0.05:
        return False
    # Thinking should not be too short
    if len(thinking.strip()) < 10:
        return False
    return True


def load_csv_data(filepath: str) -> List[Dict[str, str]]:
    """Load QA pairs from CSV file."""
    pairs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Detect if there's a header
            first_line = f.readline()
            f.seek(0)

            if 'input' in first_line and 'output' in first_line:
                reader = csv.DictReader(f)
            else:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        pairs.append({'input': row[0].strip(), 'output': row[1].strip()})

            if 'input' in first_line and 'output' in first_line:
                for row in reader:
                    pairs.append({'input': row['input'].strip(), 'output': row['output'].strip()})

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")

    return pairs


def load_aiml_data(aiml_dir: str) -> List[Dict[str, str]]:
    """Load QA pairs from AIML .datasets files."""
    pairs = []
    if not os.path.exists(aiml_dir):
        logger.warning(f"AIML directory not found: {aiml_dir}")
        return pairs

    import pickle
    from datasets import Dataset

    for filename in os.listdir(aiml_dir):
        if filename.endswith('.datasets'):
            filepath = os.path.join(aiml_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, Dataset) and 'input' in data.column_names and 'output' in data.column_names:
                    for item in data:
                        pairs.append({'input': item['input'], 'output': item['output']})
                    logger.info(f"  Loaded {len(data)} pairs from {filename}")
            except Exception as e:
                logger.warning(f"  Error loading {filename}: {e}")

    return pairs


def generate_thinking_dataset(pairs: List[Dict[str, str]], lang: str = 'es', generator: str = 'template', hf_pipeline=None, ollama_model: str = 'llama3.2') -> List[Dict[str, str]]:
    """Generate thinking dataset from QA pairs.

    Args:
        pairs: List of dicts with 'input' and 'output' keys
        lang: Language for templates ('es' or 'en')
        generator: Generation mode ('template', 'hf', 'ollama')
        hf_pipeline: HuggingFace pipeline instance (required if generator='hf')
        ollama_model: Ollama model name (used if generator='ollama')
    """
    thinking_pairs = []

    for pair in pairs:
        text = pair['input']
        answer = pair['output']

        category = detect_category(text)

        if generator == 'hf' and hf_pipeline is not None:
            thinking = generate_thinking_with_hf(text, answer, hf_pipeline)
        elif generator == 'ollama':
            thinking = generate_thinking_with_ollama(text, answer, ollama_model)
        else:
            thinking = generate_thinking(text, answer, category, lang)

        thinking_text = f"<thinking>{thinking}</thinking>{answer}"
        thinking_pairs.append({
            'input': text,
            'output': answer,
            'thinking': thinking,
            'thinking_text': thinking_text,
            'category': category,
        })

    return thinking_pairs


def save_thinking_data(pairs: List[Dict[str, str]], output_path: str):
    """Save thinking data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'output', 'thinking', 'thinking_text', 'category'])
        writer.writeheader()
        writer.writerows(pairs)

    logger.info(f"  Saved {len(pairs)} thinking examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data with <thinking> tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_thinking_data.py --source csv --input datasets_source/csv/special_facts.csv
  python generate_thinking_data.py --source aiml --input datasets_source/aiml
  python generate_thinking_data.py --source all --output datasets/thinking/thinking_data.csv
        """
    )
    parser.add_argument('--source', choices=['csv', 'aiml', 'all'], default='all',
                        help='Data source type (default: all)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input file/directory path')
    parser.add_argument('--output', type=str, default='datasets/thinking/thinking_data.csv',
                        help='Output CSV path (default: datasets/thinking/thinking_data.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--lang', type=str, default='es', choices=['es', 'en'],
                        help='Language for templates (default: es)')
    parser.add_argument('--mode', type=str, default='template', choices=['template', 'hf', 'ollama'],
                        help='Generation mode (default: template)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name for HF or Ollama mode')
    parser.add_argument('--validate', action='store_true',
                        help='Validate thinking consistency with answers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples to process')

    args = parser.parse_args()
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("GENERATING THINKING TRAINING DATA")
    logger.info("=" * 60)

    all_pairs = []

    # Load data based on source
    if args.source in ('csv', 'all'):
        csv_path = args.input if args.source == 'csv' and args.input else os.path.join('datasets_source', 'csv', 'special_facts.csv')
        if os.path.exists(csv_path):
            logger.info(f"Loading CSV data from: {csv_path}")
            csv_pairs = load_csv_data(csv_path)
            all_pairs.extend(csv_pairs)
            logger.info(f"  Loaded {len(csv_pairs)} pairs")
        else:
            logger.warning(f"  CSV not found: {csv_path}")

    if args.source in ('aiml', 'all'):
        aiml_dir = args.input if args.source == 'aiml' and args.input else os.path.join('datasets_source', 'aiml')
        if os.path.exists(aiml_dir):
            logger.info(f"Loading AIML data from: {aiml_dir}")
            aiml_pairs = load_aiml_data(aiml_dir)
            all_pairs.extend(aiml_pairs)
            logger.info(f"  Loaded {len(aiml_pairs)} pairs")
        else:
            logger.warning(f"  AIML directory not found: {aiml_dir}")

    if not all_pairs:
        logger.error("No data pairs loaded. Nothing to generate.")
        return

    # Limit samples if requested
    if args.max_samples and args.max_samples > 0:
        all_pairs = all_pairs[:args.max_samples]
        logger.info(f"  Limited to {len(all_pairs)} samples")

    # Remove duplicates
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair['input'].strip().lower(), pair['output'].strip().lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    logger.info(f"Total unique pairs: {len(unique_pairs)}")

    # Initialize HF pipeline if needed
    hf_pipeline = None
    if args.mode == 'hf':
        model_name = args.model or 'Qwen/Qwen2.5-1.5B-Instruct'
        logger.info(f"Loading HuggingFace model: {model_name}")
        try:
            from transformers import pipeline as hf_pipeline_fn
            hf_pipeline = hf_pipeline_fn('text-generation', model=model_name, trust_remote_code=True)
            logger.info(f"  Model loaded successfully")
        except Exception as e:
            logger.warning(f"  Could not load model: {e}. Falling back to template mode.")
            args.mode = 'template'

    # Generate thinking data
    logger.info("Generating <thinking> data...")
    thinking_data = generate_thinking_dataset(
        unique_pairs,
        lang=args.lang,
        generator=args.mode,
        hf_pipeline=hf_pipeline,
        ollama_model=args.model or 'llama3.2'
    )

    # Validate if requested
    if args.validate:
        logger.info("Validating thinking consistency...")
        valid_count = 0
        invalid_count = 0
        for item in thinking_data:
            if validate_thinking_consistency(item['input'], item['thinking'], item['output']):
                valid_count += 1
            else:
                invalid_count += 1
        logger.info(f"  Valid: {valid_count}, Invalid: {invalid_count}")
        if invalid_count > 0:
            logger.info(f"  ({invalid_count} samples had weak thinking-answer consistency)")

    # Save
    save_thinking_data(thinking_data, args.output)

    # Show examples
    logger.info("=" * 60)
    logger.info("EXAMPLES")
    logger.info("=" * 60)
    for i, example in enumerate(thinking_data[:5]):
        logger.info(f"--- Example {i+1} [{example['category']}] ---")
        logger.info(f"Input:    {example['input']}")
        logger.info(f"Thinking: {example['thinking']}")
        logger.info(f"Output:   {example['output']}")
        logger.info(f"Full:     {example['thinking_text']}")

    # Stats
    categories = {}
    for p in thinking_data:
        cat = p['category']
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("=" * 60)
    logger.info("STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total examples: {len(thinking_data)}")
    logger.info(f"Categories:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat:15} {count:>6}")

    logger.info(f"Output: {args.output}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
