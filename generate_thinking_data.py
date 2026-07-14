"""
Generate training data with <think> tags for chain-of-thought reasoning.

Reads existing QA pairs and generates synthetic reasoning inside
<think>...</think> tags before the final answer.

Usage:
    python generate_thinking_data.py --source csv --input datasets/special_facts.csv --output datasets/thinking/thinking_data.csv
    python generate_thinking_data.py --source aiml --input aiml_dev --output datasets/thinking/thinking_data.csv
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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


def generate_thinking(text: str, answer: str, category: str) -> str:
    """Generate a thinking block for a given text and answer."""
    templates = THINKING_TEMPLATES.get(category, THINKING_TEMPLATES['default'])
    template = random.choice(templates)

    # Add variety with patterns
    if random.random() > 0.5 and category == 'question':
        context = text[:80] + ('...' if len(text) > 80 else '')
        topic = text[:40]
        pattern = random.choice(THINKING_PATTERNS_ES)
        thinking = pattern.format(context=context, topic=topic, answer=answer[:60])
    else:
        thinking = template

    return thinking


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


def generate_thinking_dataset(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Generate thinking dataset from QA pairs."""
    thinking_pairs = []

    for pair in pairs:
        text = pair['input']
        answer = pair['output']

        category = detect_category(text)
        thinking = generate_thinking(text, answer, category)

        thinking_text = f"<think>{thinking}</think>{answer}"
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
        description='Generate training data with <think> tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_thinking_data.py --source csv --input datasets/special_facts.csv
  python generate_thinking_data.py --source aiml --input aiml_dev
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

    args = parser.parse_args()
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("GENERATING THINKING TRAINING DATA")
    logger.info("=" * 60)

    all_pairs = []

    # Load data based on source
    if args.source in ('csv', 'all'):
        csv_path = args.input if args.source == 'csv' and args.input else 'datasets/special_facts.csv'
        if os.path.exists(csv_path):
            logger.info(f"\nLoading CSV data from: {csv_path}")
            csv_pairs = load_csv_data(csv_path)
            all_pairs.extend(csv_pairs)
            logger.info(f"  Loaded {len(csv_pairs)} pairs")
        else:
            logger.warning(f"  CSV not found: {csv_path}")

    if args.source in ('aiml', 'all'):
        aiml_dir = args.input if args.source == 'aiml' and args.input else 'aiml_dev'
        if os.path.exists(aiml_dir):
            logger.info(f"\nLoading AIML data from: {aiml_dir}")
            aiml_pairs = load_aiml_data(aiml_dir)
            all_pairs.extend(aiml_pairs)
            logger.info(f"  Loaded {len(aiml_pairs)} pairs")
        else:
            logger.warning(f"  AIML directory not found: {aiml_dir}")

    if not all_pairs:
        logger.error("No data pairs loaded. Nothing to generate.")
        return

    # Remove duplicates
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair['input'].strip().lower(), pair['output'].strip().lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    logger.info(f"\nTotal unique pairs: {len(unique_pairs)}")

    # Generate thinking data
    logger.info("\nGenerating <think> data...")
    thinking_data = generate_thinking_dataset(unique_pairs)

    # Save
    save_thinking_data(thinking_data, args.output)

    # Show examples
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLES")
    logger.info("=" * 60)
    for i, example in enumerate(thinking_data[:5]):
        logger.info(f"\n--- Example {i+1} [{example['category']}] ---")
        logger.info(f"Input:    {example['input']}")
        logger.info(f"Thinking: {example['thinking']}")
        logger.info(f"Output:   {example['output']}")
        logger.info(f"Full:     {example['thinking_text']}")

    # Stats
    categories = {}
    for p in thinking_data:
        cat = p['category']
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("\n" + "=" * 60)
    logger.info("STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total examples: {len(thinking_data)}")
    logger.info(f"Categories:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat:15} {count:>6}")

    logger.info(f"\nOutput: {args.output}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
