"""
AIML thinking generator: NLP-based analysis for chatbot pattern-template pairs.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
import re
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class AIMLThinkingGenerator(ThinkingGenerator):
    """Generate thinking for AIML pattern-template pairs using NLP analysis."""

    SIMPLE_CATEGORIES = {
        'greeting': ['hello', 'hi', 'hey', 'hola', 'buenos dias', 'buenas', 'saludos'],
        'farewell': ['bye', 'goodbye', 'chao', 'adios', 'hasta luego', 'nos vemos', 'hasta'],
        'identity': ['who are you', 'what is your name', 'que eres', 'como te llamas', 'your name'],
        'thanks': ['thank', 'gracias', 'agradezco'],
        'yes': ['yes', 'si', 'ok', 'correcto', 'exacto', 'affirmative'],
        'no': ['no', 'nah', 'para nada', 'negativo'],
        'weather': ['weather', 'clima', 'temperatura', 'lluvia', 'sol'],
        'time': ['time', 'hora', 'que hora', 'hora es'],
        'help': ['help', 'ayuda', 'necesito', 'puedes ayudar'],
    }

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # AIML data may have 'input'/'output' or combined 'input_ids'
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        combined = sample.get('input_ids', '')

        if input_text and output_text:
            pattern = input_text.lower().strip()
            template = output_text.strip()
            text_for_thinking = f"{pattern} {template}"
        elif combined:
            text_for_thinking = combined.strip()
            pattern = text_for_thinking.lower()
            template = text_for_thinking
        else:
            return self._format_thinking_sample(sample, '')

        # Detect category for context
        detected_category = None
        for category, keywords in self.SIMPLE_CATEGORIES.items():
            if any(kw in pattern for kw in keywords):
                detected_category = category
                break

        # Try ThinkingEngine first (always available)
        if self.engine:
            thinking = self._engine_thinking(text_for_thinking, detected_category)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher if available
        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text_for_thinking)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback: category-based thinking
        thinking = self._category_thinking(pattern, template, detected_category)
        return self._format_thinking_sample(sample, thinking)

    def _engine_thinking(self, text: str, category: Optional[str]) -> Optional[str]:
        """Generate thinking using ThinkingEngine NLP analysis."""
        try:
            context = {
                'question': text,
                'category': category,
                'type': 'chatbot_pair'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _teacher_thinking(self, text: str) -> Optional[str]:
        """Generate thinking using Ollama teacher model."""
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Analiza esta interaccion de chatbot y genera un razonamiento paso a paso.\n"
            f"Texto: {text[:400]}\n\n"
            f"Razonamiento ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones, en español):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _category_thinking(self, pattern: str, template: str, category: Optional[str]) -> str:
        """Generate thinking based on detected category."""
        if category == 'greeting':
            return (
                f"El usuario inicia la conversación con un saludo. "
                f"La intención detectada es establecer contacto. "
                f"La respuesta apropiada debe ser amigable y ofrecer ayuda."
            )
        elif category == 'farewell':
            return (
                f"El usuario se despide, indicando el fin de la conversación. "
                f"La respuesta debe despedirse de forma amable y dejar la puerta abierta."
            )
        elif category == 'identity':
            return (
                f"El usuario pregunta sobre la identidad del asistente. "
                f"La respuesta debe presentar al bot de forma clara y amigable."
            )
        elif category == 'thanks':
            return (
                f"El usuario expresa agradecimiento. "
                f"La respuesta debe aceptar las gracias de forma amable."
            )
        elif category == 'yes':
            return (
                f"El usuario confirma o acepta algo. "
                f"La respuesta debe continuar positivamente."
            )
        elif category == 'no':
            return (
                f"El usuario niega o rechaza algo. "
                f"La respuesta debe aceptar sin insistir."
            )
        elif category == 'weather':
            return (
                f"El usuario solicita información meteorológica. "
                f"La respuesta debe proporcionar datos del clima."
            )
        elif category == 'time':
            return (
                f"El usuario pregunta la hora. "
                f"La respuesta debe indicar la hora actual o la incapacidad de proporcionarla."
            )
        elif category == 'help':
            return (
                f"El usuario necesita asistencia. "
                f"La respuesta debe ofrecer ayuda de forma proactiva y clara."
            )
        else:
            # Generic thinking for unknown patterns
            words = pattern.split()
            if len(words) <= 2:
                return (
                    f"El usuario hace una consulta breve. "
                    f"La respuesta debe ser directa y clara."
                )
            elif '?' in pattern:
                return (
                    f"El usuario formula una pregunta. "
                    f"La respuesta debe proporcionar información relevante."
                )
            else:
                return (
                    f"El usuario envía un mensaje. "
                    f"La respuesta debe procesar la solicitud de forma apropiada."
                )
