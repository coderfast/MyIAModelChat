"""
AIML thinking generator: hybrid rule-based + teacher model.
Simple patterns use rules, complex patterns use Ollama teacher.
"""
import re
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class AIMLThinkingGenerator(ThinkingGenerator):
    """Generate thinking for AIML pattern-template pairs."""

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

    def __init__(self, teacher: Optional[OllamaTeacher] = None):
        super().__init__(teacher)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pattern = sample.get('input', '').lower().strip()
        template = sample.get('output', '').strip()

        if not pattern or not template:
            return self._format_thinking_sample(sample, '')

        for category, keywords in self.SIMPLE_CATEGORIES.items():
            if any(kw in pattern for kw in keywords):
                thinking = self._rule_based_thinking(category, pattern, template)
                return self._format_thinking_sample(sample, thinking)

        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(pattern, template)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = self._fallback_thinking(pattern, template)
        return self._format_thinking_sample(sample, thinking)

    def _rule_based_thinking(self, category: str, pattern: str, template: str) -> str:
        rules = {
            'greeting': (
                f"El usuario hace un saludo con '{pattern}'. "
                f"Respondo con un saludo amigual y cortes."
            ),
            'farewell': (
                f"El usuario se despide. "
                f"Respondo con una despedida apropiada y amigable."
            ),
            'identity': (
                f"El usuario pregunta sobre mi identidad o nombre. "
                f"Presento al asistente de forma amigable."
            ),
            'thanks': (
                f"El usuario expresa agradecimiento. "
                f"Acepto las gracias de forma amable."
            ),
            'yes': (
                f"El usuario da una respuesta afirmativa. "
                f"Confirmo la información positivamente."
            ),
            'no': (
                f"El usuario da una respuesta negativa. "
                f"Acepto la respuesta sin insistir."
            ),
            'weather': (
                f"El usuario pregunta sobre el clima o temperatura. "
                f"Proporciono información meteorológica general."
            ),
            'time': (
                f"El usuario pregunta la hora. "
                f"Indico que no tengo acceso al reloj en tiempo real."
            ),
            'help': (
                f"El usuario necesita ayuda. "
                f"Ofrezco asistencia de forma amigable."
            ),
        }
        return rules.get(category, f"Procesando solicitud del usuario sobre {category}.")

    def _teacher_thinking(self, pattern: str, template: str) -> Optional[str]:
        prompt = (
            f"Analiza esta pregunta de chatbot y genera un razonamiento paso a paso.\n"
            f"Pregunta del usuario: {pattern}\n"
            f"Respuesta del bot: {template[:200]}\n\n"
            f"Razonamiento (2-4 oraciones, en español):"
        )
        return self.teacher.generate(prompt, max_tokens=100)

    def _fallback_thinking(self, pattern: str, template: str) -> str:
        words = pattern.split()
        if len(words) <= 2:
            return f"El usuario hace una consulta breve. Respondo de forma directa y clara."
        elif '?' in pattern:
            return f"El usuario hace una pregunta. Proporciono una respuesta informativa basada en el contexto."
        else:
            return f"El usuario envía un mensaje. Proceso la solicitud y respondo de forma apropiada."
