"""
AIML thinking generator: hybrid rule-based + teacher model.
Simple patterns use rules, complex patterns use Ollama teacher.
"""
import re
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher


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

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)

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
                f"El usuario se dirige a mí con un saludo '{pattern}'. "
                f"Esto indica que quiere iniciar una conversación. "
                f"Debo responder de forma amigable y ofrecer mi ayuda."
            ),
            'farewell': (
                f"El usuario se despide con '{pattern}'. "
                f"Esto indica que la conversación está terminando. "
                f"Debo despedirme de forma amable y dejar la puerta abierta."
            ),
            'identity': (
                f"El usuario pregunta sobre mi identidad o nombre con '{pattern}'. "
                f"Quiere saber quién soy. "
                f"Debo presentarme de forma clara y amigable."
            ),
            'thanks': (
                f"El usuario expresa agradecimiento con '{pattern}'. "
                f"Alguien agradece algo que hice. "
                f"Debo aceptar las gracias de forma amable."
            ),
            'yes': (
                f"El usuario da una respuesta afirmativa con '{pattern}'. "
                f"Esto indica conformidad o acuerdo. "
                f"Debo confirmar la información positivamente."
            ),
            'no': (
                f"El usuario da una respuesta negativa con '{pattern}'. "
                f"Esto indica desacuerdo o rechazo. "
                f"Debo aceptar la respuesta sin insistir."
            ),
            'weather': (
                f"El usuario pregunta sobre el clima con '{pattern}'. "
                f"Quiere información meteorológica. "
                f"Debo proporcionar datos generales del clima."
            ),
            'time': (
                f"El usuario pregunta la hora con '{pattern}'. "
                f"Necesita saber la hora actual. "
                f"Debo indicar que no tengo acceso al reloj en tiempo real."
            ),
            'help': (
                f"El usuario necesita ayuda con '{pattern}'. "
                f"Está solicitando asistencia. "
                f"Debo ofrecer ayuda de forma proactiva y clara."
            ),
        }
        return rules.get(category, f"Procesando solicitud del usuario sobre {category}.")

    def _teacher_thinking(self, pattern: str, template: str) -> Optional[str]:
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Analiza esta pregunta de chatbot y genera un razonamiento paso a paso.\n"
            f"Pregunta del usuario: {pattern}\n"
            f"Respuesta del bot: {template[:200]}\n\n"
            f"Razonamiento ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones, en español):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _fallback_thinking(self, pattern: str, template: str) -> str:
        words = pattern.split()
        if len(words) <= 2:
            return (
                f"El usuario hace una consulta breve: '{pattern}'. "
                f"Esto indica que busca información específica. "
                f"Debo responder de forma directa y clara."
            )
        elif '?' in pattern:
            return (
                f"El usuario hace una pregunta: '{pattern}'. "
                f"Esto indica que necesita información. "
                f"Debo proporcionar una respuesta informativa basada en el contexto."
            )
        else:
            return (
                f"El usuario envía un mensaje: '{pattern}'. "
                f"Esto indica una solicitud o comentario. "
                f"Debo procesar la solicitud y responder de forma apropiada."
            )
