"""
CSV thinking generator: teacher model for structured QA pairs.
"""
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class CSVThinkingGenerator(ThinkingGenerator):
    """Generate thinking for CSV QA pairs using teacher model."""

    def __init__(self, teacher: Optional[OllamaTeacher] = None):
        super().__init__(teacher)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('output', sample.get('answer', ''))

        if not question or not answer:
            return self._format_thinking_sample(sample, '')

        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(question, answer)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = self._rule_thinking(question, answer)
        return self._format_thinking_sample(sample, thinking)

    def _teacher_thinking(self, question: str, answer: str) -> Optional[str]:
        prompt = (
            f"Analiza esta pregunta y respuesta. Genera un razonamiento paso a paso\n"
            f"que lleve logicamente de la pregunta a la respuesta.\n\n"
            f"Pregunta: {question}\n"
            f"Respuesta correcta: {answer[:300]}\n\n"
            f"Razonamiento (2-4 oraciones, en espanol):"
        )
        return self.teacher.generate(prompt, max_tokens=120)

    def _rule_thinking(self, question: str, answer: str) -> str:
        q_lower = question.lower()
        if '?' in question:
            return (
                f"La pregunta es: {question[:80]}. "
                f"La respuesta indica que {answer[:100]}. "
                f"Por lo tanto, la informacion solicitada se presenta de forma clara."
            )
        return (
            f"Se presenta informacion sobre: {question[:80]}. "
            f"La respuesta proporciona los detalles relevantes."
        )
