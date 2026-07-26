"""
HuggingFace thinking generator: teacher model or passthrough for QA datasets.
"""
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class HFThinkingGenerator(ThinkingGenerator):
    """Generate thinking for HuggingFace dataset samples."""

    def __init__(self, teacher: Optional[OllamaTeacher] = None):
        super().__init__(teacher)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'question' in sample and 'answer' in sample:
            return self._generate_qa_thinking(sample)

        if 'context' in sample and ('question' in sample or 'answer' in sample):
            return self._generate_context_thinking(sample)

        text = sample.get('text', sample.get('input_ids', ''))
        if not text or len(str(text).strip()) < 10:
            return self._format_thinking_sample(sample, '')

        return self._generate_text_thinking(sample, text)

    def _generate_qa_thinking(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get('question', '')
        answer = sample.get('answer', '')

        if self.teacher and self.teacher.is_available():
            prompt = (
                f"Pregunta: {question}\n"
                f"Respuesta: {answer[:300]}\n\n"
                f"Genera un razonamiento paso a paso que lleve de la pregunta a la respuesta:\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_tokens=120)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = (
            f"La pregunta es: {question[:100]}. "
            f"La respuesta correcta es: {answer[:100]}. "
            f"Esta informacion es consistente y puede ser utilizada para entrenamiento."
        )
        return self._format_thinking_sample(sample, thinking)

    def _generate_context_thinking(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        context = sample.get('context', '')
        question = sample.get('question', sample.get('answer', ''))

        if self.teacher and self.teacher.is_available():
            prompt = (
                f"Contexto: {context[:400]}\n"
                f"Pregunta/Respuesta: {question[:200]}\n\n"
                f"Razona sobre como el contexto respuesta la pregunta (2-3 oraciones):\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_tokens=100)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = (
            f"El contexto proporciona informacion relevante. "
            f"La pregunta se responde basandose en los datos del contexto."
        )
        return self._format_thinking_sample(sample, thinking)

    def _generate_text_thinking(self, sample: Dict[str, Any], text: str) -> Dict[str, Any]:
        text_str = str(text)[:600]

        if self.teacher and self.teacher.is_available():
            prompt = (
                f"Texto: {text_str}\n\n"
                f"Resume y razona sobre la informacion clave (2-3 oraciones):\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_tokens=100)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        word_count = len(str(text).split())
        thinking = (
            f"Texto con {word_count} palabras. "
            f"Contiene informacion que puede ser procesada para entrenamiento."
        )
        return self._format_thinking_sample(sample, thinking)
