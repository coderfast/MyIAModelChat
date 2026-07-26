"""
HuggingFace thinking generator: teacher model or passthrough for QA datasets.
"""
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class HFThinkingGenerator(ThinkingGenerator):
    """Generate thinking for HuggingFace dataset samples."""

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)

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
            max_tokens = self.depth_config['max_tokens']
            prompt = (
                f"Pregunta: {question}\n"
                f"Respuesta: {answer[:300]}\n\n"
                f"Genera un razonamiento paso a paso que lleve de la pregunta a la respuesta:\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_tokens=max_tokens)
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
            max_tokens = self.depth_config['max_tokens']
            prompt = (
                f"Contexto: {context[:400]}\n"
                f"Pregunta/Respuesta: {question[:200]}\n\n"
                f"Razona sobre como el contexto respuesta la pregunta ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones):\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_tokens=max_tokens)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = (
            f"El contexto proporciona información relevante sobre el tema. "
            f"La pregunta se responde basándose en los datos del contexto. "
            f"Por lo tanto, la información es consistente y puede ser utilizada."
        )
        return self._format_thinking_sample(sample, thinking)

    def _generate_text_thinking(self, sample: Dict[str, Any], text: str) -> Dict[str, Any]:
        text_str = str(text)[:600]

        if self.teacher and self.teacher.is_available():
            max_tokens = self.depth_config['max_tokens']
            prompt = (
                f"Texto: {text_str}\n\n"
                f"Resume y razona sobre la información clave ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones):\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_tokens=max_tokens)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        word_count = len(str(text).split())
        thinking = (
            f"Este texto contiene {word_count} palabras con información relevante. "
            f"El contenido puede ser procesado para extraer conocimiento útil."
        )
        return self._format_thinking_sample(sample, thinking)
