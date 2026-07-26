"""
EPUB thinking generator: teacher model with chapter context.
"""
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class EPUBThinkingGenerator(ThinkingGenerator):
    """Generate thinking for EPUB book content using teacher model."""

    def __init__(self, teacher: Optional[OllamaTeacher] = None):
        super().__init__(teacher)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))
        chapter = sample.get('chapter', sample.get('title', ''))

        if not text or len(text.strip()) < 10:
            return self._format_thinking_sample(sample, '')

        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text, chapter)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = self._rule_thinking(text, chapter)
        return self._format_thinking_sample(sample, thinking)

    def _teacher_thinking(self, text: str, chapter: str) -> Optional[str]:
        context = text[:600]
        prompt = (
            f"Contenido de un libro electronico (EPUB).\n"
            f"{f'Capitulo: {chapter}\n' if chapter else ''}"
            f"Texto: {context}\n\n"
            f"Resume y razona sobre la informacion clave (2-4 oraciones):"
        )
        return self.teacher.generate(prompt, max_tokens=120)

    def _rule_thinking(self, text: str, chapter: str) -> str:
        word_count = len(text.split())
        if chapter:
            return (
                f"Este fragmento pertenece al capítulo '{chapter}' del libro. "
                f"El contenido presenta información narrativa o informativa. "
                f"El texto contiene {word_count} palabras que describen aspectos relevantes del tema."
            )
        return (
            f"Fragmento de libro electrónico con {word_count} palabras. "
            f"El contenido contiene información que debe ser analizada para extraer conocimiento."
        )
