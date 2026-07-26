"""
PDF thinking generator: teacher model with section/chunk context.
"""
import re
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class PDFThinkingGenerator(ThinkingGenerator):
    """Generate thinking for PDF document chunks using teacher model."""

    HEADING_PATTERN = re.compile(r'^(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}|.+\n[=\-]{3,})', re.MULTILINE)

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))

        if not text or len(text.strip()) < 20:
            return self._format_thinking_sample(sample, '')

        title = self._extract_title(text)

        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text, title)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = self._rule_thinking(text, title)
        return self._format_thinking_sample(sample, thinking)

    def _extract_title(self, text: str) -> str:
        match = self.HEADING_PATTERN.search(text)
        if match:
            return match.group(0).strip().rstrip('=-').strip()
        first_line = text.split('\n')[0].strip()
        if len(first_line) < 100:
            return first_line
        return ''

    def _teacher_thinking(self, text: str, title: str) -> Optional[str]:
        context = text[:600]
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Basado en el siguiente texto de un documento PDF, resume y razona "
            f"sobre la informacion clave.\n\n"
            f"{f'Titulo/Seccion: {title}\n' if title else ''}"
            f"Texto: {context}\n\n"
            f"Razonamiento paso a paso ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _rule_thinking(self, text: str, title: str) -> str:
        word_count = len(text.split())
        if title:
            return (
                f"Este fragmento del documento PDF corresponde a la sección '{title}'. "
                f"El contenido presenta información técnica o académica. "
                f"El texto contiene {word_count} palabras que cubren aspectos fundamentales del tema."
            )
        return (
            f"Fragmento de documento PDF con {word_count} palabras. "
            f"El contenido contiene información que debe ser procesada para extraer conocimiento relevante."
        )
