"""
Web thinking generator: teacher model with page context.
"""
from typing import Dict, Any, Optional
from thinking_generators import ThinkingGenerator, OllamaTeacher


class WebThinkingGenerator(ThinkingGenerator):
    """Generate thinking for web-scraped content using teacher model."""

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))
        title = sample.get('title', '')
        url = sample.get('url', '')

        if not text or len(text.strip()) < 10:
            return self._format_thinking_sample(sample, '')

        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text, title, url)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        thinking = self._rule_thinking(text, title)
        return self._format_thinking_sample(sample, thinking)

    def _teacher_thinking(self, text: str, title: str, url: str) -> Optional[str]:
        context = text[:700]
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Documento web.\n"
            f"{f'Titulo: {title}\n' if title else ''}"
            f"{f'URL: {url}\n' if url else ''}"
            f"Contenido: {context}\n\n"
            f"Resume y razona sobre el contenido principal ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _rule_thinking(self, text: str, title: str) -> str:
        word_count = len(text.split())
        if title:
            return (
                f"Esta página web titulada '{title}' contiene información relevante. "
                f"El contenido incluye {word_count} palabras con datos sobre el tema. "
                f"La información puede ser utilizada para responder preguntas relacionadas."
            )
        return (
            f"Contenido web de {word_count} palabras. "
            f"El texto contiene información que puede ser procesada para extraer conocimiento."
        )
