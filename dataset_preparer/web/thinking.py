"""
Web thinking generator: NLP-based analysis for web-scraped content.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class WebThinkingGenerator(ThinkingGenerator):
    """Generate thinking for web-scraped content using NLP analysis."""

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))
        title = sample.get('title', '')
        url = sample.get('url', '')

        if not text or len(text.strip()) < 10:
            return self._format_thinking_sample(sample, '')

        # Try ThinkingEngine first (always available)
        if self.engine:
            thinking = self._engine_thinking(text, title, url)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher if available
        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text, title, url)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback: structured analysis
        thinking = self._structured_thinking(text, title)
        return self._format_thinking_sample(sample, thinking)

    def _engine_thinking(self, text: str, title: str, url: str) -> Optional[str]:
        """Generate thinking using ThinkingEngine NLP analysis."""
        try:
            context = {
                'title': title,
                'url': url,
                'type': 'web_documentation'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _teacher_thinking(self, text: str, title: str, url: str) -> Optional[str]:
        """Generate thinking using Ollama teacher model."""
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

    def _structured_thinking(self, text: str, title: str) -> str:
        """Generate structured thinking based on web content analysis."""
        word_count = len(text.split())
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)

        # Detect content type
        technical_indicators = ['API', 'SDK', 'HTTP', 'JSON', 'REST', 'endpoint',
                              'function', 'class', 'method', 'parameter']
        has_technical = any(ind.lower() in text.lower() for ind in technical_indicators)

        # Detect documentation elements
        has_code = '```' in text or 'def ' in text or 'class ' in text or 'import ' in text
        has_lists = '- ' in text or '* ' in text or '1.' in text

        # Detect links
        has_links = 'http' in text.lower() or '[link' in text.lower() or 'click here' in text.lower()

        # Build thinking
        parts = []
        if title:
            parts.append(f"La página web titulada '{title}' contiene información relevante.")

        if sentence_count > 1:
            parts.append(f"El contenido incluye {word_count} palabras en {sentence_count} oraciones.")

        if has_technical:
            parts.append("El contenido es técnico y contiene terminología especializada.")

        if has_code:
            parts.append("Se incluyen fragmentos de código de programación.")

        if has_lists:
            parts.append("El contenido utiliza listas para organizar la información.")

        if has_links:
            parts.append("Se incluyen referencias a recursos externos.")

        if not parts:
            parts.append(f"Contenido web de {word_count} palabras con información procesable.")

        return " ".join(parts)
