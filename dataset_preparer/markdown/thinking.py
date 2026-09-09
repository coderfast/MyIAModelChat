"""
Markdown thinking generator: NLP-based analysis for markdown document chunks.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class MarkdownThinkingGenerator(ThinkingGenerator):
    """Generate thinking for Markdown document chunks using NLP analysis."""

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))

        if not text:
            return self._format_thinking_sample(sample, '')

        # Try ThinkingEngine first (always available)
        if self.engine:
            thinking = self._engine_thinking(text)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher if available
        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback: minimal structured thinking
        thinking = self._minimal_thinking(text)
        return self._format_thinking_sample(sample, thinking)

    def _engine_thinking(self, text: str) -> Optional[str]:
        """Generate thinking using ThinkingEngine NLP analysis."""
        try:
            context = {
                'text': text,
                'type': 'markdown_chunk'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _teacher_thinking(self, text: str) -> Optional[str]:
        """Generate thinking using Ollama teacher model."""
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Analiza este fragmento de documento markdown. Genera un razonamiento paso a paso\n"
            f"que resuma los conceptos clave y su relaci\u00f3n l\u00f3gica.\n\n"
            f"Texto:\n{text[:800]}\n\n"
            f"Razonamiento ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones, en espa\u00f1ol):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _minimal_thinking(self, text: str) -> str:
        """Generate minimal thinking when no engine or teacher is available."""
        words = text.split()
        word_count = len(words)

        # Detect headings
        headings = []
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#'):
                heading_text = stripped.lstrip('#').strip()
                if heading_text:
                    headings.append(heading_text)

        result = [f"Fragmento de documento con {word_count} palabras."]
        if headings:
            result.append(f"Contiene secciones: {', '.join(headings[:3])}.")
        result.append("El contenido proporciona informaci\u00f3n t\u00e9cnica o explicativa sobre el tema tratado.")

        return " ".join(result)
