"""
EPUB thinking generator: NLP-based analysis for book content.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class EPUBThinkingGenerator(ThinkingGenerator):
    """Generate thinking for EPUB book content using NLP analysis."""

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))
        chapter = sample.get('chapter', sample.get('title', ''))

        if not text or len(text.strip()) < 10:
            return self._format_thinking_sample(sample, '')

        # Try ThinkingEngine first (always available)
        if self.engine:
            thinking = self._engine_thinking(text, chapter)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher if available
        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text, chapter)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback: structured analysis
        thinking = self._structured_thinking(text, chapter)
        return self._format_thinking_sample(sample, thinking)

    def _engine_thinking(self, text: str, chapter: str) -> Optional[str]:
        """Generate thinking using ThinkingEngine NLP analysis."""
        try:
            context = {
                'title': chapter,
                'type': 'epub_book'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _teacher_thinking(self, text: str, chapter: str) -> Optional[str]:
        """Generate thinking using Ollama teacher model."""
        context = text[:600]
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Contenido de un libro electronico (EPUB).\n"
            f"{f'Capitulo: {chapter}\n' if chapter else ''}"
            f"Texto: {context}\n\n"
            f"Resume y razona sobre la informacion clave ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _structured_thinking(self, text: str, chapter: str) -> str:
        """Generate structured thinking based on text analysis."""
        word_count = len(text.split())
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)

        # Detect narrative elements
        narrative_indicators = ['dijo', 'respondió', 'preguntó', 'pensó', 'sintió',
                              'said', 'replied', 'asked', 'thought', 'felt']
        has_narrative = any(ind in text.lower() for ind in narrative_indicators)

        # Detect dialogue
        has_dialogue = '"' in text or "'" in text or '«' in text or '—' in text

        # Detect descriptive content
        descriptive_indicators = ['bonito', 'hermoso', 'oscuro', 'claro', 'grande',
                                 'pequeño', 'bello', 'feo', 'bright', 'dark', 'beautiful']
        has_descriptive = any(ind in text.lower() for ind in descriptive_indicators)

        # Build thinking
        parts = []
        if chapter:
            parts.append(f"El fragmento pertenece al capítulo '{chapter}'.")

        if sentence_count > 1:
            parts.append(f"El texto contiene {sentence_count} oraciones con {word_count} palabras.")

        if has_narrative:
            parts.append("El contenido presenta elementos narrativos con personajes y acciones.")

        if has_dialogue:
            parts.append("Se incluyen diálogos entre personajes.")

        if has_descriptive:
            parts.append("El texto contiene descripciones detalladas del entorno o personajes.")

        # Analyze text structure
        if sentence_count > 0:
            avg_words = word_count / sentence_count
            if avg_words > 25:
                parts.append("El estilo narrativo utiliza oraciones largas y descriptivas.")
            elif avg_words < 10:
                parts.append("El estilo narrativo utiliza oraciones cortas y dinámicas.")

        if not parts:
            parts.append(f"Fragmento de libro electrónico con {word_count} palabras de contenido narrativo.")

        return " ".join(parts)
