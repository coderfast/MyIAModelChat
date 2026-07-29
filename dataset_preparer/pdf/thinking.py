"""
PDF thinking generator: NLP-based analysis for document chunks.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
import re
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class PDFThinkingGenerator(ThinkingGenerator):
    """Generate thinking for PDF document chunks using NLP analysis."""

    HEADING_PATTERN = re.compile(r'^(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}|.+\n[=\-]{3,})', re.MULTILINE)

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample.get('input_ids', sample.get('text', ''))

        if not text or len(text.strip()) < 20:
            return self._format_thinking_sample(sample, '')

        title = self._extract_title(text)

        # Try ThinkingEngine first (always available)
        if self.engine:
            thinking = self._engine_thinking(text, title)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher if available
        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(text, title)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback: structured analysis
        thinking = self._structured_thinking(text, title)
        return self._format_thinking_sample(sample, thinking)

    def _extract_title(self, text: str) -> str:
        """Extract title or section heading from text."""
        match = self.HEADING_PATTERN.search(text)
        if match:
            return match.group(0).strip().rstrip('=-').strip()
        first_line = text.split('\n')[0].strip()
        if len(first_line) < 100:
            return first_line
        return ''

    def _engine_thinking(self, text: str, title: str) -> Optional[str]:
        """Generate thinking using ThinkingEngine NLP analysis."""
        try:
            context = {
                'title': title,
                'type': 'pdf_document'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _teacher_thinking(self, text: str, title: str) -> Optional[str]:
        """Generate thinking using Ollama teacher model."""
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

    def _structured_thinking(self, text: str, title: str) -> str:
        """Generate structured thinking based on text analysis."""
        word_count = len(text.split())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)

        # Detect technical content
        technical_indicators = [
            r'\bAPI\b', r'\balgoritmo\b', r'\bfórmula\b', r'\becuación\b',
            r'\bdato\b', r'\bresultado\b', r'\banálisis\b', r'\bporcentaje\b',
        ]
        has_technical = any(re.search(p, text, re.IGNORECASE) for p in technical_indicators)

        # Detect numbers
        has_numbers = bool(re.search(r'\d+', text))

        # Build thinking
        parts = []
        if title:
            parts.append(f"El fragmento corresponde a la sección '{title}'.")

        if sentence_count > 1:
            parts.append(f"El texto contiene {sentence_count} oraciones con {word_count} palabras.")

        if has_technical:
            parts.append("El contenido presenta información técnica o especializada.")

        if has_numbers:
            parts.append("Se incluyen datos numéricos cuantificables.")

        # Analyze sentence structure
        if sentence_count > 0:
            avg_words = word_count / sentence_count
            if avg_words > 20:
                parts.append("Las oraciones tienen una estructura compleja.")
            elif avg_words < 10:
                parts.append("El texto utiliza oraciones cortas y directas.")

        if not parts:
            parts.append(f"Fragmento de documento PDF con {word_count} palabras de contenido relevante.")

        return " ".join(parts)
