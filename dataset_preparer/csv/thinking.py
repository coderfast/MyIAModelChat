"""
CSV thinking generator: NLP-based analysis for structured QA pairs.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class CSVThinkingGenerator(ThinkingGenerator):
    """Generate thinking for CSV QA pairs using NLP analysis."""

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('output', sample.get('answer', ''))

        if not question or not answer:
            return self._format_thinking_sample(sample, '')

        # Try ThinkingEngine first (always available)
        if self.engine:
            thinking = self._engine_thinking(question, answer)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher if available
        if self.teacher and self.teacher.is_available():
            thinking = self._teacher_thinking(question, answer)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback: minimal structured thinking
        thinking = self._minimal_thinking(question, answer)
        return self._format_thinking_sample(sample, thinking)

    def _engine_thinking(self, question: str, answer: str) -> Optional[str]:
        """Generate thinking using ThinkingEngine NLP analysis."""
        try:
            # Combine question and answer for analysis
            text = f"{question} {answer}"
            context = {
                'question': question,
                'answer': answer,
                'type': 'qa_pair'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _teacher_thinking(self, question: str, answer: str) -> Optional[str]:
        """Generate thinking using Ollama teacher model."""
        max_tokens = self.depth_config['max_tokens']
        prompt = (
            f"Analiza esta pregunta y respuesta. Genera un razonamiento paso a paso\n"
            f"que lleve logicamente de la pregunta a la respuesta.\n\n"
            f"Pregunta: {question}\n"
            f"Respuesta correcta: {answer[:300]}\n\n"
            f"Razonamiento ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones, en espanol):"
        )
        return self.teacher.generate(prompt, max_tokens=max_tokens)

    def _minimal_thinking(self, question: str, answer: str) -> str:
        """Generate minimal thinking when no engine or teacher is available."""
        q_words = question.split()
        a_words = answer.split()

        # Detect question type
        q_lower = question.lower()
        if '?' in question:
            if any(w in q_lower for w in ['que', 'qué', 'what']):
                q_type = "una consulta de información"
            elif any(w in q_lower for w in ['como', 'cómo', 'how']):
                q_type = "una solicitud de procedimiento"
            elif any(w in q_lower for w in ['por que', 'por qué', 'why']):
                q_type = "una consulta causal"
            elif any(w in q_lower for w in ['donde', 'dónde', 'where']):
                q_type = "una consulta de ubicación"
            elif any(w in q_lower for w in ['cuando', 'cuándo', 'when']):
                q_type = "una consulta temporal"
            else:
                q_type = "una pregunta específica"
        else:
            q_type = "una declaración de información"

        # Find common words between question and answer
        q_set = set(w.lower() for w in q_words if len(w) > 3)
        a_set = set(w.lower() for w in a_words if len(w) > 3)
        common = q_set.intersection(a_set)

        # Build minimal reasoning
        result = [f"La consulta del usuario es {q_type}."]
        if common:
            result.append(f"Términos compartidos: {', '.join(list(common)[:3])}.")
        result.append(f"La respuesta proporciona información específica sobre el tema consultado.")

        return " ".join(result)
