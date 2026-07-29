"""
HuggingFace thinking generator: NLP-based analysis for QA datasets and text.
Uses ThinkingEngine for real reasoning instead of rule-based meta-commentary.
"""
from typing import Dict, Any, Optional
from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher
from dataset_preparer.thinking_engine import ThinkingEngine


class HFThinkingGenerator(ThinkingGenerator):
    """Generate thinking for HuggingFace dataset samples using NLP analysis."""

    def __init__(self, engine: Optional[ThinkingEngine] = None,
                 teacher: Optional[OllamaTeacher] = None,
                 depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self.engine = engine or ThinkingEngine(depth=depth)

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
        """Generate thinking for Q&A samples."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')

        # Try ThinkingEngine first
        if self.engine:
            thinking = self._engine_qa_thinking(question, answer)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher
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

        # Fallback
        thinking = self._minimal_qa_thinking(question, answer)
        return self._format_thinking_sample(sample, thinking)

    def _generate_context_thinking(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thinking for context-based Q&A samples."""
        context = sample.get('context', '')
        question = sample.get('question', sample.get('answer', ''))

        # Try ThinkingEngine first
        if self.engine:
            thinking = self._engine_context_thinking(context, question)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher
        if self.teacher and self.teacher.is_available():
            max_tokens = self.depth_config['max_tokens']
            prompt = (
                f"Contexto: {context[:400]}\n"
                f"Pregunta/Respuesta: {question[:200]}\n\n"
                f"Razona sobre como el contexto respuesta la pregunta ({self.depth_config['min_sentences']}-{self.depth_config['max_sentences']} oraciones):\n"
                f"Razonamiento:"
            )
            thinking = self.teacher.generate(prompt, max_max=max_tokens)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Fallback
        thinking = self._minimal_context_thinking(context, question)
        return self._format_thinking_sample(sample, thinking)

    def _generate_text_thinking(self, sample: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Generate thinking for plain text samples."""
        text_str = str(text)[:600]

        # Try ThinkingEngine first
        if self.engine:
            thinking = self._engine_text_thinking(text_str)
            if thinking:
                return self._format_thinking_sample(sample, thinking)

        # Try Ollama teacher
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

        # Fallback
        thinking = self._minimal_text_thinking(text_str)
        return self._format_thinking_sample(sample, thinking)

    def _engine_qa_thinking(self, question: str, answer: str) -> Optional[str]:
        """Generate Q&A thinking using ThinkingEngine."""
        try:
            text = f"{question} {answer}"
            context = {
                'question': question,
                'answer': answer,
                'type': 'huggingface_qa'
            }
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _engine_context_thinking(self, context: str, question: str) -> Optional[str]:
        """Generate context thinking using ThinkingEngine."""
        try:
            text = f"{context} {question}"
            ctx = {
                'question': question,
                'type': 'huggingface_context_qa'
            }
            return self.engine.generate_thinking(text, ctx)
        except Exception:
            return None

    def _engine_text_thinking(self, text: str) -> Optional[str]:
        """Generate text thinking using ThinkingEngine."""
        try:
            context = {'type': 'huggingface_text'}
            return self.engine.generate_thinking(text, context)
        except Exception:
            return None

    def _minimal_qa_thinking(self, question: str, answer: str) -> str:
        """Generate minimal Q&A thinking."""
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        common = q_words.intersection(a_words)

        result = [f"La consulta solicita información específica."]
        if common:
            result.append(f"Conceptos clave compartidos: {', '.join(list(common)[:3])}.")
        result.append(f"La respuesta proporciona la información solicitada de forma directa.")
        return " ".join(result)

    def _minimal_context_thinking(self, context: str, question: str) -> str:
        """Generate minimal context thinking."""
        word_count = len(context.split())
        return (
            f"El contexto proporcionado contiene {word_count} palabras. "
            f"La información del contexto responde directamente a la consulta realizada."
        )

    def _minimal_text_thinking(self, text: str) -> str:
        """Generate minimal text thinking."""
        word_count = len(text.split())
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        result = [f"El texto contiene {word_count} palabras en {len(sentences)} oraciones."]
        result.append("El contenido presenta información relevante sobre el tema.")
        return " ".join(result)
