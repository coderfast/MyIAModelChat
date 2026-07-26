"""
Base thinking generator and Ollama teacher model integration.
Each source has its own generator that produces real chain-of-thought reasoning.
"""
import re
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OllamaTeacher:
    """Teacher model via Ollama API for generating thinking."""

    def __init__(self, model: str = 'qwen2.5:1.5b', url: str = 'http://localhost:11434'):
        self.model = model
        self.url = url.rstrip('/')
        self.cache: Dict[str, str] = {}
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f'{self.url}/api/tags', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> Optional[str]:
        if prompt in self.cache:
            return self.cache[prompt]

        if not self.is_available():
            return None

        try:
            payload = json.dumps({
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            }).encode('utf-8')

            req = urllib.request.Request(
                f'{self.url}/api/generate',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                result = data.get('response', '').strip()
                self.cache[prompt] = result
                return result
        except Exception as e:
            logger.debug(f"Ollama generation failed: {e}")
            return None


class ThinkingGenerator:
    """Base class for source-specific thinking generators."""

    def __init__(self, teacher: Optional[OllamaTeacher] = None):
        self.teacher = teacher

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thinking for a sample. Returns sample with 'thinking' key added."""
        raise NotImplementedError

    def validate_thinking(self, thinking: str, answer: str, question: str = '') -> bool:
        """Validate that thinking is real reasoning, not meta-commentary."""
        if not thinking or len(thinking) < 15:
            return False

        meta_patterns = [
            re.compile(r'^(el usuario|the user|el humano|the human)\s*(me\s+)?(saluda|despide|pregunta|pide)', re.IGNORECASE),
            re.compile(r'^(saludo|greeting|despedida|farewell)$', re.IGNORECASE),
            re.compile(r'^(respondo|i respond|contestando|answering)\s*(con|with)', re.IGNORECASE),
        ]
        is_meta = False
        for pattern in meta_patterns:
            if pattern.match(thinking.strip()):
                is_meta = True
                break

        answer_words = set(answer.lower().split()[:5])
        thinking_words = set(thinking.lower().split())
        has_answer_derivation = bool(answer_words and answer_words.intersection(thinking_words))

        step_indicators = [
            'porque', 'por lo tanto', 'primero', 'paso', 'análisis',
            'entonces', 'sin embargo', 'además', 'en cambio', 'consiste',
            'because', 'therefore', 'first', 'step', 'analysis',
            'however', 'additionally', 'furthermore', 'consists',
        ]
        has_steps = any(ind in thinking.lower() for ind in step_indicators)
        has_length = len(thinking.split()) >= 8

        if is_meta and not has_steps and not has_answer_derivation:
            return False

        score = 0.0
        if len(thinking) >= 30:
            score += 0.4
        elif len(thinking) >= 15:
            score += 0.2
        if has_steps:
            score += 0.3
        if has_answer_derivation:
            score += 0.3
        if has_length:
            score += 0.2

        return score >= 0.5

    def _format_thinking_sample(self, sample: Dict[str, Any], thinking: str) -> Dict[str, Any]:
        """Format a sample with thinking block, updating input_ids for training."""
        result = dict(sample)
        result['thinking'] = thinking

        answer = sample.get('output', sample.get('input_ids', ''))
        if thinking and answer:
            thinking_text = f"<think>{thinking}</think>{answer}"
            result['thinking_text'] = thinking_text
            result['input_ids'] = thinking_text
        elif answer:
            result['input_ids'] = answer

        return result
