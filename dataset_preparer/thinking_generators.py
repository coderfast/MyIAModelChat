"""
Base thinking generator and Ollama teacher model integration.
Each source has its own generator that produces real chain-of-thought reasoning.
Now includes ThinkingEngine for NLP-based analysis without external LLM dependency.
"""
import re
import json
import logging
import urllib.request
import urllib.error
from collections import OrderedDict
from typing import Optional, Dict, Any
from config import OLLAMA_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)


class OllamaTeacher:
    """Teacher model via Ollama API for generating thinking."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip('/')
        self.cache: OrderedDict = OrderedDict()
        self._cache_maxsize = 1000
        self._available: Optional[bool] = None
        self._model_valid: Optional[bool] = None

    def is_model_available(self) -> bool:
        """Check if Ollama is running and the specific model is available (single request)."""
        if self._model_valid is not None:
            return self._model_valid
        try:
            req = urllib.request.Request(f'{self.url}/api/tags', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                self._available = True
                available_models = [m.get('name', '') for m in data.get('models', [])]
                self._model_valid = self.model in available_models
                if not self._model_valid:
                    logger.warning("MODEL '%s' NOT FOUND IN OLLAMA. AVAILABLE: %s", self.model, available_models)
                    logger.info("SWITCHING AUTOMATICALLY TO NLP MODE")
                return self._model_valid
        except Exception:
            self._available = False
            self._model_valid = False
            logger.warning("OLLAMA SERVER NOT REACHABLE AT %s", self.url)
            logger.info("SWITCHING AUTOMATICALLY TO NLP MODE")
            return False

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        return self.is_model_available()

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> Optional[str]:
        cache_key = (prompt, max_tokens, temperature)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.is_model_available():
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
                self.cache[cache_key] = result
                # LRU eviction
                if len(self.cache) > self._cache_maxsize:
                    self.cache.popitem(last=False)
                return result
        except Exception as e:
            logger.debug(f"Ollama generation failed: {e}")
            return None


class ThinkingGenerator:
    """
    Base class for source-specific thinking generators.
    Now supports ThinkingEngine for NLP-based analysis.
    """

    DEPTH_CONFIG = {
        'basic': {'max_tokens': 80, 'min_sentences': 1, 'max_sentences': 2},
        'adaptive': {'max_tokens': 120, 'min_sentences': 2, 'max_sentences': 3},
        'detailed': {'max_tokens': 200, 'min_sentences': 3, 'max_sentences': 5},
    }

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        self.teacher = teacher
        self.depth = depth
        self.depth_config = self.DEPTH_CONFIG.get(depth, self.DEPTH_CONFIG['adaptive'])

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thinking for a sample. Returns sample with 'thinking' key added."""
        raise NotImplementedError

    def _format_thinking_sample(self, sample: Dict[str, Any], thinking: str) -> Dict[str, Any]:
        """Format a sample with thinking block for training.

        GPT-2 standard format (Formato 3, no <|user|>/<|assistant|>):
        - THINKING: <|problem|>question<|thinking|>reasoning<|final|>answer
        - TEXT: <|problem|>question<|final|>answer
        """
        result = dict(sample)

        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('output', sample.get('answer', sample.get('input_ids', '')))

        result['question'] = question
        result['answer'] = answer

        if thinking and answer:
            result['thinking'] = thinking
            result['type'] = 'THINKING'
            thinking_text = f"<|problem|>{question}<|thinking|>{thinking}<|final|>{answer}"
            result['thinking_text'] = thinking_text
            result['input_ids'] = thinking_text
            result['original_text'] = answer
        else:
            result['thinking'] = ''
            result['type'] = 'TEXT'
            context_text = f"<|problem|>{question}<|final|>{answer}"
            result['input_ids'] = context_text
            result['original_text'] = answer

        return result
