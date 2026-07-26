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
        self._model_valid: Optional[bool] = None

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

    def is_model_available(self) -> bool:
        """Check if the specific model is available in Ollama."""
        if self._model_valid is not None:
            return self._model_valid
        if not self.is_available():
            self._model_valid = False
            return False
        try:
            req = urllib.request.Request(f'{self.url}/api/tags', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                available_models = [m.get('name', '') for m in data.get('models', [])]
                self._model_valid = self.model in available_models
                if not self._model_valid:
                    logger.warning(f"Model '{self.model}' not found in Ollama. Available: {available_models}")
                return self._model_valid
        except Exception:
            self._model_valid = False
            return False

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
