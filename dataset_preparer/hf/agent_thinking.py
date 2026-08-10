"""HuggingFace agentic thinking generator — wraps AgentThinkingGenerator for HF sources."""

from typing import Dict, Any, Optional
from dataset_preparer.agent.thinking import AgentThinkingGenerator
from dataset_preparer.thinking_generators import OllamaTeacher


class HFAgentThinkingGenerator(AgentThinkingGenerator):
    """Generate agentic thinking for HuggingFace datasets.

    HF data can be QA, text generation, or other formats that may need:
    - calculator for numeric QA
    - web_search for factual QA
    - current_date for temporal questions
    """

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)
