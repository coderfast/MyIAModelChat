"""AIML agentic thinking generator — wraps AgentThinkingGenerator for AIML sources."""

from typing import Dict, Any, Optional
from dataset_preparer.agent.thinking import AgentThinkingGenerator
from dataset_preparer.thinking_generators import OllamaTeacher


class AIMLAgentThinkingGenerator(AgentThinkingGenerator):
    """Generate agentic thinking for AIML patterns.

    AIML data has pattern/template pairs that may need:
    - calculator for math-related patterns
    - web_search for information-seeking patterns
    """

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)
