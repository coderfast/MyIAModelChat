"""CSV agentic thinking generator — wraps AgentThinkingGenerator for CSV sources."""

from typing import Dict, Any, Optional
from dataset_preparer.agent.thinking import AgentThinkingGenerator
from dataset_preparer.thinking_generators import OllamaTeacher


class CSVAgentThinkingGenerator(AgentThinkingGenerator):
    """Generate agentic thinking for CSV QA pairs.

    CSV data typically has question/answer pairs that may need:
    - calculator for math questions
    - current_date for date questions
    - web_search for factual questions
    """

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)
