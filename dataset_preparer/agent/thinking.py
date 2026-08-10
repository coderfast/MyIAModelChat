"""Agent thinking generator — produces tool_call + observation data for training."""

import json
import random
import logging
from typing import Dict, Any, Optional, List

from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher

logger = logging.getLogger(__name__)


class AgentThinkingGenerator(ThinkingGenerator):
    """Generate agentic thinking data with tool calls and observations.

    Produces training samples in the format:
        <|thinking|>reasoning<tool_call>{"name":"tool","arguments":{...}}</tool_call>
        <observation>result</observation></tool_call><|answer|>response

    For samples that don't need tools:
        <|thinking|>reasoning</thinking><|answer|>response
    """

    TOOL_CATEGORIES = {
        'math': {
            'tools': ['calculator'],
            'patterns': [
                (r'(\d+)\s*\*\s*(\d+)', 'calculator', lambda m: f"{m.group(1)} * {m.group(2)}"),
                (r'(\d+)\s*[\+]\s*(\d+)', 'calculator', lambda m: f"{m.group(1)} + {m.group(2)}"),
                (r'(\d+)\s*[\-]\s*(\d+)', 'calculator', lambda m: f"{m.group(1)} - {m.group(2)}"),
                (r'cuanto es (\d+)', 'calculator', lambda m: f"{m.group(1)}"),
                (r'how much is (\d+)', 'calculator', lambda m: f"{m.group(1)}"),
            ],
        },
        'date': {
            'tools': ['current_date'],
            'patterns': [
                (r'qu[eé] d[ií]a es|hoy|today|current date', 'current_date', lambda m: ''),
                (r'fecha actual|current time', 'current_date', lambda m: ''),
            ],
        },
        'files': {
            'tools': ['list_directory', 'read_file'],
            'patterns': [
                (r'qu[eé] hay en|list files|carpeta', 'list_directory', lambda m: '.'),
                (r'leer archivo|read file|contenido de', 'read_file', lambda m: ''),
            ],
        },
        'search': {
            'tools': ['web_search'],
            'patterns': [
                (r'busca|search|buscar|investiga', 'web_search', lambda m: ''),
            ],
        },
    }

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self._tool_call_count = 0
        self._no_tool_count = 0

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agentic thinking for a sample."""
        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('output', sample.get('answer', ''))

        if not question or not answer:
            return self._format_agent_sample(sample, '', with_tool=False)

        # Decide if this sample needs a tool (probabilistic)
        needs_tool, tool_name, arguments, simulated_result = self._analyze_tool_need(question, answer)

        if needs_tool and tool_name:
            thinking = self._generate_agent_thinking(question, answer, tool_name, arguments, simulated_result)
            return self._format_agent_sample(sample, thinking, with_tool=True,
                                             tool_name=tool_name, tool_args=arguments,
                                             observation=simulated_result)

        # No tool needed — generate normal thinking
        thinking = self._generate_normal_thinking(question, answer)
        return self._format_agent_sample(sample, thinking, with_tool=False)

    def _analyze_tool_need(self, question: str, answer: str) -> tuple:
        """Analyze if a question needs a tool. Returns (needs_tool, tool_name, args, result)."""
        q_lower = question.lower()

        for category, config in self.TOOL_CATEGORIES.items():
            for pattern, tool_name, args_fn in config['patterns']:
                import re
                if re.search(pattern, q_lower):
                    arguments = args_fn(re.search(pattern, q_lower))
                    simulated = self._simulate_tool_result(tool_name, arguments, answer)
                    return True, tool_name, arguments, simulated

        return False, None, None, None

    def _simulate_tool_result(self, tool_name: str, arguments: str, answer: str) -> str:
        """Simulate a realistic tool result based on the answer."""
        if tool_name == 'calculator':
            # Extract numbers from answer to simulate result
            import re
            nums = re.findall(r'[\d,]+\.?\d*', answer.replace(',', ''))
            if nums:
                return nums[0].replace(',', '')
            return answer[:50]
        elif tool_name == 'current_date':
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif tool_name == 'list_directory':
            return "file1.txt  file2.py  directory1/"
        elif tool_name == 'read_file':
            return answer[:200] if len(answer) > 200 else answer
        elif tool_name == 'web_search':
            return f"Result for '{arguments}': {answer[:200]}"
        return answer[:100]

    def _generate_agent_thinking(self, question: str, answer: str, tool_name: str,
                                  arguments: str, result: str) -> str:
        """Generate thinking that includes tool call reasoning."""
        thinking_parts = []

        # Analyze the question
        q_type = self._classify_question(question)
        thinking_parts.append(f"La consulta del usuario es {q_type}.")

        # Explain tool choice
        thinking_parts.append(f"Para responder correctamente, necesito usar la herramienta '{tool_name}'.")

        # Explain arguments
        if arguments:
            thinking_parts.append(f"Los parámetros para la herramienta son: {arguments}.")

        # Explain result processing
        thinking_parts.append(f"El resultado obtenido es: {result[:100]}.")
        thinking_parts.append(f"Proceso este resultado para dar una respuesta completa al usuario.")

        return " ".join(thinking_parts)

    def _generate_normal_thinking(self, question: str, answer: str) -> str:
        """Generate normal (non-agentic) thinking."""
        q_type = self._classify_question(question)
        thinking = f"La consulta del usuario es {q_type}. "
        thinking += "No necesito usar herramientas externas para responder. "
        thinking += f"La respuesta apropiada es: {answer[:150]}."
        return thinking

    def _classify_question(self, question: str) -> str:
        """Classify the question type."""
        q_lower = question.lower()
        if '?' in question:
            if any(w in q_lower for w in ['qué', 'que', 'what']):
                return "una consulta de información"
            elif any(w in q_lower for w in ['cómo', 'como', 'how']):
                return "una solicitud de procedimiento"
            elif any(w in q_lower for w in ['por qué', 'por que', 'why']):
                return "una consulta causal"
            elif any(w in q_lower for w in ['dónde', 'donde', 'where']):
                return "una consulta de ubicación"
            elif any(w in q_lower for w in ['cuándo', 'cuando', 'when']):
                return "una consulta temporal"
            elif any(w in q_lower for w in ['cuántos', 'cuantos', 'how many']):
                return "una consulta cuantitativa"
        return "una declaración de información"

    def _format_agent_sample(self, sample: Dict[str, Any], thinking: str,
                              with_tool: bool = False, tool_name: str = '',
                              tool_args: str = '', observation: str = '') -> Dict[str, Any]:
        """Format a sample with agentic thinking block for training."""
        result = dict(sample)
        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('output', sample.get('answer', sample.get('input_ids', '')))

        result['question'] = question
        result['answer'] = answer

        if with_tool and tool_name:
            # Build tool call JSON
            tool_call_obj = {"name": tool_name, "arguments": {}}
            if tool_args:
                # Parse arguments - could be expression, path, query, etc.
                if tool_name == 'calculator':
                    tool_call_obj["arguments"] = {"expression": tool_args}
                elif tool_name in ('read_file', 'list_directory'):
                    tool_call_obj["arguments"] = {"path": tool_args}
                elif tool_name == 'web_search':
                    tool_call_obj["arguments"] = {"query": tool_args}
                else:
                    tool_call_obj["arguments"] = {}

            tool_call_json = json.dumps(tool_call_obj, ensure_ascii=False)

            # Full agentic text
            agent_text = (
                f"<|thinking|>{question}"
                f"<thinking>{thinking}</thinking>"
                f"<tool_call>{tool_call_json}</tool_call>"
                f"<observation>{observation}</observation>"
                f"<|answer|>{answer}"
            )

            result['thinking'] = thinking
            result['has_tool_call'] = True
            result['tool_name'] = tool_name
            result['type'] = 'AGENT'
            result['thinking_text'] = agent_text
            result['input_ids'] = agent_text
            result['original_text'] = answer
            self._tool_call_count += 1
        else:
            # Normal thinking sample
            normal_text = f"<|thinking|>{question}<thinking>{thinking}</thinking><|answer|>{answer}"
            result['thinking'] = thinking
            result['has_tool_call'] = False
            result['tool_name'] = ''
            result['type'] = 'THINKING'
            result['thinking_text'] = normal_text
            result['input_ids'] = normal_text
            result['original_text'] = answer
            self._no_tool_count += 1

        return result

    def get_stats(self) -> Dict[str, int]:
        """Return generation statistics."""
        return {
            'tool_call_samples': self._tool_call_count,
            'normal_samples': self._no_tool_count,
            'total': self._tool_call_count + self._no_tool_count,
        }
