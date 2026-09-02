"""Agent thinking generator — produces tool_call + observation data for training."""

import re
import random
import json
import os
import logging
from typing import Dict, Any, Optional, List

from dataset_preparer.thinking_generators import ThinkingGenerator, OllamaTeacher

logger = logging.getLogger(__name__)

TOOLS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tools_config.json')


def _load_agent_patterns() -> dict:
    """Load agent patterns from tools_config.json."""
    if not os.path.exists(TOOLS_CONFIG_PATH):
        logger.warning(f"Tools config not found: {TOOLS_CONFIG_PATH}, using hardcoded patterns")
        return {}
    try:
        with open(TOOLS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('agent_patterns', {})
    except Exception as e:
        logger.error(f"Failed to load agent patterns: {e}")
        return {}


def _compile_patterns(patterns_data: dict) -> dict:
    """Compile JSON patterns into the format expected by _analyze_tool_need."""
    compiled = {}
    for category, config in patterns_data.items():
        compiled_patterns = []
        default_tool = config['tools'][0] if config.get('tools') else None
        
        # Compile regex patterns from JSON
        for p in config.get('patterns', []):
            regex = p['regex']
            args_groups = p.get('args_group', [])
            tool_for_pattern = p.get('tool', default_tool)
            
            def make_args_fn(groups):
                def args_fn(match):
                    parts = []
                    for g in groups:
                        if isinstance(g, int):
                            parts.append(match.group(g) if match.lastindex and g <= match.lastindex else '')
                        else:
                            parts.append(g)
                    return ''.join(parts).strip()
                return args_fn
            
            compiled_patterns.append((re.compile(regex, re.IGNORECASE), tool_for_pattern, make_args_fn(args_groups)))
        
        # Add keyword-based patterns
        all_keywords = config.get('keywords_es', []) + config.get('keywords_en', [])
        if all_keywords:
            keyword_pattern = '|'.join(re.escape(kw) for kw in all_keywords)
            tool_name = config['tools'][0] if config.get('tools') else None
            if tool_name:
                compiled_patterns.append((re.compile(keyword_pattern, re.IGNORECASE), tool_name, lambda m: ''))
        
        compiled[category] = {
            'tools': config.get('tools', []),
            'patterns': compiled_patterns
        }
    
    return compiled


class AgentThinkingGenerator(ThinkingGenerator):
    """Generate agentic thinking data with tool calls and observations.

    GPT-2 standard format (Formato 2):
        <|user|>{question}<|end|>
        <|assistant|><tool_call>{tool_name}({args})</tool_call><|tool_result|>{result}<|end|>
        <|assistant|>{response}<|end|>

    For samples that don't need tools (Formato 3):
        <|problem|>{question}<|thinking|>{reasoning}<|final|>{response}
    """

    # Default hardcoded patterns (fallback if JSON not available)
    DEFAULT_TOOL_CATEGORIES = {
        'math': {
            'tools': ['calculator'],
            'patterns': [
                (re.compile(r'(\d+)\s*\*\s*(\d+)', re.IGNORECASE), 'calculator', lambda m: f"{m.group(1)} * {m.group(2)}"),
                (re.compile(r'(\d+)\s*[\+]\s*(\d+)', re.IGNORECASE), 'calculator', lambda m: f"{m.group(1)} + {m.group(2)}"),
                (re.compile(r'(\d+)\s*[\-]\s*(\d+)', re.IGNORECASE), 'calculator', lambda m: f"{m.group(1)} - {m.group(2)}"),
                (re.compile(r'cuanto es (\d+)', re.IGNORECASE), 'calculator', lambda m: f"{m.group(1)}"),
                (re.compile(r'how much is (\d+)', re.IGNORECASE), 'calculator', lambda m: f"{m.group(1)}"),
            ],
        },
        'date': {
            'tools': ['current_date'],
            'patterns': [
                (re.compile(r'qu[eé] d[ií]a es|hoy|today|current date', re.IGNORECASE), 'current_date', lambda m: ''),
                (re.compile(r'fecha actual|current time', re.IGNORECASE), 'current_date', lambda m: ''),
            ],
        },
        'files': {
            'tools': ['list_directory', 'read_file'],
            'patterns': [
                (re.compile(r'qu[eé] hay en|list files|carpeta', re.IGNORECASE), 'list_directory', lambda m: '.'),
                (re.compile(r'leer archivo|read file|contenido de', re.IGNORECASE), 'read_file', lambda m: ''),
            ],
        },
        'search': {
            'tools': ['web_search'],
            'patterns': [
                (re.compile(r'busca|search|buscar|investiga', re.IGNORECASE), 'web_search', lambda m: ''),
            ],
        },
        'shell': {
            'tools': ['shell'],
            'patterns': [
                (re.compile(r'ejecuta|run command|execute|corre comando', re.IGNORECASE), 'shell', lambda m: ''),
            ],
        },
    }

    def __init__(self, teacher: Optional[OllamaTeacher] = None, depth: str = 'adaptive'):
        super().__init__(teacher, depth)
        self._tool_call_count = 0
        self._no_tool_count = 0
        
        # Load patterns from JSON or use defaults
        patterns_data = _load_agent_patterns()
        if patterns_data:
            self.TOOL_CATEGORIES = _compile_patterns(patterns_data)
            logger.info(f"Loaded agent patterns from JSON ({len(self.TOOL_CATEGORIES)} categories)")
        else:
            self.TOOL_CATEGORIES = self.DEFAULT_TOOL_CATEGORIES
            logger.info("Using default hardcoded agent patterns")

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
                if pattern.search(q_lower):
                    match = pattern.search(q_lower)
                    arguments = args_fn(match)
                    simulated = self._simulate_tool_result(tool_name, arguments, answer)
                    return True, tool_name, arguments, simulated

        return False, None, None, None

    def _simulate_tool_result(self, tool_name: str, arguments: str, answer: str) -> str:
        """Simulate a realistic tool result based on the answer."""
        if tool_name == 'calculator':
            import re as _re
            # Try to evaluate the expression if possible
            expr = arguments.strip('()')
            try:
                import ast
                result = eval(compile(ast.parse(expr, mode='eval'), '<expr>', 'eval'))
                return str(result)
            except Exception:
                pass
            # Fallback: extract numbers from answer
            nums = _re.findall(r'[\d,]+\.?\d*', answer.replace(',', ''))
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
        from commons.language_utils import detect_language
        lang = detect_language(question)
        is_es = lang == 'es'

        thinking_parts = []

        # Analyze the question
        q_type = self._classify_question(question)
        if is_es:
            thinking_parts.append(f"La consulta del usuario es {q_type}.")
            thinking_parts.append(f"Para responder correctamente, necesito usar la herramienta '{tool_name}'.")
            if arguments:
                thinking_parts.append(f"Los parámetros para la herramienta son: {arguments}.")
            thinking_parts.append(f"El resultado obtenido es: {result[:100]}.")
            thinking_parts.append(f"Proceso este resultado para dar una respuesta completa al usuario.")
        else:
            thinking_parts.append(f"The user's query is {q_type}.")
            thinking_parts.append(f"To respond correctly, I need to use the tool '{tool_name}'.")
            if arguments:
                thinking_parts.append(f"The parameters for the tool are: {arguments}.")
            thinking_parts.append(f"The result obtained is: {result[:100]}.")
            thinking_parts.append(f"I process this result to give a complete response to the user.")

        return " ".join(thinking_parts)

    def _generate_normal_thinking(self, question: str, answer: str) -> str:
        """Generate normal (non-agentic) thinking."""
        from commons.language_utils import detect_language
        lang = detect_language(question)
        is_es = lang == 'es'

        q_type = self._classify_question(question)
        if is_es:
            thinking = f"La consulta del usuario es {q_type}. "
            thinking += "No necesito usar herramientas externas para responder. "
            thinking += f"La respuesta apropiada es: {answer[:150]}."
        else:
            thinking = f"The user's query is {q_type}. "
            thinking += "I don't need external tools to respond. "
            thinking += f"The appropriate answer is: {answer[:150]}."
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

        lang = sample.get('language', 'unknown')
        lang_token = f'<|{lang}|>' if lang and lang not in ('unknown', 'Unknown', '') else ''

        result['question'] = question
        result['answer'] = answer

        if with_tool and tool_name:
            # Build tool call string in function notation: tool_name(args)
            args_str = f"({tool_args})" if tool_args else "()"
            tool_call_str = f"{tool_name}{args_str}"

            # GPT-2 standard agentic format (Formato 2) with thinking included
            agent_text = (
                f"{lang_token}<|user|>{question}<|end|>"
                f"<|assistant|><tool_call>{tool_call_str}</tool_call><|tool_result|>{observation}<|end|>"
                f"<|assistant|><|thinking|>{thinking}<|final|>{answer}<|end|>"
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
            # GPT-2 standard thinking format (Formato 3, no tool)
            # NOTE: <|lang|> is NOT added here — _create_bpe_text_column adds it
            normal_text = f"<|problem|>{question}<|thinking|>{thinking}<|final|>{answer}"
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
