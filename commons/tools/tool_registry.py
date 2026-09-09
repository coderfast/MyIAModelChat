"""Centralized tool registry with platform detection."""

import platform
import json
import os
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)

TOOLS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tools_config.json')


@dataclass
class ToolDef:
    """Definition of a callable tool."""
    name: str
    description: str
    parameters: dict  # JSON Schema-like parameter definitions
    func: Callable
    category: str  # 'search', 'computation', 'file', 'shell', 'api'
    platform: Optional[str] = None  # None = cross-platform, 'windows'/'linux'/'darwin'
    requires_permission: bool = False  # True for shell/file write operations


class ToolRegistry:
    """Registry of available tools with platform-aware filtering."""

    def __init__(self):
        self.tools: dict[str, ToolDef] = {}
        self.current_platform = platform.system().lower()  # 'windows', 'linux', 'darwin'

    def register(self, tool: ToolDef):
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name} ({tool.category})")

    def get(self, name: str) -> Optional[ToolDef]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        """List all registered tools."""
        return list(self.tools.values())

    def list_platform_tools(self) -> list[ToolDef]:
        """List tools compatible with current platform."""
        result = []
        for tool in self.tools.values():
            if tool.platform is None or tool.platform == self.current_platform:
                result.append(tool)
        return result

    def get_descriptions(self, include_platform: bool = False) -> str:
        """Get formatted tool descriptions for prompt injection."""
        lines = []
        for tool in self.list_platform_tools():
            params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in tool.parameters.items())
            platform_info = f" [{tool.platform}]" if tool.platform else ""
            lines.append(f"- {tool.name}{platform_info}: {tool.description}({params})")
        return "\n".join(lines)

    def call(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with arguments."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'. Available: {', '.join(self.tools.keys())}"
        try:
            result = tool.func(**arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return f"Error executing {name}: {str(e)}"


def _calculator(expression: str) -> str:
    """Safe math evaluation."""
    import ast
    import operator
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    tree = ast.parse(expression, mode='eval')
    result = _eval(tree)
    return str(result)


def _current_date() -> str:
    """Get current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _word_count(text: str) -> str:
    """Count words in text."""
    return str(len(text.split()))


def _get_platform() -> str:
    """Get current platform."""
    return platform.system().lower()


def _read_file(path: str) -> str:
    """Read file content."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def _list_directory(path: str = ".") -> str:
    """List directory contents."""
    try:
        entries = []
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            prefix = "[DIR] " if os.path.isdir(full) else "      "
            entries.append(f"{prefix}{entry}")
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"


def _web_search(query: str) -> str:
    """Web search stub — returns placeholder."""
    return f"Search results for '{query}': [web_search requires trafilatura integration]"


def _shell(command: str) -> str:
    """Execute a shell command (stub for training data generation)."""
    return f"[shell output for: {command}]"


# Built-in tool functions mapping
TOOL_FUNCTIONS = {
    'calculator': _calculator,
    'current_date': _current_date,
    'word_count': _word_count,
    'get_platform': _get_platform,
    'read_file': _read_file,
    'list_directory': _list_directory,
    'web_search': _web_search,
    'shell': _shell,
}


def load_tools_from_json(config_path: str = None) -> dict:
    """Load tool definitions from JSON config file."""
    if config_path is None:
        config_path = TOOLS_CONFIG_PATH
    
    if not os.path.exists(config_path):
        logger.warning(f"Tools config not found: {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded tools config from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load tools config: {e}")
        return {}


def register_default_tools(registry: ToolRegistry):
    """Register tools from JSON config, falling back to hardcoded defaults."""
    config = load_tools_from_json()
    
    if config and 'tools' in config:
        # Load from JSON
        for tool_name, tool_def in config['tools'].items():
            if not tool_def.get('enabled', True):
                continue
            
            func = TOOL_FUNCTIONS.get(tool_name)
            if func is None:
                logger.warning(f"No function implementation for tool: {tool_name}")
                continue
            
            registry.register(ToolDef(
                name=tool_name,
                description=tool_def['description'],
                parameters=tool_def.get('parameters', {}),
                func=func,
                category=tool_def.get('category', 'general'),
                platform=tool_def.get('platform'),
                requires_permission=tool_def.get('requires_permission', False),
            ))
        logger.info(f"Registered {len(registry.tools)} tools from JSON config")
    else:
        # Fallback to hardcoded defaults
        registry.register(ToolDef(
            name="calculator",
            description="Evaluate math expressions safely",
            parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
            func=_calculator,
            category="computation",
        ))
        registry.register(ToolDef(
            name="current_date",
            description="Get current date and time",
            parameters={},
            func=_current_date,
            category="computation",
        ))
        registry.register(ToolDef(
            name="word_count",
            description="Count words in text",
            parameters={"text": {"type": "string", "description": "Text to count words in"}},
            func=_word_count,
            category="computation",
        ))
        registry.register(ToolDef(
            name="get_platform",
            description="Detect current operating system",
            parameters={},
            func=_get_platform,
            category="computation",
        ))
        registry.register(ToolDef(
            name="read_file",
            description="Read content of a file",
            parameters={"path": {"type": "string", "description": "File path to read"}},
            func=_read_file,
            category="file",
            requires_permission=True,
        ))
        registry.register(ToolDef(
            name="list_directory",
            description="List files in a directory",
            parameters={"path": {"type": "string", "description": "Directory path (default: current)"}},
            func=_list_directory,
            category="file",
            requires_permission=True,
        ))
        registry.register(ToolDef(
            name="web_search",
            description="Search the web for information",
            parameters={"query": {"type": "string", "description": "Search query"}},
            func=_web_search,
            category="search",
        ))
        logger.info("Registered 7 default tools (hardcoded fallback)")
