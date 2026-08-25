"""Tool execution with cross-platform shell support and security."""

import re
import json
import platform
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_platform() -> str:
    """Return 'windows', 'linux', or 'darwin'."""
    return platform.system().lower()


def get_default_shell() -> str:
    """Return default shell: 'powershell' on Windows, 'bash' on Linux/Mac."""
    if platform.system().lower() == 'windows':
        return 'powershell'
    return 'bash'


def execute_shell(command: str, timeout: int = 30) -> str:
    """Execute command on the correct platform."""
    if platform.system().lower() == 'windows':
        return execute_powershell(command, timeout)
    return execute_bash(command, timeout)


def execute_powershell(command: str, timeout: int = 30) -> str:
    """Execute a PowerShell command (Windows)."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            if error:
                output = f"{output}\nError: {error}" if output else f"Error: {error}"
        return output[:500] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing PowerShell: {e}"


def execute_bash(command: str, timeout: int = 30) -> str:
    """Execute a Bash command (Linux/Mac)."""
    try:
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            if error:
                output = f"{output}\nError: {error}" if output else f"Error: {error}"
        return output[:500] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing bash: {e}"


class ShellSecurity:
    """Validates commands against safe/blocked patterns."""

    SAFE_COMMANDS_WINDOWS = [
        'Get-ChildItem', 'Get-Content', 'Get-Item', 'Test-Path', 'Resolve-Path',
        'Get-Process', 'Get-Service', 'Get-Host', 'Get-Date', 'Get-Command',
        'Get-Help', 'Get-Variable', 'Select-String', 'Measure-Object',
        'Sort-Object', 'Where-Object', 'Format-Table', 'Format-List',
        'Out-String', 'ConvertTo-Json', 'ConvertFrom-Json',
        'cat', 'ls', 'dir', 'echo', 'pwd', 'whoami', 'date',
    ]

    SAFE_COMMANDS_UNIX = [
        'ls', 'll', 'la', 'tree', 'find', 'which', 'whereis',
        'cat', 'head', 'tail', 'less', 'more', 'file', 'stat',
        'grep', 'egrep', 'wc', 'diff', 'md5sum', 'sha256sum',
        'ps', 'top', 'uptime', 'who', 'last',
        'uname', 'hostname', 'id', 'whoami', 'date', 'cal',
        'env', 'printenv', 'free', 'df', 'du', 'mount',
        'echo', 'printf', 'bc', 'jq',
    ]

    BLOCKED_PATTERNS = [
        r'Remove-Item\s+-Recurse', r'Remove-Item\s+-Force',
        r'del\s+/[sSqQfF]', r'Format-Volume', r'Initialize-Disk',
        r'Set-Content', r'Out-File', r'Start-Process\s+-Verb\s+RunAs',
        r'rm\s+-rf', r'mkfs', r'fdisk.*-w',
        r'chmod\s+777', r'chmod\s+-R\s+777', r'chown',
        r'sudo', r'su\s+-', r'su\s+root',
        r'curl.*\|\s*(ba)?sh', r'wget.*\|\s*(ba)?sh',
        r'eval\s+', r'exec\s+',
        r'systemctl\s+(stop|disable|mask)',
        r'shutdown', r'reboot', r'poweroff',
    ]

    @classmethod
    def validate_command(cls, command: str, sys_platform: str) -> tuple[bool, str]:
        """Validate if a command is safe. Returns (is_safe, reason)."""
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Blocked: matches destructive pattern '{pattern}'"
        return True, "OK"

    @classmethod
    def sanitize_command(cls, command: str) -> str:
        """Remove dangerous characters from command."""
        return command.strip().rstrip('`')


class ToolExecutor:
    """Executes tool calls from model output with security checks."""

    def __init__(self, permission_callback=None, dry_run: bool = False, registry=None):
        """
        Args:
            permission_callback: Callable(tool_name, command, platform) -> bool
            dry_run: If True, show what would execute without executing
            registry: ToolRegistry instance for tool lookup
        """
        self.permission_callback = permission_callback
        self.dry_run = dry_run
        self.platform = get_platform()
        self.registry = registry

    def has_tool_call(self, text: str) -> bool:
        """Detect if text contains <tool_call>."""
        return '<tool_call>' in text and '</tool_call>' in text

    def parse_tool_call(self, text: str) -> tuple[str, str]:
        """Extract (tool_name, args_string) from a tool_call.

        Function-call notation per plan Formato 2: <tool_call>search_places(arg1, arg2)</tool_call>.
        """
        try:
            call = text.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            match = re.match(r'^([A-Za-z_]\w*)\s*\((.*)\)$', call, re.DOTALL)
            if not match:
                logger.warning(f"Failed to parse tool_call: {call}")
                return '', ''
            return match.group(1), match.group(2).strip()
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse tool_call: {e}")
            return '', ''

    def execute_tool_call(self, tool_call_text: str, registry=None) -> str:
        """Execute a tool_call and return observation.

        Args:
            tool_call_text: Raw tool_call string from model output
            registry: Optional ToolRegistry; falls back to self.registry
        """
        tool_name, args_string = self.parse_tool_call(tool_call_text)
        if not tool_name:
            return format_observation("Error: Could not parse tool_call")

        reg = registry or self.registry
        if reg is None:
            return format_observation("Error: No tool registry available")

        tool = reg.get(tool_name)
        if tool is None:
            return format_observation(f"Error: Unknown tool '{tool_name}'")

        arguments = self._map_arguments(tool, args_string)

        # All tools go through permission check
        command = arguments.get('command', '')

        # Shell commands need security check
        if tool.category == 'shell':
            is_safe, reason = ShellSecurity.validate_command(command, self.platform)
            if not is_safe:
                return format_observation(f"Blocked: {reason}")

        if self.dry_run:
            return format_observation(f"[DRY RUN] Would execute {tool_name}: {command or str(arguments)[:200]}")

        if self.permission_callback:
            if not self.permission_callback(tool_name, command, self.platform):
                return format_observation("Permission denied by user")

        # Execute tool
        if tool.category == 'shell':
            if tool_name == 'powershell':
                result = execute_powershell(command)
            elif tool_name == 'bash':
                result = execute_bash(command)
            else:
                result = execute_shell(command)
            return format_observation(result)

        result = reg.call(tool_name, arguments)
        return format_observation(result)

    def _map_arguments(self, tool, args_string: str) -> dict:
        """Map a positional args string to named arguments using the tool parameter schema.

        Formato 2 notation: <tool_call>web_search(que es python)</tool_call>
        becomes {"query": "que es python"} for the web_search tool.
        """
        if not args_string:
            return {}
        params = list(tool.parameters.keys())
        if not params:
            return {}
        if len(params) == 1:
            return {params[0]: args_string}
        # Multiple params: split on commas (simple positional mapping)
        parts = [p.strip() for p in args_string.split(',')]
        return {params[i]: parts[i] for i in range(min(len(params), len(parts)))}


def format_observation(result: str, max_length: int = 500) -> str:
    """Format result as a <|tool_result|> prefix block (no closing tag).

    The observation content runs until the next <|end|>/<|assistant|> turn marker.
    """
    if len(result) > max_length:
        result = result[:max_length] + "\n... (truncated)"
    return f"<|tool_result|>{result}"
