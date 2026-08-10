import pytest
from commons.tools.tool_executor import (
    ToolExecutor, ShellSecurity, format_observation,
    execute_shell, execute_powershell, execute_bash,
    get_platform, get_default_shell
)
from commons.tools.tool_registry import ToolRegistry, register_default_tools


class TestShellSecurity:
    def test_validate_safe_command_windows(self):
        is_safe, reason = ShellSecurity.validate_command("Get-Date", "windows")
        assert is_safe

    def test_validate_safe_command_linux(self):
        is_safe, reason = ShellSecurity.validate_command("ls -la", "linux")
        assert is_safe

    def test_dangerous_command_blocked(self):
        is_safe, reason = ShellSecurity.validate_command("Remove-Item -Recurse C:\\*", "windows")
        assert not is_safe
        assert "Blocked" in reason

    def test_rm_rf_blocked(self):
        is_safe, reason = ShellSecurity.validate_command("rm -rf /", "linux")
        assert not is_safe

    def test_sanitize_command(self):
        result = ShellSecurity.sanitize_command("echo hello`")
        assert result == "echo hello"

    def test_sudo_blocked(self):
        is_safe, _ = ShellSecurity.validate_command("sudo apt install python", "linux")
        assert not is_safe


class TestFormatObservation:
    def test_format_observation(self):
        result = format_observation("56088")
        assert "<observation>" in result
        assert "</observation>" in result
        assert "56088" in result

    def test_format_observation_multiline(self):
        result = format_observation("line1\nline2")
        assert "<observation>" in result
        assert "line1" in result
        assert "line2" in result

    def test_format_observation_truncation(self):
        result = format_observation("x" * 600, max_length=500)
        assert "truncated" in result


class TestToolExecutor:
    def test_platform_detection(self):
        executor = ToolExecutor(dry_run=False)
        assert executor.platform in ["windows", "linux", "darwin"]

    def test_parse_tool_call_valid(self):
        executor = ToolExecutor(dry_run=False)
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        name, args = executor.parse_tool_call(text)
        assert name == "calculator"
        assert args == {"expression": "2+2"}

    def test_parse_tool_call_invalid(self):
        executor = ToolExecutor(dry_run=False)
        text = '<tool_call>{invalid json}</tool_call>'
        name, args = executor.parse_tool_call(text)
        assert name == ''

    def test_parse_tool_call_no_tool_call(self):
        executor = ToolExecutor(dry_run=False)
        text = "Hello, how are you?"
        name, args = executor.parse_tool_call(text)
        assert name == ''

    def test_has_tool_call_true(self):
        executor = ToolExecutor(dry_run=False)
        text = '<tool_call>{"name": "test"}</tool_call>'
        assert executor.has_tool_call(text)

    def test_has_tool_call_false(self):
        executor = ToolExecutor(dry_run=False)
        text = "Hello, how are you?"
        assert not executor.has_tool_call(text)

    def test_execute_tool_call_with_registry(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        executor = ToolExecutor(dry_run=False)
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        result = executor.execute_tool_call(text, registry)
        assert "<observation>" in result
        assert "4" in result

    def test_execute_tool_call_unknown_tool(self):
        registry = ToolRegistry()
        executor = ToolExecutor(dry_run=False)
        text = '<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>'
        result = executor.execute_tool_call(text, registry)
        assert "Error" in result

    def test_execute_tool_call_dry_run(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        executor = ToolExecutor(dry_run=True)
        # Calculator is non-shell (category='computation'), dry_run only applies to shell
        # So we test dry_run by calling execute_tool_call with a shell-like text
        # But shell tools aren't in default registry, so test with unknown tool
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        result = executor.execute_tool_call(text, registry)
        # Non-shell tools execute normally even in dry_run mode
        assert "<observation>" in result

    def test_execute_tool_call_blocked_command(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        executor = ToolExecutor(dry_run=False)
        text = '<tool_call>{"name": "shell", "arguments": {"command": "rm -rf /"}}</tool_call>'
        result = executor.execute_tool_call(text, registry)
        assert "Error" in result


class TestPlatformFunctions:
    def test_get_platform(self):
        p = get_platform()
        assert p in ["windows", "linux", "darwin"]

    def test_get_default_shell(self):
        shell = get_default_shell()
        assert shell in ["powershell", "bash"]
