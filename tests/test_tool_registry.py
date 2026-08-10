import pytest
from commons.tools.tool_registry import ToolRegistry, register_default_tools, ToolDef


def _dummy_func(**kwargs):
    return "ok"


class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()
        tool = ToolDef(name="test_tool", description="A test tool", parameters={}, func=_dummy_func, category="computation")
        registry.register(tool)
        names = [t.name for t in registry.list_tools()]
        assert "test_tool" in names

    def test_get_tool(self):
        registry = ToolRegistry()
        tool = ToolDef(name="test_tool", description="A test tool", parameters={}, func=_dummy_func, category="computation")
        registry.register(tool)
        retrieved = registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_get_unknown_tool(self):
        registry = ToolRegistry()
        assert registry.get("unknown") is None

    def test_register_default_tools(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        names = [t.name for t in registry.list_tools()]
        assert "calculator" in names
        assert "current_date" in names
        assert "word_count" in names
        assert "get_platform" in names
        assert "read_file" in names
        assert "list_directory" in names
        assert "web_search" in names

    def test_tool_descriptions(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        descriptions = registry.get_descriptions()
        assert isinstance(descriptions, str)
        assert "calculator" in descriptions
        assert "current_date" in descriptions

    def test_calculator_tool(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        result = registry.call("calculator", {"expression": "15 * 37"})
        assert "555" in result

    def test_current_date_tool(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        result = registry.call("current_date", {})
        assert result
        assert any(c.isdigit() for c in result)

    def test_get_platform_tool(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        result = registry.call("get_platform", {})
        assert result.lower() in ["windows", "linux", "darwin"]

    def test_word_count_tool(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        result = registry.call("word_count", {"text": "hello world foo"})
        assert "3" in result

    def test_register_duplicate_tool_overwrites(self):
        registry = ToolRegistry()
        tool1 = ToolDef(name="test", description="First", parameters={}, func=_dummy_func, category="computation")
        tool2 = ToolDef(name="test", description="Second", parameters={}, func=_dummy_func, category="computation")
        registry.register(tool1)
        registry.register(tool2)
        assert registry.get("test").description == "Second"

    def test_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        result = registry.call("nonexistent", {})
        assert "Error" in result
        assert "nonexistent" in result

    def test_list_platform_tools(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        tools = registry.list_platform_tools()
        assert len(tools) > 0
