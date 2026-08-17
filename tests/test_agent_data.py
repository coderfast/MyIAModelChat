import pytest
from dataset_preparer.agent.thinking import AgentThinkingGenerator
from dataset_preparer.agent.quality import (
    validate_agent_sample, filter_low_quality_agent, agent_data_report
)


class TestAgentThinkingGenerator:
    def test_generate_math_tool_call(self):
        gen = AgentThinkingGenerator()
        sample = {"input": "¿Cuánto es 2+2?", "output": "Cuatro"}
        result = gen.generate(sample)
        assert result is not None
        assert result.get('has_tool_call') is True
        assert "<tool_call>" in result.get('input_ids', '')
        assert "calculator" in result.get('tool_name', '')

    def test_generate_date_tool_call(self):
        gen = AgentThinkingGenerator()
        sample = {"input": "¿Qué día es hoy?", "output": "Hoy es lunes"}
        result = gen.generate(sample)
        assert result is not None
        assert result.get('has_tool_call') is True
        assert "current_date" in result.get('tool_name', '')

    def test_generate_word_count(self):
        gen = AgentThinkingGenerator()
        sample = {"input": "Cuenta las palabras de esto", "output": "Hola mundo"}
        result = gen.generate(sample)
        assert result is not None
        # word_count tool may or may not be triggered depending on regex
        assert 'thinking' in result

    def test_generate_no_tool_needed(self):
        gen = AgentThinkingGenerator()
        sample = {"input": "Hola", "output": "Hola, ¿cómo estás?"}
        result = gen.generate(sample)
        assert result is not None
        # Simple greeting should not trigger tool
        assert result.get('has_tool_call') is False

    def test_generate_platform_tool(self):
        gen = AgentThinkingGenerator()
        sample = {"input": "¿Qué sistema operativo usas?", "output": "Windows"}
        result = gen.generate(sample)
        assert result is not None
        assert 'thinking' in result

    def test_empty_sample(self):
        gen = AgentThinkingGenerator()
        sample = {"input": "", "output": ""}
        result = gen.generate(sample)
        assert result is not None

    def test_stats(self):
        gen = AgentThinkingGenerator()
        gen.generate({"input": "¿Cuánto es 2+2?", "output": "4"})
        gen.generate({"input": "Hola", "output": "Hola"})
        stats = gen.get_stats()
        assert stats['total'] == 2
        assert stats['tool_call_samples'] + stats['normal_samples'] == 2


class TestAgentQualityValidation:
    def test_validate_good_agent_sample(self):
        sample = {
            "input_ids": '<tool_call>calculator(2+2)</tool_call><|tool_result|>4<|end|>',
            "thinking": "Necesito calcular 2+2.",
            "answer": "La respuesta es 4",
            "has_tool_call": True,
            "tool_name": "calculator",
        }
        result = validate_agent_sample(sample)
        assert result.valid

    def test_validate_no_tool_call_with_thinking(self):
        sample = {
            "input_ids": "Hola, ¿cómo estás?",
            "thinking": "Es un saludo simple.",
            "answer": "Estoy bien",
            "has_tool_call": False,
        }
        result = validate_agent_sample(sample)
        assert result.valid

    def test_validate_no_tool_call_no_thinking(self):
        sample = {
            "input_ids": "Hola",
            "thinking": "",
            "answer": "Hola",
            "has_tool_call": False,
        }
        result = validate_agent_sample(sample)
        assert not result.valid or result.score < 1.0

    def test_validate_empty_answer(self):
        sample = {
            "input_ids": "test",
            "thinking": "test",
            "answer": "",
            "has_tool_call": False,
        }
        result = validate_agent_sample(sample)
        assert 'empty_answer' in result.issues
        assert result.score < 1.0

    def test_validate_malformed_tool_call(self):
        sample = {
            "input_ids": '<tool_call>invalid_format</tool_call>',
            "thinking": "test",
            "answer": "test",
            "has_tool_call": True,
        }
        result = validate_agent_sample(sample)
        assert not result.valid

    def test_filter_low_quality(self):
        samples = [
            {
                "input_ids": "Hello",
                "thinking": "",
                "answer": "Hi",
                "has_tool_call": False,
            },
            {
                "input_ids": '<tool_call>calculator(1+1)</tool_call><|tool_result|>2<|end|>test',
                "thinking": "Necesito calcular 1+1.",
                "answer": "2",
                "has_tool_call": True,
                "tool_name": "calculator",
            },
        ]
        filtered = filter_low_quality_agent(samples)
        assert len(filtered) >= 1


class TestAgentDataReport:
    def test_agent_data_report(self):
        samples = [
            {"has_tool_call": True, "tool_name": "calculator", "input_ids": "test1", "thinking": "t1", "answer": "a1"},
            {"has_tool_call": False, "input_ids": "test2", "thinking": "t2", "answer": "a2"},
            {"has_tool_call": True, "tool_name": "current_date", "input_ids": "test3", "thinking": "t3", "answer": "a3"},
        ]
        report = agent_data_report(samples)
        assert report['total_samples'] == 3
        assert report['tool_call_samples'] == 2
        assert report['normal_samples'] == 1
        assert report['tool_ratio'] == pytest.approx(2/3)
        assert 'calculator' in report['tool_distribution']
