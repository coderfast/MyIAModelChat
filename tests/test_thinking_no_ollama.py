"""
Test that thinking generation works without Ollama dependency.
This test mocks OllamaTeacher to verify ThinkingEngine works independently.
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.thinking_engine import ThinkingEngine
from dataset_preparer.csv.thinking import CSVThinkingGenerator
from dataset_preparer.thinking_quality import validate_thinking


class TestThinkingNoOllama:
    """Test thinking generation without Ollama."""

    def test_thinking_engine_works_without_ollama(self):
        """Test that ThinkingEngine works without any external dependency."""
        engine = ThinkingEngine(depth='adaptive')

        texts = [
            "Machine learning is a branch of artificial intelligence.",
            "Python es un lenguaje de programación interpretado.",
            "El análisis de datos permite tomar decisiones informadas.",
            "FastAPI provides automatic API documentation.",
        ]

        for text in texts:
            thinking = engine.generate_thinking(text)
            assert len(thinking) > 30, f"Thinking too short for: {text[:30]}"
            validation = validate_thinking(thinking)
            assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    @patch('dataset_preparer.csv.thinking.OllamaTeacher')
    def test_csv_thinking_without_ollama(self, mock_teacher_class):
        """Test CSV thinking generation without Ollama."""
        # Mock OllamaTeacher to return None (not available)
        mock_teacher = MagicMock()
        mock_teacher.is_available.return_value = False
        mock_teacher_class.return_value = mock_teacher

        engine = ThinkingEngine(depth='adaptive')
        generator = CSVThinkingGenerator(engine, mock_teacher, depth='adaptive')

        sample = {
            'input': 'What is deep learning?',
            'output': 'Deep learning is a subset of machine learning using neural networks.'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True

    @patch('dataset_preparer.aiml.thinking.OllamaTeacher')
    def test_aiml_thinking_without_ollama(self, mock_teacher_class):
        """Test AIML thinking generation without Ollama."""
        from dataset_preparer.aiml.thinking import AIMLThinkingGenerator

        mock_teacher = MagicMock()
        mock_teacher.is_available.return_value = False
        mock_teacher_class.return_value = mock_teacher

        engine = ThinkingEngine(depth='adaptive')
        generator = AIMLThinkingGenerator(engine, mock_teacher, depth='adaptive')

        sample = {
            'input': 'hello',
            'output': 'Hello! How can I help you?'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30

    def test_all_sources_generate_real_thinking(self):
        """Test that all sources generate real thinking without Ollama."""
        from dataset_preparer.aiml.thinking import AIMLThinkingGenerator
        from dataset_preparer.pdf.thinking import PDFThinkingGenerator
        from dataset_preparer.epub.thinking import EPUBThinkingGenerator
        from dataset_preparer.hf.thinking import HFThinkingGenerator
        from dataset_preparer.web.thinking import WebThinkingGenerator

        engine = ThinkingEngine(depth='adaptive')

        sources = [
            ('CSV', CSVThinkingGenerator(engine, None, depth='adaptive'),
             {'input': 'What is AI?', 'output': 'AI is artificial intelligence.'}),
            ('AIML', AIMLThinkingGenerator(engine, None, depth='adaptive'),
             {'input': 'hi', 'output': 'Hello!'}),
            ('PDF', PDFThinkingGenerator(engine, None, depth='adaptive'),
             {'input_ids': 'Python is a programming language created in 1991.'}),
            ('EPUB', EPUBThinkingGenerator(engine, None, depth='adaptive'),
             {'input_ids': 'The story begins in a small town.'}),
            ('HF', HFThinkingGenerator(engine, None, depth='adaptive'),
             {'question': 'What is ML?', 'answer': 'ML is machine learning.'}),
            ('Web', WebThinkingGenerator(engine, None, depth='adaptive'),
             {'input_ids': 'This page explains how to use Docker.'}),
        ]

        for name, generator, sample in sources:
            result = generator.generate(sample)
            assert 'thinking' in result, f"{name}: No thinking generated"
            assert len(result['thinking']) > 30, f"{name}: Thinking too short"
            validation = validate_thinking(result['thinking'])
            assert validation.valid is True, f"{name}: Thinking failed validation: {validation.issues}"


def run_tests():
    """Run all no-ollama tests."""
    test = TestThinkingNoOllama()

    tests = [
        test.test_thinking_engine_works_without_ollama,
        test.test_csv_thinking_without_ollama,
        test.test_aiml_thinking_without_ollama,
        test.test_all_sources_generate_real_thinking,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"  PASS {test_func.__name__}")
        except AssertionError as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"  FAIL {test_func.__name__}: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"  FAIL {test_func.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
