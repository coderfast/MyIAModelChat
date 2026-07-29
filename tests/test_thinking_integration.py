"""
Integration tests for the ThinkingEngine with source-specific generators.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.thinking_engine import ThinkingEngine
from dataset_preparer.csv.thinking import CSVThinkingGenerator
from dataset_preparer.aiml.thinking import AIMLThinkingGenerator
from dataset_preparer.pdf.thinking import PDFThinkingGenerator
from dataset_preparer.epub.thinking import EPUBThinkingGenerator
from dataset_preparer.hf.thinking import HFThinkingGenerator
from dataset_preparer.web.thinking import WebThinkingGenerator
from dataset_preparer.thinking_quality import validate_thinking


class TestThinkingIntegration:
    """Integration tests for thinking generation across all sources."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engine = ThinkingEngine(depth='adaptive')

    def test_csv_thinking_real(self):
        """Test CSV generates real thinking."""
        generator = CSVThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'input': 'What is machine learning?',
            'output': 'Machine learning is a subset of AI that learns from data.'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    def test_aiml_thinking_real(self):
        """Test AIML generates real thinking."""
        generator = AIMLThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'input': 'hello',
            'output': 'Hello! How can I help you?'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    def test_pdf_thinking_real(self):
        """Test PDF generates real thinking."""
        generator = PDFThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'input_ids': 'Python es un lenguaje de programación interpretado. Fue creado por Guido van Rossum en 1991.'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    def test_epub_thinking_real(self):
        """Test EPUB generates real thinking."""
        generator = EPUBThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'input_ids': 'El capítulo describe la aventura del protagonista en la ciudad.',
            'chapter': 'Capítulo 1'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    def test_hf_thinking_real(self):
        """Test HuggingFace generates real thinking."""
        generator = HFThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'question': 'What is neural network?',
            'answer': 'A neural network is a computing system inspired by biological neural networks.'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    def test_web_thinking_real(self):
        """Test Web generates real thinking."""
        generator = WebThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'input_ids': 'FastAPI is a modern web framework for building APIs with Python.',
            'title': 'FastAPI Documentation'
        }
        result = generator.generate(sample)

        assert 'thinking' in result
        assert len(result['thinking']) > 30
        validation = validate_thinking(result['thinking'], result.get('output', ''))
        assert validation.valid is True, f"Thinking failed validation: {validation.issues}"

    def test_thinking_quality_score(self):
        """Test that generated thinking passes quality validation."""
        texts = [
            "Python es un lenguaje de programación popular.",
            "Machine learning uses algorithms to learn from data.",
            "El análisis de datos permite tomar mejores decisiones.",
        ]

        engine = ThinkingEngine(depth='adaptive')
        for text in texts:
            thinking = engine.generate_thinking(text)
            assert len(thinking) > 30
            validation = validate_thinking(thinking)
            assert validation.valid is True, f"Thinking for '{text[:30]}...' failed: {validation.issues}"

    def test_thinking_no_re_declaration(self):
        """Test that thinking is not just re-declaring the answer."""
        generator = CSVThinkingGenerator(self.engine, depth='adaptive')
        sample = {
            'input': 'What is Python?',
            'output': 'Python is a programming language.'
        }
        result = generator.generate(sample)

        # Thinking should not contain "La pregunta es" or "La respuesta es"
        thinking = result['thinking'].lower()
        assert 'la pregunta es' not in thinking or 'la respuesta es' not in thinking


def run_tests():
    """Run all integration tests."""
    test = TestThinkingIntegration()
    test.setup_method()

    tests = [
        test.test_csv_thinking_real,
        test.test_aiml_thinking_real,
        test.test_pdf_thinking_real,
        test.test_epub_thinking_real,
        test.test_hf_thinking_real,
        test.test_web_thinking_real,
        test.test_thinking_quality_score,
        test.test_thinking_no_re_declaration,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_func in tests:
        try:
            test.setup_method()
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
