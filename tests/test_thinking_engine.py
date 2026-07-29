"""
Tests for the ThinkingEngine - Real NLP-based thinking generation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.thinking_engine import ThinkingEngine


class TestThinkingEngine:
    """Tests for ThinkingEngine core functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engine = ThinkingEngine(depth='adaptive')

    def test_analyze_text_entities(self):
        """Test entity detection in text."""
        text = "Microsoft fue fundada por Bill Gates en Albuquerque."
        analysis = self.engine._analyze_text(text)

        assert 'entities' in analysis
        assert len(analysis['entities']) > 0
        entity_texts = [e[0] for e in analysis['entities']]
        assert any('Microsoft' in e or 'Bill Gates' in e or 'Albuquerque' in e for e in entity_texts)

    def test_analyze_text_noun_phrases(self):
        """Test noun phrase extraction."""
        text = "El motor de búsqueda de Google utiliza algoritmos avanzados de procesamiento de lenguaje natural."
        analysis = self.engine._analyze_text(text)

        assert 'noun_phrases' in analysis
        assert len(analysis['noun_phrases']) > 0

    def test_analyze_text_structure(self):
        """Test text structure analysis."""
        text = "Primera oración. Segunda oración más larga con más palabras. Tercera oración final."
        analysis = self.engine._analyze_text(text)

        assert analysis['word_count'] > 10
        assert analysis['sentence_count'] >= 3
        assert analysis['avg_sentence_length'] > 0

    def test_analyze_text_questions(self):
        """Test question detection."""
        text_question = "¿Cuál es la capital de Francia?"
        text_statement = "La capital de Francia es París."

        analysis_q = self.engine._analyze_text(text_question)
        analysis_s = self.engine._analyze_text(text_statement)

        assert analysis_q['has_questions'] is True
        assert analysis_s['has_questions'] is False

    def test_analyze_text_numbers(self):
        """Test number detection."""
        text_with_numbers = "Python fue creado en 1991 y tiene más de 30 millones de usuarios."
        text_without = "Python es un lenguaje de programación popular."

        analysis_with = self.engine._analyze_text(text_with_numbers)
        analysis_without = self.engine._analyze_text(text_without)

        assert analysis_with['has_numbers'] is True
        assert analysis_without['has_numbers'] is False

    def test_extract_key_concepts(self):
        """Test key concept extraction."""
        text = "Machine learning algorithms can classify data using neural networks. Deep learning uses multiple layers."
        analysis = self.engine._analyze_text(text)
        concepts = self.engine._extract_key_concepts(text, analysis)

        assert len(concepts) > 0
        assert all(isinstance(c, tuple) and len(c) == 2 for c in concepts)
        assert all(isinstance(c[1], int) for c in concepts)

    def test_detect_content_type_qa(self):
        """Test Q&A content type detection."""
        text = "What is the capital of France? How does photosynthesis work?"
        analysis = self.engine._analyze_text(text)

        content_type = self.engine._detect_content_type(text, analysis)
        assert content_type == 'qa'

    def test_detect_content_type_technical(self):
        """Test technical content type detection."""
        text = "The API returns JSON data with status code 200. The SDK provides REST endpoints."
        analysis = self.engine._analyze_text(text)

        content_type = self.engine._detect_content_type(text, analysis)
        assert content_type == 'technical'

    def test_detect_content_type_narrative(self):
        """Test narrative content type detection."""
        text = "The company was founded in 1995. They had a vision for the future. It came from humble beginnings."
        analysis = self.engine._analyze_text(text)

        content_type = self.engine._detect_content_type(text, analysis)
        assert content_type == 'narrative'

    def test_detect_content_type_instructional(self):
        """Test instructional content type detection."""
        text = "You must install the package first. You should verify the installation. Check the documentation."
        analysis = self.engine._analyze_text(text)

        content_type = self.engine._detect_content_type(text, analysis)
        assert content_type == 'instructional'

    def test_detect_language_spanish(self):
        """Test Spanish language detection."""
        text = "El aprendizaje automático es una rama de la inteligencia artificial."
        language = self.engine._detect_language(text)

        assert language == 'es'

    def test_detect_language_english(self):
        """Test English language detection."""
        text = "Machine learning is a branch of artificial intelligence that learns from data."
        language = self.engine._detect_language(text)

        assert language == 'en'

    def test_build_reasoning_factual(self):
        """Test reasoning building for factual content."""
        text = "Python es un lenguaje de programación interpretado de alto nivel. Fue creado por Guido van Rossum."
        context = {'title': 'Python Programming'}

        thinking = self.engine.generate_thinking(text, context)

        assert len(thinking) > 30
        assert 'Analizando' in thinking or 'analic' in thinking.lower()

    def test_build_reasoning_qa(self):
        """Test reasoning building for Q&A content."""
        text = "¿Cuál es la capital de Francia? La capital es París."
        context = {'question': '¿Cuál es la capital de Francia?', 'answer': 'La capital es París'}

        thinking = self.engine.generate_thinking(text, context)

        assert len(thinking) > 30
        assert 'París' in thinking or 'capital' in thinking.lower()

    def test_format_steps_es(self):
        """Test Spanish step formatting."""
        steps = ["Paso uno", "Paso dos", "Paso tres"]
        result = self.engine._format_steps_es(steps)

        assert 'primer lugar' in result.lower()
        assert len(result) > 20
        assert 'paso uno' in result.lower()
        assert 'paso dos' in result.lower()

    def test_format_steps_en(self):
        """Test English step formatting."""
        steps = ["Step one", "Step two", "Step three"]
        result = self.engine._format_steps_en(steps)

        assert 'first' in result.lower()
        assert 'additionally' in result.lower()
        assert 'step one' in result.lower()
        assert 'step two' in result.lower()

    def test_empty_text(self):
        """Test handling of empty text."""
        thinking = self.engine.generate_thinking("")
        assert thinking == ""

        thinking = self.engine.generate_thinking(None)
        assert thinking == ""

    def test_short_text(self):
        """Test handling of very short text."""
        thinking = self.engine.generate_thinking("Hi")
        assert thinking == ""

    def test_cache_works(self):
        """Test that caching works correctly."""
        text = "This is a test text for caching."

        thinking1 = self.engine.generate_thinking(text)
        thinking2 = self.engine.generate_thinking(text)

        assert thinking1 == thinking2

    def test_different_depths(self):
        """Test different depth configurations."""
        text = "Machine learning is a subset of artificial intelligence that focuses on algorithms."

        engine_basic = ThinkingEngine(depth='basic')
        engine_detailed = ThinkingEngine(depth='detailed')

        thinking_basic = engine_basic.generate_thinking(text)
        thinking_detailed = engine_detailed.generate_thinking(text)

        assert len(thinking_basic) > 0
        assert len(thinking_detailed) > 0


def run_tests():
    """Run all tests."""
    test = TestThinkingEngine()
    test.setup_method()

    tests = [
        test.test_analyze_text_entities,
        test.test_analyze_text_noun_phrases,
        test.test_analyze_text_structure,
        test.test_analyze_text_questions,
        test.test_analyze_text_numbers,
        test.test_extract_key_concepts,
        test.test_detect_content_type_qa,
        test.test_detect_content_type_technical,
        test.test_detect_content_type_narrative,
        test.test_detect_content_type_instructional,
        test.test_detect_language_spanish,
        test.test_detect_language_english,
        test.test_build_reasoning_factual,
        test.test_build_reasoning_qa,
        test.test_format_steps_es,
        test.test_format_steps_en,
        test.test_empty_text,
        test.test_short_text,
        test.test_cache_works,
        test.test_different_depths,
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
