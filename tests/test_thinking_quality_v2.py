"""
Tests for the Thinking Quality Validator V2.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.thinking_quality import validate_thinking, BatchQualityReport


class TestThinkingQuality:
    """Tests for ThinkingQuality V2 functionality."""

    def test_reject_too_short(self):
        """Test rejection of too-short thinking."""
        result = validate_thinking("Hi")
        assert result.valid is False
        assert result.score == 0.0
        assert 'too_short' in result.issues

    def test_reject_empty(self):
        """Test rejection of empty thinking."""
        result = validate_thinking("")
        assert result.valid is False

    def test_reject_qa_redeclaration(self):
        """Test rejection of Q&A re-declaration."""
        thinking = "La pregunta es: ¿Cuál es la capital? La respuesta correcta es: París."
        result = validate_thinking(thinking, answer="París")
        assert result.valid is False
        assert 're_declaration' in result.issues

    def test_reject_information_restatement(self):
        """Test rejection of information re-statement."""
        thinking = "Se presenta información sobre: el clima. La respuesta contiene los detalles relevantes."
        result = validate_thinking(thinking)
        assert result.valid is False
        assert 're_declaration' in result.issues

    def test_reject_placeholder(self):
        """Test rejection of placeholder content."""
        thinking = "Este texto contiene 150 palabras con información relevante. El contenido puede ser procesada."
        result = validate_thinking(thinking)
        assert result.valid is False
        assert 'placeholder' in result.issues

    def test_reject_generic_technical(self):
        """Test rejection of generic technical placeholder."""
        thinking = "El contenido presenta información técnica o académica. El texto contiene 200 palabras."
        result = validate_thinking(thinking)
        assert result.valid is False
        assert 'placeholder' in result.issues

    def test_accept_real_reasoning(self):
        """Test acceptance of real reasoning."""
        thinking = (
            "Analizando el contenido, identificamos que el texto presenta información "
            "sobre machine learning. Los conceptos principales incluyen redes neuronales "
            "y algoritmos de clasificación. Además, se mencionan aplicaciones prácticas "
            "en el ámbito de la inteligencia artificial."
        )
        result = validate_thinking(thinking)
        assert result.valid is True
        assert result.score >= 0.5

    def test_accept_entity_analysis(self):
        """Test acceptance of thinking with entity analysis."""
        thinking = (
            "El fragmento corresponde a la sección 'Introducción al Python'. "
            "Se identifican entidades como Guido van Rossum y Python Software Foundation. "
            "El contenido presenta información técnica sobre el lenguaje de programación."
        )
        result = validate_thinking(thinking)
        assert result.valid is True

    def test_accept_logical_connectors(self):
        """Test acceptance of thinking with logical connectors."""
        thinking = (
            "En primer lugar, el texto contiene información sobre bases de datos. "
            "Además, se presentan conceptos técnicos como SQL y normalización. "
            "Finalmente, el análisis muestra que el contenido es de tipo educativo."
        )
        result = validate_thinking(thinking)
        assert result.valid is True

    def test_accept_question_analysis(self):
        """Test acceptance of thinking with question analysis."""
        thinking = (
            "La consulta del usuario es una solicitud de información. "
            "Se detecta que la pregunta busca datos específicos sobre el tema. "
            "La respuesta proporciona información directa y relevante."
        )
        result = validate_thinking(thinking, question="¿Qué es machine learning?")
        assert result.valid is True

    def test_reject_meta_commentary(self):
        """Test rejection of pure meta-commentary."""
        thinking = "El usuario saluda con hola. Esto indica que quiere iniciar una conversación."
        result = validate_thinking(thinking)
        assert result.valid is False
        assert 'meta_commentary' in result.issues

    def test_accept_with_steps(self):
        """Test acceptance of thinking with step indicators."""
        thinking = (
            "Primero, analizamos la estructura del documento. "
            "Luego, identificamos los conceptos clave. "
            "Finalmente, construimos el razonamiento paso a paso."
        )
        result = validate_thinking(thinking)
        assert result.valid is True

    def test_vocabulary_diversity(self):
        """Test vocabulary diversity check."""
        # Low diversity: just re-declaring the answer (should be rejected)
        thinking_low = "La pregunta es: ¿Qué es? La respuesta correcta es: machine learning."
        result_low = validate_thinking(thinking_low, answer="machine learning")

        # High diversity: adding new vocabulary (should be accepted)
        thinking_high = (
            "El análisis identifica que el contenido trata sobre aprendizaje automático. "
            "Los conceptos principales incluyen redes neuronales y clasificación."
        )
        result_high = validate_thinking(thinking_high, answer="machine learning")

        # Low diversity should be rejected (re-declaration)
        assert result_low.valid is False
        # High diversity should be accepted
        assert result_high.valid is True

    def test_batch_report(self):
        """Test batch quality report."""
        samples = [
            {'thinking': 'This is a valid thinking with enough content and reasoning.', 'output': 'answer'},
            {'thinking': '', 'output': 'answer'},  # too short
            {'thinking': 'La pregunta es: ¿Qué es? La respuesta correcta es: machine learning.', 'output': 'Y'},  # re-declaration
        ]

        report = BatchQualityReport()
        for sample in samples:
            result = validate_thinking(sample['thinking'], sample.get('output', ''))
            report.total += 1
            if result.valid:
                report.valid += 1
            else:
                if 'too_short' in result.issues:
                    report.too_short += 1
                if 're_declaration' in result.issues:
                    report.re_declaration += 1

        assert report.total == 3
        assert report.too_short == 1
        assert report.re_declaration == 1


def run_tests():
    """Run all tests."""
    test = TestThinkingQuality()

    tests = [
        test.test_reject_too_short,
        test.test_reject_empty,
        test.test_reject_qa_redeclaration,
        test.test_reject_information_restatement,
        test.test_reject_placeholder,
        test.test_reject_generic_technical,
        test.test_accept_real_reasoning,
        test.test_accept_entity_analysis,
        test.test_accept_logical_connectors,
        test.test_accept_question_analysis,
        test.test_reject_meta_commentary,
        test.test_accept_with_steps,
        test.test_vocabulary_diversity,
        test.test_batch_report,
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
