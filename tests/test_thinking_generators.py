"""
Tests for thinking generators.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinking_generators import ThinkingGenerator, OllamaTeacher
from aiml_thinking import AIMLThinkingGenerator
from csv_thinking import CSVThinkingGenerator
from pdf_thinking import PDFThinkingGenerator
from epub_thinking import EPUBThinkingGenerator
from web_thinking import WebThinkingGenerator
from hf_thinking import HFThinkingGenerator
from thinking_quality import validate_thinking, validate_thinking_batch


def test_aiml_generator_greeting():
    gen = AIMLThinkingGenerator(teacher=None)
    sample = {'input': 'hello', 'output': 'Hello! How are you?'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    assert len(result['thinking']) > 10
    print("PASS: test_aiml_generator_greeting")


def test_aiml_generator_farewell():
    gen = AIMLThinkingGenerator(teacher=None)
    sample = {'input': 'goodbye', 'output': 'See you later!'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_aiml_generator_farewell")


def test_aiml_generator_unknown():
    gen = AIMLThinkingGenerator(teacher=None)
    sample = {'input': 'what is the meaning of life', 'output': '42'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_aiml_generator_unknown")


def test_csv_generator():
    gen = CSVThinkingGenerator(teacher=None)
    sample = {'input': 'What is Python?', 'output': 'Python is a programming language.'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    assert 'Python' in result['thinking'] or 'pregunta' in result['thinking'].lower()
    print("PASS: test_csv_generator")


def test_pdf_generator():
    gen = PDFThinkingGenerator(teacher=None)
    sample = {'input_ids': 'This is a section about machine learning. It covers supervised and unsupervised learning.'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_pdf_generator")


def test_pdf_generator_with_title():
    gen = PDFThinkingGenerator(teacher=None)
    sample = {'input_ids': 'Chapter 1: Introduction to AI\n\nArtificial intelligence is a field of computer science.'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_pdf_generator_with_title")


def test_epub_generator():
    gen = EPUBThinkingGenerator(teacher=None)
    sample = {'input_ids': 'The story continues with the hero facing new challenges.', 'chapter': 'Chapter 5'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    assert 'Chapter 5' in result['thinking']
    print("PASS: test_epub_generator")


def test_web_generator():
    gen = WebThinkingGenerator(teacher=None)
    sample = {'input_ids': 'This article discusses the latest trends in AI technology.', 'title': 'AI Trends 2024', 'url': 'https://example.com/ai'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_web_generator")


def test_hf_generator_qa():
    gen = HFThinkingGenerator(teacher=None)
    sample = {'question': 'What is 2+2?', 'answer': '4'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_hf_generator_qa")


def test_hf_generator_text():
    gen = HFThinkingGenerator(teacher=None)
    sample = {'text': 'Python is a high-level programming language known for its simplicity.'}
    result = gen.generate(sample)
    assert 'thinking' in result
    assert result['thinking'] is not None
    print("PASS: test_hf_generator_text")


def test_validate_thinking_good():
    result = validate_thinking(
        'La pregunta es sobre Python. Python es un lenguaje de programacion. Por lo tanto, la respuesta es 4.',
        'Python es un lenguaje de programacion.'
    )
    assert result.valid is True
    assert result.score > 0.5
    print("PASS: test_validate_thinking_good")


def test_validate_thinking_meta():
    result = validate_thinking(
        'El usuario me saluda.',
        'Hola!'
    )
    assert result.valid is False
    assert 'meta_commentary' in result.issues
    print("PASS: test_validate_thinking_meta")


def test_validate_thinking_too_short():
    result = validate_thinking('Hi', 'Hello')
    assert result.valid is False
    assert 'too_short' in result.issues
    print("PASS: test_validate_thinking_too_short")


def test_validate_thinking_batch():
    samples = [
        {'thinking': 'La pregunta es sobre X. Por lo tanto Y.', 'output': 'X es Y'},
        {'thinking': 'El usuario me saluda.', 'output': 'Hola'},
        {'thinking': 'Short', 'output': 'Hi'},
    ]
    report = validate_thinking_batch(samples)
    assert report.total == 3
    assert report.meta_commentary >= 1
    assert report.too_short >= 1
    print("PASS: test_validate_thinking_batch")


def test_format_thinking_sample():
    gen = AIMLThinkingGenerator(teacher=None)
    sample = {'input': 'hello', 'output': 'Hello!'}
    result = gen.generate(sample)
    assert 'thinking_text' in result
    assert '<think>' in result['thinking_text']
    assert '</think>' in result['thinking_text']
    print("PASS: test_format_thinking_sample")


if __name__ == '__main__':
    test_aiml_generator_greeting()
    test_aiml_generator_farewell()
    test_aiml_generator_unknown()
    test_csv_generator()
    test_pdf_generator()
    test_pdf_generator_with_title()
    test_epub_generator()
    test_web_generator()
    test_hf_generator_qa()
    test_hf_generator_text()
    test_validate_thinking_good()
    test_validate_thinking_meta()
    test_validate_thinking_too_short()
    test_validate_thinking_batch()
    test_format_thinking_sample()
    print("\nAll thinking generator tests passed!")
