"""
Tests for source validators.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.source_validators import (
    GenericValidator, AIMLValidator, PDFValidator,
    EPUBValidator, WebValidator, CSVValidator, get_validator
)


def test_generic_validator_classifies_correctly():
    v = GenericValidator('test')
    assert v.classify({'input': 'hello world'}) == 'good'
    assert v.classify({'input': ''}) == 'discardable'
    assert v.classify({'input': 'hi'}) == 'discardable'
    print("PASS: test_generic_validator_classifies_correctly")


def test_generic_validator_fixes_html():
    v = GenericValidator('test')
    sample = {'input': 'Hello <b>world</b> &amp; friends'}
    fixed = v.fix(sample)
    assert '<b>' not in fixed['input']
    assert '&amp;' not in fixed['input']
    assert 'Hello world' in fixed['input']
    print("PASS: test_generic_validator_fixes_html")


def test_aiml_validator_cleans_html_tags():
    v = AIMLValidator()
    sample = {'input': 'HELLO', 'output': 'Hello! <br/>How are you? <b>Good</b>'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert '<br/>' not in fixed['output']
    assert '<b>' not in fixed['output']
    assert 'Hello!' in fixed['output']
    print("PASS: test_aiml_validator_cleans_html_tags")


def test_aiml_validator_cleans_aiml_tags():
    v = AIMLValidator()
    sample = {'input': 'WHAT IS YOUR NAME', 'output': '<random><li>I am a bot</li><li>I am Alice</li></random>'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert '<random>' not in fixed['output']
    assert '<li>' not in fixed['output']
    print("PASS: test_aiml_validator_cleans_aiml_tags")


def test_aiml_validator_discards_wildcards():
    v = AIMLValidator()
    sample = {'input': '_', 'output': 'I do not understand'}
    assert v.classify(sample) == 'discardable'
    print("PASS: test_aiml_validator_discards_wildcards")


def test_aiml_validator_good_sample():
    v = AIMLValidator()
    sample = {'input': 'HELLO', 'output': 'Hello! How are you?'}
    assert v.classify(sample) == 'good'
    print("PASS: test_aiml_validator_good_sample")


def test_pdf_validator_detects_mojibake():
    v = PDFValidator()
    sample = {'input_ids': 'This is \ufffd\ufffd\ufffd garbled text with replacement chars'}
    assert v.classify(sample) == 'discardable'
    print("PASS: test_pdf_validator_detects_mojibake")


def test_pdf_validator_fixes_hyphens():
    v = PDFValidator()
    sample = {'input_ids': 'Artifi-\ncial intelligence is import-\nant.'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert 'Artificial' in fixed['input_ids']
    assert 'important' in fixed['input_ids']
    print("PASS: test_pdf_validator_fixes_hyphens")


def test_pdf_validator_fixes_page_numbers():
    v = PDFValidator()
    sample = {'input_ids': 'Some content here.\n42\nMore content.'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert '42' not in fixed['input_ids']
    print("PASS: test_pdf_validator_fixes_page_numbers")


def test_pdf_validator_good_sample():
    v = PDFValidator()
    sample = {'input_ids': 'This is a well-formed PDF text with enough content to pass validation checks.'}
    assert v.classify(sample) == 'good'
    print("PASS: test_pdf_validator_good_sample")


def test_epub_validator_decodes_entities():
    v = EPUBValidator()
    sample = {'input_ids': 'Hello &amp;world&lt; &nbsp; testing'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert '&amp;' not in fixed['input_ids']
    assert '&lt;' not in fixed['input_ids']
    print("PASS: test_epub_validator_decodes_entities")


def test_epub_validator_removes_html():
    v = EPUBValidator()
    sample = {'input_ids': '<p>Hello</p> <b>world</b>'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert '<p>' not in fixed['input_ids']
    assert '<b>' not in fixed['input_ids']
    print("PASS: test_epub_validator_removes_html")


def test_epub_validator_discards_navigation():
    v = EPUBValidator()
    sample = {'input_ids': 'Table of Contents'}
    assert v.classify(sample) == 'discardable'
    print("PASS: test_epub_validator_discards_navigation")


def test_web_validator_removes_boilerplate():
    v = WebValidator()
    sample = {'input_ids': 'Cookie notice: This site uses cookies. Privacy policy. Copyright 2024.'}
    assert v.classify(sample) in ('fixable', 'discardable')
    print("PASS: test_web_validator_removes_boilerplate")


def test_web_validator_removes_html():
    v = WebValidator()
    sample = {'input_ids': '<div>Hello</div> <script>alert("hi")</script> content'}
    assert v.classify(sample) == 'fixable'
    fixed = v.fix(sample)
    assert '<div>' not in fixed['input_ids']
    print("PASS: test_web_validator_removes_html")


def test_web_validator_good_sample():
    v = WebValidator()
    sample = {'input_ids': 'This is a well-written article about machine learning with enough content.'}
    assert v.classify(sample) == 'good'
    print("PASS: test_web_validator_good_sample")


def test_csv_validator_discards_missing_fields():
    v = CSVValidator()
    sample = {'input': '', 'output': 'some answer'}
    assert v.classify(sample) == 'discardable'
    print("PASS: test_csv_validator_discards_missing_fields")


def test_csv_validator_good_sample():
    v = CSVValidator()
    sample = {'input': 'What is Python?', 'output': 'Python is a programming language.'}
    assert v.classify(sample) == 'good'
    print("PASS: test_csv_validator_good_sample")


def test_get_validator_factory():
    assert isinstance(get_validator('aiml'), AIMLValidator)
    assert isinstance(get_validator('pdf'), PDFValidator)
    assert isinstance(get_validator('epub'), EPUBValidator)
    assert isinstance(get_validator('web'), WebValidator)
    assert isinstance(get_validator('csv'), CSVValidator)
    assert isinstance(get_validator('unknown'), GenericValidator)
    print("PASS: test_get_validator_factory")


def test_validate_batch():
    v = AIMLValidator()
    samples = [
        {'input': 'HELLO', 'output': 'Hello!'},
        {'input': '_', 'output': 'I do not understand'},
        {'input': 'BYE', 'output': 'Goodbye! <br/>'},
    ]
    cleaned, report = v.validate_batch(samples)
    assert report.total_samples == 3
    assert report.discardable >= 1
    print("PASS: test_validate_batch")


if __name__ == '__main__':
    test_generic_validator_classifies_correctly()
    test_generic_validator_fixes_html()
    test_aiml_validator_cleans_html_tags()
    test_aiml_validator_cleans_aiml_tags()
    test_aiml_validator_discards_wildcards()
    test_aiml_validator_good_sample()
    test_pdf_validator_detects_mojibake()
    test_pdf_validator_fixes_hyphens()
    test_pdf_validator_fixes_page_numbers()
    test_pdf_validator_good_sample()
    test_epub_validator_decodes_entities()
    test_epub_validator_removes_html()
    test_epub_validator_discards_navigation()
    test_web_validator_removes_boilerplate()
    test_web_validator_removes_html()
    test_web_validator_good_sample()
    test_csv_validator_discards_missing_fields()
    test_csv_validator_good_sample()
    test_get_validator_factory()
    test_validate_batch()
    print("\nAll source validator tests passed!")
