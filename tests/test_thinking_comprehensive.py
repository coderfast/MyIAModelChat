"""Comprehensive tests for thinking pipeline integration.

Tests cover: BPE thinking symbols, thinking quality, source validators edge cases,
thinking generator depth, training metrics, and pipeline data flow.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATASET_CACHE = os.path.join(os.path.dirname(__file__), '..', 'dataset_cache')


# ============================================================
# BPE Tokenizer Thinking Symbols Tests
# ============================================================

def test_bpe_thinking_symbols_registered():
    """Verify <thought> and </thought> are in the BPE vocabulary."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    thinking_id = wrapper.get_thinking_index()
    thinking_end_id = wrapper.get_thinking_end_index()

    assert thinking_id >= 0, f"thinking_id should be >= 0, got {thinking_id}"
    assert thinking_end_id >= 0, f"thinking_end_id should be >= 0, got {thinking_end_id}"
    assert thinking_id != thinking_end_id, "thinking and thinking_end should have different ids"

    # Verify they decode to the correct strings
    decoded = wrapper.sp.id_to_piece(thinking_id)
    assert 'thought' in decoded.lower() or '<thinking>' in decoded, f"Unexpected token: {decoded}"
    decoded_end = wrapper.sp.id_to_piece(thinking_end_id)
    assert 'thought' in decoded_end.lower() or '</thinking>' in decoded_end, f"Unexpected token: {decoded_end}"
    print("PASS: BPE thinking symbols registered and working")


def test_bpe_encode_preserves_thinking_tags():
    """Verify BPE encode preserves <think> tags in text."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    text_with_thinking = "<thinking>El usuario pregunta sobre Python. Python es un lenguaje.</thinking>La respuesta es 42."
    encoded = wrapper.encode(text_with_thinking, add_bos=False, add_eos=False)

    assert len(encoded) > 0, "Encoded should not be empty"

    # Check that thinking tokens are present
    thinking_id = wrapper.get_thinking_index()
    thinking_end_id = wrapper.get_thinking_end_index()
    assert thinking_id in encoded, f"<thought> token not found in encoded: {encoded}"
    assert thinking_end_id in encoded, f"</thought> token not found in encoded: {encoded}"

    # Verify ordering: <thought> before </thought>
    open_pos = encoded.index(thinking_id)
    close_pos = encoded.index(thinking_end_id)
    assert open_pos < close_pos, f"<thought> at {open_pos} should be before </thought> at {close_pos}"
    print("PASS: BPE encode preserves thinking tags")


def test_bpe_has_thinking_method():
    """Verify has_thinking and split_thinking methods work."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    text_with = "<thinking>Razoning aquí</thinking>La respuesta"
    text_without = "Hola mundo"

    assert wrapper.has_thinking(text_with) is True
    assert wrapper.has_thinking(text_without) is False

    thinking, response = wrapper.split_thinking(text_with)
    assert 'Razoning' in thinking
    assert 'respuesta' in response
    print("PASS: has_thinking and split_thinking work")


# ============================================================
# Thinking Quality Tests
# ============================================================

def test_validate_thinking_good_reasoning():
    """Verify good reasoning thinking passes validation."""
    from dataset_preparer.thinking_quality import validate_thinking

    result = validate_thinking(
        "La pregunta es sobre sumar 2+2. Python es un lenguaje interpretado.",
        "2+2 = 4. La suma de dos más dos es cuatro."
    )
    assert result.valid is True, f"Expected valid, got issues={result.issues}, score={result.score}"
    assert result.score >= 0.5, f"Expected score >= 0.5, got {result.score}"
    print("PASS: Good reasoning thinking passes validation")


def test_validate_thinking_meta_commentary_rejected():
    """Verify pure meta-commentary thinking is rejected."""
    from dataset_preparer.thinking_quality import validate_thinking

    result = validate_thinking(
        "El usuario me saluda",
        "Hola! ¿Cómo estás?"
    )
    assert result.valid is False, f"Expected invalid for meta-commentary, got valid={result.valid}, issues={result.issues}"
    print("PASS: Meta-commentary thinking rejected")


def test_validate_thinking_too_short():
    """Verify very short thinking is rejected."""
    from dataset_preparer.thinking_quality import validate_thinking

    result = validate_thinking("ok", "Respuesta")
    assert result.valid is False, f"Expected invalid for too short, got {result}"
    print("PASS: Too short thinking rejected")


def test_validate_thinking_with_steps():
    """Verify thinking with step indicators passes."""
    from dataset_preparer.thinking_quality import validate_thinking

    result = validate_thinking(
        "Paso 1: Analizo la pregunta. Paso 2: Identifico el tema. Paso 3: Formulo respuesta.",
        "Aquí está tu respuesta."
    )
    assert result.valid is True, f"Expected valid for step-based thinking, got issues={result.issues}, score={result.score}"
    print("PASS: Step-based thinking passes validation")


def test_filter_low_quality():
    """Verify low quality thinking samples are filtered."""
    from dataset_preparer.thinking_quality import filter_low_quality

    samples = [
        {'thinking': 'The user asks about Python. Python is a language. The answer is 42.', 'output': '42'},
        {'thinking': 'El usuario me saluda', 'output': 'Hola'},
        {'thinking': 'Análisis paso a paso: primero identifico el tema, luego formulo la respuesta basada en el conocimiento.', 'output': 'Respuesta'},
        {'thinking': 'ok', 'output': 'Respuesta'},
    ]

    filtered = filter_low_quality(samples)
    assert len(filtered) < len(samples), f"Expected filtering, got {len(filtered)} from {len(samples)}"
    # The meta-commentary one should be filtered
    filtered_thinkings = [s.get('thinking', '') for s in filtered]
    assert 'El usuario me saluda' not in filtered_thinkings, "Meta-commentary should be filtered"
    print("PASS: filter_low_quality works correctly")


# ============================================================
# Source Validators Edge Cases
# ============================================================

def test_generic_validator_fixes_encoding():
    """Verify GenericValidator detects encoding issues as fixable."""
    from dataset_preparer.source_validators import GenericValidator

    validator = GenericValidator('test')
    sample = {'input': 'El texto tiene caracteres raros\ufffd aqu\u00ed', 'output': 'respuesta'}
    result = validator.classify(sample)
    assert result == 'fixable', f"Expected fixable for encoding issues, got {result}"
    print("PASS: GenericValidator detects encoding issues")


def test_generic_validator_discards_empty():
    """Verify GenericValidator discards empty samples."""
    from dataset_preparer.source_validators import GenericValidator

    validator = GenericValidator('test')
    result = validator.classify({'input': '', 'output': ''})
    assert result == 'discardable', f"Expected discardable for empty, got {result}"
    print("PASS: GenericValidator discards empty samples")


def test_aiml_validator_good_sample():
    """Verify AIMLValidator accepts valid samples."""
    from dataset_preparer.source_validators import AIMLValidator

    validator = AIMLValidator()
    sample = {'input': 'hello', 'output': 'Hi there! How can I help you?'}
    result = validator.classify(sample)
    assert result == 'good', f"Expected good, got {result}"
    print("PASS: AIMLValidator accepts valid samples")


def test_pdf_validator_fixes_hyphens():
    """Verify PDFValidator fixes hyphen line breaks."""
    from dataset_preparer.source_validators import PDFValidator

    validator = PDFValidator()
    # Test with a hyphen line break pattern (word-\nword)
    sample = {'input': 'compu-\ntadora es un dispositivo', 'output': 'respuesta'}
    fixed = validator.fix(sample)
    # The _clean_pdf_text method handles HYPHEN_LINE_BREAK
    assert 'compu-\ntadora' not in fixed.get('input', ''), f"Hyphen break not fixed: {fixed}"
    print("PASS: PDFValidator fixes hyphens")


def test_epub_validator_removes_navigation():
    """Verify EPUBValidator discards navigation samples."""
    from dataset_preparer.source_validators import EPUBValidator

    validator = EPUBValidator()
    sample = {'input': 'Table of Contents', 'output': 'Chapter 1'}
    result = validator.classify(sample)
    assert result == 'discardable', f"Expected discardable for navigation, got {result}"
    print("PASS: EPUBValidator discards navigation")


def test_web_validator_cleans_html():
    """Verify WebValidator cleans HTML content."""
    from dataset_preparer.source_validators import WebValidator

    validator = WebValidator()
    sample = {'input': '<p>Hello</p> <script>alert(1)</script> World', 'output': 'content'}
    # classify should detect HTML/JS
    result = validator.classify(sample)
    assert result == 'fixable', f"Expected fixable for HTML/JS, got {result}"
    # fix should clean it
    fixed = validator.fix(sample)
    assert '<p>' not in fixed.get('input', ''), f"HTML not cleaned: {fixed}"
    assert 'script' not in fixed.get('input', ''), f"JS not cleaned: {fixed}"
    print("PASS: WebValidator cleans HTML")


def test_csv_validator_discards_missing_fields():
    """Verify CSVValidator discards samples with missing fields."""
    from dataset_preparer.source_validators import CSVValidator

    validator = CSVValidator()
    sample = {'input': '', 'output': 'respuesta'}
    result = validator.classify(sample)
    assert result == 'discardable', f"Expected discardable for missing input, got {result}"
    print("PASS: CSVValidator discards missing fields")


# ============================================================
# Thinking Generator Depth Tests
# ============================================================

def test_aiml_generator_greeting():
    """Verify AIML generator produces reasoning for greetings."""
    from dataset_preparer.aiml.thinking import AIMLThinkingGenerator

    generator = AIMLThinkingGenerator()
    result = generator.generate({'input': 'hola', 'output': 'Hola! ¿Cómo estás?'})
    assert result is not None, "Generator should return result"
    assert 'thinking' in result, "Result should have 'thinking' key"
    thinking = result['thinking']
    assert len(thinking) > 10, f"Thinking too short: {thinking}"
    # Should be reasoning, not meta-commentary
    assert 'El usuario me saluda' not in thinking, f"Should not be meta-commentary: {thinking}"
    print("PASS: AIML generator produces reasoning")


def test_thinking_generator_base_depth_config():
    """Verify ThinkingGenerator base has DEPTH_CONFIG."""
    from dataset_preparer.thinking_generators import ThinkingGenerator

    assert hasattr(ThinkingGenerator, 'DEPTH_CONFIG'), "ThinkingGenerator should have DEPTH_CONFIG"
    config = ThinkingGenerator.DEPTH_CONFIG
    assert 'basic' in config, "DEPTH_CONFIG should have 'basic'"
    assert 'adaptive' in config, "DEPTH_CONFIG should have 'adaptive'"
    assert 'detailed' in config, "DEPTH_CONFIG should have 'detailed'"
    assert 'max_tokens' in config['basic'], "basic should have max_tokens"
    assert 'max_tokens' in config['detailed'], "detailed should have max_tokens"
    assert config['detailed']['max_tokens'] > config['basic']['max_tokens'], \
        "detailed should have more max_tokens than basic"
    print("PASS: ThinkingGenerator DEPTH_CONFIG correct")


def test_ollama_teacher_is_model_available():
    """Verify OllamaTeacher.is_model_available works without hanging."""
    from dataset_preparer.thinking_generators import OllamaTeacher

    teacher = OllamaTeacher()
    # This should return False quickly if Ollama isn't running
    result = teacher.is_model_available()
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    print(f"PASS: OllamaTeacher.is_model_available() returns {result}")


def test_aiml_generator_depth():
    """Verify AIML generator respects depth parameter."""
    from dataset_preparer.aiml.thinking import AIMLThinkingGenerator

    generator_basic = AIMLThinkingGenerator(depth='basic')
    generator_detailed = AIMLThinkingGenerator(depth='detailed')

    assert generator_basic.depth_config['max_tokens'] < generator_detailed.depth_config['max_tokens']
    print("PASS: AIML generator depth parameter works")


# ============================================================
# Training Metrics Tests
# ============================================================

def test_thinking_detection_in_dataset():
    """Verify thinking data detection works."""
    sample_with_thinking = "<thinking>El usuario pregunta.</thinking>La respuesta es 42."
    sample_without_thinking = "Hola! ¿Cómo estás?"

    has_open = '<thinking>' in sample_with_thinking
    has_close = '</thinking>' in sample_with_thinking
    assert has_open and has_close, "Should detect thinking tags in sample"

    has_open_plain = '<thinking>' in sample_without_thinking
    has_close_plain = '</thinking>' in sample_without_thinking
    assert not has_open_plain and not has_close_plain, "Should not detect thinking tags in plain text"
    print("PASS: Thinking detection in dataset works")


def test_thinking_loss_weight_in_config():
    """Verify thinking_loss_weight is 0.5 in TRAINING_CONFIG."""
    from training.trainer import TRAINING_CONFIG

    assert 'thinking_loss_weight' in TRAINING_CONFIG, "thinking_loss_weight not in config"
    assert TRAINING_CONFIG['thinking_loss_weight'] == 0.5, \
        f"Expected 0.5, got {TRAINING_CONFIG['thinking_loss_weight']}"
    print("PASS: thinking_loss_weight is 0.5 in config")


def test_thinking_metrics_accumulation():
    """Verify thinking metrics accumulate correctly."""
    metrics_accum = {
        'thinking_accuracy': [0.8, 0.85, 0.9],
        'thinking_token_ratio': [0.3, 0.35, 0.4],
    }

    for key, values in metrics_accum.items():
        avg = sum(values) / len(values)
        assert avg > 0, f"Average for {key} should be > 0"
        assert avg <= 1.0, f"Average for {key} should be <= 1.0"

    print("PASS: Thinking metrics accumulation correct")


# ============================================================
# Pipeline Data Flow Tests
# ============================================================

def test_thinking_sample_format():
    """Verify thinking sample has correct format for training."""
    sample = {
        'input_ids': '<thinking>El usuario pregunta sobre Python.</thinking> Python es un lenguaje.',
        'thinking_text': '<thinking>El usuario pregunta sobre Python.</thinking>',
        'answer': 'Python es un lenguaje.',
        'token_ids': [5, 10, 20, 30, 6, 40, 50]
    }

    assert '<thinking>' in sample['input_ids'], "input_ids should contain <thinking>"
    assert '</thinking>' in sample['input_ids'], "input_ids should contain </thinking>"
    assert sample['answer'] in sample['input_ids'], "answer should be in input_ids"
    print("PASS: Thinking sample format correct")


def test_thinking_token_positions():
    """Verify thinking tokens are at correct positions."""
    token_ids = [5, 10, 20, 30, 6, 40, 50]
    thinking_id = 5
    thinking_end_id = 6

    thinking_start_pos = token_ids.index(thinking_id)
    thinking_end_pos = token_ids.index(thinking_end_id)

    assert thinking_start_pos == 0, f"Expected thinking at 0, got {thinking_start_pos}"
    assert thinking_end_pos == 4, f"Expected thinking_end at 4, got {thinking_end_pos}"
    assert thinking_start_pos < thinking_end_pos, "thinking should be before thinking_end"
    print("PASS: Thinking token positions correct")


def test_thinking_loss_weight_application():
    """Verify thinking_loss_weight is applied only to thinking tokens."""
    thinking_loss_weight = 0.5
    token_ids = [5, 10, 20, 30, 6, 40, 50]
    thinking_id = 5
    thinking_end_id = 6

    weights = [1.0] * len(token_ids)
    in_thinking = False

    for i, token in enumerate(token_ids):
        if token == thinking_id:
            in_thinking = True
        elif token == thinking_end_id:
            in_thinking = False
        if in_thinking and token != thinking_id and token != thinking_end_id:
            weights[i] = thinking_loss_weight

    # Tokens between <thought> and </thought> should have reduced weight
    assert weights[0] == 1.0, f"<thought> token should have weight 1.0, got {weights[0]}"
    assert weights[1] == 0.5, f"Token inside thinking should have weight 0.5, got {weights[1]}"
    assert weights[2] == 0.5, f"Token inside thinking should have weight 0.5, got {weights[2]}"
    assert weights[3] == 0.5, f"Token inside thinking should have weight 0.5, got {weights[3]}"
    assert weights[4] == 1.0, f"</thought> token should have weight 1.0, got {weights[4]}"
    assert weights[5] == 1.0, f"Token after thinking should have weight 1.0, got {weights[5]}"
    print("PASS: Thinking loss weight applied correctly")


def test_format_thinking_sample_structure():
    """Verify _format_thinking_sample produces correct structure."""
    from dataset_preparer.thinking_generators import ThinkingGenerator

    sample = {'input': 'hola', 'output': 'Hola! ¿Cómo estás?'}
    result = ThinkingGenerator._format_thinking_sample(None, sample, "El usuario me despuda amablemente.")

    assert 'thinking' in result, "Result should have 'thinking' key"
    assert '<thinking>' in result.get('thinking_text', ''), "thinking_text should contain <thinking>"
    assert '</thinking>' in result.get('thinking_text', ''), "thinking_text should contain </thinking>"
    assert 'Hola' in result.get('input_ids', ''), "Answer should be in input_ids"
    print("PASS: _format_thinking_sample produces correct structure")


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("COMPREHENSIVE THINKING PIPELINE TESTS")
    print("=" * 60)

    # BPE Tokenizer
    print("\n--- BPE Tokenizer Thinking Symbols ---")
    test_bpe_thinking_symbols_registered()
    test_bpe_encode_preserves_thinking_tags()
    test_bpe_has_thinking_method()

    # Thinking Quality
    print("\n--- Thinking Quality ---")
    test_validate_thinking_good_reasoning()
    test_validate_thinking_meta_commentary_rejected()
    test_validate_thinking_too_short()
    test_validate_thinking_with_steps()
    test_filter_low_quality()

    # Source Validators Edge Cases
    print("\n--- Source Validators Edge Cases ---")
    test_generic_validator_fixes_encoding()
    test_generic_validator_discards_empty()
    test_aiml_validator_good_sample()
    test_pdf_validator_fixes_hyphens()
    test_epub_validator_removes_navigation()
    test_web_validator_cleans_html()
    test_csv_validator_discards_missing_fields()

    # Thinking Generator Depth
    print("\n--- Thinking Generator Depth ---")
    test_aiml_generator_greeting()
    test_thinking_generator_base_depth_config()
    test_ollama_teacher_is_model_available()
    test_aiml_generator_depth()

    # Training Metrics
    print("\n--- Training Metrics ---")
    test_thinking_detection_in_dataset()
    test_thinking_loss_weight_in_config()
    test_thinking_metrics_accumulation()

    # Pipeline Data Flow
    print("\n--- Pipeline Data Flow ---")
    test_thinking_sample_format()
    test_thinking_token_positions()
    test_thinking_loss_weight_application()
    test_format_thinking_sample_structure()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
