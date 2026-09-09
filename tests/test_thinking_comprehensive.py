"""Comprehensive tests for thinking pipeline integration.

Tests cover: BPE thinking symbols, thinking quality, source validators edge cases,
thinking generator depth, training metrics, and pipeline data flow.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATASET_CACHE = os.path.join(os.path.dirname(__file__), '..', 'dataset_cache')


# ============================================================
# BPE Tokenizer Thinking Symbols Tests
# ============================================================

def test_bpe_thinking_symbols_registered():
    """Verify <|problem|>, <|thinking|>, <|final|> are in the BPE vocabulary."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    problem_id = wrapper.get_problem_index()
    thinking_mode_id = wrapper.get_thinking_mode_index()
    final_id = wrapper.get_final_index()

    assert problem_id >= 0, f"problem_id should be >= 0, got {problem_id}"
    assert thinking_mode_id >= 0, f"thinking_mode_id should be >= 0, got {thinking_mode_id}"
    assert final_id >= 0, f"final_id should be >= 0, got {final_id}"

    # Verify they decode to the correct strings
    decoded_problem = wrapper.sp.id_to_piece(problem_id)
    assert '<|problem|>' in decoded_problem, f"Unexpected token: {decoded_problem}"
    decoded_thinking = wrapper.sp.id_to_piece(thinking_mode_id)
    assert '<|thinking|>' in decoded_thinking, f"Unexpected token: {decoded_thinking}"
    decoded_final = wrapper.sp.id_to_piece(final_id)
    assert '<|final|>' in decoded_final, f"Unexpected token: {decoded_final}"
    print("PASS: BPE GPT-2 standard symbols registered and working")


def test_bpe_encode_preserves_thinking_tags():
    """Verify BPE encode preserves <|problem|>, <|thinking|>, <|final|> tokens."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    text_with_thinking = "<|problem|>Pregunta<|thinking|>El usuario pregunta sobre Python. Python es un lenguaje.<|final|>La respuesta es 42."
    encoded = wrapper.encode(text_with_thinking)

    assert len(encoded) > 0, "Encoded should not be empty"

    # Check that problem and final tokens are present (GPT-2 standard)
    problem_id = wrapper.get_problem_index()
    final_id = wrapper.get_final_index()
    assert problem_id in encoded, f"<|problem|> token not found in encoded: {encoded}"
    assert final_id in encoded, f"<|final|> token not found in encoded: {encoded}"

    # Verify ordering: <|problem|> before <|final|>
    problem_pos = encoded.index(problem_id)
    final_pos = encoded.index(final_id)
    assert problem_pos < final_pos, f"<|problem|> at {problem_pos} should be before <|final|> at {final_pos}"
    print("PASS: BPE encode preserves GPT-2 standard tokens")


def test_bpe_has_thinking_method():
    """Verify has_thinking and split_thinking methods work."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    text_with = "<|thinking|>Razoning aquí<|final|>La respuesta"
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
# Training Metrics Tests (testing real project code)
# ============================================================

def test_thinking_detection_in_dataset():
    """Verify has_thinking method detects thinking tags correctly."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    sample_with = "<|thinking|>El usuario pregunta.<|final|>La respuesta es 42."
    sample_without = "Hola! ¿Cómo estás?"

    assert wrapper.has_thinking(sample_with) is True, "Should detect thinking tags"
    assert wrapper.has_thinking(sample_without) is False, "Should not detect thinking tags in plain text"

    thinking, response = wrapper.split_thinking(sample_with)
    assert 'El usuario pregunta' in thinking
    assert 'La respuesta es 42' in response
    print("PASS: Thinking detection works correctly")


def test_thinking_loss_weight_in_config():
    """Verify TrainingConfig defaults to thinking_loss_weight=1.0."""
    from training.trainer import TrainingConfig

    config = TrainingConfig()
    assert config.thinking_loss_weight == 1.0, \
        f"Expected 1.0, got {config.thinking_loss_weight}"
    print("PASS: thinking_loss_weight is 1.0 in config")


def test_thinking_metrics_accumulation():
    """Verify thinking metrics are computed from real tokenizer output."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)

    # Encode a text with GPT-2 standard tokens
    text = "<|problem|>Pregunta<|thinking|>Análizo la pregunta.<|final|>La respuesta es 42."
    token_ids = wrapper.encode(text)

    problem_id = wrapper.get_problem_index()
    thinking_mode_id = wrapper.get_thinking_mode_index()
    final_id = wrapper.get_final_index()

    # Count GPT-2 standard token positions
    gpt2_positions = sum(1 for t in token_ids if t in (problem_id, thinking_mode_id, final_id))
    assert gpt2_positions == 3, f"Expected 3 GPT-2 standard positions, got {gpt2_positions}"

    # Verify ordering
    first_problem = token_ids.index(problem_id)
    first_thinking = token_ids.index(thinking_mode_id)
    first_final = token_ids.index(final_id)
    assert first_problem < first_thinking < first_final, "Order should be problem -> thinking -> final"
    print("PASS: Thinking metrics computed correctly")


# ============================================================
# Pipeline Data Flow Tests
# ============================================================

def test_thinking_sample_format():
    """Verify ThinkingGenerator._format_thinking_sample produces valid structure."""
    from dataset_preparer.thinking_generators import ThinkingGenerator

    sample = {'input': 'hola', 'output': 'Hola! ¿Cómo estás?'}
    result = ThinkingGenerator._format_thinking_sample(None, sample, "El usuario me despuda amablemente.")

    assert 'thinking' in result, "Result should have 'thinking' key"
    assert '<|thinking|>' in result.get('thinking_text', ''), "thinking_text should contain <|thinking|>"
    assert '<|final|>' in result.get('thinking_text', ''), "thinking_text should contain <|final|>"
    assert 'Hola' in result.get('input_ids', ''), "Answer should be in input_ids"
    print("PASS: Thinking sample format correct")


def test_thinking_token_positions():
    """Verify tokenizer correctly identifies GPT-2 standard token positions."""
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

    model_path = os.path.join(DATASET_CACHE, 'sentencepiece.model')
    if not os.path.exists(model_path):
        print("SKIP: sentencepiece.model not found")
        return

    wrapper = SentencePieceTokenizerWrapper(model_path)
    text = "<|problem|>Pregunta<|thinking|>Razonamiento aquí<|final|>La respuesta es 42."
    token_ids = wrapper.encode(text)

    problem_id = wrapper.get_problem_index()
    thinking_mode_id = wrapper.get_thinking_mode_index()
    final_id = wrapper.get_final_index()

    assert problem_id in token_ids, f"<|problem|> token not found in {token_ids}"
    assert thinking_mode_id in token_ids, f"<|thinking|> token not found in {token_ids}"
    assert final_id in token_ids, f"<|final|> token not found in {token_ids}"

    problem_pos = token_ids.index(problem_id)
    thinking_pos = token_ids.index(thinking_mode_id)
    final_pos = token_ids.index(final_id)
    assert problem_pos < thinking_pos < final_pos, f"Order should be <|problem|> ({problem_pos}) < <|thinking|> ({thinking_pos}) < <|final|> ({final_pos})"
    print("PASS: GPT-2 standard token positions correct")


def test_thinking_loss_weight_application():
    """Verify thinking_loss_weight affects training config."""
    from training.trainer import TrainingConfig

    config_default = TrainingConfig()
    config_custom = TrainingConfig(thinking_loss_weight=0.5)

    assert config_default.thinking_loss_weight == 1.0, "Default should be 1.0"
    assert config_custom.thinking_loss_weight == 0.5, "Custom should be 0.5"
    print("PASS: Thinking loss weight applied correctly")


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

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
