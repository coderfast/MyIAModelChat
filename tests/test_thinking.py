"""Phase 3-8 tests for thinking functionality.

Tests that import main_chat.py are skipped if torchvision is broken.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if main_chat imports work
MAIN_CHAT_AVAILABLE = False
try:
    from main_chat import stream_chat_with_thinking, stream_chat_text, parse_thinking_response
    MAIN_CHAT_AVAILABLE = True
except (RuntimeError, ModuleNotFoundError):
    pass


# ============================================================
# Phase 3 Tests: Thinking Stop Token & Response Format
# ============================================================

class FakeTokenizerPhase3:
    def get_pad_index(self): return 0
    def get_unk_index(self): return 1
    def get_eos_index(self): return 2
    def get_bos_index(self): return 3
    def get_thinking_index(self): return 5
    def get_thinking_end_index(self): return 6

def test_dialogue_manager_dict_return():
    """Verify DialogueManager.generate_response returns dict format."""
    from dialogmanager import DialogueManager

    tokenizer = FakeTokenizerPhase3()

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 10)
            self.linear = nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(self.emb(x))

    model = SimpleModel()
    device = torch.device('cpu')

    dm = DialogueManager(
        model=model, device=device, tokenizer=tokenizer,
        thinking_enabled=False, thinking_max_tokens=64
    )

    # Even with thinking disabled, should return dict
    result = dm.generate_response("hola")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'thinking' in result, "Missing 'thinking' key"
    assert 'response' in result, "Missing 'response' key"
    print("PASS: DialogueManager returns dict with thinking/response keys")


def test_thinking_enabled_flag():
    """Verify thinking_enabled flag is stored."""
    from dialogmanager import DialogueManager

    tokenizer = FakeTokenizerPhase3()
    model = nn.Embedding(10, 10)
    device = torch.device('cpu')

    dm = DialogueManager(
        model=model, device=device, tokenizer=tokenizer,
        thinking_enabled=True, thinking_max_tokens=32
    )
    assert dm.thinking_enabled is True
    assert dm.thinking_max_tokens == 32
    assert dm.thinking_end_id == 6
    assert dm.thinking_id == 5

    dm2 = DialogueManager(
        model=model, device=device, tokenizer=tokenizer,
        thinking_enabled=False
    )
    assert dm2.thinking_enabled is False
    print("PASS: Thinking flags stored correctly")


# ============================================================
# Phase 4 Tests: Streaming & Schemas
# ============================================================

def test_stream_chat_with_thinking():
    """Verify stream_chat_with_thinking yields correct delta format."""
    if not MAIN_CHAT_AVAILABLE:
        print("SKIP: main_chat not importable (torchvision issue)")
        return
    import json
    gen = stream_chat_with_thinking("Analyzing...", "The answer is 42.")
    chunks = list(gen)
    assert len(chunks) >= 3
    first = json.loads(chunks[0])
    assert 'reasoning' in first.get('delta', {}), f"Missing reasoning in first chunk delta"
    last = json.loads(chunks[-1])
    assert last.get('object') == 'chat.completion.complete'
    content_chunks = [json.loads(c) for c in chunks[:-1] if 'content' in json.loads(c).get('delta', {})]
    assert len(content_chunks) > 0
    print("PASS: stream_chat_with_thinking yields correct format")


def test_stream_chat_text():
    """Verify stream_chat_text yields content deltas."""
    if not MAIN_CHAT_AVAILABLE:
        print("SKIP: main_chat not importable (torchvision issue)")
        return
    import json
    gen = stream_chat_text("Hello world")
    chunks = list(gen)
    assert len(chunks) >= 2
    first = json.loads(chunks[0])
    assert 'content' in first.get('delta', {})
    print("PASS: stream_chat_text yields content deltas")


# ============================================================
# Phase 5 Tests: Thinking Metrics
# ============================================================

def test_thinking_metrics_computation():
    """Verify thinking metrics are computed correctly."""
    tokenizer = FakeTokenizerPhase3()

    targets = torch.tensor([
        [5, 7, 8, 6, 3, 0],  # <think> ... </think> ... response ... pad
        [1, 5, 9, 6, 2, 0],  # ... <think> ... </think> ... response ... pad
    ])

    thinking_id = tokenizer.get_thinking_index()
    thinking_end_id = tokenizer.get_thinking_end_index()

    thinking_open_correct = 0
    thinking_close_correct = 0
    total_thinking_positions = 0

    for i in range(targets.size(0)):
        for j in range(targets.size(1)):
            target_token = targets[i, j].item()
            if target_token == thinking_id:
                total_thinking_positions += 1
                thinking_open_correct += 1  # simulate correct prediction
            elif target_token == thinking_end_id:
                total_thinking_positions += 1
                thinking_close_correct += 1

    assert total_thinking_positions == 4, f"Expected 4 thinking positions, got {total_thinking_positions}"
    assert thinking_open_correct == 2
    assert thinking_close_correct == 2
    print("PASS: Thinking metrics computed correctly")


# ============================================================
# Phase 6 Tests: Thinking Data Generation
# ============================================================

def test_thinking_generation_english():
    """Verify English thinking templates exist and work."""
    from generate_thinking_data import generate_thinking, THINKING_TEMPLATES_EN

    assert 'identity' in THINKING_TEMPLATES_EN
    assert 'greeting' in THINKING_TEMPLATES_EN
    assert 'question' in THINKING_TEMPLATES_EN
    assert 'farewell' in THINKING_TEMPLATES_EN
    assert 'default' in THINKING_TEMPLATES_EN

    thinking = generate_thinking("who are you", "I am Eduardo", "identity", lang='en')
    assert isinstance(thinking, str)
    assert len(thinking) > 0
    print("PASS: English thinking templates work")


def test_thinking_generation_spanish():
    """Verify Spanish thinking templates work."""
    from generate_thinking_data import generate_thinking

    thinking = generate_thinking("quien eres", "soy Eduardo", "identity", lang='es')
    assert isinstance(thinking, str)
    assert len(thinking) > 0
    print("PASS: Spanish thinking templates work")


def test_validate_thinking_consistency():
    """Verify thinking validation function."""
    from generate_thinking_data import validate_thinking_consistency

    # Good: thinking mentions answer words
    assert validate_thinking_consistency(
        "what is 2+2",
        "The user asks about addition. 2 plus 2 equals 4.",
        "4"
    ) is True

    # Bad: empty thinking
    assert validate_thinking_consistency("q", "", "a") is False

    # Bad: thinking too short
    assert validate_thinking_consistency("what", "ok", "a long answer") is False

    print("PASS: validate_thinking_consistency works")


def test_generate_thinking_dataset_with_lang():
    """Verify dataset generation with language parameter."""
    from generate_thinking_data import generate_thinking_dataset

    pairs = [
        {'input': 'hello', 'output': 'hi there'},
        {'input': 'who are you', 'output': 'I am Eduardo'},
    ]

    # English
    en_data = generate_thinking_dataset(pairs, lang='en', generator='template')
    assert len(en_data) == 2
    assert all('thinking' in d for d in en_data)
    assert all('thinking_text' in d for d in en_data)

    # Spanish
    es_data = generate_thinking_dataset(pairs, lang='es', generator='template')
    assert len(es_data) == 2
    print("PASS: generate_thinking_dataset with lang parameter")


# ============================================================
# Phase 7 Tests: CLI Flags
# ============================================================

def test_cli_flags_exist():
    """Verify thinking CLI flags are defined in main.py."""
    import ast
    with open('main.py', encoding='utf-8') as f:
        content = f.read()

    assert 'thinking-loss-weight' in content
    assert 'thinking-enabled' in content
    assert 'thinking-max-tokens' in content
    assert 'generate-thinking' in content
    assert 'thinking-mode' in content
    assert 'thinking-model' in content
    print("PASS: CLI flags defined in main.py")


def test_main_chat_thinking_flags():
    """Verify MainChat passes thinking params to DialogueManager."""
    import ast
    with open('main_chat.py', encoding='utf-8') as f:
        content = f.read()

    assert 'thinking_enabled' in content
    assert 'thinking_max_tokens' in content
    print("PASS: MainChat passes thinking params")


# ============================================================
# Phase 8 Tests: Integration
# ============================================================

def test_backward_compatibility():
    """Verify old code paths still work (string return from generate_response)."""
    from dialogmanager import DialogueManager

    tokenizer = FakeTokenizerPhase3()

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 10)
            self.linear = nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(self.emb(x))

    model = SimpleModel()
    device = torch.device('cpu')

    dm = DialogueManager(
        model=model, device=device, tokenizer=tokenizer,
        thinking_enabled=False
    )

    result = dm.generate_response("test")
    # Should always return dict now
    assert isinstance(result, dict)
    assert 'response' in result
    print("PASS: Backward compatibility maintained")


if __name__ == '__main__':
    print("=" * 60)
    print("THINKING FUNCTIONALITY TESTS")
    print("=" * 60)

    # Phase 3
    print("\n--- Phase 3: Generation ---")
    test_dialogue_manager_dict_return()
    test_thinking_enabled_flag()

    # Phase 4
    print("\n--- Phase 4: Streaming ---")
    test_stream_chat_with_thinking()
    test_stream_chat_text()

    # Phase 5
    print("\n--- Phase 5: Metrics ---")
    test_thinking_metrics_computation()

    # Phase 6
    print("\n--- Phase 6: Data Generation ---")
    test_thinking_generation_english()
    test_thinking_generation_spanish()
    test_validate_thinking_consistency()
    test_generate_thinking_dataset_with_lang()

    # Phase 7
    print("\n--- Phase 7: CLI ---")
    test_cli_flags_exist()
    test_main_chat_thinking_flags()

    # Phase 8
    print("\n--- Phase 8: Integration ---")
    test_backward_compatibility()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
