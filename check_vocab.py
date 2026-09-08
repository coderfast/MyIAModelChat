"""Check which special tokens are in the SentencePiece vocabulary."""
import os
from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

w = SentencePieceTokenizerWrapper(os.path.join('dataset_cache', 'sentencepiece.model'))

# Check which special tokens are in vocab
tokens = [
    '<|problem|>', '<|final|>',
    '<|thinking|>', '<tool_call>', '</tool_call>',
    '<|tool_result|>',
    '<|user|>', '<|assistant|>',
    '<|system|>', '<|end|>', '<|sep|>',
]
print("=== piece_to_id lookup ===")
for tok in tokens:
    try:
        idx = w.sp.piece_to_id(tok)
    except Exception:
        idx = -1
    print(f"  {tok:25s} -> {idx}")

print()
print("=== get_*_index() results ===")
print(f"  get_thinking_index()      = {w.get_thinking_index()}")
print(f"  get_thinking_end_index()  = {w.get_thinking_end_index()}")
print(f"  get_thinking_mode_index() = {w.get_thinking_mode_index()}")
print(f"  get_problem_index()       = {w.get_problem_index()}")
print(f"  get_final_index()         = {w.get_final_index()}")
print(f"  get_tool_call_index()     = {w.get_tool_call_index()}")
print(f"  get_tool_call_end_index() = {w.get_tool_call_end_index()}")
print(f"  get_tool_result_index()   = {w.get_tool_result_index()}")
print(f"  get_user_index()          = {w.get_user_index()}")
print(f"  get_assistant_index()     = {w.get_assistant_index()}")

print()
print(f"  idx2word[0] = {w.idx2word.get(0, '?')}")
print(f"  idx2word[1] = {w.idx2word.get(1, '?')}")

print()
print("=== Vocab scan for think/problem/final/tool tokens ===")
for i in range(w.vocab_size):
    word = w.idx2word.get(i, '')
    wl = word.lower()
    if any(kw in wl for kw in ['think', 'prob', 'final', 'tool']):
        print(f"  vocab[{i}] = {word}")

print()
print("=== Thinking_id == thinking_end_id? ===")
print(f"  thinking_id={w.get_thinking_index()}, thinking_end_id={w.get_thinking_end_index()}")
print(f"  Same? {w.get_thinking_index() == w.get_thinking_end_index()}")
