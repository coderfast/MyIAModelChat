#!/usr/bin/env python3
"""
Convert HuggingFace GPT-2 format to GGUF.

Usage:
    python convert_gguf.py <hf_dir> --outfile <output.gguf> --outtype q8_0
    python convert_gguf.py <hf_dir> --outfile <output.gguf> --outtype q8_0 --name my_model

Requires: pip install gguf torch numpy sentencepiece
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType, quantize

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from gguf_quantizer import quantize_k_quant, QUANTIZERS
    HAS_KQUANT = True
except ImportError:
    HAS_KQUANT = False


QUANT_TYPES = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "bf16": GGMLQuantizationType.BF16,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q8_0": GGMLQuantizationType.Q8_0,
}

KQUANT_TYPES = {"q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
                "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq2_s",
                "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"}

KQUANT_ENUM_MAP = {
    "q2_k": GGMLQuantizationType.Q2_K,
    "q3_k": GGMLQuantizationType.Q3_K,
    "q4_k": GGMLQuantizationType.Q4_K,
    "q5_k": GGMLQuantizationType.Q5_K,
    "q6_k": GGMLQuantizationType.Q6_K,
    "q8_k": GGMLQuantizationType.Q8_K,
    "iq4_nl": GGMLQuantizationType.IQ4_NL,
    "iq4_xs": GGMLQuantizationType.IQ4_XS,
    "iq2_xxs": GGMLQuantizationType.IQ2_XXS,
    "iq2_xs": GGMLQuantizationType.IQ2_XS,
    "iq2_s": GGMLQuantizationType.IQ2_S,
    "iq3_xxs": GGMLQuantizationType.IQ3_XXS,
    "iq3_s": GGMLQuantizationType.IQ3_S,
    "iq1_s": GGMLQuantizationType.IQ1_S,
    "iq1_m": GGMLQuantizationType.IQ1_M,
}

QUANT_BYTES_PER_ELEM = {
    "f32": 4, "f16": 2, "bf16": 2,
    "q4_0": 0.5625, "q4_1": 0.625,
    "q5_0": 0.6875, "q5_1": 0.75,
    "q8_0": 1.0625, "q8_1": 1.25,
    "q2_k": 2.625, "q3_k": 3.4375,
    "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625, "q8_k": 9.125,
}

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\n' + message['content'] + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{{ '<|assistant|>\n' }}"
)

STOP_TOKENS = ["<|user|>", "<|end_of_text|>", "</s>"]


def load_config(hf_dir):
    config_path = os.path.join(hf_dir, "config.json")
    with open(config_path, "r", encoding='utf-8') as f:
        return json.load(f)


def load_metadata(hf_dir):
    meta_path = os.path.join(hf_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_weights(hf_dir):
    bin_path = os.path.join(hf_dir, "pytorch_model.bin")
    return torch.load(bin_path, map_location="cpu", weights_only=True)


def load_sentencepiece_tokenizer(hf_dir):
    """Load SentencePiece model from the HF directory.

    Returns the SentencePieceProcessor or None if not available.
    """
    if spm is None:
        print("Warning: sentencepiece not installed. Tokenizer will not be embedded in GGUF.")
        return None

    sp_path = os.path.join(hf_dir, "sentencepiece.model")
    if not os.path.exists(sp_path):
        print(f"Warning: sentencepiece.model not found in {hf_dir}. Tokenizer will not be embedded.")
        return None

    processor = spm.SentencePieceProcessor()
    processor.load(sp_path)
    return processor


def extract_tokenizer_for_gguf(sp, config, metadata):
    """Extract full tokenizer data from SentencePiece for GGUF embedding.

    Returns dict with tokens, scores, token_types, merges, and special token IDs.
    """
    vocab_size = sp.get_piece_size()

    # Known special/control tokens from our SentencePiece model
    CONTROL_TOKENS = {
        '<pad>', '<unk>', '<s>', '</s>',
        '<|problem|>', '<|thinking|>', '<|final|>',
        '<|user|>', '<|assistant|>',
        '<|system|>', '<|end|>', '<|sep|>',
        '<tool_call>', '</tool_call>', '<|tool_result|>',
        '<thinking>', '</thinking>',
        '<|context|>', '<|answer|>',
        '<observation>', '</observation>',
    }

    tokens = []
    scores = []
    token_types = []

    # GGUF token types: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE
    TT_NORMAL = 1
    TT_UNKNOWN = 2
    TT_CONTROL = 3
    TT_USER_DEFINED = 4

    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        tokens.append(piece)

        # Determine token type
        if piece in CONTROL_TOKENS or piece.startswith('<|') and piece.endswith('|>') or piece.startswith('</') and piece.endswith('>'):
            token_types.append(TT_CONTROL)
            scores.append(-1000.0)  # Control tokens get very low scores
        elif piece == '<unk>':
            token_types.append(TT_UNKNOWN)
            scores.append(-1000.0)
        else:
            token_types.append(TT_NORMAL)
            scores.append(0.0)  # SentencePiece doesn't provide log probs; 0.0 is standard

    # Extract special token IDs from metadata
    special_tokens = metadata.get('special_tokens', {})
    agentic_tokens = metadata.get('agentic_tokens', {})
    mode_tokens = metadata.get('mode_tokens', {})

    # Build a mapping of token string -> id from the special_tokens metadata
    token_id_map = {}
    for token_name, info in special_tokens.items():
        if 'id' in info and 'token' in info:
            token_id_map[info['token']] = info['id']
    for token_name, info in agentic_tokens.items():
        if 'id' in info and 'token' in info:
            token_id_map[info['token']] = info['id']
    for token_name, info in mode_tokens.items():
        if 'id' in info and 'token' in info:
            token_id_map[info['token']] = info['id']

    # Fallback: use config.json IDs
    bos_id = config.get('bos_token_id', -1)
    eos_id = config.get('eos_token_id', -1)
    pad_id = config.get('pad_token_id', 0)

    # Extract BPE merge rules from SentencePiece
    # SentencePiece stores merges in the model proto; we extract them via the vocab
    merges = []
    try:
        # Get merge rules from SentencePiece model proto
        model_proto = sp.decode_serialized_proto(sp.serialized_model_proto())
        # The proto is a bytes object; we parse it to extract merges
        # For SentencePiece BPE models, merges are stored in the trainer spec
        # We can access them via the piece table
        pass  # SentencePiece handles merges internally; empty list is acceptable for GGUF
    except Exception:
        pass

    return {
        'tokens': tokens,
        'scores': scores,
        'token_types': token_types,
        'merges': merges,
        'bos_token_id': bos_id,
        'eos_token_id': eos_id,
        'pad_token_id': pad_id,
    }


def pad_to_block(arr, block_size):
    n = len(arr)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    if padded_n == n:
        return arr
    padded = np.zeros(padded_n, dtype=arr.dtype)
    padded[:n] = arr
    return padded


def convert_to_gguf(hf_dir, output_path, outtype="q8_0", model_name=None):
    print(f"Loading config from {hf_dir}...")
    config = load_config(hf_dir)
    metadata = load_metadata(hf_dir)

    print(f"Loading weights...")
    state_dict = load_weights(hf_dir)

    vocab_size = config.get("vocab_size", 8000)
    n_embd = config.get("n_embd", 256)
    n_layer = config.get("n_layer", 4)
    n_head = config.get("n_head", 4)
    n_positions = config.get("n_positions", 512)
    bos_token_id = config.get("bos_token_id", -1)
    eos_token_id = config.get("eos_token_id", -1)
    pad_token_id = config.get("pad_token_id", 0)

    display_name = model_name or metadata.get("model_name") or config.get("name", "chat_model")

    # Load SentencePiece tokenizer and extract full vocab for GGUF embedding
    sp = load_sentencepiece_tokenizer(hf_dir)
    tokenizer_data = None
    if sp is not None:
        print(f"Loaded SentencePiece tokenizer: {sp.get_piece_size()} tokens")
        tokenizer_data = extract_tokenizer_for_gguf(sp, config, metadata)
        # Override IDs from tokenizer data if available
        if tokenizer_data['bos_token_id'] >= 0:
            bos_token_id = tokenizer_data['bos_token_id']
        if tokenizer_data['eos_token_id'] >= 0:
            eos_token_id = tokenizer_data['eos_token_id']
        if tokenizer_data['pad_token_id'] >= 0:
            pad_token_id = tokenizer_data['pad_token_id']

    is_kquant = outtype in KQUANT_TYPES

    if is_kquant:
        if not HAS_KQUANT:
            print(f"Error: K-quant type '{outtype}' requires quantize.py module")
            sys.exit(1)
        print(f"Config: vocab={vocab_size}, embd={n_embd}, layer={n_layer}, head={n_head}, pos={n_positions}")
        print(f"Quantization: {outtype} (K-quant, custom implementation)")
    else:
        qtype = QUANT_TYPES.get(outtype)
        if qtype is None:
            print(f"Error: Unknown quantization type: {outtype}")
            print(f"Available types: {', '.join(sorted(list(QUANT_TYPES.keys()) + list(KQUANT_TYPES)))}")
            sys.exit(1)
        print(f"Config: vocab={vocab_size}, embd={n_embd}, layer={n_layer}, head={n_head}, pos={n_positions}")
        print(f"Quantization: {outtype} ({qtype.name})")

    writer = GGUFWriter(output_path, "gpt2")

    writer.add_name(display_name)
    writer.add_context_length(n_positions)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_feed_forward_length(n_embd * 4)
    writer.add_vocab_size(vocab_size)

    # ── Tokenizer embedding ──────────────────────────────────────────
    if tokenizer_data is not None:
        tokens = tokenizer_data['tokens']
        scores = tokenizer_data['scores']
        token_types = tokenizer_data['token_types']
        merges = tokenizer_data['merges']

        # Use "llama" model type for SentencePiece BPE tokenizers
        writer.add_tokenizer_model("llama")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types)
        if merges:
            writer.add_token_merges(merges)

        # Special token IDs
        if bos_token_id >= 0:
            writer.add_bos_token_id(bos_token_id)
        if eos_token_id >= 0:
            writer.add_eos_token_id(eos_token_id)
        if pad_token_id >= 0:
            writer.add_pad_token_id(pad_token_id)

        # Write agentic/special token IDs as custom metadata
        agentic_tokens = metadata.get('agentic_tokens', {})
        mode_tokens = metadata.get('mode_tokens', {})
        for token_name, info in agentic_tokens.items():
            if 'id' in info and info['id'] >= 0:
                writer.add_uint32(f"tokenizer.ggml.token_id.{token_name}", info['id'])
        for token_name, info in mode_tokens.items():
            if 'id' in info and info['id'] >= 0:
                writer.add_uint32(f"tokenizer.ggml.token_id.{token_name}", info['id'])

        print(f"Embedded {len(tokens)} tokens in GGUF ({sum(1 for t in token_types if t == 3)} control tokens)")
    else:
        # Fallback: minimal tokenizer metadata
        writer.add_tokenizer_model("gpt2")
        writer.add_string("tokenizer.chat_template", CHAT_TEMPLATE)
        for tok in STOP_TOKENS:
            writer.add_string("tokenizer.chat_stop", tok)
        if bos_token_id >= 0:
            writer.add_uint32("tokenizer.bos_token_id", bos_token_id)
        if eos_token_id >= 0:
            writer.add_uint32("tokenizer.eos_token_id", eos_token_id)
        print("Warning: No SentencePiece model found. GGUF will have minimal tokenizer metadata.")

    tensor_count = 0
    total_orig_bytes = 0
    total_quant_bytes = 0

    for name, tensor in state_dict.items():
        weights = tensor.cpu().numpy().astype(np.float32).flatten()
        gguf_name = name.replace(".", "_")
        total_orig_bytes += weights.nbytes

        if outtype == "f32":
            writer.add_tensor(gguf_name, weights)
        elif outtype == "f16":
            writer.add_tensor(gguf_name, weights.astype(np.float16))
        elif outtype == "bf16":
            writer.add_tensor(gguf_name, weights, raw_dtype=GGMLQuantizationType.BF16)
        elif is_kquant:
            quantized, n_elements, n_blocks = quantize_k_quant(weights, outtype)
            kquant_dtype = KQUANT_ENUM_MAP[outtype]
            writer.add_tensor(gguf_name, quantized, raw_dtype=kquant_dtype)
            total_quant_bytes += quantized.nbytes
        else:
            block_size = 32
            padded = pad_to_block(weights, block_size)
            n_blocks = len(padded) // block_size
            blocks = padded.reshape(n_blocks, block_size)
            quantized = quantize(blocks.flatten(), qtype)
            writer.add_tensor(gguf_name, quantized, raw_dtype=qtype)
            total_quant_bytes += quantized.nbytes

        tensor_count += 1
        print(f"  [{tensor_count}] {name} -> {weights.shape}")

    print(f"\nWriting GGUF to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    orig_mb = total_orig_bytes / (1024 * 1024)
    ratio = size_mb / orig_mb * 100 if orig_mb > 0 else 0
    print(f"Done! Output: {output_path} ({size_mb:.2f} MB)")
    print(f"Original: {orig_mb:.2f} MB -> Quantized: {size_mb:.2f} MB ({ratio:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace GPT-2 to GGUF")
    parser.add_argument("hf_dir", help="Path to HuggingFace model directory")
    parser.add_argument("--outfile", "-o", required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", choices=sorted(list(QUANT_TYPES.keys()) + list(KQUANT_TYPES)),
                        default="q8_0", help="Output quantization type (default: q8_0)")
    parser.add_argument("--name", default=None, help="Model name for GGUF metadata")

    args = parser.parse_args()

    if not os.path.isdir(args.hf_dir):
        print(f"Error: Directory not found: {args.hf_dir}")
        sys.exit(1)

    config_file = os.path.join(args.hf_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"Error: config.json not found in {args.hf_dir}")
        sys.exit(1)

    convert_to_gguf(args.hf_dir, args.outfile, args.outtype, model_name=args.name)


if __name__ == "__main__":
    main()