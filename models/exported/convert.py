#!/usr/bin/env python3
"""
Convert HuggingFace GPT-2 format to GGUF.

Usage:
    python convert.py <hf_dir> --outfile <output.gguf> --outtype q8_0

Requires: pip install gguf torch numpy
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType, quantize

# Import our custom K-quant implementations
try:
    from quantize import quantize_k_quant, QUANTIZERS
    HAS_KQUANT = True
except ImportError:
    HAS_KQUANT = False


# Map string names to GGMLQuantizationType
QUANT_TYPES = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q8_0": GGMLQuantizationType.Q8_0,
}

# K-quant types (implemented in quantize.py)
KQUANT_TYPES = {"q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
                "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq2_s",
                "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"}

# Map K-quant string to GGMLQuantizationType enum
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

# Approximate bytes per element for each quant type (for size estimation)
QUANT_BYTES_PER_ELEM = {
    "f32": 4, "f16": 2, "bf16": 2,
    "q4_0": 0.5625, "q4_1": 0.625,
    "q5_0": 0.6875, "q5_1": 0.75,
    "q8_0": 1.0625, "q8_1": 1.25,
    "q2_k": 2.625, "q3_k": 3.4375,
    "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625, "q8_k": 9.125,
}


def load_config(hf_dir):
    config_path = os.path.join(hf_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_weights(hf_dir):
    bin_path = os.path.join(hf_dir, "pytorch_model.bin")
    return torch.load(bin_path, map_location="cpu", weights_only=True)


def pad_to_block(arr, block_size):
    """Pad array to multiple of block_size."""
    n = len(arr)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    if padded_n == n:
        return arr
    padded = np.zeros(padded_n, dtype=arr.dtype)
    padded[:n] = arr
    return padded


def convert_to_gguf(hf_dir, output_path, outtype="q8_0"):
    print(f"Loading config from {hf_dir}...")
    config = load_config(hf_dir)

    print(f"Loading weights...")
    state_dict = load_weights(hf_dir)

    vocab_size = config.get("vocab_size", 8000)
    n_embd = config.get("n_embd", 256)
    n_layer = config.get("n_layer", 4)
    n_head = config.get("n_head", 4)
    n_positions = config.get("n_positions", 512)

    # Check if it's a K-quant type
    is_kquant = outtype in KQUANT_TYPES

    if is_kquant:
        if not HAS_KQUANT:
            print(f"Error: K-quant type '{outtype}' requires quantize.py module")
            print(f"Make sure quantize.py is in the same directory as convert.py")
            sys.exit(1)
        qtype_name = outtype.upper()
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

    # Add metadata
    writer.add_name("chat_model")
    writer.add_context_length(n_positions)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_feed_forward_length(n_embd * 4)
    writer.add_vocab_size(vocab_size)

    # Convert and write tensors
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
            # Use our custom K-quant implementation
            quantized, n_elements, n_blocks = quantize_k_quant(weights, outtype)
            kquant_dtype = KQUANT_ENUM_MAP[outtype]
            writer.add_tensor(gguf_name, quantized, raw_dtype=kquant_dtype)
            total_quant_bytes += quantized.nbytes
        else:
            # Quantize using gguf.quantize
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

    args = parser.parse_args()

    if not os.path.isdir(args.hf_dir):
        print(f"Error: Directory not found: {args.hf_dir}")
        sys.exit(1)

    config_file = os.path.join(args.hf_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"Error: config.json not found in {args.hf_dir}")
        sys.exit(1)

    convert_to_gguf(args.hf_dir, args.outfile, args.outtype)


if __name__ == "__main__":
    main()
