#!/usr/bin/env python3
"""
Convert HuggingFace model to ONNX with quantization options.

Usage:
    python convert_onnx.py <hf_dir> --outfile <output.onnx>
    python convert_onnx.py <hf_dir> --outfile <output.onnx> --quant-type int8
    python convert_onnx.py <hf_dir> --outfile <output.onnx> --quant-type fp8_e4m3fn --static

Requires: pip install torch onnxruntime numpy
"""
import argparse
import os
import sys

import numpy as np


def export_to_onnx(hf_dir: str, output_path: str, opset: int = 17, seq_len: int = 32) -> str:
    """Export HuggingFace model to ONNX."""
    import torch

    print(f"Loading model from {hf_dir}...")

    # Try project model first, fall back to HuggingFace
    model = None
    config = None
    try:
        from commons.model.chatmodel import ChatModel
        from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
        import json as json_mod
        cfg_path = os.path.join(hf_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg_data = json_mod.load(f)
            tokenizer_path = os.path.join(hf_dir, 'sentencepiece.model')
            if os.path.exists(tokenizer_path):
                tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
                model = ChatModel(tokenizer,
                    embed_size=cfg_data.get('n_embd', 256),
                    num_layers=cfg_data.get('n_layer', 4))
                # Try loading weights
                bin_path = os.path.join(hf_dir, 'pytorch_model.bin')
                if os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(state_dict, strict=False)
                model.eval()
                config = cfg_data
                print(f"Loaded project ChatModel: vocab={cfg_data.get('vocab_size')}, "
                      f"embd={cfg_data.get('n_embd')}, layers={cfg_data.get('n_layer')}")
    except Exception as e:
        print(f"  Could not load as project model: {e}")
        model = None

    if model is None:
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config.from_pretrained(hf_dir)
        model = GPT2LMHeadModel.from_pretrained(hf_dir, config=config)
        model.eval()
        print(f"Loaded HuggingFace GPT2LMHeadModel: vocab={config.vocab_size}, "
              f"embd={config.n_embd}, layers={config.n_layer}, heads={config.n_head}")

    vocab_size = config.get('vocab_size', 8000) if isinstance(config, dict) else config.vocab_size
    dummy_input = torch.randint(0, vocab_size, (1, seq_len))

    print(f"Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX exported: {output_path} ({size_mb:.2f} MB)")
    return output_path


def quantize_onnx_model(
    model_path: str,
    output_path: str,
    quant_type: str = "int8",
    static: bool = False,
    per_channel: bool = False,
    calibration_samples: int = 100,
    seq_len: int = 32,
    hf_dir: str = None,
    block_size: int = 128,
) -> str:
    """Quantize ONNX model using onnx_quantizer module."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from onnx_quantizer import (
        quantize_dynamic_int8, quantize_dynamic_uint8,
        quantize_dynamic_int4, quantize_dynamic_uint4,
        quantize_static_int8_qdq, quantize_static_int8_qoperator,
        quantize_static_int4_qoperator, quantize_static_int4_qdq,
        quantize_static_fp8_e4m3fn, quantize_static_fp8_e5m2,
        quantize_static_fp8_mixed, quantize_static_fp8_e4m3fnuz,
        quantize_static_fp8_e5m2fnuz,
        quantize_per_channel_int8, quantize_per_channel_int4,
        quantize_mixed_int8_int4, quantize_mixed_fp8_int8,
        quantize_tensor_overrides,
        TokenCalibrationReader,
    )

    if not static:
        # Dynamic quantization — no calibration needed
        dynamic_map = {
            "int8": quantize_dynamic_int8,
            "uint8": quantize_dynamic_uint8,
            "int4": quantize_dynamic_int4,
            "uint4": quantize_dynamic_uint4,
        }
        if quant_type not in dynamic_map:
            raise ValueError(f"Dynamic quant_type must be one of: {list(dynamic_map.keys())}")
        return dynamic_map[quant_type](model_path, output_path)

    # Static quantization — need calibration data
    if hf_dir is None:
        raise ValueError("Static quantization requires --hf-dir for tokenizer")

    print(f"Loading tokenizer from {hf_dir}...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(hf_dir)

    print(f"Generating {calibration_samples} calibration samples...")
    calibration_texts = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I need help with my computer.",
        "What is machine learning?",
        "Tell me about Python programming.",
        "How do I train a neural network?",
        "What is the capital of France?",
        "Explain quantum computing simply.",
        "Write a function in Python.",
        "What are the benefits of exercise?",
    ] * (calibration_samples // 10 + 1)
    calibration_texts = calibration_texts[:calibration_samples]

    reader = TokenCalibrationReader(tokenizer, calibration_texts, seq_len=seq_len)

    # Static quantization map
    static_map = {
        "static_int8_qdq": lambda m, o: quantize_static_int8_qdq(m, o, reader, per_channel=per_channel),
        "static_int8_qoperator": lambda m, o: quantize_static_int8_qoperator(m, o, reader, per_channel=per_channel),
        "static_int4_qoperator": lambda m, o: quantize_static_int4_qoperator(m, o, reader, block_size=block_size),
        "static_int4_qdq": lambda m, o: quantize_static_int4_qdq(m, o, reader, block_size=block_size),
        "fp8_e4m3fn": lambda m, o: quantize_static_fp8_e4m3fn(m, o, reader),
        "fp8_e5m2": lambda m, o: quantize_static_fp8_e5m2(m, o, reader),
        "fp8_mixed": lambda m, o: quantize_static_fp8_mixed(m, o, reader),
        "fp8_e4m3fnuz": lambda m, o: quantize_static_fp8_e4m3fnuz(m, o, reader),
        "fp8_e5m2fnuz": lambda m, o: quantize_static_fp8_e5m2fnuz(m, o, reader),
        "per_channel_int8": lambda m, o: quantize_per_channel_int8(m, o, reader),
        "per_channel_int4": lambda m, o: quantize_per_channel_int4(m, o, reader),
        "mixed_int8_int4": lambda m, o: quantize_mixed_int8_int4(m, o, reader),
        "mixed_fp8_int8": lambda m, o: quantize_mixed_fp8_int8(m, o, reader),
        "tensor_overrides": lambda m, o: quantize_tensor_overrides(m, o, reader),
    }

    if quant_type not in static_map:
        valid = sorted(static_map.keys())
        raise ValueError(f"quant_type must be one of: {valid}")

    return static_map[quant_type](model_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to ONNX with quantization")
    parser.add_argument("hf_dir", help="HuggingFace model directory")
    parser.add_argument("--outfile", "-o", required=True, help="Output ONNX file")

    # Quantization options
    quant_choices = [
        "int8", "uint8", "int4", "uint4",
        "static_int8_qdq", "static_int8_qoperator",
        "static_int4_qoperator", "static_int4_qdq",
        "fp8_e4m3fn", "fp8_e5m2", "fp8_mixed",
        "fp8_e4m3fnuz", "fp8_e5m2fnuz",
        "per_channel_int8", "per_channel_int4",
        "mixed_int8_int4", "mixed_fp8_int8",
        "tensor_overrides",
    ]
    parser.add_argument("--quant-type", default=None, choices=quant_choices,
                        help="Quantization type (default: no quantization)")
    parser.add_argument("--static", action="store_true",
                        help="Use static quantization (requires calibration)")
    parser.add_argument("--per-channel", action="store_true",
                        help="Use per-channel quantization")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size for INT4")
    parser.add_argument("--calibration-samples", type=int, default=100,
                        help="Number of calibration samples")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Sequence length for dummy input")

    args = parser.parse_args()

    if not os.path.isdir(args.hf_dir):
        print(f"Error: Directory not found: {args.hf_dir}")
        sys.exit(1)

    config_file = os.path.join(args.hf_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"Error: config.json not found in {args.hf_dir}")
        sys.exit(1)

    # Export to ONNX
    onnx_path = args.outfile
    if args.quant_type:
        # If quantizing, first export to temp ONNX
        temp_path = args.outfile + ".fp32.onnx"
        export_to_onnx(args.hf_dir, temp_path, opset=args.opset, seq_len=args.seq_len)

        # Then quantize
        quantize_onnx_model(
            temp_path, onnx_path,
            quant_type=args.quant_type,
            static=args.static,
            per_channel=args.per_channel,
            calibration_samples=args.calibration_samples,
            seq_len=args.seq_len,
            hf_dir=args.hf_dir,
            block_size=args.block_size,
        )

        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Removed temp: {temp_path}")
    else:
        export_to_onnx(args.hf_dir, onnx_path, opset=args.opset, seq_len=args.seq_len)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nDone! Output: {onnx_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
