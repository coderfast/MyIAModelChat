"""
Model export: converts .pth checkpoints to GGUF, ONNX, and quantized formats.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch

from commons.registry.model_registry import load_model_metadata, MODELS_DIR, EXPORTED_DIR

logger = logging.getLogger(__name__)


def export_to_onnx(pth_path: str, output_path: Optional[str] = None, seq_len: int = 512) -> str:
    """Export a .pth checkpoint to ONNX format."""
    try:
        from commons.model.chatmodel import ChatModel
    except ImportError:
        raise RuntimeError("Cannot import ChatModel. Ensure chatmodel.py is in the project root.")

    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    tokenizer = ckpt.get('tokenizer') if isinstance(ckpt, dict) else None

    # Strip module. prefix
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v

    if tokenizer is None:
        tokenizer_path = ckpt.get('tokenizer_path')
        if tokenizer_path and os.path.exists(tokenizer_path):
            from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
            tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
        else:
            raise RuntimeError("No tokenizer found in checkpoint. Cannot export to ONNX.")

    arch = ckpt.get('architecture', {}) if isinstance(ckpt, dict) else {}
    model = ChatModel(tokenizer,
        embed_size=arch.get('embed_size', 256),
        num_layers=arch.get('num_layers', 4))
    model.load_state_dict(new_state)
    model.eval()

    os.makedirs(EXPORTED_DIR, exist_ok=True)
    if output_path is None:
        stem = Path(pth_path).stem
        output_path = os.path.join(EXPORTED_DIR, f"{stem}.onnx")

    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, seq_len))

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'},
        },
        opset_version=14,
    )

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"ONNX exported: {output_path} ({size_mb} MB)")
    return output_path


def export_to_onnx_quantized(pth_path: str, output_path: Optional[str] = None, quant_type: str = 'int8') -> str:
    """Export a .pth checkpoint to quantized ONNX format."""
    # First export to regular ONNX
    onnx_path = export_to_onnx(pth_path)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise RuntimeError("onnxruntime is required for ONNX quantization. Install: pip install onnxruntime")

    if output_path is None:
        stem = Path(pth_path).stem
        output_path = os.path.join(EXPORTED_DIR, f"{stem}_{quant_type}.onnx")

    quant_type_map = {
        'int8': QuantType.QInt8,
        'uint8': QuantType.QUInt8,
    }
    qt = quant_type_map.get(quant_type, QuantType.QInt8)

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=qt,
    )

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"ONNX quantized ({quant_type}): {output_path} ({size_mb} MB)")
    return output_path


def export_to_gguf(pth_path: str, output_path: Optional[str] = None, quantization: str = 'q8_0') -> str:
    """Export a .pth checkpoint to GGUF format.

    This generates a HuggingFace-compatible directory that can be converted
    to GGUF using llama.cpp's convert.py script.
    """
    try:
        from commons.model.chatmodel import ChatModel
    except ImportError:
        raise RuntimeError("Cannot import ChatModel. Ensure chatmodel.py is in the project root.")

    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    tokenizer = ckpt.get('tokenizer') if isinstance(ckpt, dict) else None
    arch_meta = ckpt.get('architecture', {}) if isinstance(ckpt, dict) else {}

    # Strip module. prefix
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v

    if tokenizer is None:
        tokenizer_path = ckpt.get('tokenizer_path')
        if tokenizer_path and os.path.exists(tokenizer_path):
            from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
            tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
        else:
            raise RuntimeError("No tokenizer found in checkpoint. Cannot export to GGUF.")

    stem = Path(pth_path).stem
    os.makedirs(EXPORTED_DIR, exist_ok=True)

    # Save as HuggingFace-compatible format
    hf_dir = os.path.join(EXPORTED_DIR, f"{stem}_hf")
    os.makedirs(hf_dir, exist_ok=True)

    # Save model weights
    torch.save(new_state, os.path.join(hf_dir, "pytorch_model.bin"))

    # Save config.json
    vocab_size = arch_meta.get('vocab_size', tokenizer.vocab_size)
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": vocab_size,
        "n_positions": arch_meta.get('n_positions', 512),
        "n_embd": arch_meta.get('embed_size', 256),
        "n_layer": arch_meta.get('num_layers', 4),
        "n_head": arch_meta.get('n_head', 4),
        "activation_function": "gelu_new",
        "transformers_version": "4.0.0",
    }
    with open(os.path.join(hf_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Save tokenizer files
    tokenizer.save_vocabulary(os.path.join(hf_dir, "tokenizer_vocab.json"))

    # Generate conversion scripts for the user (with absolute paths)
    hf_dir_abs = os.path.abspath(hf_dir)
    exported_dir = os.path.abspath(EXPORTED_DIR)
    convert_py = os.path.join(exported_dir, 'convert.py')
    gguf_out = os.path.abspath(os.path.join(exported_dir, f"{stem}_{quantization}.gguf"))
    convert_cmd = f"python \"{convert_py}\" \"{hf_dir_abs}\" --outfile \"{gguf_out}\" --outtype {quantization.lower()}"
    convert_cmd_unix = convert_cmd.replace('\\', '/')

    # .sh for Linux/Mac
    sh_path = os.path.join(hf_dir, "convert_to_gguf.sh")
    with open(sh_path, 'w', newline='\n') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# Convert {stem} to GGUF\n")
        f.write(f"{convert_cmd_unix}\n")

    # .bat for Windows
    bat_path = os.path.join(hf_dir, "convert_to_gguf.bat")
    with open(bat_path, 'w', newline='\r\n') as f:
        f.write(f"@echo off\n")
        f.write(f"REM Convert {stem} to GGUF\n")
        f.write(f"{convert_cmd}\n")

    logger.info(f"HuggingFace format exported to: {hf_dir}")
    logger.info(f"To convert to GGUF:")
    logger.info(f"  Windows:   {bat_path}")
    logger.info(f"  Linux/Mac: bash {sh_path}")
    logger.info(f"  Or manually: {convert_cmd}")

    return hf_dir


def export_model(pth_path: str, formats: List[str], quantization: str = 'q8_0') -> dict:
    """Export a model to multiple formats.

    Args:
        pth_path: Path to the .pth checkpoint.
        formats: List of format strings: 'gguf', 'onnx', 'onnx_int8'
        quantization: GGUF quantization type (e.g. 'q8_0', 'q4_k_m', 'f16')

    Returns:
        Dict mapping format to output path.
    """
    results = {}
    for fmt in formats:
        fmt = fmt.strip().lower()
        try:
            if fmt == 'onnx':
                results['onnx'] = export_to_onnx(pth_path)
            elif fmt == 'onnx_int8':
                results['onnx_int8'] = export_to_onnx_quantized(pth_path, quant_type='int8')
            elif fmt == 'gguf':
                results['gguf'] = export_to_gguf(pth_path, quantization=quantization)
            else:
                logger.warning(f"Unknown format: {fmt}")
        except Exception as e:
            logger.error(f"Export to {fmt} failed: {e}")
            results[fmt] = f"ERROR: {e}"

    return results


def export_cli(model_spec: str, formats: List[str], quantization: str = 'q8_0'):
    """CLI entry point for model export."""
    # Parse model spec (may contain + for merge)
    if '+' in model_spec:
        from model_merge import parse_merge_spec, merge_from_names
        names, weights = parse_merge_spec(model_spec)
        logger.info(f"Merging models: {names}")
        merged_path = merge_from_names(names, weights)
        pth_path = merged_path
    else:
        # Find the model file
        direct_path = model_spec if model_spec.endswith('.pth') else os.path.join(MODELS_DIR, f"{model_spec}.pth")
        if not os.path.exists(direct_path):
            logger.error(f"Model not found: {direct_path}")
            sys.exit(1)
        pth_path = direct_path

    results = export_model(pth_path, formats, quantization=quantization)

    print(f"\nExport results for {Path(pth_path).stem}:")
    print("-" * 50)
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")
    print()
