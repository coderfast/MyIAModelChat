"""
Model export: converts .pth checkpoints to GGUF, ONNX, and quantized formats.
Includes license files and metadata for GPT-2 compliance.
"""
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import torch

from commons.registry.model_registry import load_model_metadata, MODELS_DIR, EXPORTED_DIR

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _copy_license_files(output_dir: str, used_pretrained: bool = False):
    """Copy LICENSE and NOTICE files to the output directory.
    
    Args:
        output_dir: Directory where license files will be copied
        used_pretrained: Whether GPT-2 pretrained weights were used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy LICENSE file
    license_src = os.path.join(PROJECT_ROOT, "LICENSE")
    if os.path.exists(license_src):
        shutil.copy2(license_src, os.path.join(output_dir, "LICENSE"))
    
    # Copy NOTICE file
    notice_src = os.path.join(PROJECT_ROOT, "NOTICE")
    if os.path.exists(notice_src):
        shutil.copy2(notice_src, os.path.join(output_dir, "NOTICE"))


def _get_license_metadata(used_pretrained: bool = False) -> dict:
    """Get license metadata for the model.
    
    Args:
        used_pretrained: Whether GPT-2 pretrained weights were used
        
    Returns:
        Dictionary with license information
    """
    license_info = {
        "project_license": "MIT",
        "project_copyright": "Copyright (c) 2026 MyIAModelChat"
    }
    
    if used_pretrained:
        license_info["pretrained_weights"] = {
            "model": "GPT-2",
            "license": "Modified MIT",
            "copyright": "Copyright (c) 2019 OpenAI",
            "notice": "This model uses pre-trained weights from GPT-2, licensed under the Modified MIT License."
        }
    
    return license_info


def export_to_onnx(pth_path: str, output_path: Optional[str] = None, seq_len: int = 512) -> str:
    """Export a .pth checkpoint to ONNX format."""
    try:
        from commons.model.chatmodel import ChatModel
        from commons.model.chatmodel_moe import ChatModelMoE
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
    
    # Check if this is a MoE model
    is_moe = ckpt.get('is_moe', False) or any('moe' in k.lower() for k in new_state.keys())
    
    if is_moe:
        # Use MoE model
        model = ChatModelMoE(tokenizer,
            embed_size=arch.get('embed_size', 256),
            num_layers=arch.get('num_layers', 2),
            num_experts=arch.get('num_experts', 4),
            top_k=arch.get('top_k', 2))
    else:
        model = ChatModel(tokenizer,
            embed_size=arch.get('embed_size', 256),
            num_layers=arch.get('num_layers', 4))
    
    model.load_state_dict(new_state, strict=False)
    model.eval()

    os.makedirs(EXPORTED_DIR, exist_ok=True)
    if output_path is None:
        stem = Path(pth_path).stem
        output_path = os.path.join(EXPORTED_DIR, f"{stem}.onnx")

    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, seq_len))

    # For MoE models, we need to handle the tuple output
    if is_moe:
        # Create a wrapper that only returns logits for ONNX export
        class MoEWrapper(torch.nn.Module):
            def __init__(self, moe_model):
                super().__init__()
                self.moe_model = moe_model
            
            def forward(self, input_ids):
                logits, _ = self.moe_model(input_ids)
                return logits
        
        export_model = MoEWrapper(model)
    else:
        export_model = model

    torch.onnx.export(
        export_model,
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

    # Save agentic token metadata for inference
    agentic_metadata = _extract_agentic_metadata(tokenizer, ckpt)
    if is_moe:
        agentic_metadata['is_moe'] = True
        agentic_metadata['moe_config'] = {
            'num_experts': arch.get('num_experts', 4),
            'top_k': arch.get('top_k', 2),
            'load_balance_weight': arch.get('load_balance_weight', 0.01)
        }
    
    # Add license metadata
    used_pretrained = ckpt.get('used_pretrained', False)
    agentic_metadata['license'] = _get_license_metadata(used_pretrained)
    
    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(agentic_metadata, f, indent=2, ensure_ascii=False)
    
    # Copy license files to exported directory
    _copy_license_files(os.path.dirname(output_path), used_pretrained)
    
    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"ONNX exported: {output_path} ({size_mb} MB)")
    return output_path


def _extract_agentic_metadata(tokenizer, ckpt: dict) -> dict:
    """Extract agentic token metadata from tokenizer for inference."""
    metadata = {
        "agentic_tokens": {},
        "special_tokens": {},
        "mode_tokens": {}
    }
    
    # Extract agentic tokens
    agentic_token_names = [
        'thinking', 'thinking_end', 'tool_call', 'tool_call_end',
        'observation', 'observation_end', 'action', 'action_end'
    ]
    
    for token_name in agentic_token_names:
        getter_name = f"get_{token_name}_index"
        if hasattr(tokenizer, getter_name):
            token_id = getattr(tokenizer, getter_name)()
            if token_id >= 0:
                token_str = f"<{token_name}>" if not token_name.startswith('thinking') else f"<{token_name.replace('_', ' ')}>"
                metadata["agentic_tokens"][token_name] = {
                    "id": token_id,
                    "token": token_str
                }
    
    # Extract mode tokens
    mode_token_names = ['context', 'answer', 'thinking_mode']
    for token_name in mode_token_names:
        getter_name = f"get_{token_name}_index"
        if hasattr(tokenizer, getter_name):
            token_id = getattr(tokenizer, getter_name)()
            if token_id >= 0:
                metadata["mode_tokens"][token_name] = {
                    "id": token_id,
                    "token": f"<|{token_name}|>"
                }
    
    # Extract standard special tokens
    special_token_names = ['pad', 'unk', 'bos', 'eos']
    for token_name in special_token_names:
        getter_name = f"get_{token_name}_index"
        if hasattr(tokenizer, getter_name):
            token_id = getattr(tokenizer, getter_name)()
            if token_id >= 0:
                metadata["special_tokens"][token_name] = {
                    "id": token_id,
                    "token": f"<{token_name}>"
                }
    
    # Add training config if present
    if 'training_config' in ckpt:
        metadata["training_config"] = ckpt['training_config']
    
    # Add architecture info
    if 'architecture' in ckpt:
        metadata["architecture"] = ckpt['architecture']
    
    return metadata


def export_to_onnx_quantized(
    pth_path: str,
    output_path: Optional[str] = None,
    quant_type: str = 'int8',
    static: bool = False,
    per_channel: bool = False,
    block_size: int = 128,
) -> str:
    """Export a .pth checkpoint to quantized ONNX format.

    Supports all 6 phases:
    - Dynamic: int8, uint8, int4, uint4
    - Static INT8: static_int8_qdq, static_int8_qoperator
    - Static INT4: static_int4_qoperator, static_int4_qdq
    - FP8: fp8_e4m3fn, fp8_e5m2, fp8_mixed, fp8_e4m3fnuz, fp8_e5m2fnuz
    - Per-Channel: per_channel_int8, per_channel_int4
    - Mixed: mixed_int8_int4, mixed_fp8_int8, tensor_overrides

    Args:
        pth_path: Path to .pth checkpoint
        output_path: Output ONNX path (auto-generated if None)
        quant_type: Quantization type string
        static: Use static quantization (default: dynamic)
        per_channel: Use per-channel quantization
        block_size: Block size for INT4/UINT4

    Returns:
        Path to quantized ONNX model
    """
    # First export to regular ONNX
    onnx_path = export_to_onnx(pth_path)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.exported.onnx_quantizer import quantize_onnx
    except ImportError:
        # Fallback: try relative import
        try:
            from onnx_quantizer import quantize_onnx
        except ImportError:
            raise RuntimeError("onnx_quantizer module not found. Ensure models/exported/ is accessible.")

    if output_path is None:
        stem = Path(pth_path).stem
        output_path = os.path.join(EXPORTED_DIR, f"{stem}_{quant_type}.onnx")

    # Resolve quant_type: if user passes simple "int8" with static=True, use static_int8_qdq
    resolved_type = quant_type
    if static and quant_type in ('int8', 'uint8'):
        resolved_type = f"static_{quant_type}_qdq"
    elif static and quant_type in ('int4', 'uint4'):
        resolved_type = f"static_{quant_type}_qdq"

    quantize_onnx(onnx_path, output_path, quant_type=resolved_type)

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"ONNX quantized ({quant_type}): {output_path} ({size_mb} MB)")
    return output_path


def export_to_gguf(pth_path: str, output_path: Optional[str] = None, quantization: str = 'q8_0') -> str:
    """Export a .pth checkpoint to GGUF format.

    This generates a HuggingFace-compatible directory that can be converted
    to GGUF using llama.cpp's convert_gguf.py script.
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
    
    # Copy license files
    used_pretrained = ckpt.get('used_pretrained', False)
    _copy_license_files(hf_dir, used_pretrained)
    
    # Add license metadata
    license_metadata = _get_license_metadata(used_pretrained)
    with open(os.path.join(hf_dir, "LICENSE_INFO.json"), 'w') as f:
        json.dump(license_metadata, f, indent=2)

    # Generate conversion scripts for the user (with absolute paths)
    hf_dir_abs = os.path.abspath(hf_dir)
    exported_dir = os.path.abspath(EXPORTED_DIR)
    convert_py = os.path.join(exported_dir, 'convert_gguf.py')
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


def export_model(
    pth_path: str,
    formats: List[str],
    quantization: str = 'q8_0',
    onnx_quant_type: str = 'int8',
    onnx_static: bool = False,
    onnx_per_channel: bool = False,
    onnx_block_size: int = 128,
) -> dict:
    """Export a model to multiple formats.

    Args:
        pth_path: Path to the .pth checkpoint.
        formats: List of format strings:
            - 'gguf': GGUF format
            - 'onnx': FP32 ONNX
            - 'onnx_int8': Dynamic INT8
            - 'onnx_uint8': Dynamic UINT8
            - 'onnx_int4': Dynamic INT4
            - 'onnx_uint4': Dynamic UINT4
            - 'onnx_static_int8': Static INT8 QDQ
            - 'onnx_static_int4': Static INT4
            - 'onnx_fp8': FP8 E4M3FN
            - 'onnx_fp8_mixed': FP8 mixed
            - 'onnx_per_channel': Per-channel INT8
            - 'onnx_mixed': Mixed INT8+INT4
        quantization: GGUF quantization type (e.g. 'q8_0', 'q4_k_m', 'f16')
        onnx_quant_type: ONNX quantization type (int8, uint8, int4, uint4, fp8_*)
        onnx_static: Use static quantization for ONNX
        onnx_per_channel: Use per-channel quantization for ONNX
        onnx_block_size: Block size for INT4/UINT4 ONNX quantization

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
                results['onnx_int8'] = export_to_onnx_quantized(
                    pth_path, quant_type='int8')
            elif fmt == 'onnx_uint8':
                results['onnx_uint8'] = export_to_onnx_quantized(
                    pth_path, quant_type='uint8')
            elif fmt == 'onnx_int4':
                results['onnx_int4'] = export_to_onnx_quantized(
                    pth_path, quant_type='int4')
            elif fmt == 'onnx_uint4':
                results['onnx_uint4'] = export_to_onnx_quantized(
                    pth_path, quant_type='uint4')
            elif fmt == 'onnx_static_int8':
                results['onnx_static_int8'] = export_to_onnx_quantized(
                    pth_path, quant_type='int8', static=True,
                    per_channel=onnx_per_channel)
            elif fmt == 'onnx_static_int4':
                results['onnx_static_int4'] = export_to_onnx_quantized(
                    pth_path, quant_type='int4', static=True,
                    block_size=onnx_block_size)
            elif fmt == 'onnx_fp8':
                results['onnx_fp8'] = export_to_onnx_quantized(
                    pth_path, quant_type='fp8_e4m3fn', static=True)
            elif fmt == 'onnx_fp8_mixed':
                results['onnx_fp8_mixed'] = export_to_onnx_quantized(
                    pth_path, quant_type='fp8_mixed', static=True)
            elif fmt == 'onnx_per_channel':
                results['onnx_per_channel'] = export_to_onnx_quantized(
                    pth_path, quant_type='per_channel_int8', static=True)
            elif fmt == 'onnx_mixed':
                results['onnx_mixed'] = export_to_onnx_quantized(
                    pth_path, quant_type='mixed_int8_int4', static=True)
            elif fmt == 'gguf':
                results['gguf'] = export_to_gguf(pth_path, quantization=quantization)
            else:
                logger.warning(f"Unknown format: {fmt}")
        except Exception as e:
            logger.error(f"Export to {fmt} failed: {e}")
            results[fmt] = f"ERROR: {e}"

    return results


def export_cli(
    model_spec: str,
    formats: List[str],
    quantization: str = 'q8_0',
    onnx_quant_type: str = 'int8',
    onnx_static: bool = False,
    onnx_per_channel: bool = False,
    onnx_block_size: int = 128,
):
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

    results = export_model(
        pth_path,
        formats,
        quantization=quantization,
        onnx_quant_type=onnx_quant_type,
        onnx_static=onnx_static,
        onnx_per_channel=onnx_per_channel,
        onnx_block_size=onnx_block_size,
    )

    print(f"\nExport results for {Path(pth_path).stem}:")
    print("-" * 50)
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")
    print()
