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


def _copy_license_files(output_dir: str):
    """Copy LICENSE and NOTICE files to the output directory.
    
    Args:
        output_dir: Directory where license files will be copied
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


def _get_license_metadata() -> dict:
    """Get license metadata for the model.
        
    Returns:
        Dictionary with license information
    """
    license_info = {
        "project_license": "MIT",
        "project_copyright": "Copyright (c) 2026 MyIAModelChat"
    }
    
    return license_info


def _generate_onnx_inference_script(model_name, config, metadata):
    """Generate a standalone ONNX inference script."""
    vocab_size = config.get('vocab_size', 8000)
    n_positions = config.get('n_positions', 512)
    eos_id = config.get('eos_token_id', -1)
    pad_id = config.get('pad_token_id', 0)

    # Build special tokens dict for the script
    special_tokens = metadata.get('special_tokens', {})
    mode_tokens = metadata.get('mode_tokens', {})
    agentic_tokens = metadata.get('agentic_tokens', {})

    script = f'''#!/usr/bin/env python3
"""
Standalone inference script for {model_name} ONNX model.

Usage:
    python inference.py                          # Interactive chat
    python inference.py --prompt "Hello"         # Single prompt
    python inference.py --prompt "Hello" --max-tokens 100

Requirements:
    pip install -r requirements.txt
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    import sentencepiece as spm
except ImportError:
    print("Error: sentencepiece not installed. Run: pip install sentencepiece")
    sys.exit(1)


# Model config
VOCAB_SIZE = {vocab_size}
N_POSITIONS = {n_positions}
EOS_ID = {eos_id}
PAD_ID = {pad_id}

# Special tokens
SPECIAL_TOKENS = {json.dumps({k: v.get('id', -1) for k, v in special_tokens.items()}, indent=4) if special_tokens else '{}'}


class ONNXChatInference:
    """Standalone ONNX inference for GPT-2 chat model."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_dir = model_dir

        # Load SentencePiece tokenizer
        sp_path = os.path.join(model_dir, "sentencepiece.model")
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"Tokenizer not found: {{sp_path}}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)

        # Load ONNX model
        onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in {{model_dir}}")
        model_path = os.path.join(model_dir, onnx_files[0])

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, text):
        """Encode text to token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self.sp.decode_ids(token_ids)

    def generate(self, prompt, max_tokens=150, temperature=0.7, top_k=50, top_p=0.9):
        """Generate a response given a prompt."""
        token_ids = self.encode(prompt)
        if not token_ids:
            token_ids = [self.sp.bos_id()] if self.sp.bos_id() >= 0 else []

        # Truncate to context window
        token_ids = token_ids[-(N_POSITIONS - 1):]

        for _ in range(max_tokens):
            # Pad to fixed length for ONNX
            input_ids = token_ids.copy()
            if len(input_ids) < N_POSITIONS:
                input_ids = [PAD_ID] * (N_POSITIONS - len(input_ids)) + input_ids

            input_array = np.array([input_ids], dtype=np.int64)
            outputs = self.session.run(None, {{self.input_name: input_array}})
            logits = outputs[0]

            # Get next token logits (last position)
            next_logits = logits[0, min(len(token_ids) - 1, N_POSITIONS - 1), :]

            # Temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_indices = np.argpartition(next_logits, -top_k)[-top_k:]
                top_k_logits = np.full_like(next_logits, -np.inf)
                top_k_logits[top_k_indices] = next_logits[top_k_indices]
                next_logits = top_k_logits

            # Greedy or sampling
            if temperature <= 0:
                next_id = int(np.argmax(next_logits))
            else:
                probs = np.exp(next_logits) / np.sum(np.exp(next_logits))
                next_id = int(np.random.choice(len(probs), p=probs))

            # Stop conditions
            if next_id == EOS_ID and EOS_ID >= 0:
                break
            if next_id == self.sp.unk_id():
                break
            if len(token_ids) >= N_POSITIONS:
                break

            token_ids.append(next_id)

        return self.decode(token_ids)

    def chat(self):
        """Interactive chat loop."""
        print(f"\n=== {model_name} (ONNX) ===")
        print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if not user_input:
                continue

            prompt = f"<|user|>\n{{user_input}}\n<|assistant|>\n"
            response = self.generate(prompt)
            print(f"Assistant: {{response}}\n")


def main():
    parser = argparse.ArgumentParser(description="{model_name} ONNX inference")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-tokens", "-m", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Temperature (0=greedy)")
    parser.add_argument("--model-dir", "-d", type=str, default=None, help="Model directory path")
    args = parser.parse_args()

    try:
        engine = ONNXChatInference(model_dir=args.model_dir)
    except FileNotFoundError as e:
        print(f"Error: {{e}}")
        sys.exit(1)

    if args.prompt:
        prompt = f"<|user|>\n{{args.prompt}}\n<|assistant|>\n"
        response = engine.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)
    else:
        engine.chat()


if __name__ == "__main__":
    main()
'''
    return script


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

    # Build stem and output directory
    stem = Path(pth_path).stem
    onnx_dir = os.path.join(EXPORTED_DIR, f"{stem}_onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Move ONNX file to the package directory
    final_onnx_path = os.path.join(onnx_dir, f"{stem}.onnx")
    if output_path != final_onnx_path:
        shutil.move(output_path, final_onnx_path)
    output_path = final_onnx_path

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
    agentic_metadata['license'] = _get_license_metadata()

    # Add architecture info to metadata
    agentic_metadata['architecture'] = {
        'model_type': 'gpt2',
        'vocab_size': arch.get('vocab_size', tokenizer.vocab_size),
        'n_positions': arch.get('n_positions', 512),
        'n_embd': arch.get('embed_size', 256),
        'n_layer': arch.get('num_layers', 4),
        'n_head': arch.get('n_head', 4),
    }
    agentic_metadata['model_name'] = stem
    agentic_metadata['format'] = 'onnx'

    metadata_path = os.path.join(onnx_dir, f"{stem}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(agentic_metadata, f, indent=2, ensure_ascii=False)

    # Copy license files
    _copy_license_files(onnx_dir)

    # Generate config.json for standalone usage
    vocab_size = arch.get('vocab_size', tokenizer.vocab_size)
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": vocab_size,
        "n_positions": arch.get('n_positions', 512),
        "n_embd": arch.get('embed_size', 256),
        "n_layer": arch.get('num_layers', 4),
        "n_head": arch.get('n_head', 4),
        "activation_function": "gelu_new",
        "bos_token_id": getattr(tokenizer, '_bos_id', -1),
        "eos_token_id": tokenizer.get_eos_index(),
        "pad_token_id": tokenizer.get_pad_index(),
    }
    with open(os.path.join(onnx_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # Copy sentencepiece.model
    tokenizer_src = ckpt.get('tokenizer_path', '')
    if tokenizer_src and os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, os.path.join(onnx_dir, "sentencepiece.model"))

    # Generate HuggingFace tokenizer files
    tokenizer.save_as_huggingface(onnx_dir)

    # Generate metadata.json
    model_metadata = {
        "model_name": stem,
        "format": "onnx",
        "vocab_size": vocab_size,
        "n_positions": arch.get('n_positions', 512),
        "n_embd": arch.get('embed_size', 256),
        "n_layer": arch.get('num_layers', 4),
        "n_head": arch.get('n_head', 4),
        "dataset_source": ckpt.get('dataset_source', 'unknown'),
    }
    with open(os.path.join(onnx_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=2)

    # Generate standalone inference script
    inference_script = _generate_onnx_inference_script(stem, config, agentic_metadata)
    with open(os.path.join(onnx_dir, "inference.py"), 'w', encoding='utf-8') as f:
        f.write(inference_script)

    # Generate requirements.txt
    requirements = """onnxruntime>=1.16.0
sentencepiece>=0.1.99
numpy>=1.24.0
"""
    with open(os.path.join(onnx_dir, "requirements.txt"), 'w', encoding='utf-8') as f:
        f.write(requirements)

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"ONNX package exported to: {onnx_dir}")
    logger.info(f"  Model: {output_path} ({size_mb} MB)")
    logger.info(f"  To use: cd {onnx_dir} && pip install -r requirements.txt && python inference.py")
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
    # First export to regular ONNX (creates _onnx package directory)
    stem = Path(pth_path).stem
    onnx_path = export_to_onnx(pth_path)

    # The base export moved the .onnx to _onnx/stem.onnx
    onnx_dir = os.path.join(EXPORTED_DIR, f"{stem}_onnx")
    base_onnx = os.path.join(onnx_dir, f"{stem}.onnx")

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.exported.onnx_quantizer import quantize_onnx
    except ImportError:
        try:
            from onnx_quantizer import quantize_onnx
        except ImportError:
            raise RuntimeError("onnx_quantizer module not found. Ensure models/exported/ is accessible.")

    # Quantized output goes to the package directory
    if output_path is None:
        output_path = os.path.join(onnx_dir, f"{stem}_{quant_type}.onnx")

    # Resolve quant_type: if user passes simple "int8" with static=True, use static_int8_qdq
    resolved_type = quant_type
    if static and quant_type in ('int8', 'uint8'):
        resolved_type = f"static_{quant_type}_qdq"
    elif static and quant_type in ('int4', 'uint4'):
        resolved_type = f"static_{quant_type}_qdq"

    quantize_onnx(base_onnx, output_path, quant_type=resolved_type)

    # Update inference.py to use the quantized model by default
    inference_path = os.path.join(onnx_dir, "inference.py")
    if os.path.exists(inference_path):
        with open(inference_path, 'r', encoding='utf-8') as f:
            inf_content = f.read()
        quant_filename = f"{stem}_{quant_type}.onnx"
        old_detect = "onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]"
        new_lines = [
            "onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]",
            "        # Prefer quantized model if available",
            f"        quant_file = '{quant_filename}'",
            "        if quant_file in onnx_files:",
            "            onnx_files = [quant_file]",
        ]
        new_detect = "\n".join(new_lines)
        if old_detect in inf_content:
            inf_content = inf_content.replace(old_detect, new_detect)
            with open(inference_path, 'w', encoding='utf-8') as f:
                f.write(inf_content)

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"ONNX quantized ({quant_type}): {output_path} ({size_mb} MB)")
    logger.info(f"  Package: {onnx_dir}")
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
    pad_id = tokenizer.get_pad_index()
    eos_id = tokenizer.get_eos_index()
    bos_id = getattr(tokenizer, '_bos_id', -1)
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": vocab_size,
        "n_positions": arch_meta.get('n_positions', 512),
        "n_embd": arch_meta.get('embed_size', 256),
        "n_layer": arch_meta.get('num_layers', 4),
        "n_head": arch_meta.get('n_head', 4),
        "activation_function": "gelu_new",
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "name": stem,
    }
    with open(os.path.join(hf_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # Save tokenizer files
    tokenizer.save_vocabulary(os.path.join(hf_dir, "tokenizer_vocab.json"))

    # Copy real sentencepiece.model to HF directory
    tokenizer_src = ckpt.get('tokenizer_path', '')
    if tokenizer_src and os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, os.path.join(hf_dir, "sentencepiece.model"))
        logger.info(f"Copied SentencePiece model: {tokenizer_src}")

    # Generate HuggingFace tokenizer files
    tokenizer.save_as_huggingface(hf_dir)
    
    # Copy license files
    _copy_license_files(hf_dir)
    
    # Add license metadata
    license_metadata = _get_license_metadata()
    with open(os.path.join(hf_dir, "LICENSE_INFO.json"), 'w', encoding='utf-8') as f:
        json.dump(license_metadata, f, indent=2)

    # Generate metadata.json for convert_gguf.py (includes agentic token info)
    agentic_metadata = _extract_agentic_metadata(tokenizer, ckpt)
    model_metadata = {
        "model_name": stem,
        "vocab_size": vocab_size,
        "n_positions": arch_meta.get('n_positions', 512),
        "n_embd": arch_meta.get('embed_size', 256),
        "n_layer": arch_meta.get('num_layers', 4),
        "n_head": arch_meta.get('n_head', 4),
        "dataset_source": ckpt.get('dataset_source', 'unknown'),
        "agentic_tokens": agentic_metadata.get("agentic_tokens", {}),
        "special_tokens": agentic_metadata.get("special_tokens", {}),
        "mode_tokens": agentic_metadata.get("mode_tokens", {}),
    }
    with open(os.path.join(hf_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)

    # Generate conversion scripts for the user (with absolute paths)
    hf_dir_abs = os.path.abspath(hf_dir)
    exported_dir = os.path.abspath(EXPORTED_DIR)
    convert_py = os.path.join(exported_dir, 'convert_gguf.py')
    gguf_out = os.path.abspath(os.path.join(exported_dir, f"{stem}_{quantization}.gguf"))
    convert_cmd = f"python \"{convert_py}\" \"{hf_dir_abs}\" --outfile \"{gguf_out}\" --outtype {quantization.lower()} --name {stem}"
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

    # Generate Modelfile for Ollama
    gguf_rel = os.path.relpath(gguf_out, hf_dir).replace('\\', '/')
    modelfile_path = os.path.join(hf_dir, "Modelfile")
    modelfile_content = f"FROM {gguf_rel}\n"
    modelfile_content += "\n"
    modelfile_content += 'TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>\n{{{{ .System }}}}\n{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>\n{{{{ .Prompt }}}}\n{{{{ end }}}}<|assistant|>\n{{{{ .Response }}}}\n\"\"\"\n'
    modelfile_content += "\n"
    modelfile_content += 'PARAMETER stop "<|user|>"\n'
    modelfile_content += 'PARAMETER stop "<|end_of_text|>"\n'
    modelfile_content += 'PARAMETER stop "</s>"\n'
    modelfile_content += "\n"
    modelfile_content += 'SYSTEM "You are a helpful AI assistant. Think step by step before answering."\n'
    with open(modelfile_path, 'w', newline='\n') as f:
        f.write(modelfile_content)

    logger.info(f"HuggingFace format exported to: {hf_dir}")
    logger.info(f"To convert to GGUF:")
    logger.info(f"  Windows:   {bat_path}")
    logger.info(f"  Linux/Mac: bash {sh_path}")
    logger.info(f"  Or manually: {convert_cmd}")
    logger.info(f"To use with Ollama:")
    logger.info(f"  1. Convert to GGUF first")
    logger.info(f"  2. Copy Modelfile to GGUF directory")
    logger.info(f"  3. Run: ollama create {stem} -f Modelfile")

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
