"""
ONNX Quantization Implementation — All 6 Phases.

Supports:
- Phase 1: Dynamic INT8, UINT8, INT4, UINT4
- Phase 2: Static INT8 (QDQ, QOperator, Per-Channel)
- Phase 3: Static INT4 (QOperator, QDQ)
- Phase 4: Static FP8 (E4M3FN, E5M2, E4M3FN+E5M2, FNUZ variants)
- Phase 5: Per-Channel INT8/INT4
- Phase 6: Mixed Precision (INT8+INT4, FP8+INT8, Tensor Overrides)

Requires: pip install onnxruntime>=1.17 onnx>=1.16 numpy
Optional: onnxruntime-gpu (FP8 support), optimum
"""
import os
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class QuantFormat(Enum):
    QOPERATOR = "qoperator"
    QDQ = "qdq"


class CalibrationMethod(Enum):
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    ENTROPY = "entropy"
    KL = "kl"


class QuantType(Enum):
    QINT8 = "int8"
    QUINT8 = "uint8"
    QINT16 = "int16"
    QUINT16 = "uint16"
    QINT4 = "int4"
    QUINT4 = "uint4"
    QFLOAT8E4M3FN = "fp8_e4m3fn"
    QFLOAT8E5M2 = "fp8_e5m2"
    QFLOAT8E4M3FNUZ = "fp8_e4m3fnuz"
    QFLOAT8E5M2FNUZ = "fp8_e5m2fnuz"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ONNXQuantConfig:
    """Configuration for ONNX quantization."""
    quant_type: str = "int8"
    quant_format: QuantFormat = QuantFormat.QDQ
    calibration_method: CalibrationMethod = CalibrationMethod.MINMAX
    activation_type: Optional[str] = None
    weight_type: Optional[str] = None
    per_channel: bool = False
    block_size: int = 128
    reduce_range: bool = False
    symmetric: bool = False
    op_types_to_quantize: Optional[List[str]] = None
    nodes_to_exclude: Optional[List[str]] = None
    extra_options: Optional[Dict[str, Any]] = None


# =============================================================================
# Calibration Data Reader
# =============================================================================

class TokenCalibrationReader:
    """Calibration data reader for ONNX static quantization.

    Feeds real token sequences to calibrate activation ranges.
    """

    def __init__(self, tokenizer, texts: list, seq_len: int = 32):
        """Initialize with tokenizer and calibration texts.

        Args:
            tokenizer: Tokenizer with encode() method
            texts: List of text strings for calibration
            seq_len: Sequence length (pad/truncate to this)
        """
        self.seq_len = seq_len
        self._index = 0

        self.encodings = []
        for text in texts:
            tokens = tokenizer.encode(text)[:seq_len]
            padded = np.zeros((1, seq_len), dtype=np.int64)
            padded[0, :len(tokens)] = tokens
            self.encodings.append(padded)

    def get_next(self) -> Optional[dict]:
        """Get next calibration sample."""
        if self._index >= len(self.encodings):
            return None
        sample = self.encodings[self._index]
        self._index += 1
        return {"input_ids": sample}


class NumpyCalibrationReader:
    """Calibration data reader from numpy arrays."""

    def __init__(self, data: np.ndarray):
        """Initialize with numpy array of input_ids.

        Args:
            data: Array of shape (n_samples, seq_len) with int64 dtype
        """
        self.data = data
        self._index = 0

    def get_next(self) -> Optional[dict]:
        if self._index >= len(self.data):
            return None
        sample = self.data[self._index]
        self._index += 1
        return {"input_ids": sample.reshape(1, -1)}


# =============================================================================
# Platform Detection
# =============================================================================

def detect_platform() -> dict:
    """Detect hardware platform for optimal quantization settings."""
    import platform
    import subprocess

    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "has_vnni": False,
        "has_cuda": False,
        "cuda_version": None,
        "gpu_name": None,
    }

    # Check VNNI (x86)
    if platform.machine() in ("AMD64", "x86_64", "x86"):
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "Name"],
                    capture_output=True, text=True, timeout=5
                )
                cpu_name = result.stdout.lower()
            elif platform.system() == "Linux":
                result = subprocess.run(
                    ["lscpu"], capture_output=True, text=True, timeout=5
                )
                cpu_name = result.stdout.lower()
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                cpu_name = result.stdout.lower()
            else:
                cpu_name = ""
            # Skylake-SP+ and AMD Zen4+ have VNNI
            info["has_vnni"] = any(x in cpu_name for x in [
                "skylake", "cannonlake", "icelake", "rocketlake",
                "alderlake", "raptorlake", "zen4", "epyc genoa"
            ])
        except Exception:
            pass

    # Check CUDA
    try:
        import torch
        info["has_cuda"] = torch.cuda.is_available()
        if info["has_cuda"]:
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return info


def get_recommended_config(platform_info: dict) -> ONNXQuantConfig:
    """Get recommended quantization config for detected platform."""
    if platform_info.get("has_cuda"):
        if platform_info.get("gpu_name", "").lower().find("h100") >= 0:
            return ONNXQuantConfig(
                quant_type="fp8_e4m3fn",
                weight_type="fp8_e4m3fn",
                activation_type="uint8",
            )
        return ONNXQuantConfig(
            quant_type="int8",
            weight_type="int8",
            activation_type="uint8",
            reduce_range=not platform_info.get("has_vnni", False),
        )

    return ONNXQuantConfig(
        quant_type="int8",
        weight_type="int8",
        activation_type="uint8",
        reduce_range=not platform_info.get("has_vnni", False),
    )


# =============================================================================
# Phase 1: Dynamic Quantization
# =============================================================================

def quantize_dynamic_int8(model_path: str, output_path: str) -> str:
    """Phase 1.1: Dynamic INT8 — simplest, no calibration needed."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Dynamic INT8: {output_path}")
    return output_path


def quantize_dynamic_uint8(model_path: str, output_path: str) -> str:
    """Phase 1.1: Dynamic UINT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )
    logger.info(f"Dynamic UINT8: {output_path}")
    return output_path


def quantize_dynamic_int4(model_path: str, output_path: str) -> str:
    """Phase 1.2: Dynamic INT4 — weight-only, requires onnxruntime >= 1.17."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt4,
    )
    logger.info(f"Dynamic INT4: {output_path}")
    return output_path


def quantize_dynamic_uint4(model_path: str, output_path: str) -> str:
    """Phase 1.2: Dynamic UINT4 — weight-only, requires onnxruntime >= 1.17."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt4,
    )
    logger.info(f"Dynamic UINT4: {output_path}")
    return output_path


# =============================================================================
# Phase 2: Static INT8 Quantization
# =============================================================================

def quantize_static_int8_qdq(
    model_path: str,
    output_path: str,
    calibration_reader,
    per_channel: bool = False,
    reduce_range: bool = False,
    calibration_method: str = "minmax",
) -> str:
    """Phase 2.1: Static INT8 with QDQ format — general purpose."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    calib_method = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
        "kl": CalibrationMethod.Entropy,  # KL maps to Entropy in onnxruntime
    }.get(calibration_method, CalibrationMethod.MinMax)

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_method=calib_method,
        per_channel=per_channel,
        reduce_range=reduce_range,
        format=QuantFormat.QDQ,
    )
    logger.info(f"Static INT8 QDQ: {output_path}")
    return output_path


def quantize_static_int8_qoperator(
    model_path: str,
    output_path: str,
    calibration_reader,
    per_channel: bool = False,
    reduce_range: bool = False,
) -> str:
    """Phase 2.2: Static INT8 with QOperator format — native INT8 ops."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_method=CalibrationMethod.MinMax,
        per_channel=per_channel,
        reduce_range=reduce_range,
        format=QuantFormat.QOperator,
    )
    logger.info(f"Static INT8 QOperator: {output_path}")
    return output_path


# =============================================================================
# Phase 3: Static INT4 Quantization
# =============================================================================

def quantize_static_int4_qoperator(
    model_path: str,
    output_path: str,
    calibration_reader,
    block_size: int = 128,
) -> str:
    """Phase 3.1: Static INT4 with QOperator format — best compression."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt4,
        calibration_method=CalibrationMethod.MinMax,
        format=QuantFormat.QOperator,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    logger.info(f"Static INT4 QOperator: {output_path}")
    return output_path


def quantize_static_int4_qdq(
    model_path: str,
    output_path: str,
    calibration_reader,
    block_size: int = 128,
) -> str:
    """Phase 3.2: Static INT4 with QDQ format."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt4,
        calibration_method=CalibrationMethod.MinMax,
        format=QuantFormat.QDQ,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    logger.info(f"Static INT4 QDQ: {output_path}")
    return output_path


# =============================================================================
# Phase 4: Static FP8 Quantization
# =============================================================================

def quantize_static_fp8_e4m3fn(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 4.1: FP8 E4M3FN weights — for H100/Blackwell GPUs."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QFLOAT8E4M3FN,
        activation_type=QuantType.QUInt8,
        calibration_method=CalibrationMethod.MinMax,
    )
    logger.info(f"FP8 E4M3FN: {output_path}")
    return output_path


def quantize_static_fp8_e5m2(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 4.2: FP8 E5M2 activations — wider dynamic range."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QFLOAT8E5M2,
        calibration_method=CalibrationMethod.MinMax,
    )
    logger.info(f"FP8 E5M2: {output_path}")
    return output_path


def quantize_static_fp8_mixed(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 4.3: FP8 mixed — E4M3FN weights + E5M2 activations."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QFLOAT8E4M3FN,
        activation_type=QuantType.QFLOAT8E5M2,
        calibration_method=CalibrationMethod.MinMax,
    )
    logger.info(f"FP8 Mixed E4M3FN+E5M2: {output_path}")
    return output_path


def quantize_static_fp8_e4m3fnuz(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 4.4: FP8 E4M3FNUZ — for GraphCore, some GPUs."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QFLOAT8E4M3FNUZ,
        activation_type=QuantType.QUInt8,
        calibration_method=CalibrationMethod.MinMax,
    )
    logger.info(f"FP8 E4M3FNUZ: {output_path}")
    return output_path


def quantize_static_fp8_e5m2fnuz(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 4.4: FP8 E5M2FNUZ — for GraphCore, some GPUs."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QFLOAT8E5M2FNUZ,
        activation_type=QuantType.QUInt8,
        calibration_method=CalibrationMethod.MinMax,
    )
    logger.info(f"FP8 E5M2FNUZ: {output_path}")
    return output_path


# =============================================================================
# Phase 5: Per-Channel Quantization
# =============================================================================

def quantize_per_channel_int8(
    model_path: str,
    output_path: str,
    calibration_reader,
    reduce_range: bool = False,
) -> str:
    """Phase 5.1: Per-Channel INT8 — better precision for variable distributions."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=reduce_range,
        format=QuantFormat.QDQ,
    )
    logger.info(f"Per-Channel INT8: {output_path}")
    return output_path


def quantize_per_channel_int4(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 5.2: Per-Channel INT4 — block quantization."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt4,
        calibration_method=CalibrationMethod.MinMax,
        per_channel=True,
        format=QuantFormat.QDQ,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    logger.info(f"Per-Channel INT4: {output_path}")
    return output_path


# =============================================================================
# Phase 6: Mixed Precision
# =============================================================================

def quantize_mixed_int8_int4(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 6.1: INT8 activations + INT4 weights — balance compression/precision."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt4,
        calibration_method=CalibrationMethod.MinMax,
        format=QuantFormat.QDQ,
    )
    logger.info(f"Mixed INT8+INT4: {output_path}")
    return output_path


def quantize_mixed_fp8_int8(
    model_path: str,
    output_path: str,
    calibration_reader,
) -> str:
    """Phase 6.2: FP8 weights + INT8 activations — high precision weights."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QFLOAT8E4M3FN,
        calibration_method=CalibrationMethod.MinMax,
        format=QuantFormat.QDQ,
    )
    logger.info(f"Mixed FP8+INT8: {output_path}")
    return output_path


def quantize_tensor_overrides(
    model_path: str,
    output_path: str,
    calibration_reader,
    quantize_op_types: Optional[List[str]] = None,
    nodes_to_exclude: Optional[List[str]] = None,
) -> str:
    """Phase 6.3: Tensor Quantization Overrides — per-layer control.

    Use quantize_op_types to limit which ops get quantized.
    Use nodes_to_exclude to exclude specific nodes from quantization.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_method=CalibrationMethod.MinMax,
        format=QuantFormat.QDQ,
        op_types_to_quantize=quantize_op_types,
        nodes_to_exclude=nodes_to_exclude,
    )
    logger.info(f"Tensor Overrides: {output_path}")
    return output_path


# =============================================================================
# Unified API
# =============================================================================

# Map quant_type string to (phase, function)
QUANT_REGISTRY = {
    # Phase 1: Dynamic
    "int8":           (1, quantize_dynamic_int8),
    "uint8":          (1, quantize_dynamic_uint8),
    "int4":           (1, quantize_dynamic_int4),
    "uint4":          (1, quantize_dynamic_uint4),
    # Phase 2: Static INT8
    "static_int8_qdq":      (2, quantize_static_int8_qdq),
    "static_int8_qoperator": (2, quantize_static_int8_qoperator),
    # Phase 3: Static INT4
    "static_int4_qoperator": (3, quantize_static_int4_qoperator),
    "static_int4_qdq":      (3, quantize_static_int4_qdq),
    # Phase 4: FP8
    "fp8_e4m3fn":     (4, quantize_static_fp8_e4m3fn),
    "fp8_e5m2":       (4, quantize_static_fp8_e5m2),
    "fp8_mixed":      (4, quantize_static_fp8_mixed),
    "fp8_e4m3fnuz":   (4, quantize_static_fp8_e4m3fnuz),
    "fp8_e5m2fnuz":   (4, quantize_static_fp8_e5m2fnuz),
    # Phase 5: Per-Channel
    "per_channel_int8": (5, quantize_per_channel_int8),
    "per_channel_int4": (5, quantize_per_channel_int4),
    # Phase 6: Mixed Precision
    "mixed_int8_int4":  (6, quantize_mixed_int8_int4),
    "mixed_fp8_int8":   (6, quantize_mixed_fp8_int8),
    "tensor_overrides": (6, quantize_tensor_overrides),
}


def quantize_onnx(
    model_path: str,
    output_path: str,
    quant_type: str = "int8",
    calibration_reader=None,
    **kwargs,
) -> str:
    """Unified quantization API.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to output quantized ONNX model
        quant_type: One of the keys in QUANT_REGISTRY
        calibration_reader: Required for static/FP8/mixed types
        **kwargs: Extra args passed to the quantization function

    Returns:
        Path to quantized model
    """
    if quant_type not in QUANT_REGISTRY:
        valid = sorted(QUANT_REGISTRY.keys())
        raise ValueError(f"Unknown quant_type: '{quant_type}'. Valid: {valid}")

    phase, func = QUANT_REGISTRY[quant_type]

    # Dynamic quantization (Phase 1) doesn't need calibration
    if phase == 1:
        return func(model_path, output_path, **kwargs)

    # Static quantization needs calibration reader
    if calibration_reader is None:
        raise ValueError(f"quant_type '{quant_type}' requires calibration_reader")

    return func(model_path, output_path, calibration_reader, **kwargs)


def get_quant_types() -> List[str]:
    """Return all available quantization type strings."""
    return sorted(QUANT_REGISTRY.keys())


def get_quant_info() -> Dict[str, dict]:
    """Return info about each quantization type."""
    info = {}
    for qt, (phase, func) in QUANT_REGISTRY.items():
        needs_calib = phase > 1
        info[qt] = {
            "phase": phase,
            "needs_calibration": needs_calib,
            "description": func.__doc__.strip().split("\n")[0] if func.__doc__ else "",
        }
    return info
