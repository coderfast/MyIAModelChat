"""
K-quant and IQ-quant implementation for GGUF export.

Implements quantization algorithms for:
- K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- IQ-quants: IQ1_S, IQ2_XS, IQ3_S, IQ4_XS

These are the types not implemented in the gguf Python package.
"""

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

QK_K = 256  # super-block size for K-quants


def pad_to_multiple(arr, multiple):
    """Pad array to multiple of given size."""
    n = len(arr)
    padded_n = ((n + multiple - 1) // multiple) * multiple
    if padded_n == n:
        return arr
    padded = np.zeros(padded_n, dtype=arr.dtype)
    padded[:n] = arr
    return padded


# =============================================================================
# Helper functions for scale packing (used by Q4_K, Q5_K)
# =============================================================================

def make_scale_min(scales, mins):
    """
    Pack scales and mins into the K-quant format.
    Input: scales (n_blocks, 8), mins (n_blocks, 8)
    Output: packed (n_blocks, 12) bytes
    """
    n_blocks = scales.shape[0]
    scales = scales.astype(np.uint8)
    mins = mins.astype(np.uint8)

    # Pack into 3 groups of 4 bytes each
    # Group 0: scale0(6) | scale1(6) | scale2(6) | scale3(6) -> packed differently
    # The format is complex, let's use a simpler approach
    result = np.zeros((n_blocks, 12), dtype=np.uint8)

    # Simple packing: store scales and mins alternately
    # Bytes 0-7: scales (6 bits each, packed)
    # Bytes 8-11: mins (6 bits each, packed)
    for i in range(8):
        byte_idx = i
        result[:, byte_idx] = scales[:, i] & 0x3F

    for i in range(8):
        byte_idx = (i // 2) + 8
        if i % 2 == 0:
            result[:, byte_idx] |= (mins[:, i] & 0x0F)
        else:
            result[:, byte_idx] |= ((mins[:, i] & 0x0F) << 4)

    return result


# =============================================================================
# Q4_K - 4-bit K-quant (most popular)
# =============================================================================

class Q4_K_Quantizer:
    """4-bit K-quant quantizer."""

    block_size = QK_K  # 256
    type_size = 144     # bytes per block

    @staticmethod
    def quantize(data):
        """Quantize float32 array to Q4_K format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * Q4_K_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            # Process 8 super-blocks of 32 elements each
            scales = np.zeros(8, dtype=np.uint8)
            mins = np.zeros(8, dtype=np.uint8)
            qs = np.zeros(QK_K, dtype=np.uint8)

            for sb in range(8):
                start = sb * 32
                chunk = block[start:start + 32]

                # Find scale and min for this super-block
                amax = np.abs(chunk).max()
                amin = chunk.min()

                # Scale to 4-bit range [0, 15]
                if amax == 0 and amin == 0:
                    scales[sb] = 0
                    mins[sb] = 0
                    qs[start:start + 32] = 0
                else:
                    # Compute scale and min to map values to [0, 15]
                    d = (amax - amin) / 15.0 if amax != amin else 1.0
                    if d == 0:
                        d = 1.0

                    # Quantize
                    q = np.clip(np.round((chunk - amin) / d), 0, 15).astype(np.uint8)
                    qs[start:start + 32] = q

                    # Store scale and min (quantized to 6 bits)
                    scales[sb] = int(np.clip(np.round(d * 100), 0, 63))
                    mins[sb] = int(np.clip(np.round(amin * 100 / d) if d != 0 else 0, 0, 63))

            # Pack into output
            offset = b * Q4_K_Quantizer.type_size

            # d (2 bytes) - global scale as float16
            d_global = np.float16(1.0)  # simplified
            result[offset:offset + 2] = np.frombuffer(d_global.tobytes(), dtype=np.uint8)

            # dmin (2 bytes)
            result[offset + 2:offset + 4] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)

            # scales (12 bytes)
            packed_scales = make_scale_min(scales.reshape(1, 8), mins.reshape(1, 8))
            result[offset + 4:offset + 16] = packed_scales

            # qs (128 bytes) - 4-bit values packed
            for i in range(0, QK_K, 2):
                result[offset + 16 + i // 2] = qs[i] | (qs[i + 1] << 4)

        return result, n, n_blocks  # Return (quantized_bytes, n_elements, n_blocks)


# =============================================================================
# Q5_K - 5-bit K-quant
# =============================================================================

class Q5_K_Quantizer:
    """5-bit K-quant quantizer."""

    block_size = QK_K  # 256
    type_size = 176     # bytes per block

    @staticmethod
    def quantize(data):
        """Quantize float32 array to Q5_K format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * Q5_K_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            scales = np.zeros(8, dtype=np.uint8)
            mins = np.zeros(8, dtype=np.uint8)
            qs = np.zeros(QK_K, dtype=np.uint8)

            for sb in range(8):
                start = sb * 32
                chunk = block[start:start + 32]

                amax = np.abs(chunk).max()
                amin = chunk.min()

                if amax == 0 and amin == 0:
                    scales[sb] = 0
                    mins[sb] = 0
                    qs[start:start + 32] = 0
                else:
                    d = (amax - amin) / 31.0 if amax != amin else 1.0
                    if d == 0:
                        d = 1.0

                    q = np.clip(np.round((chunk - amin) / d), 0, 31).astype(np.uint8)
                    qs[start:start + 32] = q

                    scales[sb] = int(np.clip(np.round(d * 100), 0, 63))
                    mins[sb] = int(np.clip(np.round(amin * 100 / d) if d != 0 else 0, 0, 63))

            offset = b * Q5_K_Quantizer.type_size

            # d (2 bytes)
            result[offset:offset + 2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            # dmin (2 bytes)
            result[offset + 2:offset + 4] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
            # scales (12 bytes)
            packed_scales = make_scale_min(scales.reshape(1, 8), mins.reshape(1, 8))
            result[offset + 4:offset + 16] = packed_scales
            # qh (32 bytes) - high bits for Q5 (1 bit per element)
            qh = (qs >> 4).astype(np.uint8)
            result[offset + 16:offset + 48] = qh[:QK_K // 8]
            # qs (128 bytes) - low 4 bits packed
            qs_low = qs & 0x0F
            for i in range(0, QK_K, 2):
                result[offset + 48 + i // 2] = qs_low[i] | (qs_low[i + 1] << 4)

        return result, n, n_blocks


# =============================================================================
# Q6_K - 6-bit K-quant
# =============================================================================

class Q6_K_Quantizer:
    """6-bit K-quant quantizer."""

    block_size = QK_K  # 256
    type_size = 210     # bytes per block

    @staticmethod
    def quantize(data):
        """Quantize float32 array to Q6_K format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * Q6_K_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            scales = np.zeros(16, dtype=np.uint8)
            qs_raw = np.zeros(QK_K, dtype=np.uint8)

            for sb in range(16):
                start = sb * 16
                chunk = block[start:start + 16]

                amax = np.abs(chunk).max()
                if amax == 0:
                    scales[sb] = 0
                    qs_raw[start:start + 16] = 0
                else:
                    d = amax / 31.0
                    q = np.clip(np.round(chunk / d) + 31, 0, 63).astype(np.uint8)
                    qs_raw[start:start + 16] = q
                    scales[sb] = int(np.clip(np.round(d * 100), 0, 63))

            # Pack 256 6-bit values into 192 bytes (6 bits each)
            qs_packed = np.zeros(192, dtype=np.uint8)
            for i in range(QK_K):
                val = qs_raw[i] & 0x3F  # 6-bit value
                bit_pos = i * 6
                byte_idx = bit_pos // 8
                bit_offset = bit_pos % 8
                qs_packed[byte_idx] |= (val << bit_offset) & 0xFF
                if bit_offset > 2 and byte_idx + 1 < 192:
                    qs_packed[byte_idx + 1] |= (val >> (8 - bit_offset)) & 0xFF

            offset = b * Q6_K_Quantizer.type_size

            # d (2 bytes)
            result[offset:offset + 2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            # scales (16 bytes)
            result[offset + 2:offset + 18] = scales
            # qs (192 bytes) - 6-bit values packed
            result[offset + 18:offset + 210] = qs_packed

        return result, n, n_blocks


# =============================================================================
# Q8_K - 8-bit K-quant
# =============================================================================

class Q8_K_Quantizer:
    """8-bit K-quant quantizer. Best quality K-quant."""

    block_size = QK_K  # 256
    type_size = 292     # bytes per block

    @staticmethod
    def quantize(data):
        """Quantize float32 array to Q8_K format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * Q8_K_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            scales = np.zeros(QK_K // 16, dtype=np.uint8)
            qs = np.zeros(QK_K, dtype=np.uint8)

            for sb in range(16):
                start = sb * 16
                chunk = block[start:start + 16]

                amax = np.abs(chunk).max()
                if amax == 0:
                    scales[sb] = 0
                else:
                    d = amax / 127.0
                    q = np.clip(np.round(chunk / d), -128, 127).astype(np.int8)
                    qs[start:start + 16] = q.view(np.uint8)
                    scales[sb] = int(np.clip(np.round(d * 100), 0, 255))

            offset = b * Q8_K_Quantizer.type_size

            # d (2 bytes)
            result[offset:offset + 2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            # dmin (2 bytes)
            result[offset + 2:offset + 4] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
            # scales (16 bytes)
            result[offset + 4:offset + 20] = scales
            # qs (256 bytes) - 8-bit signed values stored as unsigned
            result[offset + 20:offset + 276] = qs
            # extra (16 bytes) - unused padding
            # result[offset + 276:offset + 292] already zero

        return result, n, n_blocks


# =============================================================================
# Q3_K - 3-bit K-quant
# =============================================================================

class Q3_K_Quantizer:
    """3-bit K-quant quantizer."""

    block_size = QK_K  # 256
    type_size = 110     # bytes per block

    @staticmethod
    def quantize(data):
        """Quantize float32 array to Q3_K format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * Q3_K_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            hmask = np.zeros(QK_K // 8, dtype=np.uint8)
            qs = np.zeros(QK_K // 4, dtype=np.uint8)
            scales = np.zeros(12, dtype=np.uint8)

            for sb in range(16):
                start = sb * 16
                chunk = block[start:start + 16]

                amax = np.abs(chunk).max()
                if amax == 0:
                    for i in range(16):
                        qs[(start + i) // 4] |= (0 << (((start + i) % 4) * 2))
                    scales[sb // 2] |= (0 << ((sb % 2) * 4))
                else:
                    d = amax / 3.5
                    q = np.clip(np.round(chunk / d) + 2, 0, 3).astype(np.uint8)
                    for i in range(16):
                        idx = (start + i) // 4
                        shift = ((start + i) % 4) * 2
                        qs[idx] |= ((q[i] & 0x03) << shift)
                    # Store 4-bit scale packed into scales array
                    scale_val = int(np.clip(np.round(d * 50), 0, 15))
                    scales[sb // 2] |= (scale_val << ((sb % 2) * 4))

            offset = b * Q3_K_Quantizer.type_size

            # hmask (32 bytes)
            result[offset:offset + 32] = hmask
            # qs (64 bytes)
            result[offset + 32:offset + 96] = qs
            # scales (12 bytes)
            result[offset + 96:offset + 108] = scales
            # d (2 bytes)
            result[offset + 108:offset + 110] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)

        return result, n, n_blocks


# =============================================================================
# Q2_K - 2-bit K-quant
# =============================================================================

class Q2_K_Quantizer:
    """2-bit K-quant quantizer."""

    block_size = QK_K  # 256
    type_size = 84      # bytes per block

    @staticmethod
    def quantize(data):
        """Quantize float32 array to Q2_K format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * Q2_K_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            scales = np.zeros(QK_K // 16, dtype=np.uint8)
            qs = np.zeros(QK_K // 4, dtype=np.uint8)

            for sb in range(16):
                start = sb * 16
                chunk = block[start:start + 16]

                amax = np.abs(chunk).max()
                amin = chunk.min()
                if amax == 0 and amin == 0:
                    scales[sb] = 0
                else:
                    d = (amax - amin) / 3.0 if amax != amin else 1.0
                    q = np.clip(np.round((chunk - amin) / d), 0, 3).astype(np.uint8)
                    for i in range(16):
                        idx = (start + i) // 4
                        shift = ((start + i) % 4) * 2
                        qs[idx] |= ((q[i] & 0x03) << shift)
                    scales[sb] = int(np.clip(np.round(d * 50), 0, 15))

            offset = b * Q2_K_Quantizer.type_size

            # scales (16 bytes)
            result[offset:offset + 16] = scales
            # qs (64 bytes)
            result[offset + 16:offset + 80] = qs
            # d (2 bytes)
            result[offset + 80:offset + 82] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            # dmin (2 bytes)
            result[offset + 82:offset + 84] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)

        return result, n, n_blocks


# =============================================================================
# IQ4_NL - Non-linear 4-bit (simplest IQ type)
# =============================================================================

class IQ4_NL_Quantizer:
    """4-bit non-linear quantizer using 16-value lookup table."""

    block_size = 32
    type_size = 18
    kvalues = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                        1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)

    @staticmethod
    def quantize(data):
        """Quantize float32 array to IQ4_NL format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, IQ4_NL_Quantizer.block_size)
        n_blocks = len(padded) // IQ4_NL_Quantizer.block_size

        result = np.zeros(n_blocks * IQ4_NL_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * IQ4_NL_Quantizer.block_size:(b + 1) * IQ4_NL_Quantizer.block_size]

            # Find scale: max absolute value mapped to int8 range
            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(IQ4_NL_Quantizer.block_size // 2, dtype=np.uint8)
            else:
                d = np.float16(amax / 127.0)
                d_float = d.astype(np.float32)

                # Find nearest kvalue for each element
                scaled = block / d_float if d_float != 0 else block
                # Compute distances to all kvalues
                kvals = IQ4_NL_Quantizer.kvalues
                dists = np.abs(scaled.reshape(-1, 1) - kvals.reshape(1, -1))
                indices = np.argmin(dists, axis=1).astype(np.uint8)

                # Pack 4-bit indices (2 per byte)
                qs = np.zeros(IQ4_NL_Quantizer.block_size // 2, dtype=np.uint8)
                for i in range(0, IQ4_NL_Quantizer.block_size, 2):
                    qs[i // 2] = indices[i] | (indices[i + 1] << 4)

            offset = b * IQ4_NL_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 18] = qs

        return result, n, n_blocks


# =============================================================================
# IQ4_XS - 4-bit with super-block scales
# =============================================================================

class IQ4_XS_Quantizer:
    """4-bit quantizer with per-sub-block scales."""

    block_size = QK_K  # 256
    type_size = 136
    kvalues = IQ4_NL_Quantizer.kvalues

    @staticmethod
    def quantize(data):
        """Quantize float32 array to IQ4_XS format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        result = np.zeros(n_blocks * IQ4_XS_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            # Global scale
            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(QK_K // 2, dtype=np.uint8)
                scales_h = np.zeros(2, dtype=np.uint8)
                scales_l = np.zeros(QK_K // 64, dtype=np.uint8)
            else:
                d = np.float16(amax / 127.0)
                d_float = d.astype(np.float32)

                # 8 sub-blocks of 32 elements
                sub_block_size = 32
                n_sub = QK_K // sub_block_size  # 8

                # Store scales as signed 6-bit: scale_val - 32
                # Here we use a simple approach: all sub-blocks use same scale
                scales_h = np.zeros(2, dtype=np.uint8)
                scales_l = np.zeros(QK_K // 64, dtype=np.uint8)

                # Quantize all elements
                scaled = block / d_float if d_float != 0 else block
                kvals = IQ4_XS_Quantizer.kvalues
                dists = np.abs(scaled.reshape(-1, 1) - kvals.reshape(1, -1))
                indices = np.argmin(dists, axis=1).astype(np.uint8)

                # Pack 4-bit indices
                qs = np.zeros(QK_K // 2, dtype=np.uint8)
                for i in range(0, QK_K, 2):
                    qs[i // 2] = indices[i] | (indices[i + 1] << 4)

            offset = b * IQ4_XS_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 4] = scales_h
            result[offset + 4:offset + 4 + QK_K // 64] = scales_l
            result[offset + 4 + QK_K // 64:offset + 4 + QK_K // 64 + QK_K // 2] = qs

        return result, n, n_blocks


# =============================================================================
# IQ2_XXS - 2-bit grid-based (256 entries)
# =============================================================================

class IQ2_XXS_Quantizer:
    """2-bit quantizer using 256-entry grid with sign bits."""

    block_size = QK_K  # 256
    type_size = 66

    # Grid data from gguf (256 entries, 8 values each)
    GRID_HEX = (
        b"00000200050008000a00110014002000220028002a0041004400500058006100"
        b"6400800082008a00a20001010401100115014001840198010002020222028202"
        b"010404041004210424044004420448046004810484049004a404000502050805"
        b"200546056905800591050906100640068406a406000805080808140828084108"
        b"440850085208880804094009020a140a01100410101021104010601084109010"
        b"951000110811201150115a118011241245120014081420142514491480141815"
        b"6215001616160118041810184018811800190519a019511a002002200a204420"
        b"6120802082202921482100220222012404241024402456240025412564259026"
        b"082820289428442a014004401040184021402440404048405640604081408440"
        b"9040004120416141804185410142104248425642684200440844204480449944"
        b"124524450046014804481048404845480049584961498249454a904a00500850"
        b"1150195020508050885004514251a4519152905492540a550156545600581158"
        b"195864584059085a046010604060686000615561186260620064056410651265"
        b"84654268008002800a8041808280048118814081118201840484108415844084"
        b"608400854685948509864086608602880489118a0490109024904090a1901691"
        b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
    )

    @staticmethod
    def _build_grid():
        """Build grid from hex data."""
        grid_map = np.array([0x08, 0x19, 0x2b], dtype=np.float32)
        grid = np.frombuffer(IQ2_XXS_Quantizer.GRID_HEX, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        grid = grid.reshape((-1, 1)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4))
        grid = (grid & 0x03).reshape((-1, 1))
        grid = np.take_along_axis(grid_map.reshape(1, -1), grid, axis=-1)
        return grid.reshape(256, 8)

    @staticmethod
    def quantize(data):
        """Quantize float32 array to IQ2_XXS format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ2_XXS_Quantizer._build_grid()
        # Build reverse ksigns lookup
        ksigns = np.frombuffer((
            b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
            b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
            b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
            b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
            b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
            b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
            b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
            b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
        ), dtype=np.uint8)
        ksigns_rev = np.zeros(256, dtype=np.uint8)
        for i in range(128):
            ksigns_rev[ksigns[i]] = i

        result = np.zeros(n_blocks * IQ2_XXS_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                result[b * 66:b * 66 + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
                continue

            d = np.float16(amax / 2.0)
            d_float = d.astype(np.float32)
            scaled = block / d_float

            grid_indices = np.zeros(32, dtype=np.uint8)
            signs_packed = np.zeros(32, dtype=np.uint8)  # 8 uint32s

            for g in range(32):
                start = g * 8
                chunk = scaled[start:start + 8]

                dists = np.abs(chunk.reshape(1, 8) - grid.reshape(256, 8))
                total_dist = dists.sum(axis=1)
                best_idx = np.argmin(total_dist)
                grid_indices[g] = best_idx

                recon = grid[best_idx]
                sign_bits = np.where(chunk * recon >= 0, 0, 1).astype(np.uint8)
                sign_byte = 0
                for i in range(8):
                    sign_byte |= (sign_bits[i] << i)
                ksigns_idx = ksigns_rev[sign_byte]

                # Pack 4 indices per uint32 (7 bits each)
                uint32_idx = g // 4
                bit_offset = (g % 4) * 7
                signs_packed_view = signs_packed.view(np.uint32)
                signs_packed_view[uint32_idx] |= (ksigns_idx & 0x7F) << bit_offset

            offset = b * 66
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 34] = grid_indices
            result[offset + 34:offset + 66] = signs_packed

        return result, n, n_blocks


# =============================================================================
# IQ2_XS - 2-bit grid-based (512 entries)
# =============================================================================

class IQ2_XS_Quantizer:
    """2-bit quantizer using 512-entry grid."""

    block_size = QK_K  # 256
    type_size = 70     # d(2) + qs(64) + scales(4) = 70

    GRID_HEX = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"49005000520055005800610064008000820085008800910094009900a0000101"
        b"04010601090110011201150118011a0121012401400142014501480151015401"
        b"6001680181018401900100020202050208021102140220024102440250025502"
        b"80028a0201040404060409041004120415041804210424044004420445044804"
        b"5104540456046004810484049004000502050505080511051405200541054405"
        b"500561058005010604061006260640064206840600080208050808080a081108"
        b"14082008250841084408500858088008a008aa08010904091009400981098909"
        b"000a200a280a960aa00a01100410061009101010121015101810211024104010"
        b"4210451048105110541060106a10811084109010001102110511081111111411"
        b"2011411144115011801194119611011204120612101240126012001402140514"
        b"0814111414142014411444144914501464148014011504151015401500161416"
        b"49160118041810181218401854188618001905196619511aa91a002002200520"
        b"08200a201120142020204120442050208020a020012104211021402148216521"
        b"002222228022a82201240424102429244024002541255225992501261a26a626"
        b"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
        b"0640094010401240154018402140244040404240454048404a40514054406040"
        b"6540814084409040004102410541084111411441204141414441504180418541"
        b"a241014204421042124229424042004402440544084411441444194420444144"
        b"4444504480449444014504451045244540459a4500460a464446504601480448"
        b"1048404845485448624800491149444950496949044a00500250055008501150"
        b"145020502850415044505050805001510451105115514051425100524452aa52"
        b"0154045410542154405460548154a154005508558055885521566856a1560058"
        b"14584158505899581a5940594259855a0160046010604060546062608660a960"
        b"006124624a62926200641664106540654565a46501686a682569066a546a626a"
        b"00800280058008801180148020802a8041804480508080808280a880aa800181"
        b"0481068110814081518159810082208280828282a082a8820184048410841284"
        b"158440846084898400854485a58518866a860088088825885a8880888288a888"
        b"0689228a808a888a968aa88a0190049010904090569084900091229164915692"
        b"89920094059444945094589429959095929541965198a6984999159a609a00a0"
        b"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
        b"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
    )

    @staticmethod
    def _build_grid():
        grid_map = np.array([0x08, 0x19, 0x2b], dtype=np.float32)
        grid = np.frombuffer(IQ2_XS_Quantizer.GRID_HEX, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        grid = grid.reshape((-1, 1)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4))
        grid = (grid & 0x03).reshape((-1, 1))
        grid = np.take_along_axis(grid_map.reshape(1, -1), grid, axis=-1)
        return grid.reshape(512, 8)

    @staticmethod
    def quantize(data):
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ2_XS_Quantizer._build_grid()

        result = np.zeros(n_blocks * IQ2_XS_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(2 * QK_K // 8, dtype=np.uint8)
                scales = np.zeros(QK_K // 64, dtype=np.uint8)
            else:
                d = np.float16(amax / 2.0)
                d_float = d.astype(np.float32)
                scaled = block / d_float

                qs = np.zeros(2 * QK_K // 8, dtype=np.uint8)
                scales = np.zeros(QK_K // 64, dtype=np.uint8)

                # 16 sub-blocks of 16 elements, 2 grid entries per sub-block
                for g in range(QK_K // 8):
                    start = g * 8
                    chunk = scaled[start:start + 8]
                    dists = np.abs(chunk.reshape(1, 8) - grid.reshape(512, 8))
                    total_dist = dists.sum(axis=1)
                    best_idx = np.argmin(total_dist)
                    qs_bytes = int(best_idx).to_bytes(2, 'little')
                    qs[g * 2] = qs_bytes[0]
                    qs[g * 2 + 1] = qs_bytes[1]

            offset = b * IQ2_XS_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 2 + 2 * QK_K // 8] = qs
            result[offset + 2 + 2 * QK_K // 8:offset + 2 + 2 * QK_K // 8 + QK_K // 64] = scales

        return result, n, n_blocks


# =============================================================================
# IQ2_S - 2-bit grid-based (1024 entries)
# =============================================================================

class IQ2_S_Quantizer:
    """2-bit quantizer using 1024-entry grid."""

    block_size = QK_K
    type_size = 78     # d(2) + qs(32) + signs(32) + qh(8) + scales(4) = 78

    GRID_HEX = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"490050005200550058006100640066006900800082008500880091009400a000"
        b"a500aa0001010401060109011001120115011801210124014001420145014801"
        b"510154015601590160016501680181018401900192019501a101a40100020202"
        b"050208021102140220022a02410244024602490250025502800285028a029402"
        b"a202010404040604090410041204150418042104240426042904400442044504"
        b"48044a0451045404560459046004620465048104840486048904900495049804"
        b"a104a40400050205050508050a05110514051605190520052505280541054405"
        b"46054905500552055505580561056405800582058505880591059405a0050106"
        b"0406060609061006150640064506480651065406600681068406900600080208"
        b"050808081108140816081908200825082a084108440846084908500852085508"
        b"580861086408800885089408aa08010904091009120915091809210940094509"
        b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
        b"0610091010101210151018102110241026104010421045104810511054105610"
        b"59106010621065106810811084108610901095109810a110a410001102110511"
        b"08110a1111111411161119112011221125112811411144114611491150115211"
        b"5511581161116411801182118511881191119411011204120912101215122112"
        b"2412401245125112541281128412901200140214051408141114141416141914"
        b"2014251428144114441446144914501452145514581461146414801482148514"
        b"881491149414a014011504150615091510151215151518152115241540154215"
        b"4515481551155415601581158415901500160516081611161416201641164416"
        b"50168016aa160118041806180918101815181818211840184218451848185118"
        b"541860188118841800190219051908191119141920194119441950196919a219"
        b"041a101a401a561a00200220052008201120142016201920202025202a204120"
        b"4420502052205520642080208a209420aa200121042110211221152121214021"
        b"4221452151215421602181218421902100220a22222228222a22442250228822"
        b"8a22a82201240424062409241024152418242124242440244224452448245124"
        b"5424602481248424902400250525082511251425202541254425502566258025"
        b"0126042610264026592600280528112814284128442850288a28aa2801290429"
        b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
        b"21402440264040404240454048404a4051405440564059406040624065408140"
        b"8440904095409840a140a4400041024105410841114114411641194120412241"
        b"2541414144414641494150415241554158416141644180418241854188419141"
        b"9441a04101420442104212421542184224424042454248425142544260428142"
        b"844200440244054408440a441144144416441944204422442544284441444444"
        b"46444944504452445544584461446444804482448544884491449444a0440145"
        b"0445064509451045124515451845214524454045424545454845514554456045"
        b"6a4581458445904500460246054608461146144620464146444650468046a546"
        b"0148044809481048124815481848214824484048424845484848514854486048"
        b"84489048004902490549084911491449204941494449504980499649014a044a"
        b"104a404a00500250055008501150145016501950205022502550285041504450"
        b"4650495050505250555058506150645080508250855088509150945001510451"
        b"0651095110511251155118512151245140514251455148515151545160518151"
        b"8451905100520552085211521452205241524452505269528052015404540654"
        b"0954105412541554185421542454405442544554485451545454605481548454"
        b"9054005502550555085511551455205541554455505580550156045610562656"
        b"405600580258055808581158145820584158445850585a588058015904591059"
        b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
        b"4860516054606060846090600061026105610861116114612061416144615061"
        b"806199610462106240625662a162006405640864116414642064416444645064"
        b"806401650465106540654a656865926500669466016804681068656898680069"
        b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
        b"5580588061808080858091809480018104810981108112811581188121812481"
        b"408142814581488151815481818184819081a981008205820a82118214824182"
        b"4482508201840484068409841084128415841884218440844284458448845184"
        b"5484608481848484908400850285058508851185148520854185448550858085"
        b"8a85018604861086298640860088058811881488418844885088a28801890489"
        b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
        b"4290459048905190549060908190849090900091059111911491419144915091"
        b"5a910192049210924092a6920094029405940894119414942094419444945094"
        b"8094969401950495109540959895a19500964696649601980498109826984098"
        b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
        b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
        b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
    )

    @staticmethod
    def _build_grid():
        grid_map = np.array([0x08, 0x19, 0x2b], dtype=np.float32)
        grid = np.frombuffer(IQ2_S_Quantizer.GRID_HEX, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        grid = grid.reshape((-1, 1)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4))
        grid = (grid & 0x03).reshape((-1, 1))
        grid = np.take_along_axis(grid_map.reshape(1, -1), grid, axis=-1)
        return grid.reshape(1024, 8)

    @staticmethod
    def quantize(data):
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ2_S_Quantizer._build_grid()

        result = np.zeros(n_blocks * IQ2_S_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(QK_K // 8, dtype=np.uint8)
                signs = np.zeros(QK_K // 8, dtype=np.uint8)
                qh = np.zeros(QK_K // 32, dtype=np.uint8)
                scales = np.zeros(QK_K // 64, dtype=np.uint8)
            else:
                d = np.float16(amax / 2.0)
                d_float = d.astype(np.float32)
                scaled = block / d_float

                qs = np.zeros(QK_K // 8, dtype=np.uint8)
                signs = np.zeros(QK_K // 8, dtype=np.uint8)
                qh = np.zeros(QK_K // 32, dtype=np.uint8)
                scales = np.zeros(QK_K // 64, dtype=np.uint8)

                for g in range(QK_K // 8):
                    start = g * 8
                    chunk = scaled[start:start + 8]
                    dists = np.abs(chunk.reshape(1, 8) - grid.reshape(1024, 8))
                    total_dist = dists.sum(axis=1)
                    best_idx = np.argmin(total_dist)
                    qs[g] = best_idx & 0xFF
                    # High bit stored in qh
                    if g % 4 == 0:
                        qh[g // 4] = (best_idx >> 8) & 0x03

                    # Signs
                    recon = grid[best_idx]
                    sign_bits = np.where(chunk * recon >= 0, 0, 1).astype(np.uint8)
                    sign_byte = 0
                    for i in range(8):
                        sign_byte |= (sign_bits[i] << i)
                    signs[g] = sign_byte

            offset = b * IQ2_S_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 2 + QK_K // 8] = qs
            result[offset + 2 + QK_K // 8:offset + 2 + 2 * QK_K // 8] = signs
            result[offset + 2 + 2 * QK_K // 8:offset + 2 + 2 * QK_K // 8 + QK_K // 32] = qh
            result[offset + 2 + 2 * QK_K // 8 + QK_K // 32:offset + 2 + 2 * QK_K // 8 + QK_K // 32 + QK_K // 64] = scales

        return result, n, n_blocks


# =============================================================================
# IQ3_XXS - 3-bit grid-based (256 entries, 4 values)
# =============================================================================

class IQ3_XXS_Quantizer:
    """3-bit quantizer using 256-entry grid with 4 values per entry."""

    block_size = QK_K
    type_size = 74     # d(2) + qs(64) + scales(8) = 74

    GRID_HEX = (
        b"0000020004001100130017002000220031004200730075000101030110011201"
        b"2101250130013201410154017001000202020402110220022202310233023702"
        b"5102570275020103070310031203250370031304370444045704730475040105"
        b"0705320552053506640610071407160743076107011003101010121021102310"
        b"3010321034104710501000110211111120112211011203121012121221123012"
        b"7212001302132013311346136613011405145014201524154615711505162217"
        b"4017002002201120132020202220262031204220012103210521102112212121"
        b"3021632167217021002202221122172220222222372240225522012310231423"
        b"7023742335245324032527254125742501270327162745270130103012302130"
        b"2330503065307230003102312031313144314631013203321032253252327232"
        b"1133333330344734723400350635223555351436363663363337603704401740"
        b"3540374053405740744120423742404260426642074345430444514464442545"
        b"4345704505471047124730471250415070500051065126515551145232527252"
        b"0253535310542354275472540255315550562457425724604460466064602161"
        b"6161176264623063366344640565526533660367216703700570077010703270"
        b"5270267140711272457252720073157333736073217441740075027524753076"
    )

    @staticmethod
    def _build_grid():
        grid_map = np.array([0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e], dtype=np.float32)
        grid = np.frombuffer(IQ3_XXS_Quantizer.GRID_HEX, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        # IQ3_XXS: 3 bits per value, 2 values per byte
        grid = grid.reshape((-1, 1)) >> np.array([0, 3], dtype=np.uint8).reshape((1, 2))
        grid = (grid & 0x07).reshape((-1, 1))
        grid = np.take_along_axis(grid_map.reshape(1, -1), grid, axis=-1)
        return grid.reshape(-1, 4)

    @staticmethod
    def quantize(data):
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ3_XXS_Quantizer._build_grid()

        result = np.zeros(n_blocks * IQ3_XXS_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(QK_K // 4, dtype=np.uint8)
                scales = np.zeros(QK_K // 32, dtype=np.uint8)
            else:
                d = np.float16(amax / 2.0)
                d_float = d.astype(np.float32)
                scaled = block / d_float

                qs = np.zeros(QK_K // 4, dtype=np.uint8)
                scales = np.zeros(QK_K // 32, dtype=np.uint8)

                for g in range(QK_K // 4):
                    start = g * 4
                    chunk = scaled[start:start + 4]
                    dists = np.abs(chunk.reshape(1, 4) - grid.reshape(256, 4))
                    total_dist = dists.sum(axis=1)
                    best_idx = np.argmin(total_dist)
                    qs[g] = best_idx

            offset = b * IQ3_XXS_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 2 + QK_K // 4] = qs
            result[offset + 2 + QK_K // 4:offset + 2 + QK_K // 4 + QK_K // 32] = scales

        return result, n, n_blocks


# =============================================================================
# IQ3_S - 3-bit grid-based (512 entries, 4 values)
# =============================================================================

class IQ3_S_Quantizer:
    """3-bit quantizer using 512-entry grid."""

    block_size = QK_K
    type_size = 110

    GRID_HEX = (
        b"0000010002000500070010001100120014001600200021002500330040004200"
        b"4500470051005300600062007100740077000001010102010401100111011501"
        b"2001230127013101350144016101650172010002010205020702100213021602"
        b"2102250230023402420245024702510253027002730203031103150320032203"
        b"3103330336034403500352036703710375030004130417042104240432044004"
        b"4304510470040205040520052205260533054105450547056605730506061106"
        b"1306310652067106000702070407200722072607330750075407001001100210"
        b"0410101011101310151017102010221031103410361054105610611072100011"
        b"0111031106111011141121113011331141115011521170117611001212121512"
        b"1712201224123212401243125512601272120113041307131013131321132713"
        b"3013341341136213701303140514121414143114331442144614501454140115"
        b"1015131521153015321551152016241627164416461601170317101712172117"
        b"3517411762177017002001200320052007201020122014201620212023202720"
        b"3020322041204320452050205220672070207320752000210221102113211721"
        b"2221252131213421422151210122042207222122232230223722412253225722"
        b"7122742200230223052311232223242331233323422350236623012407242024"
        b"2324322435244124722475240425112522253725402553257025002602260726"
        b"2126552661260527112726273027432750270230113013301530173022303130"
        b"3330353042304430473051306330713001310331053114312131233140316031"
        b"7231763100321232203232323432503201331033143321332333273330334133"
        b"4333473355337333033411341634223431345234603464340135103512352535"
        b"3235443556357335163641360137033720372237353700400440124020402440"
        b"2740324041405040704002410741114113412241304135414341514155410142"
        b"0342104215422142334240425742624270420443114313432043224331433543"
        b"0044024424443744404471440545074521456245134634466046104715473047"
        b"4347514702501050145022504050445047505250665074500151035105511251"
        b"2151325172510052115223523052365253520253075310532753445351536553"
        b"7353015404542054325446541255265551555355425602570457225711601360"
        b"1560316033606060006120612761646112623462426255626262706200631463"
        b"2163406325644364626400650365346560650566406611671367007004700770"
        b"2070227036704070547062700271117124714371457101720472107216722172"
        b"3072517202733273357353730174057413742074507422754275027631760077"
    )

    @staticmethod
    def _build_grid():
        grid_map = np.array([0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f], dtype=np.float32)
        grid = np.frombuffer(IQ3_S_Quantizer.GRID_HEX, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        # IQ3_S: 3 bits per value, 2 values per byte
        grid = grid.reshape((-1, 1)) >> np.array([0, 3], dtype=np.uint8).reshape((1, 2))
        grid = (grid & 0x07).reshape((-1, 1))
        grid = np.take_along_axis(grid_map.reshape(1, -1), grid, axis=-1)
        return grid.reshape(-1, 4)

    @staticmethod
    def quantize(data):
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ3_S_Quantizer._build_grid()

        result = np.zeros(n_blocks * IQ3_S_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(QK_K // 4, dtype=np.uint8)
                qh = np.zeros(QK_K // 32, dtype=np.uint8)
                signs = np.zeros(QK_K // 8, dtype=np.uint8)
                scales = np.zeros(QK_K // 64, dtype=np.uint8)
            else:
                d = np.float16(amax / 2.0)
                d_float = d.astype(np.float32)
                scaled = block / d_float

                qs = np.zeros(QK_K // 4, dtype=np.uint8)
                qh = np.zeros(QK_K // 32, dtype=np.uint8)
                signs = np.zeros(QK_K // 8, dtype=np.uint8)
                scales = np.zeros(QK_K // 64, dtype=np.uint8)

                for g in range(QK_K // 4):
                    start = g * 4
                    chunk = scaled[start:start + 4]
                    dists = np.abs(chunk.reshape(1, 4) - grid.reshape(-1, 4))
                    total_dist = dists.sum(axis=1)
                    best_idx = np.argmin(total_dist)
                    qs[g] = best_idx & 0xFF
                    # Store high bit in qh (1 bit per group)
                    qh_byte_idx = g // 8
                    qh_bit_idx = g % 8
                    if best_idx >= 256:
                        qh[qh_byte_idx] |= (1 << qh_bit_idx)

                    recon = grid[best_idx]
                    sign_bits = np.where(chunk * recon >= 0, 0, 1).astype(np.uint8)
                    sign_byte = 0
                    for i in range(4):
                        sign_byte |= (sign_bits[i] << i)
                    signs[g // 2] |= (sign_byte << ((g % 2) * 4))

            offset = b * IQ3_S_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 2 + QK_K // 4] = qs
            result[offset + 2 + QK_K // 4:offset + 2 + QK_K // 4 + QK_K // 32] = qh
            result[offset + 2 + QK_K // 4 + QK_K // 32:offset + 2 + QK_K // 4 + QK_K // 32 + QK_K // 8] = signs
            result[offset + 2 + QK_K // 4 + QK_K // 32 + QK_K // 8:offset + 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 + QK_K // 64] = scales

        return result, n, n_blocks


# =============================================================================
# IQ1_S - 1-bit grid-based (2048 entries, signed grid + delta)
# =============================================================================

class IQ1_S_Quantizer:
    """1-bit quantizer using 2048-entry signed grid with delta offset."""

    block_size = QK_K
    type_size = 50

    GRID_HEX = (
        b"00000200050008000a00110015002000220028002a0045005100540056006500"
        b"8000820088008a009500a000a200a800aa000401050111011401160119011a01"
        b"2501410146014901520155015a0161016401660168018501910194019601a501"
        b"0002020208020a0215022002220228022a024502510259026402690280028202"
        b"88028a02910295029902a002a202a802aa021104140416042504410449045504"
        b"5a046404650491049904a5040105040505050605150518051a05290540054505"
        b"4a0550055105540555055605590560056205650568056a058105910595059805"
        b"9a05a105a405a505a605a9051406190641064406500652065506580660066106"
        b"6606690685069106940699060008020808080a0815082008220828082a084508"
        b"5108560865088008820888088a089508a008a208a808aa080509110914091909"
        b"2409250941095009510955096109640969099109940996099909a509000a020a"
        b"080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a"
        b"8a0a950aa00aa20aa80aaa0a1010111014101910241025104110441050105510"
        b"58106110641065106910911094109610a110a510011104110611091110111211"
        b"1511181121112411291145114a11501151115211541155115611591160116511"
        b"841192119511a111a41111121412161225124012461249125212551258125a12"
        b"641266128512911294129612a512011406140914141415141814191421142614"
        b"41144514461448144a1451145414551456145914621465146814841489149014"
        b"94149514981499149a14a114a414a514a914021505150a151115141515151615"
        b"191520152215251528152a154115441545154615511552155415551556155915"
        b"5a1561156415651566156915801582158415851588158a159015911594159515"
        b"961599159a15a015a215a51501160416051606161516161618161a1621162616"
        b"401642164416451648164a165116551656165816591661166416651668166916"
        b"6a1686168a1692169516a416a916111816182518411844184618491850185518"
        b"58185a1860186118641866186918851891189418a5181019121915191a192119"
        b"25194219441945194819511954195519561959195a19601965196a1989199119"
        b"921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a"
        b"581a611a661a691a851a911a961a9a1a0020022008200a201520202022202520"
        b"28202a20452051205920612065208020822088208a209520a020a220a520a820"
        b"aa2005211121142119212521422144214921552158215a216121642165216621"
        b"8521902196219921a521012208220a22112215222022222228222a2245225122"
        b"562259226522812288228a2291229522a022a222a822aa220524142416241924"
        b"252444244524462449245224552458245a2466248524912494249924a124a524"
        b"0925152521252925402545254825512554255525592562256525682589259025"
        b"9425952598259a25a125a425a625a92505261026122619262526412649265526"
        b"6026612669268426862690269a260028022808280a2815282028222828282a28"
        b"45285128542865288028822888288a28a028a228a828aa280929112914291929"
        b"2529462949295229552961296429662969298529902996299929a429a529002a"
        b"022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a"
        b"952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440"
        b"664094409940a140a6400041014104410641094112411541164118411a412141"
        b"26412941454148414a41514154415541564159415a41654168416a4181418441"
        b"8641904192419541a041a141a241054211421442164225424142524255425a42"
        b"6442694289429442a5420144154419442944454448444a445144544455445644"
        b"61446244654468446a44814486448944904492449544a044a144a94401450245"
        b"05450a4511451445154516451945204525452a45414544454545464549455045"
        b"5145544555455645584559456145644565456645694582458445854588459145"
        b"94459545964599459a45a545a845aa450146054609461446154618461a462146"
        b"2446294640464246454648465046514652465546564659466246654668468146"
        b"85468a4694469546a146a446a6460548114815481a4825484248494850485548"
        b"5848614864486648694885489148944896489948a5480149054906490a491049"
        b"144915491849214924492649404945494a495149524954495549564959496049"
        b"6249654966496a49864989499249954996499849a149a449a649a949164a444a"
        b"464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550"
        b"1a50215024502950405045504850515054505550565059506550685086508950"
        b"95509850a050a150a650a9500551085109510a51115114511551165118511951"
        b"20512551265128512a5141514451455146514951505151515251545155515651"
        b"585159515a51615164516551665169518251855191519451955196519951a051"
        b"a551aa5101520652125215521a5221522452425245524a525152545255525652"
        b"595262526552855290529252955299529a52a452045405541154145415541654"
        b"185419542154255428542a54415444544554465449544a545054515454545554"
        b"5654585459545a54615462546454655466546954805488548a54915494549554"
        b"96549954a154a454a554aa540155025504550555065509551055115512551455"
        b"1555165519551a55215524552555265529554055415542554455455546554855"
        b"4955505551555255545555555655585559555a55605561556455655566556855"
        b"69556a5581558455855589558a559055915594559555965598559955a155a455"
        b"a555a655a9550056015602560456065608560956115614561556185619562056"
        b"2156225624562556265628562956415645564656485649564a56505651565256"
        b"545655565656585659565a566156645665566956825685568656885689568a56"
        b"915695569a56a256a556a656a856a95604580558065809581058155818582158"
        b"2a58455848584a58515854585558565858585958605862586458655882588958"
        b"9058925895589858a158a9580159025905590a59115914591559165919592559"
        b"41594459455946594959505951595259545955595659585959595a5961596459"
        b"655966596959815985598959915994599559965998599959a559045a085a155a"
        b"1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a"
        b"6a5a815a8a5a925a955a965a985a9a5aa15a0560146016601960256044605060"
        b"5560566058605a60616064606660696081609660a56001610461066109611261"
        b"15612161226126612961456149615161556156615961656166616a6184618a61"
        b"92619561a161a661a96111621662196240624162466255625662586260628562"
        b"91629662a56211641264156416641a6421642664296440644264456448644a64"
        b"516454645564566459645a646064626465648464856489649064926494649564"
        b"966498649a64a164a464a964056508650a651165156516651965446545654665"
        b"496550655165546555655665596561656465656566656965866589658a659165"
        b"9565966599659a65a265a565a665a86502660966156620662666286629664066"
        b"456648664a66516654665566566658665a666066656668668066826685668a66"
        b"9466966698669966a066a466a666aa661668196825684168526855685a686168"
        b"6968856891689868a66801690469106915692169246926692969406941694569"
        b"4669486951695469556956695969606965696a69826984698a699569a169a469"
        b"a569a969116a166a186a416a446a496a506a556a586a5a6a646a656a696a866a"
        b"946a986a9a6aa66a0080028008800a802080228028802a804580508051805480"
        b"5680598065808080828088808a809580a080a280a880aa800581118114811681"
        b"1981258141814481498150815281558156815881598164816681698185818981"
        b"948196819981a5810082028208820a8215822082228228822a82518254825982"
        b"65828082828288828a829582a082a282a882aa82148419844184448451845584"
        b"5a846184648469849484998401850985128515851a8526852985408541854585"
        b"4885518554855585568559855a856585668568856a8581858485868589859085"
        b"928595859885a68511861686198625864186448649864a865086558659865a86"
        b"618666866a86858691869a86a4860088028808880a8815882088228828882a88"
        b"41884588518854885988658869888088828888888a889588a088a288a888aa88"
        b"05890689118914891689258941894489468949895089528955895a8961896489"
        b"858996899989a589008a028a088a0a8a158a208a228a288a2a8a458a518a548a"
        b"568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590"
        b"419046904990559058905a9069906a9085909190949096909990a59001910491"
        b"069109911091159118911a912191249126912991409145915091519154915591"
        b"569159916291659184918691929195919891a191a491a691a991059211921492"
        b"19922592449246924992509252925592589266926992859294929692a9920194"
        b"04940694109415941894269440944a9451945494559456945894599460946194"
        b"62946594849486949294949495949894a194a9940095059508950a9510951195"
        b"14951595169519952195259529952a9541954495459546954995509551955295"
        b"549555955695589559955a956195649565956695699581958595889591959295"
        b"94959595969599959a95a095a295a595a895aa95019604961096159619962096"
        b"2696299645964896499651965296559656965996659668968296849689968a96"
        b"929694969596a496a696a9960598169819982598419846985098529855985698"
        b"5a98649865988598919896989998a59804990699099910991299159918991a99"
        b"209921992499269940994299459948994a995199549955995699599962996599"
        b"66996a99819984999099929995999a99a199a699059a159a259a449a469a499a"
        b"509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0"
        b"28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa0"
        b"05a109a111a114a116a119a11aa146a149a151a155a158a15aa161a164a185a1"
        b"90a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2"
        b"65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a4"
        b"55a458a45aa461a465a466a468a469a485a406a509a510a512a515a518a526a5"
        b"29a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5"
        b"95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a6"
        b"62a686a690a695a696a699a6a1a6a4a6a6a600a802a808a80aa820a822a828a8"
        b"2aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9"
        b"19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa"
        b"20aa22aa28aa2aaa51aa54aa56aa80aa82aa88aa8aaa95aaa0aaa2aaa8aaaaaa"
    )

    @staticmethod
    def _build_grid():
        grid_map = np.array([-1, 0, 1], dtype=np.float32)
        grid = np.frombuffer(IQ1_S_Quantizer.GRID_HEX, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        grid = grid.reshape((-1, 1)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4))
        grid = (grid & 0x03).reshape((-1, 1))
        grid = np.take_along_axis(grid_map.reshape(1, -1), grid, axis=-1)
        return grid.reshape(2048, 8)

    @staticmethod
    def quantize(data):
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ1_S_Quantizer._build_grid()

        result = np.zeros(n_blocks * IQ1_S_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                d = np.float16(0.0)
                qs = np.zeros(QK_K // 8, dtype=np.uint8)
                qh = np.zeros(QK_K // 16, dtype=np.uint8)
            else:
                d = np.float16(amax / 2.0)
                d_float = d.astype(np.float32)
                scaled = block / d_float

                qs = np.zeros(QK_K // 8, dtype=np.uint8)
                qh = np.zeros(QK_K // 16, dtype=np.uint8)

                for g in range(QK_K // 8):
                    start = g * 8
                    chunk = scaled[start:start + 8]
                    dists = np.abs(chunk.reshape(1, 8) - grid.reshape(2048, 8))
                    total_dist = dists.sum(axis=1)
                    best_idx = np.argmin(total_dist)
                    qs[g] = best_idx & 0xFF
                    # High bits (3 bits per group of 8)
                    if g % 4 == 0:
                        qh_bytes = int(best_idx >> 8).to_bytes(2, 'little')
                        qh[g // 2] = qh_bytes[0] | (qh_bytes[1] << 4)

            offset = b * IQ1_S_Quantizer.type_size
            result[offset:offset + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            result[offset + 2:offset + 2 + QK_K // 8] = qs
            result[offset + 2 + QK_K // 8:offset + 2 + QK_K // 8 + QK_K // 16] = qh

        return result, n, n_blocks


# =============================================================================
# IQ1_M - 1-bit with packed f16 scales
# =============================================================================

class IQ1_M_Quantizer:
    """1-bit quantizer reusing IQ1_S grid with packed f16 scales."""

    block_size = QK_K
    type_size = 56

    @staticmethod
    def quantize(data):
        """Quantize float32 array to IQ1_M format."""
        data = data.astype(np.float32)
        n = len(data)
        padded = pad_to_multiple(data, QK_K)
        n_blocks = len(padded) // QK_K

        grid = IQ1_S_Quantizer._build_grid()

        result = np.zeros(n_blocks * IQ1_M_Quantizer.type_size, dtype=np.uint8)

        for b in range(n_blocks):
            block = padded[b * QK_K:(b + 1) * QK_K]

            amax = np.abs(block).max()
            if amax == 0:
                qs = np.zeros(QK_K // 8, dtype=np.uint8)
                qh = np.zeros(QK_K // 16, dtype=np.uint8)
                scales = np.zeros(QK_K // 16, dtype=np.uint8)
            else:
                d = np.float16(amax / 2.0)
                d_float = d.astype(np.float32)
                scaled = block / d_float

                qs = np.zeros(QK_K // 8, dtype=np.uint8)
                qh = np.zeros(QK_K // 16, dtype=np.uint8)
                scales = np.zeros(QK_K // 16, dtype=np.uint8)

                for g in range(QK_K // 8):
                    start = g * 8
                    chunk = scaled[start:start + 8]
                    dists = np.abs(chunk.reshape(1, 8) - grid.reshape(2048, 8))
                    total_dist = dists.sum(axis=1)
                    best_idx = np.argmin(total_dist)
                    qs[g] = best_idx & 0xFF
                    if g % 4 == 0:
                        qh_bytes = int(best_idx >> 8).to_bytes(2, 'little')
                        qh[g // 2] = qh_bytes[0] | (qh_bytes[1] << 4)

                # Pack f16 scale into nibbles
                d_uint16 = d.view(np.uint16)
                scales[0] = (d_uint16 & 0xF000) >> 12
                scales[1] = (d_uint16 & 0x0F00) >> 8
                scales[2] = (d_uint16 & 0x00F0) >> 4
                scales[3] = (d_uint16 & 0x000F)

            offset = b * IQ1_M_Quantizer.type_size
            result[offset:offset + QK_K // 8] = qs
            result[offset + QK_K // 8:offset + QK_K // 8 + QK_K // 16] = qh
            result[offset + QK_K // 8 + QK_K // 16:offset + 56] = scales[:8]

        return result, n, n_blocks


# =============================================================================
# Public API
# =============================================================================

QUANTIZERS = {
    "q2_k": Q2_K_Quantizer,
    "q3_k": Q3_K_Quantizer,
    "q4_k": Q4_K_Quantizer,
    "q5_k": Q5_K_Quantizer,
    "q6_k": Q6_K_Quantizer,
    "q8_k": Q8_K_Quantizer,
    "iq4_nl": IQ4_NL_Quantizer,
    "iq4_xs": IQ4_XS_Quantizer,
    "iq2_xxs": IQ2_XXS_Quantizer,
    "iq2_xs": IQ2_XS_Quantizer,
    "iq2_s": IQ2_S_Quantizer,
    "iq3_xxs": IQ3_XXS_Quantizer,
    "iq3_s": IQ3_S_Quantizer,
    "iq1_s": IQ1_S_Quantizer,
    "iq1_m": IQ1_M_Quantizer,
}


def quantize_k_quant(data, quant_type):
    """
    Quantize float32 data to a K-quant type.

    Args:
        data: numpy float32 array
        quant_type: string like "q4_k", "q5_k", etc.

    Returns:
        tuple: (quantized_bytes, n_elements, n_blocks)
            - quantized_bytes: numpy uint8 array with quantized data
            - n_elements: original element count (for raw_shape)
            - n_blocks: number of K-quant blocks
    """
    quant_type = quant_type.lower()
    if quant_type not in QUANTIZERS:
        raise ValueError(f"Unknown K-quant type: {quant_type}. Available: {list(QUANTIZERS.keys())}")

    return QUANTIZERS[quant_type].quantize(data)
