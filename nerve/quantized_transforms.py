from __future__ import annotations

from fractions import Fraction

from nerve.compilation import ModelCompileError


# MXFP4 E2M1 stores sign in bit 3 and one of these non-negative magnitudes in
# bits 0..2.  Multiplication by two makes every finite value an exact SINT8
# integer.  The original value is recovered by halving the associated E8M0
# group scale.  This is a representation transform, not requantization.
MXFP4_E2M1_SINT8_CODES = (
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    0,
    -1,
    -2,
    -3,
    -4,
    -6,
    -8,
    -12,
)

# Every finite E2M1 code is exactly representable in E4M3. Keeping the table in
# the shared numeric module lets the compiler, proof verifier, and runtime tests
# agree on one bit-level contract.
MXFP4_E2M1_FP8_E4M3_BITS = (
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x80,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
)


def mxfp4_e2m1_sint8_code(nibble: int) -> int:
    if isinstance(nibble, bool) or not isinstance(nibble, int) or not 0 <= nibble < 16:
        raise ModelCompileError(f"MXFP4 nibble must be in [0, 15], got {nibble!r}")
    return MXFP4_E2M1_SINT8_CODES[nibble]


def expand_mxfp4_e2m1_to_sint8(packed: bytes) -> bytes:
    """Expand low-nibble-first MXFP4 storage into exact signed-byte codes."""

    expanded = bytearray(len(packed) * 2)
    for index, value in enumerate(packed):
        expanded[index * 2] = mxfp4_e2m1_sint8_code(value & 0x0F) & 0xFF
        expanded[index * 2 + 1] = mxfp4_e2m1_sint8_code(value >> 4) & 0xFF
    return bytes(expanded)


def mxfp4_e2m1_fp8_e4m3_bits(nibble: int) -> int:
    if isinstance(nibble, bool) or not isinstance(nibble, int) or not 0 <= nibble < 16:
        raise ModelCompileError(f"MXFP4 nibble must be in [0, 15], got {nibble!r}")
    return MXFP4_E2M1_FP8_E4M3_BITS[nibble]


def expand_mxfp4_e2m1_to_fp8_e4m3(packed: bytes) -> bytes:
    """Expand low-nibble-first MXFP4 into exact E4M3 byte codes."""

    expanded = bytearray(len(packed) * 2)
    for index, value in enumerate(packed):
        expanded[index * 2] = mxfp4_e2m1_fp8_e4m3_bits(value & 0x0F)
        expanded[index * 2 + 1] = mxfp4_e2m1_fp8_e4m3_bits(value >> 4)
    return bytes(expanded)


def e8m0_scale(scale_byte: int) -> Fraction:
    """Return the finite power-of-two value consumed by NERVE's E8M0 ABI."""

    _validate_e8m0_scale_byte(scale_byte)
    exponent = -127 if scale_byte == 0 else scale_byte - 127
    return _power_of_two(exponent)


def mxfp4_sint8_reconstruction_scale(scale_byte: int) -> Fraction:
    """Return the exact group scale for the expanded SINT8 code stream."""

    return e8m0_scale(scale_byte) / 2


def mxfp4_value(nibble: int, scale_byte: int) -> Fraction:
    """Decode one finite logical MXFP4 value exactly as a rational number."""

    code = mxfp4_e2m1_sint8_code(nibble)
    return Fraction(code, 2) * e8m0_scale(scale_byte)


def reconstructed_mxfp4_value(nibble: int, scale_byte: int) -> Fraction:
    """Decode through the exact SINT8 representation used by a candidate."""

    return mxfp4_e2m1_sint8_code(nibble) * mxfp4_sint8_reconstruction_scale(scale_byte)


def fp8_e4m3_value(bits: int) -> Fraction:
    """Decode a finite E4M3 byte exactly as a rational number."""

    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits <= 0xFF:
        raise ModelCompileError(f"FP8 E4M3 bits must be in [0, 255], got {bits!r}")
    sign = -1 if bits & 0x80 else 1
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    if exponent == 0:
        magnitude = Fraction(mantissa, 8) * _power_of_two(-6)
    else:
        magnitude = Fraction(8 + mantissa, 8) * _power_of_two(exponent - 7)
    return sign * magnitude


def resident_mxfp4_value(nibble: int, scale_byte: int) -> Fraction:
    """Decode through the exact resident FP8 form used by the runtime."""

    return fp8_e4m3_value(mxfp4_e2m1_fp8_e4m3_bits(nibble)) * e8m0_scale(scale_byte)


def _validate_e8m0_scale_byte(scale_byte: int) -> None:
    if (
        isinstance(scale_byte, bool)
        or not isinstance(scale_byte, int)
        or not 0 <= scale_byte <= 0xFE
    ):
        raise ModelCompileError(
            f"finite E8M0 scale byte must be in [0, 254], got {scale_byte!r}"
        )


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent, 1)
    return Fraction(1, 1 << -exponent)
