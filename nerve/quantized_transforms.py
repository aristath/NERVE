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

    return mxfp4_e2m1_sint8_code(nibble) * mxfp4_sint8_reconstruction_scale(
        scale_byte
    )


def _validate_e8m0_scale_byte(scale_byte: int) -> None:
    if (
        isinstance(scale_byte, bool)
        or not isinstance(scale_byte, int)
        or not 0 <= scale_byte <= 0xFE
    ):
        raise ModelCompileError(
            "finite E8M0 scale byte must be in [0, 254], "
            f"got {scale_byte!r}"
        )


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent, 1)
    return Fraction(1, 1 << -exponent)
