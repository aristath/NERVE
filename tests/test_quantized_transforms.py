from __future__ import annotations

import pytest

from nerve.compilation import ModelCompileError
from nerve.quantized_transforms import (
    MXFP4_E2M1_FP8_E4M3_BITS,
    MXFP4_E2M1_SINT8_CODES,
    e8m0_scale,
    expand_mxfp4_e2m1_to_fp8_e4m3,
    expand_mxfp4_e2m1_to_sint8,
    fp8_e4m3_value,
    mxfp4_e2m1_fp8_e4m3_bits,
    mxfp4_e2m1_sint8_code,
    mxfp4_sint8_reconstruction_scale,
    mxfp4_value,
    reconstructed_mxfp4_value,
    resident_mxfp4_value,
)


def test_mxfp4_expands_low_nibble_first_to_signed_int8_codes() -> None:
    packed = bytes((0x10, 0x87, 0xFE))

    expanded = expand_mxfp4_e2m1_to_sint8(packed)

    assert expanded == bytes((0, 1, 12, 0, 248, 244))


def test_sint8_reconstruction_is_exact_for_every_finite_mxfp4_value() -> None:
    for scale_byte in range(0xFF):
        for nibble in range(16):
            assert reconstructed_mxfp4_value(nibble, scale_byte) == mxfp4_value(
                nibble, scale_byte
            )


def test_fp8_resident_expansion_is_exact_for_every_finite_mxfp4_value() -> None:
    assert len(MXFP4_E2M1_FP8_E4M3_BITS) == 16
    assert expand_mxfp4_e2m1_to_fp8_e4m3(bytes((0x10, 0xF8))) == bytes(
        (0x00, 0x30, 0x80, 0xCC)
    )
    for nibble in range(16):
        bits = mxfp4_e2m1_fp8_e4m3_bits(nibble)
        assert fp8_e4m3_value(bits) == mxfp4_value(nibble, 127)
        for scale_byte in range(0xFF):
            assert resident_mxfp4_value(
                nibble,
                scale_byte,
            ) == mxfp4_value(nibble, scale_byte)


def test_sint8_reconstruction_halves_even_the_two_subnormal_boundary_scales() -> None:
    assert mxfp4_sint8_reconstruction_scale(0) * 2 == e8m0_scale(0)
    assert mxfp4_sint8_reconstruction_scale(1) * 2 == e8m0_scale(1)


@pytest.mark.parametrize("value", (-1, 16, True, 1.5, None))
def test_mxfp4_code_rejects_invalid_nibbles(value: object) -> None:
    with pytest.raises(ModelCompileError, match="MXFP4 nibble"):
        mxfp4_e2m1_sint8_code(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (-1, 255, 256, True, 1.5, None))
def test_e8m0_reconstruction_rejects_nonfinite_or_invalid_scales(
    value: object,
) -> None:
    with pytest.raises(ModelCompileError, match="finite E8M0 scale byte"):
        mxfp4_sint8_reconstruction_scale(value)  # type: ignore[arg-type]


def test_codebook_contract_covers_all_nibbles_without_overflow() -> None:
    assert len(MXFP4_E2M1_SINT8_CODES) == 16
    assert all(-128 <= value <= 127 for value in MXFP4_E2M1_SINT8_CODES)
