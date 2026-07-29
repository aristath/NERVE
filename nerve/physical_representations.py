from __future__ import annotations

from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError


FP8_PREQUANTIZATION_CONTRACT = "bf16_blockwise_fp8_e4m3_f32_scale.v1"
INT8_PREQUANTIZATION_CONTRACT = "bf16_blockwise_symmetric_int8_f32_scale.v1"
PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT = (
    "bf16_blockwise_symmetric_int8_pairpacked_f32_scale_i32_sum.v1"
)


@dataclass(frozen=True)
class PhysicalRepresentationContract:
    id: str
    helper_op: str
    output_signal_suffixes: tuple[str, ...]
    output_element_bytes: tuple[int, ...]


_CONTRACTS = {
    contract.id: contract
    for contract in (
        PhysicalRepresentationContract(
            id=FP8_PREQUANTIZATION_CONTRACT,
            helper_op="quantize_fp8_e4m3",
            output_signal_suffixes=("fp8_e4m3", "scale_f32"),
            output_element_bytes=(1, 4),
        ),
        PhysicalRepresentationContract(
            id=INT8_PREQUANTIZATION_CONTRACT,
            helper_op="quantize_int8_symmetric",
            output_signal_suffixes=("int8", "scale_f32"),
            output_element_bytes=(1, 4),
        ),
        PhysicalRepresentationContract(
            id=PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT,
            helper_op="quantize_int8_symmetric_pairpacked",
            output_signal_suffixes=("int8_pairpacked", "scale_f32", "sum_i32"),
            output_element_bytes=(1, 4, 4),
        ),
    )
}


def physical_representation_contract(contract_id: str) -> PhysicalRepresentationContract:
    try:
        return _CONTRACTS[contract_id]
    except KeyError as error:
        raise ModelCompileError(
            f"unsupported physical representation contract {contract_id!r}"
        ) from error


def physical_representation_contract_for_helper(
    helper_op: str,
) -> PhysicalRepresentationContract | None:
    return next(
        (contract for contract in _CONTRACTS.values() if contract.helper_op == helper_op),
        None,
    )


def prequantization_spec(
    contract_id: str,
    *,
    input_size: int,
    block_columns: int,
) -> Json:
    physical_representation_contract(contract_id)
    return {
        "contract": contract_id,
        "input_size": int(input_size),
        "block_columns": int(block_columns),
    }
