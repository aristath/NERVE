from __future__ import annotations

from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError


FP8_PREQUANTIZATION_CONTRACT = "bf16_blockwise_fp8_e4m3_f32_scale.v1"
SPARSE_MOE_FP8_INTERMEDIATE_CONTRACT = (
    "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_u32_route_map.v1"
)
INT8_PREQUANTIZATION_CONTRACT = "bf16_blockwise_symmetric_int8_f32_scale.v1"
PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT = (
    "bf16_blockwise_symmetric_int8_pairpacked_f32_scale_i32_sum.v1"
)
ATTENTION_PARTIALS_CONTRACT = "bf16_attention_partition_partials_f32.v1"


@dataclass(frozen=True)
class PhysicalRepresentationContract:
    id: str
    helper_op: str | None
    output_signal_suffixes: tuple[str, ...]
    output_element_bytes: tuple[int, ...]
    metadata_fields: tuple[str, ...] = ("element_count", "block_columns")
    logical_input_count: int = 1
    mirrors_consumer_state_reads: bool = False


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
            id=SPARSE_MOE_FP8_INTERMEDIATE_CONTRACT,
            helper_op=None,
            output_signal_suffixes=(
                "expert_fp8_e4m3",
                "expert_scale_f32",
                "route_map_u32",
            ),
            output_element_bytes=(1, 4, 4),
            metadata_fields=(
                "element_count",
                "block_columns",
                "experts_per_token",
            ),
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
        PhysicalRepresentationContract(
            id=ATTENTION_PARTIALS_CONTRACT,
            helper_op="attention_partition_partials",
            output_signal_suffixes=("attention_partials_f32",),
            output_element_bytes=(4,),
            metadata_fields=(
                "query_heads",
                "key_value_heads",
                "head_width",
                "partition_count",
                "scale",
                "window_size",
                "attention_sinks",
            ),
            logical_input_count=4,
            mirrors_consumer_state_reads=True,
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
        (
            contract
            for contract in _CONTRACTS.values()
            if contract.helper_op is not None
            and contract.helper_op == helper_op
        ),
        None,
    )


def prequantization_spec(
    contract_id: str,
    *,
    input_size: int,
    block_columns: int,
    **metadata: int,
) -> Json:
    contract = physical_representation_contract(contract_id)
    spec = {
        "contract": contract_id,
        "input_size": int(input_size),
        "block_columns": int(block_columns),
    }
    spec.update({key: int(value) for key, value in metadata.items()})
    missing = set(contract.metadata_fields) - {
        "element_count" if key == "input_size" else key for key in spec
    }
    if missing:
        raise ModelCompileError(
            f"physical representation contract {contract_id!r} is missing metadata "
            f"{sorted(missing)}"
        )
    return spec
