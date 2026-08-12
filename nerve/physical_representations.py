from __future__ import annotations

from dataclasses import dataclass
import re

from nerve.compilation import Json, ModelCompileError


FP8_PREQUANTIZATION_CONTRACT = "bf16_blockwise_fp8_e4m3_f32_scale.v1"
FP8_E8M0_PREQUANTIZATION_CONTRACT = (
    "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
)
SPARSE_MOE_FP8_INTERMEDIATE_CONTRACT = (
    "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_u32_route_map.v1"
)
INT8_PREQUANTIZATION_CONTRACT = "bf16_blockwise_symmetric_int8_f32_scale.v1"
PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT = (
    "bf16_blockwise_symmetric_int8_pairpacked_f32_scale_i32_sum.v1"
)
ATTENTION_PARTIALS_CONTRACT = "bf16_attention_partition_partials_f32.v1"
KERNEL_RESOURCE_REPRESENTATION_DISPATCH_SCHEMA = (
    "nerve.kernel_resource_representation_dispatch.v2"
)
MXFP4_E2M1_G32_RESOURCE_REPRESENTATION = "mxfp4_e2m1_g32"
SELECTOR_MAPPED_MXFP4_OR_NATIVE_FP8_RESOURCE_REPRESENTATION = (
    "selector_mapped_mxfp4_e2m1_g32_or_fp8_e4m3_e8m0_b128"
)
MXFP4_E2M1_TO_FP8_E4M3_RESIDENT_DERIVATION = "mxfp4_e2m1_to_fp8_e4m3"


def fixed_mxfp4_resource_representation_dispatch() -> Json:
    return {
        "schema": KERNEL_RESOURCE_REPRESENTATION_DISPATCH_SCHEMA,
        "source_representation": MXFP4_E2M1_G32_RESOURCE_REPRESENTATION,
        "source_representation_boundary": None,
        "resident_derivation": None,
        "selection": "fixed_source",
    }


def adaptive_mxfp4_resource_representation_dispatch() -> Json:
    return {
        "schema": KERNEL_RESOURCE_REPRESENTATION_DISPATCH_SCHEMA,
        "source_representation": MXFP4_E2M1_G32_RESOURCE_REPRESENTATION,
        "source_representation_boundary": None,
        "resident_derivation": (MXFP4_E2M1_TO_FP8_E4M3_RESIDENT_DERIVATION),
        "selection": "resource_address_tag",
    }


def independent_expert_resource_representation_dispatch(
    shader_file: str,
    *,
    adaptive: bool = False,
) -> Json:
    mixed = re.search(r"_native_fp8_e4m3_se8m0_b128_nf(\d+)_", shader_file)
    source_representation = MXFP4_E2M1_G32_RESOURCE_REPRESENTATION
    boundary = None
    if mixed is not None:
        source_representation = (
            SELECTOR_MAPPED_MXFP4_OR_NATIVE_FP8_RESOURCE_REPRESENTATION
        )
        boundary = int(mixed.group(1))
    return {
        "schema": KERNEL_RESOURCE_REPRESENTATION_DISPATCH_SCHEMA,
        "source_representation": source_representation,
        "source_representation_boundary": boundary,
        "resident_derivation": (
            MXFP4_E2M1_TO_FP8_E4M3_RESIDENT_DERIVATION if adaptive else None
        ),
        "selection": "resource_address_tag" if adaptive else "fixed_source",
    }


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
            id=FP8_E8M0_PREQUANTIZATION_CONTRACT,
            helper_op="quantize_fp8_e4m3_e8m0",
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
