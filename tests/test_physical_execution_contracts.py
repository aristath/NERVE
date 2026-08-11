from copy import deepcopy
import json
from pathlib import Path

import pytest

from nerve.physical_execution_contracts import (
    PhysicalExecutionContractError,
    artifact_sha256,
    build_kernel_physical_execution_contracts,
    implementation_digest,
    seal_physical_execution_contract,
    validate_physical_execution_contract,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "execution-contracts"
    / "fixtures"
    / "tensor_parallel_projection.json"
)


def fixture_contract() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def test_shared_tensor_parallel_contract_is_valid() -> None:
    validate_physical_execution_contract(fixture_contract())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(strategy="model_named_deepseek"), "strategy"),
        (lambda value: value.update(parameter_partitions=[]), "partition"),
        (
            lambda value: value["inputs"][0].update(
                distribution="sharded", dimension=0
            ),
            "alignment",
        ),
        (
            lambda value: value["outputs"][0].update(
                collection="reduced", reduction=None
            ),
            "dimension",
        ),
        (lambda value: value.update(model_name="deepseek"), "unknown"),
    ],
)
def test_invalid_or_model_specific_distribution_fails_closed(mutate, message: str) -> None:
    contract = fixture_contract()
    mutate(contract)
    with pytest.raises(PhysicalExecutionContractError, match=message):
        validate_physical_execution_contract(contract)


def partial_output_contract() -> dict[str, object]:
    contract = fixture_contract()
    contract["execution_form"] = "partitioned_input_partial_output"
    contract["inputs"][0] = {
        "binding": 0,
        "distribution": "sharded",
        "dimension": 0,
        "alignment_elements": 128,
    }
    contract["outputs"][0] = {
        "binding": 1,
        "collection": "reduced",
        "reduction": {
            "operation": "sum_f32",
            "dimension_name": "output_rows",
            "finalization": {"kind": "store_f32"},
        },
    }
    contract["partition_extent"] = {
        "dimension_name": "input_width",
        "elements": contract["geometry"]["dimensions"]["input_width"],
        "alignment_elements": 128,
    }
    contract["partition_launch"] = {
        "workgroup_x": "repeated",
        "origin": "push_constant_u32",
        "origin_push_constant": "input_start",
        "count_push_constant": "input_count",
    }
    return contract


def test_partial_output_requires_typed_f32_sum_reduction() -> None:
    contract = partial_output_contract()
    validate_physical_execution_contract(contract)

    contract["outputs"][0]["reduction"] = {
        "operation": "vendor_magic",
        "dimension_name": "output_rows",
        "finalization": {"kind": "store_f32"},
    }
    with pytest.raises(PhysicalExecutionContractError, match="reduction"):
        validate_physical_execution_contract(contract)

    contract = partial_output_contract()
    contract["formats"]["accumulation"] = "bf16"
    with pytest.raises(PhysicalExecutionContractError, match="f32 accumulation"):
        validate_physical_execution_contract(contract)

    contract = partial_output_contract()
    contract["outputs"][0]["reduction"]["dimension_name"] = "missing"
    with pytest.raises(PhysicalExecutionContractError, match="geometry dimension"):
        validate_physical_execution_contract(contract)


def test_bf16_residual_reduction_finalization_is_typed_and_replicated() -> None:
    contract = partial_output_contract()
    contract["inputs"].append({"binding": 3, "distribution": "replicated"})
    contract["outputs"][0]["reduction"]["finalization"] = {
        "kind": "add_bf16_residual_to_bf16",
        "residual_binding": 3,
    }
    validate_physical_execution_contract(contract)

    contract["inputs"][1] = {
        "binding": 3,
        "distribution": "sharded",
        "dimension": 0,
        "alignment_elements": 128,
    }
    with pytest.raises(PhysicalExecutionContractError, match="must be replicated"):
        validate_physical_execution_contract(contract)

    contract = partial_output_contract()
    contract["outputs"][0]["reduction"]["finalization"] = {
        "kind": "add_bf16_residual_to_bf16",
        "residual_binding": 99,
    }
    with pytest.raises(PhysicalExecutionContractError, match="contract input"):
        validate_physical_execution_contract(contract)

    contract = partial_output_contract()
    contract["inputs"].append({"binding": 3, "distribution": "replicated"})
    contract["outputs"][0]["reduction"]["finalization"] = {
        "kind": "add_bf16_residual_to_bf16",
        "residual_binding": 3,
    }
    contract["geometry"]["dimensions"]["output_rows"] -= 1
    with pytest.raises(PhysicalExecutionContractError, match="even element count"):
        validate_physical_execution_contract(contract)

    contract = partial_output_contract()
    contract["outputs"][0]["reduction"]["finalization"] = {
        "kind": "store_f32",
        "residual_binding": 3,
    }
    with pytest.raises(PhysicalExecutionContractError, match="unknown fields"):
        validate_physical_execution_contract(contract)


def test_reduced_output_and_partial_output_form_are_bidirectional() -> None:
    reduced_with_wrong_form = partial_output_contract()
    reduced_with_wrong_form["execution_form"] = (
        "replicated_input_partitioned_output"
    )
    with pytest.raises(PhysicalExecutionContractError, match="reduced outputs"):
        validate_physical_execution_contract(reduced_with_wrong_form)

    partial_without_reduction = fixture_contract()
    partial_without_reduction["execution_form"] = (
        "partitioned_input_partial_output"
    )
    with pytest.raises(PhysicalExecutionContractError, match="requires a reduced output"):
        validate_physical_execution_contract(partial_without_reduction)

    partial_without_sharded_input = partial_output_contract()
    partial_without_sharded_input["inputs"][0] = {
        "binding": 0,
        "distribution": "replicated",
    }
    with pytest.raises(PhysicalExecutionContractError, match="requires a sharded input"):
        validate_physical_execution_contract(partial_without_sharded_input)


def test_repeated_partition_requires_distinct_declared_range_controls() -> None:
    contract = partial_output_contract()
    del contract["partition_launch"]["count_push_constant"]
    with pytest.raises(PhysicalExecutionContractError, match="push constants"):
        validate_physical_execution_contract(contract)

    contract["partition_launch"]["count_push_constant"] = "input_start"
    with pytest.raises(PhysicalExecutionContractError, match="must differ"):
        validate_physical_execution_contract(contract)


def test_lazy_resources_cannot_be_declared_permanent() -> None:
    contract = fixture_contract()
    contract["resources"][0].update(kind="lazy_resource")
    with pytest.raises(PhysicalExecutionContractError, match="demand resident"):
        validate_physical_execution_contract(contract)


def test_block_scaled_partition_requires_logically_aligned_slices() -> None:
    contract = fixture_contract()
    contract["parameter_partitions"][0].update(
        alignment_elements=1, logical_elements_per_index=256
    )
    with pytest.raises(PhysicalExecutionContractError, match="logical extent"):
        validate_physical_execution_contract(contract)


def test_parameter_partition_names_its_exact_physical_resource() -> None:
    contract = fixture_contract()
    contract["parameter_partitions"][0]["resource"] = "missing"
    with pytest.raises(
        PhysicalExecutionContractError, match="exactly one declared parameter resource"
    ):
        validate_physical_execution_contract(contract)

    contract = fixture_contract()
    contract["resources"][0]["binding"] = 99
    with pytest.raises(PhysicalExecutionContractError, match="binding must match"):
        validate_physical_execution_contract(contract)

    contract = fixture_contract()
    del contract["parameter_partitions"][0]["resource"]
    with pytest.raises(PhysicalExecutionContractError, match="missing fields"):
        validate_physical_execution_contract(contract)


def test_contract_sealing_is_deterministic_and_covers_semantics() -> None:
    contract = fixture_contract()
    contract.pop("contract_id")
    first = seal_physical_execution_contract(contract)
    second = seal_physical_execution_contract(deepcopy(contract))
    assert first["contract_id"] == second["contract_id"]

    changed = deepcopy(contract)
    changed["geometry"]["dimensions"]["output_rows"] += 128
    changed["partition_extent"]["elements"] += 128
    assert seal_physical_execution_contract(changed)["contract_id"] != first["contract_id"]


def test_implementation_digest_covers_artifact_phase_format_geometry_and_strategy() -> None:
    contract = fixture_contract()
    digest = implementation_digest(
        artifacts=contract["artifacts"],
        phases=contract["phases"],
        formats=contract["formats"],
        geometry=contract["geometry"],
        strategy=contract["strategy"],
        execution_form=contract["execution_form"],
        partition_extent=contract["partition_extent"],
        partition_launch=contract["partition_launch"],
        parameter_partitions=contract["parameter_partitions"],
        inputs=contract["inputs"],
        outputs=contract["outputs"],
        local_intermediates=contract["local_intermediates"],
    )
    assert digest.startswith("sha256:")
    assert digest != implementation_digest(
        artifacts=[
            {
                **contract["artifacts"][0],
                "sha256": artifact_sha256(b"different artifact"),
            }
        ],
        phases=contract["phases"],
        formats=contract["formats"],
        geometry=contract["geometry"],
        strategy=contract["strategy"],
        execution_form=contract["execution_form"],
        partition_extent=contract["partition_extent"],
        partition_launch=contract["partition_launch"],
        parameter_partitions=contract["parameter_partitions"],
        inputs=contract["inputs"],
        outputs=contract["outputs"],
        local_intermediates=contract["local_intermediates"],
    )


def projection_compiler_fixture(tmp_path: Path, dtype: str = "BF16"):
    shader = tmp_path / "shaders" / "parallel_projection_bf16.spv"
    batch_shader = tmp_path / "shaders" / "parallel_projection_batch_bf16.spv"
    shader.parent.mkdir()
    shader.write_bytes(b"scalar spirv")
    batch_shader.write_bytes(b"batch spirv")
    node = {
        "id": "gate_up",
        "op": "parallel_linear_silu_multiply",
        "inputs": ["hidden"],
        "outputs": ["intermediate"],
        "params": ["gate", "up"],
    }
    circuit = {
        "parameters": {
            "refs": {
                "gate": {"tensor": "gate.weight"},
                "up": {"tensor": "up.weight"},
            }
        }
    }
    tensor_index = {
        "tensors": {
            "gate.weight": {
                "dtype": dtype,
                "shape": [256, 128],
                "layout": "row_major",
            },
            "up.weight": {
                "dtype": dtype,
                "shape": [256, 128],
                "layout": "row_major",
            },
        }
    }
    kernel = {
        "source_node_ids": ["gate_projection", "up_projection", "activation"],
        "semantic_module_ids": ["layer.feed_forward.gate_up"],
        "shader_path": "shaders/parallel_projection_bf16.spv",
        "local_size_x": 64,
        "workgroup_count_x": 128,
        "batch_implementations": [
            {
                "execution_domain": "decode_and_prefill",
                "lane_tile_width": 16,
                "stages": [
                    {
                        "shader_path": "shaders/parallel_projection_batch_bf16.spv",
                        "local_size_x": 64,
                        "workgroup_count_x": 128,
                    }
                ],
            }
        ],
    }
    return node, circuit, tensor_index, kernel


def test_compiler_emits_local_batch_and_legal_distributed_contracts(tmp_path: Path) -> None:
    node, circuit, tensor_index, kernel = projection_compiler_fixture(tmp_path)
    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=tmp_path,
    )

    assert [contract["strategy"] for contract in contracts] == [
        "single_device",
        "tensor_parallel",
        "single_device",
    ]
    distributed = contracts[1]
    assert distributed["execution_form"] == "replicated_input_partitioned_output"
    assert distributed["region_family"] == "layer.feed_forward.gate_up"
    assert distributed["member_node_ids"] == [
        "gate_up",
        "gate_projection",
        "up_projection",
        "activation",
    ]
    assert [partition["binding"] for partition in distributed["parameter_partitions"]] == [
        2,
        3,
    ]
    assert distributed["outputs"] == [
        {
            "binding": 1,
            "collection": "concatenated",
            "dimension": 0,
            "alignment_elements": 2,
        }
    ]
    assert contracts[2]["phases"] == ["decode", "prefill"]
    assert contracts[0]["artifacts"][0]["sha256"] == artifact_sha256(b"scalar spirv")
    assert all("model_name" not in contract for contract in contracts)


def test_compiler_does_not_guess_distribution_for_unsupported_storage(tmp_path: Path) -> None:
    node, circuit, tensor_index, kernel = projection_compiler_fixture(tmp_path, "Q8_0")
    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=tmp_path,
    )
    assert {contract["strategy"] for contract in contracts} == {"single_device"}


def test_compiler_declares_the_exact_sparse_expert_range_abi(tmp_path: Path) -> None:
    shader = tmp_path / "shaders" / "sparse_moe_down.spv"
    shader.parent.mkdir()
    shader.write_bytes(b"sparse scalar spirv")
    node = {
        "id": "expert_down",
        "op": "sparse_moe_down",
        "inputs": ["intermediates", "routes"],
        "outputs": ["expert_outputs"],
        "params": ["expert_weight", "expert_scale"],
    }
    circuit = {
        "parameters": {
            "refs": {
                "expert_weight": {"tensor": "experts.down.weight"},
                "expert_scale": {"tensor": "experts.down.scale"},
            }
        }
    }
    tensor_index = {
        "tensors": {
            "experts.down.weight": {
                "dtype": "F8_E4M3",
                "shape": [32, 1024, 512],
                "layout": "row_major",
            },
            "experts.down.scale": {
                "dtype": "BF16",
                "shape": [32, 8, 4],
                "layout": "row_major",
            },
        }
    }
    kernel = {
        "source_node_ids": ["expert_down_projection"],
        "semantic_module_ids": ["layer.feature_transform.routed_experts"],
        "shader_path": "shaders/sparse_moe_down.spv",
        "local_size_x": 64,
        "workgroup_count_x": 8192,
        "batch_implementations": [],
    }

    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=tmp_path,
    )

    expert = next(
        contract
        for contract in contracts
        if contract["execution_form"] == "whole_expert_ownership"
    )
    assert expert["partition_launch"] == {
        "workgroup_x": "repeated",
        "origin": "push_constant_u32",
        "origin_push_constant": "expert_start",
        "count_push_constant": "expert_count",
    }
    validate_physical_execution_contract(expert)


def test_compiler_maps_block_scaled_parameter_indices_to_logical_rows(
    tmp_path: Path,
) -> None:
    node, circuit, tensor_index, kernel = projection_compiler_fixture(tmp_path)
    node["params"] = ["gate", "gate_scale", "up", "up_scale"]
    circuit["parameters"]["refs"].update(
        {
            "gate_scale": {"tensor": "gate.scale"},
            "up_scale": {"tensor": "up.scale"},
        }
    )
    tensor_index["tensors"]["gate.weight"]["dtype"] = "F8_E4M3"
    tensor_index["tensors"]["up.weight"]["dtype"] = "F8_E4M3"
    tensor_index["tensors"].update(
        {
            "gate.scale": {
                "dtype": "BF16",
                "shape": [2, 1],
                "layout": "row_major",
            },
            "up.scale": {
                "dtype": "BF16",
                "shape": [2, 1],
                "layout": "row_major",
            },
        }
    )

    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=tmp_path,
    )

    distributed = contracts[1]
    assert distributed["partition_extent"] == {
        "dimension_name": "parameter_0_dimension_0",
        "elements": 256,
        "alignment_elements": 128,
    }
    assert [
        partition["logical_elements_per_index"]
        for partition in distributed["parameter_partitions"]
    ] == [1, 128, 1, 128]


def test_compiler_fails_when_declared_artifact_is_missing(tmp_path: Path) -> None:
    node, circuit, tensor_index, kernel = projection_compiler_fixture(tmp_path)
    (tmp_path / kernel["shader_path"]).unlink()
    with pytest.raises(PhysicalExecutionContractError, match="could not read"):
        build_kernel_physical_execution_contracts(
            node=node,
            circuit=circuit,
            tensor_index=tensor_index,
            kernel=kernel,
            package_dir=tmp_path,
        )
