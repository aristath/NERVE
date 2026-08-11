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
