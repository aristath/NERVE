from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.resource_residency import (
    RESOURCE_IDENTITY_ALGORITHM,
    RESOURCE_RESIDENCY_SCHEMA,
    RESOURCE_STATE_MACHINE_SCHEMA,
    atomic_group_identity,
    build_eager_resource_residency_contract,
    checkpoint_identity,
    derived_partition_identity,
    partition_template_identity,
    residency_content_id,
    residency_state_transition_allowed,
    resource_identity,
    selector_identity,
    validate_resource_residency_contract,
)


def _fixture(
    root: Path,
    *,
    artifact_path: str = "weights/parameter.safetensors",
    selector: bool = False,
    second_parameter: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = b"0123456789abcdef"
    second_payload = b"FEDCBA9876543210"
    source = root / artifact_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(
        b"H" * 16 + payload + (second_payload if second_parameter else b"")
    )
    nodes = []
    if selector:
        nodes.append(
            {
                "id": "selector",
                "op": "top_k",
                "inputs": ["input"],
                "outputs": ["selection"],
                "params": [],
                "attrs": {
                    "selection_domain": {
                        "id": "addressable_resources",
                        "resource_count": 2,
                    }
                },
            }
        )
    parameter_ids = ["weight"]
    parameter_refs = {
        "weight": {
            "tensor": "semantic.tensor.name",
        }
    }
    tensors = {
        "semantic.tensor.name": {
            "source_file": artifact_path,
            "data_offsets": [0, len(payload)],
            "safetensors_header_bytes": 8,
            "byte_count": len(payload),
            "data_sha256": sha256(payload).hexdigest(),
        }
    }
    if second_parameter:
        parameter_ids.append("bias")
        parameter_refs["bias"] = {
            "tensor": "another.semantic.tensor",
        }
        tensors["another.semantic.tensor"] = {
            "source_file": artifact_path,
            "data_offsets": [len(payload), len(payload) + len(second_payload)],
            "safetensors_header_bytes": 8,
            "byte_count": len(second_payload),
            "data_sha256": sha256(second_payload).hexdigest(),
        }
    nodes.append(
        {
            "id": "compute",
            "op": "linear",
            "inputs": ["selection" if selector else "input"],
            "outputs": ["output"],
            "params": parameter_ids,
        }
    )
    manifest = {
        "circuit_graph": {
            "components": [
                {
                    "component_id": "component",
                    "circuit": {"nodes": nodes},
                    "params": {
                        "refs": parameter_refs
                    },
                }
            ]
        },
        "speculative_decoders": [],
    }
    tensor_index = {
        "tensors": tensors
    }
    return manifest, tensor_index


def _contract(
    root: Path,
    *,
    artifact_path: str = "weights/parameter.safetensors",
    selector: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest, tensor_index = _fixture(
        root,
        artifact_path=artifact_path,
        selector=selector,
    )
    contract = build_eager_resource_residency_contract(
        package_dir=root,
        tensor_index=tensor_index,
        manifest=manifest,
    )
    return contract, manifest


def _dynamic_template(root: Path) -> dict[str, object]:
    table = root / "integrity" / "partitions.sha256"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(b"a" * 64)
    template = {
        "id": "",
        "partition_count": 2,
        "lifetime": "dynamic",
        "group_identity_seed": residency_content_id(
            "partition_group_seed", {"semantics": "selected_partition"}
        ),
        "member_templates": [
            {
                "resource_identity_seed": residency_content_id(
                    "partition_resource_seed", {"member": 0}
                ),
                "range_templates": [
                    {
                        "artifact_path": "weights/parameter.safetensors",
                        "base_byte_offset": 16,
                        "stride_bytes": 4,
                        "byte_count": 4,
                        "alignment_bytes": 4,
                        "integrity": {
                            "algorithm": "sha256_table",
                            "digest_table_path": "integrity/partitions.sha256",
                            "digest_table_byte_offset": 0,
                            "digest_stride_bytes": 32,
                            "table_sha256": sha256(b"a" * 64).hexdigest(),
                        },
                    }
                ],
                "compatibility": {
                    "device_api": "vulkan",
                    "storage_class": "storage_buffer",
                    "read_only": True,
                    "required_features": [],
                },
            }
        ],
        "dependencies": [],
    }
    template["id"] = partition_template_identity(template)
    return template


def test_builds_a_complete_eager_contract_from_compiled_access_semantics(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path)

    assert contract["schema"] == RESOURCE_RESIDENCY_SCHEMA
    assert contract["identity_algorithm"] == RESOURCE_IDENTITY_ALGORITHM
    assert contract["state_machine_schema"] == RESOURCE_STATE_MACHINE_SCHEMA
    assert contract["supported_policies"] == ["demand_retained", "eager"]
    assert len(contract["resources"]) == 1
    assert len(contract["atomic_groups"]) == 1
    assert contract["resources"][0]["lifetime"] == "always_resident"
    assert contract["bindings"] == [
        {
            "execution_scope": "target",
            "component_id": "component",
            "node_id": "compute",
            "parameter_id": "weight",
            "mapping": {
                "kind": "atomic_group",
                "atomic_group_id": contract["atomic_groups"][0]["id"],
            },
        }
    ]
    validate_resource_residency_contract(tmp_path, contract, manifest)


def test_eager_spine_is_one_semantic_group_not_one_group_per_tensor(
    tmp_path: Path,
) -> None:
    manifest, tensor_index = _fixture(tmp_path, second_parameter=True)
    contract = build_eager_resource_residency_contract(
        package_dir=tmp_path,
        tensor_index=tensor_index,
        manifest=manifest,
    )

    assert len(contract["resources"]) == 2
    assert len(contract["atomic_groups"]) == 1
    assert contract["atomic_groups"][0]["resource_ids"] == sorted(
        resource["id"] for resource in contract["resources"]
    )
    assert {
        binding["mapping"]["atomic_group_id"]
        for binding in contract["bindings"]
    } == {
        contract["atomic_groups"][0]["id"]
    }


def test_content_identity_survives_package_relocation_and_artifact_renaming(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first, _ = _contract(
        first_root, artifact_path="weights/original.safetensors"
    )
    second, _ = _contract(
        second_root, artifact_path="relocated/renamed.safetensors"
    )

    assert [item["id"] for item in first["resources"]] == [
        item["id"] for item in second["resources"]
    ]
    assert [item["id"] for item in first["atomic_groups"]] == [
        item["id"] for item in second["atomic_groups"]
    ]


def test_failed_and_resident_states_require_explicit_lifecycle_clear() -> None:
    assert residency_state_transition_allowed("absent", "requested")
    assert residency_state_transition_allowed("requested", "loading")
    assert residency_state_transition_allowed("loading", "resident")
    assert residency_state_transition_allowed("loading", "failed")
    assert residency_state_transition_allowed("loading", "absent")
    assert not residency_state_transition_allowed("failed", "absent")
    assert residency_state_transition_allowed(
        "failed", "absent", explicit_lifecycle=True
    )
    assert not residency_state_transition_allowed("resident", "absent")
    assert residency_state_transition_allowed(
        "resident", "absent", explicit_lifecycle=True
    )


@pytest.mark.parametrize(
    "mutation, message",
    (
        (
            lambda contract: contract.update({"schema": "nerve.unknown.v9"}),
            "unsupported",
        ),
        (
            lambda contract: contract.update({"unknown": True}),
            "fields are invalid",
        ),
        (
            lambda contract: contract.update({"supported_policies": ["eager"]}),
            "policies",
        ),
        (
            lambda contract: contract["resources"][0]["ranges"][0].update(
                {"alignment_bytes": 3}
            ),
            "power of two",
        ),
        (
            lambda contract: contract["resources"][0].update({"unknown": True}),
            "fields are invalid",
        ),
        (
            lambda contract: contract["bindings"][0].update(
                {"node_id": "invented"}
            ),
            "exactly cover",
        ),
    ),
)
def test_rejects_malformed_or_semantically_stale_contracts(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    contract, manifest = _contract(tmp_path)
    mutation(contract)

    with pytest.raises(ModelCompileError, match=message):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_validates_compact_partition_selector_and_checkpoint_contract(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    contract["partition_templates"] = [template]
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "node_id": "selector",
        "domain_id": "addressable_resources",
        "resource_count": 2,
        "mapping": {
            "kind": "partition_template",
            "partition_template_id": template["id"],
        },
    }
    selector["id"] = selector_identity(selector)
    checkpoint = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "after_node_id": "selector",
        "resume_node_id": "compute",
        "selector_ids": [selector["id"]],
    }
    checkpoint["id"] = checkpoint_identity(checkpoint)
    contract["selectors"] = [selector]
    contract["checkpoints"] = [checkpoint]

    validate_resource_residency_contract(tmp_path, contract, manifest)
    assert derived_partition_identity(template["group_identity_seed"], 0) != (
        derived_partition_identity(template["group_identity_seed"], 1)
    )


def test_validates_concrete_dynamic_group_selector_and_checkpoint(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    manifest["circuit_graph"]["components"][0]["circuit"]["nodes"][0]["attrs"][
        "selection_domain"
    ]["resource_count"] = 1

    resource = contract["resources"][0]
    resource["lifetime"] = "dynamic"
    resource["id"] = resource_identity(resource)
    group = contract["atomic_groups"][0]
    group["lifetime"] = "dynamic"
    group["resource_ids"] = [resource["id"]]
    group["id"] = atomic_group_identity(group)
    contract["bindings"][0]["mapping"]["atomic_group_id"] = group["id"]
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "node_id": "selector",
        "domain_id": "addressable_resources",
        "resource_count": 1,
        "mapping": {
            "kind": "group_table",
            "atomic_group_ids": [group["id"]],
        },
    }
    selector["id"] = selector_identity(selector)
    checkpoint = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "after_node_id": "selector",
        "resume_node_id": "compute",
        "selector_ids": [selector["id"]],
    }
    checkpoint["id"] = checkpoint_identity(checkpoint)
    contract["selectors"] = [selector]
    contract["checkpoints"] = [checkpoint]

    validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_duplicate_concrete_resource_membership(tmp_path: Path) -> None:
    contract, manifest = _contract(tmp_path)
    group = contract["atomic_groups"][0]
    group["resource_ids"].append(group["resource_ids"][0])
    group["id"] = atomic_group_identity(group)

    with pytest.raises(ModelCompileError, match="unique and sorted"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_partition_digest_table_without_complete_coverage(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    template["member_templates"][0]["range_templates"][0]["integrity"][
        "digest_table_byte_offset"
    ] = 40
    template["id"] = partition_template_identity(template)
    contract["partition_templates"] = [template]

    with pytest.raises(ModelCompileError, match="digest table is too small"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_selector_count_and_checkpoint_boundary_drift(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    contract["partition_templates"] = [template]
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "node_id": "selector",
        "domain_id": "addressable_resources",
        "resource_count": 1,
        "mapping": {
            "kind": "partition_template",
            "partition_template_id": template["id"],
        },
    }
    selector["id"] = selector_identity(selector)
    checkpoint = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "after_node_id": "selector",
        "resume_node_id": "compute",
        "selector_ids": [selector["id"]],
    }
    checkpoint["id"] = checkpoint_identity(checkpoint)
    contract["selectors"] = [selector]
    contract["checkpoints"] = [checkpoint]

    with pytest.raises(ModelCompileError, match="node semantics"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_resource_identity_is_not_a_tensor_name_or_package_path(
    tmp_path: Path,
) -> None:
    contract, _ = _contract(tmp_path)
    serialized = str(
        {
            "resources": contract["resources"],
            "atomic_groups": contract["atomic_groups"],
        }
    )

    assert "semantic.tensor.name" not in serialized
    assert str(tmp_path) not in serialized
    assert contract["resources"][0]["id"].startswith("sha256:")
    assert contract["atomic_groups"][0]["id"] == atomic_group_identity(
        contract["atomic_groups"][0]
    )
