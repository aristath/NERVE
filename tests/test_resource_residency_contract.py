from __future__ import annotations

import shutil
from copy import deepcopy
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
    compiled_immutable_resource,
    derived_partition_identity,
    partition_group_identity_seed,
    partition_template_identity,
    read_verified_partition_atomic_group,
    residency_content_id,
    residency_state_transition_allowed,
    resolve_partition_atomic_group,
    resource_identity,
    selector_identity,
    validate_resource_residency_contract,
)


def test_selector_identity_matches_the_runtime_execution_record() -> None:
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "layer_00",
        "node_id": "select",
        "domain_id": "experts",
        "resource_count": 2,
        "selection_signal": "selected",
        "execution_signal": "weighted",
        "execution_calibration_word_base": 0x3F800000,
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": 1,
            "index_shift": 0,
            "index_mask": 0xFFFF,
            "calibration_word_base": 0,
        },
        "mapping": {
            "kind": "partition_template",
            "partition_template_id": (
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ),
        },
    }

    assert selector_identity(selector) == (
        "sha256:9a02eec5614224066ecf0f52c2d4074c98a3ba5770848ca571356cc6cf6d1219"
    )


def test_compiler_reuses_one_verified_partition_digest_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = "weights/bank.safetensors"
    artifact = tmp_path / artifact_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"H" * 16 + bytes(range(32)))
    table_path = "integrity/resource_partitions.sha256"
    table = tmp_path / table_path
    table.parent.mkdir(parents=True)
    table_payload = b"".join(sha256(bytes([index])).digest() for index in range(4))
    table.write_bytes(table_payload)
    table_digest = sha256(table_payload).hexdigest()
    tensors = {
        f"tensor.{index}": {
            "source_file": artifact_path,
            "data_offsets": [index * 16, (index + 1) * 16],
            "safetensors_header_bytes": 8,
            "byte_count": 16,
            "data_sha256": sha256(
                bytes(range(index * 16, (index + 1) * 16))
            ).hexdigest(),
            "partition_integrity": {
                "schema": "nerve.tensor_partition_integrity.v1",
                "partition_axis": 0,
                "partition_count": 2,
                "partition_byte_count": 8,
                "digest_table_path": table_path,
                "digest_table_byte_offset": index * 64,
                "digest_stride_bytes": 32,
                "table_sha256": table_digest,
            },
        }
        for index in range(2)
    }
    open_count = 0
    original_open = Path.open

    def count_table_opens(path: Path, *args, **kwargs):
        nonlocal open_count
        if path == table:
            open_count += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_table_opens)
    digest_catalog: dict[str, tuple[bytes, str]] = {}

    resources = [
        compiled_immutable_resource(
            package_dir=tmp_path,
            tensor_index={"tensors": tensors},
            tensor_name=tensor_name,
            lifetime="dynamic",
            source_headers={artifact_path: 8},
            artifact_byte_counts={artifact_path: artifact.stat().st_size},
            partition_digest_catalog=digest_catalog,
        )
        for tensor_name in tensors
    ]

    assert open_count == 1
    assert set(digest_catalog) == {table_path}
    assert [
        byte_range["integrity"]["digest"] for byte_range in resources[0]["ranges"]
    ] == [
        table_payload[:32].hex(),
        table_payload[32:64].hex(),
    ]


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
                        "selection_signal": "selection",
                        "encoding": {
                            "element_type": "u32",
                            "selection_count_per_activation": 1,
                            "index_shift": 0,
                            "index_mask": 0xFFFF,
                            "calibration_word_base": 0,
                        },
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
                    "params": {"refs": parameter_refs},
                }
            ]
        },
        "speculative_decoders": [],
    }
    tensor_index = {"tensors": tensors}
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
    table_payload = sha256(b"0123").digest() + sha256(b"4567").digest()
    table.write_bytes(table_payload)
    template = {
        "id": "",
        "partition_count": 2,
        "lifetime": "dynamic",
        "group_identity_seed": "",
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
                            "table_sha256": sha256(table_payload).hexdigest(),
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
    template["group_identity_seed"] = partition_group_identity_seed(
        template["partition_count"],
        [member["resource_identity_seed"] for member in template["member_templates"]],
    )
    template["id"] = partition_template_identity(template)
    return template


def _replace_eager_resource_with_dynamic_template(
    contract: dict[str, object], template: dict[str, object]
) -> None:
    contract["resources"] = []
    contract["atomic_groups"] = []
    contract["partition_templates"] = [template]
    contract["bindings"][0]["mapping"] = {
        "kind": "partition_template_member",
        "partition_template_id": template["id"],
        "resource_identity_seed": template["member_templates"][0][
            "resource_identity_seed"
        ],
        "selection_signal": "selection",
        "parameter_slot": 0,
    }


def _add_dynamic_selector_contract(
    contract: dict[str, object],
    template: dict[str, object],
    *,
    resource_count: int | None = None,
) -> None:
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "node_id": "selector",
        "domain_id": "addressable_resources",
        "resource_count": (
            template["partition_count"] if resource_count is None else resource_count
        ),
        "selection_signal": "selection",
        "execution_signal": "selection",
        "execution_calibration_word_base": 0,
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": 1,
            "index_shift": 0,
            "index_mask": 0xFFFF,
            "calibration_word_base": 0,
        },
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
                "resource_id": contract["resources"][0]["id"],
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
        binding["mapping"]["atomic_group_id"] for binding in contract["bindings"]
    } == {contract["atomic_groups"][0]["id"]}


def test_content_identity_survives_package_relocation_and_artifact_renaming(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first, _ = _contract(first_root, artifact_path="weights/original.safetensors")
    second, _ = _contract(second_root, artifact_path="relocated/renamed.safetensors")

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
            lambda contract: contract["bindings"][0].update({"node_id": "invented"}),
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
    _replace_eager_resource_with_dynamic_template(contract, template)
    _add_dynamic_selector_contract(contract, template)

    validate_resource_residency_contract(tmp_path, contract, manifest)
    assert derived_partition_identity(template["group_identity_seed"], 0) != (
        derived_partition_identity(template["group_identity_seed"], 1)
    )


def test_rejects_selector_execution_signal_not_consumed_by_selected_node(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    _replace_eager_resource_with_dynamic_template(contract, template)
    _add_dynamic_selector_contract(contract, template)
    selector = contract["selectors"][0]
    selector["execution_signal"] = "unconsumed_expert_records"
    selector["id"] = selector_identity(selector)
    checkpoint = contract["checkpoints"][0]
    checkpoint["selector_ids"] = [selector["id"]]
    checkpoint["id"] = checkpoint_identity(checkpoint)

    with pytest.raises(ModelCompileError, match="do not consume"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_selector_width_larger_than_resource_domain(tmp_path: Path) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    _replace_eager_resource_with_dynamic_template(contract, template)
    _add_dynamic_selector_contract(contract, template)
    selector = contract["selectors"][0]
    selector["encoding"]["selection_count_per_activation"] = 3
    manifest["circuit_graph"]["components"][0]["circuit"]["nodes"][0]["attrs"][
        "selection_domain"
    ]["encoding"]["selection_count_per_activation"] = 3
    selector["id"] = selector_identity(selector)
    checkpoint = contract["checkpoints"][0]
    checkpoint["selector_ids"] = [selector["id"]]
    checkpoint["id"] = checkpoint_identity(checkpoint)

    with pytest.raises(ModelCompileError, match="invalid selection index encoding"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_resolves_and_independently_verifies_one_partition(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    _replace_eager_resource_with_dynamic_template(contract, template)

    resolved = resolve_partition_atomic_group(
        tmp_path,
        contract,
        partition_template_id=template["id"],
        partition_index=1,
    )

    assert resolved["partition_index"] == 1
    assert resolved["atomic_group"]["resource_ids"] == [resolved["resources"][0]["id"]]
    assert resolved["resources"][0]["ranges"] == [
        {
            "artifact_path": "weights/parameter.safetensors",
            "byte_offset": 20,
            "byte_count": 4,
            "alignment_bytes": 4,
            "integrity": {
                "algorithm": "sha256",
                "digest": sha256(b"4567").hexdigest(),
            },
        }
    ]
    assert list(read_verified_partition_atomic_group(tmp_path, resolved).values()) == [
        [b"4567"]
    ]

    artifact = tmp_path / "weights" / "parameter.safetensors"
    payload = bytearray(artifact.read_bytes())
    payload[16] ^= 0xFF
    artifact.write_bytes(payload)

    # Corruption in partition zero does not force a read or hash of it when
    # partition one is requested.
    assert list(read_verified_partition_atomic_group(tmp_path, resolved).values()) == [
        [b"4567"]
    ]
    corrupt = resolve_partition_atomic_group(
        tmp_path,
        contract,
        partition_template_id=template["id"],
        partition_index=0,
    )
    with pytest.raises(ModelCompileError, match="failed SHA-256"):
        read_verified_partition_atomic_group(tmp_path, corrupt)


def test_resolved_partition_group_rejects_duplicate_or_mismatched_membership(
    tmp_path: Path,
) -> None:
    contract, _manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    _replace_eager_resource_with_dynamic_template(contract, template)
    resolved = resolve_partition_atomic_group(
        tmp_path,
        contract,
        partition_template_id=template["id"],
        partition_index=0,
    )

    duplicate = deepcopy(resolved)
    duplicate["resources"].append(deepcopy(duplicate["resources"][0]))
    duplicate["atomic_group"]["resource_ids"].append(duplicate["resources"][0]["id"])
    with pytest.raises(ModelCompileError, match="resource ids are invalid"):
        read_verified_partition_atomic_group(tmp_path, duplicate)

    mismatch = deepcopy(resolved)
    mismatch["atomic_group"]["resource_ids"] = []
    with pytest.raises(ModelCompileError, match="membership is inconsistent"):
        read_verified_partition_atomic_group(tmp_path, mismatch)


def test_resolved_partition_group_rejects_unsafe_or_misaligned_ranges(
    tmp_path: Path,
) -> None:
    contract, _manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    _replace_eager_resource_with_dynamic_template(contract, template)
    resolved = resolve_partition_atomic_group(
        tmp_path,
        contract,
        partition_template_id=template["id"],
        partition_index=0,
    )

    unsafe = deepcopy(resolved)
    unsafe["resources"][0]["ranges"][0]["artifact_path"] = "../outside.bin"
    with pytest.raises(ModelCompileError, match="inside the compiled package"):
        read_verified_partition_atomic_group(tmp_path, unsafe)

    misaligned = deepcopy(resolved)
    misaligned["resources"][0]["ranges"][0]["byte_offset"] = 1
    with pytest.raises(ModelCompileError, match="range is invalid"):
        read_verified_partition_atomic_group(tmp_path, misaligned)


def test_partition_contract_survives_package_relocation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    contract, manifest = _contract(source, selector=True)
    template = _dynamic_template(source)
    _replace_eager_resource_with_dynamic_template(contract, template)
    _add_dynamic_selector_contract(contract, template)
    destination = tmp_path / "relocated" / "renamed-package"
    shutil.copytree(source, destination)

    validate_resource_residency_contract(destination, contract, manifest)
    resolved = resolve_partition_atomic_group(
        destination,
        contract,
        partition_template_id=template["id"],
        partition_index=0,
    )
    assert list(
        read_verified_partition_atomic_group(destination, resolved).values()
    ) == [[b"0123"]]


def test_rejects_partition_data_or_digest_table_truncation(
    tmp_path: Path,
) -> None:
    for truncated in ("data", "digest"):
        root = tmp_path / truncated
        contract, manifest = _contract(root, selector=True)
        template = _dynamic_template(root)
        _replace_eager_resource_with_dynamic_template(contract, template)
        if truncated == "data":
            artifact = root / "weights" / "parameter.safetensors"
            artifact.write_bytes(artifact.read_bytes()[:22])
            message = "exceeds artifact"
        else:
            table = root / "integrity" / "partitions.sha256"
            table.write_bytes(table.read_bytes()[:-1])
            message = "digest table is too small"

        with pytest.raises(ModelCompileError, match=message):
            validate_resource_residency_contract(root, contract, manifest)


def test_rejects_digest_table_corruption_and_uncovered_suffix(
    tmp_path: Path,
) -> None:
    for malformed in ("corrupt", "suffix"):
        root = tmp_path / malformed
        contract, manifest = _contract(root, selector=True)
        template = _dynamic_template(root)
        table = root / "integrity" / "partitions.sha256"
        payload = bytearray(table.read_bytes())
        if malformed == "corrupt":
            payload[0] ^= 0xFF
            message = "does not match its SHA-256"
        else:
            payload.extend(b"x" * 32)
            template["member_templates"][0]["range_templates"][0]["integrity"][
                "table_sha256"
            ] = sha256(payload).hexdigest()
            template["id"] = partition_template_identity(template)
            message = "covers 64 of 96"
        table.write_bytes(payload)
        _replace_eager_resource_with_dynamic_template(contract, template)

        with pytest.raises(ModelCompileError, match=message):
            validate_resource_residency_contract(root, contract, manifest)


def test_rejects_overlapping_partition_ranges(tmp_path: Path) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    template["member_templates"][0]["range_templates"][0]["stride_bytes"] = 2
    template["member_templates"][0]["range_templates"][0]["alignment_bytes"] = 2
    template["id"] = partition_template_identity(template)
    _replace_eager_resource_with_dynamic_template(contract, template)

    with pytest.raises(ModelCompileError, match="overlaps adjacent"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_dynamic_overlap_with_concrete_resource(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    contract["partition_templates"] = [template]

    with pytest.raises(ModelCompileError, match="ranges overlap"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_partition_group_seed_that_omits_member_identity(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    template["group_identity_seed"] = residency_content_id(
        "partition_group_seed", {"incomplete": True}
    )
    template["id"] = partition_template_identity(template)
    _replace_eager_resource_with_dynamic_template(contract, template)

    with pytest.raises(ModelCompileError, match="does not exactly cover"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_concrete_binding_to_resource_outside_group(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path)
    contract["bindings"][0]["mapping"]["resource_id"] = residency_content_id(
        "resource", {"not": "a group member"}
    )

    with pytest.raises(ModelCompileError, match="outside its atomic group"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_unbound_atomic_partition_member(tmp_path: Path) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    second_artifact = tmp_path / "weights" / "second.bin"
    second_artifact.write_bytes(b"abcdefgh")
    table = tmp_path / "integrity" / "partitions.sha256"
    table_payload = table.read_bytes()
    second_digest_offset = len(table_payload)
    table_payload += sha256(b"abcd").digest() + sha256(b"efgh").digest()
    table.write_bytes(table_payload)
    for member in template["member_templates"]:
        member["range_templates"][0]["integrity"]["table_sha256"] = sha256(
            table_payload
        ).hexdigest()
    second_member = deepcopy(template["member_templates"][0])
    second_member["resource_identity_seed"] = residency_content_id(
        "partition_resource_seed", {"member": 1}
    )
    second_range = second_member["range_templates"][0]
    second_range["artifact_path"] = "weights/second.bin"
    second_range["base_byte_offset"] = 0
    second_range["integrity"]["digest_table_byte_offset"] = second_digest_offset
    second_range["integrity"]["table_sha256"] = sha256(table_payload).hexdigest()
    template["member_templates"].append(second_member)
    template["member_templates"].sort(
        key=lambda member: member["resource_identity_seed"]
    )
    template["group_identity_seed"] = partition_group_identity_seed(
        template["partition_count"],
        [member["resource_identity_seed"] for member in template["member_templates"]],
    )
    template["id"] = partition_template_identity(template)
    _replace_eager_resource_with_dynamic_template(contract, template)

    with pytest.raises(ModelCompileError, match="atomic resource membership"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


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
    contract["bindings"][0]["mapping"]["resource_id"] = resource["id"]
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "node_id": "selector",
        "domain_id": "addressable_resources",
        "resource_count": 1,
        "selection_signal": "selection",
        "execution_signal": "selection",
        "execution_calibration_word_base": 0,
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": 1,
            "index_shift": 0,
            "index_mask": 0xFFFF,
            "calibration_word_base": 0,
        },
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
    ] = 32
    template["id"] = partition_template_identity(template)
    _replace_eager_resource_with_dynamic_template(contract, template)

    with pytest.raises(ModelCompileError, match="digest table is too small"):
        validate_resource_residency_contract(tmp_path, contract, manifest)


def test_rejects_selector_count_and_checkpoint_boundary_drift(
    tmp_path: Path,
) -> None:
    contract, manifest = _contract(tmp_path, selector=True)
    template = _dynamic_template(tmp_path)
    _replace_eager_resource_with_dynamic_template(contract, template)
    selector = {
        "id": "",
        "execution_scope": "target",
        "component_id": "component",
        "node_id": "selector",
        "domain_id": "addressable_resources",
        "resource_count": 1,
        "selection_signal": "selection",
        "execution_signal": "selection",
        "execution_calibration_word_base": 0,
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": 1,
            "index_shift": 0,
            "index_mask": 0xFFFF,
            "calibration_word_base": 0,
        },
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
