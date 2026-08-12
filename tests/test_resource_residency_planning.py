from __future__ import annotations

import json
import struct
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
import nerve.resource_residency_planning as residency_planning

from nerve.compilation import ModelCompileError
from nerve.model_package_artifact_layout import pack_tensor_artifacts_by_affinity
from nerve.model_package_assets import copy_tensor_package
from nerve.resource_residency import (
    validate_resource_residency_contract,
)
from nerve.resource_residency_planning import (
    RESIDENCY_ANALYSIS_SCHEMA,
    SOURCE_INTEGRITY_PARTITION_COUNT_FIELD,
    TENSOR_PARTITION_INTEGRITY_SCHEMA,
    analyze_resource_residency_components,
    artifact_affinity_groups_for_packaging,
    build_planned_resource_residency_contract,
    partition_counts_for_packaging,
)


def _tensor(byte_count: int, shape: list[int]) -> dict[str, object]:
    return {
        "dtype": "U8",
        "shape": shape,
        "byte_count": byte_count,
        "parameter_count": byte_count,
        "source_file": "unused.safetensors",
        "data_offsets": [0, byte_count],
    }


def _routed_component() -> tuple[list[dict[str, object]], dict[str, object]]:
    nodes = [
        {
            "id": "choose",
            "op": "choose",
            "inputs": ["input"],
            "outputs": ["chosen"],
            "params": ["router"],
            "attrs": {
                "selection_domain": {
                    "id": "selectable_units",
                    "resource_count": 4,
                    "selection_signal": "chosen",
                    "encoding": {
                        "element_type": "u32",
                        "selection_count_per_activation": 2,
                        "index_shift": 0,
                        "index_mask": 0xFFFF,
                    },
                }
            },
        },
        {
            "id": "first_selected_compute",
            "op": "first_physical_operation",
            "inputs": ["input", "chosen"],
            "outputs": ["middle"],
            "params": ["bank_a", "scale_a"],
            "attrs": {
                "selected_parameter_accesses": [
                    {
                        "selection_signal": "chosen",
                        "partition_axis": 0,
                        "parameter_ids": ["bank_a", "scale_a"],
                    }
                ]
            },
        },
        {
            "id": "second_selected_compute",
            "op": "second_physical_operation",
            "inputs": ["middle", "chosen"],
            "outputs": ["output"],
            "params": ["bank_b", "scale_b"],
            "attrs": {
                "selected_parameter_accesses": [
                    {
                        "selection_signal": "chosen",
                        "partition_axis": 0,
                        "parameter_ids": ["bank_b", "scale_b"],
                    }
                ]
            },
        },
    ]
    refs = {
        parameter: {"tensor": f"tensor.{parameter}"}
        for parameter in ("router", "bank_a", "scale_a", "bank_b", "scale_b")
    }
    return nodes, refs


def _optional_component() -> tuple[list[dict[str, object]], dict[str, object]]:
    nodes = [
        {
            "id": "feature_switch",
            "op": "feature_switch",
            "inputs": ["input"],
            "outputs": ["feature_index"],
            "params": [],
            "attrs": {
                "selection_domain": {
                    "id": "optional_feature",
                    "resource_count": 3,
                    "selection_signal": "feature_index",
                    "encoding": {
                        "element_type": "u32",
                        "selection_count_per_activation": 1,
                        "index_shift": 0,
                        "index_mask": 0xFFFF,
                    },
                }
            },
        },
        {
            "id": "optional_projection",
            "op": "projection",
            "inputs": ["input", "feature_index"],
            "outputs": ["output"],
            "params": ["projection_bank", "always_bias"],
            "attrs": {
                "selected_parameter_accesses": [
                    {
                        "selection_signal": "feature_index",
                        "partition_axis": 0,
                        "parameter_ids": ["projection_bank"],
                    }
                ]
            },
        },
    ]
    return nodes, {
        "projection_bank": {"tensor": "tensor.projection_bank"},
        "always_bias": {"tensor": "tensor.always_bias"},
    }


def _independent_component() -> tuple[list[dict[str, object]], dict[str, object]]:
    nodes = [
        {
            "id": "choose",
            "op": "choose",
            "inputs": ["input"],
            "outputs": ["chosen"],
            "params": ["router"],
            "attrs": {
                "selection_domain": {
                    "id": "selectable_units",
                    "resource_count": 2,
                    "selection_signal": "chosen",
                    "encoding": {
                        "element_type": "u32",
                        "selection_count_per_activation": 1,
                        "index_shift": 0,
                        "index_mask": 0xFFFF,
                    },
                }
            },
        },
        {
            "id": "selected_compute",
            "op": "selected_compute",
            "inputs": ["input", "chosen"],
            "outputs": ["output"],
            "params": [
                "unit_0_scale",
                "unit_0_weight",
                "unit_1_scale",
                "unit_1_weight",
            ],
            "attrs": {
                "selected_parameter_accesses": [
                    {
                        "selection_signal": "chosen",
                        "mapping": [
                            {
                                "selector": 0,
                                "parameter_ids": ["unit_0_scale", "unit_0_weight"],
                            },
                            {
                                "selector": 1,
                                "parameter_ids": ["unit_1_scale", "unit_1_weight"],
                            },
                        ],
                    }
                ]
            },
        },
    ]
    refs = {
        parameter: {"tensor": f"tensor.{parameter}"}
        for parameter in (
            "router",
            "unit_0_scale",
            "unit_0_weight",
            "unit_1_scale",
            "unit_1_weight",
        )
    }
    return nodes, refs


def _component(
    nodes: list[dict[str, object]],
    refs: dict[str, object],
    *,
    component_id: str = "component",
) -> dict[str, object]:
    return {
        "execution_scope": "target",
        "component_id": component_id,
        "nodes": nodes,
        "parameter_refs": refs,
    }


def _analysis_tensors() -> dict[str, object]:
    return {
        "tensors": {
            "tensor.router": _tensor(8, [4, 2]),
            "tensor.bank_a": _tensor(32, [4, 8]),
            "tensor.scale_a": _tensor(16, [4, 4]),
            "tensor.bank_b": _tensor(48, [4, 12]),
            "tensor.scale_b": _tensor(16, [4, 4]),
            "tensor.projection_bank": _tensor(24, [3, 8]),
            "tensor.always_bias": _tensor(8, [8]),
        }
    }


def test_discovers_atomic_selected_bundle_from_multiple_physical_consumers() -> None:
    nodes, refs = _routed_component()
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=_analysis_tensors(),
        require_direct_packaging=True,
    )

    assert analysis["schema"] == RESIDENCY_ANALYSIS_SCHEMA
    assert analysis["spine_tensors"] == ["tensor.router"]
    assert analysis["dynamic_tensors"] == {
        "tensor.bank_a": {"partition_axis": 0, "partition_count": 4},
        "tensor.bank_b": {"partition_axis": 0, "partition_count": 4},
        "tensor.scale_a": {"partition_axis": 0, "partition_count": 4},
        "tensor.scale_b": {"partition_axis": 0, "partition_count": 4},
    }
    assert len(analysis["groups"]) == 1
    group = analysis["groups"][0]
    assert group["resume_node_id"] == "first_selected_compute"
    assert {access["tensor"] for access in group["accesses"]} == set(
        analysis["dynamic_tensors"]
    )


def test_discovers_structurally_different_optional_partition_pattern() -> None:
    nodes, refs = _optional_component()
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs, component_id="optional")],
        tensor_index=_analysis_tensors(),
        require_direct_packaging=True,
    )

    assert analysis["spine_tensors"] == ["tensor.always_bias"]
    assert analysis["dynamic_tensors"] == {
        "tensor.projection_bank": {
            "partition_axis": 0,
            "partition_count": 3,
        }
    }
    assert analysis["groups"][0]["domain_id"] == "optional_feature"
    assert analysis["groups"][0]["resume_node_id"] == "optional_projection"


def test_discovers_independently_stored_selected_resources() -> None:
    nodes, refs = _independent_component()
    tensors = {
        "tensors": {
            "tensor.router": _tensor(4, [2]),
            "tensor.unit_0_scale": _tensor(2, [2]),
            "tensor.unit_0_weight": _tensor(8, [2, 4]),
            "tensor.unit_1_scale": _tensor(2, [2]),
            "tensor.unit_1_weight": _tensor(8, [2, 4]),
        }
    }

    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensors,
        require_direct_packaging=True,
    )

    assert analysis["spine_tensors"] == ["tensor.router"]
    assert analysis["dynamic_tensors"] == {
        f"tensor.unit_{unit}_{kind}": {"storage": "independent_resource"}
        for unit in range(2)
        for kind in ("scale", "weight")
    }
    assert partition_counts_for_packaging(analysis) == {}
    group = analysis["groups"][0]
    assert group["storage"] == "independent_resources"
    assert group["partition_count"] == 2
    assert {(access["selector"], access["tensor"]) for access in group["accesses"]} == {
        (unit, f"tensor.unit_{unit}_{kind}")
        for unit in range(2)
        for kind in ("scale", "weight")
    }


def test_independent_resources_preserve_compiler_source_integrity_partitions() -> None:
    nodes, refs = _independent_component()
    tensors = {
        "tensors": {
            "tensor.router": _tensor(4, [2]),
            **{
                f"tensor.unit_{unit}_{kind}": {
                    **_tensor(8, [2, 4]),
                    SOURCE_INTEGRITY_PARTITION_COUNT_FIELD: 2,
                }
                for unit in range(2)
                for kind in ("scale", "weight")
            },
        }
    }

    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensors,
        require_direct_packaging=True,
    )

    assert partition_counts_for_packaging(analysis) == {
        f"tensor.unit_{unit}_{kind}": 2
        for unit in range(2)
        for kind in ("scale", "weight")
    }
    assert all(
        metadata[SOURCE_INTEGRITY_PARTITION_COUNT_FIELD] == 2
        for metadata in analysis["dynamic_tensors"].values()
    )


def test_independent_resource_contract_exposes_each_sealed_source_range(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    nodes, refs = _independent_component()
    tensor_index = {
        "tensors": {
            "tensor.router": _write_source_tensor(
                source_dir,
                tensor_name="tensor.router",
                shape=[2],
                payload=b"rt",
            ),
            **{
                f"tensor.unit_{unit}_{kind}": {
                    **_write_source_tensor(
                        source_dir,
                        tensor_name=f"tensor.unit_{unit}_{kind}",
                        shape=[2, 4],
                        payload=bytes([16 * unit + offset]) * 8,
                    ),
                    SOURCE_INTEGRITY_PARTITION_COUNT_FIELD: 2,
                }
                for unit in range(2)
                for offset, kind in enumerate(("scale", "weight"), start=1)
            },
        }
    }
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensor_index,
        require_direct_packaging=True,
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    packaged = copy_tensor_package(
        tensor_index,
        package_dir,
        partition_counts=partition_counts_for_packaging(analysis),
        artifact_affinity_groups=artifact_affinity_groups_for_packaging(analysis),
    )
    manifest = {
        "circuit_graph": {
            "components": [
                {
                    "component_id": "component",
                    "circuit": {"nodes": nodes},
                    "params": {"refs": refs},
                }
            ]
        },
        "speculative_decoders": [],
    }
    contract = build_planned_resource_residency_contract(
        package_dir=package_dir,
        tensor_index=packaged,
        manifest=manifest,
    )

    dynamic_resources = [
        resource
        for resource in contract["resources"]
        if resource["lifetime"] == "dynamic"
    ]
    assert len(dynamic_resources) == 4
    assert all(
        [byte_range["byte_count"] for byte_range in resource["ranges"]]
        == [4, 4]
        for resource in dynamic_resources
    )
    assert all(
        all(
            byte_range["integrity"]["algorithm"] == "sha256"
            for byte_range in resource["ranges"]
        )
        for resource in dynamic_resources
    )
    validate_resource_residency_contract(package_dir, contract, manifest)


def test_derives_artifact_affinity_from_selection_cohorts_not_model_identity() -> None:
    nodes, refs = _independent_component()
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index={
            "tensors": {
                "tensor.router": _tensor(4, [2]),
                "tensor.unit_0_scale": _tensor(2, [2]),
                "tensor.unit_0_weight": _tensor(8, [2, 4]),
                "tensor.unit_1_scale": _tensor(2, [2]),
                "tensor.unit_1_weight": _tensor(8, [2, 4]),
            }
        },
        require_direct_packaging=True,
    )

    assert artifact_affinity_groups_for_packaging(analysis) == [
        [
            "tensor.unit_0_scale",
            "tensor.unit_0_weight",
            "tensor.unit_1_scale",
            "tensor.unit_1_weight",
        ]
    ]


def test_artifact_affinity_merges_shared_structural_views_once() -> None:
    nodes, refs = _independent_component()
    analysis = analyze_resource_residency_components(
        components=[
            _component(nodes, refs, component_id="first_view"),
            _component(nodes, refs, component_id="second_view"),
        ],
        tensor_index={
            "tensors": {
                "tensor.router": _tensor(4, [2]),
                "tensor.unit_0_scale": _tensor(2, [2]),
                "tensor.unit_0_weight": _tensor(8, [2, 4]),
                "tensor.unit_1_scale": _tensor(2, [2]),
                "tensor.unit_1_weight": _tensor(8, [2, 4]),
            }
        },
        require_direct_packaging=True,
    )

    affinity_groups = artifact_affinity_groups_for_packaging(analysis)
    assert len(analysis["groups"]) == 2
    assert affinity_groups == [
        [
            "tensor.unit_0_scale",
            "tensor.unit_0_weight",
            "tensor.unit_1_scale",
            "tensor.unit_1_weight",
        ]
    ]


def test_rejects_incomplete_independent_selector_table() -> None:
    nodes, refs = _independent_component()
    nodes[1]["attrs"]["selected_parameter_accesses"][0]["mapping"].pop()

    with pytest.raises(ModelCompileError, match="must map every selector"):
        analyze_resource_residency_components(
            components=[_component(nodes, refs)],
            tensor_index={
                "tensors": {
                    "tensor.router": _tensor(4, [2]),
                    "tensor.unit_0_scale": _tensor(2, [2]),
                    "tensor.unit_0_weight": _tensor(8, [2, 4]),
                    "tensor.unit_1_scale": _tensor(2, [2]),
                    "tensor.unit_1_weight": _tensor(8, [2, 4]),
                }
            },
            require_direct_packaging=True,
        )


def test_rejects_selector_without_physical_selected_access() -> None:
    nodes, refs = _optional_component()
    nodes[1]["attrs"] = {}

    with pytest.raises(ModelCompileError, match="does not select any"):
        analyze_resource_residency_components(
            components=[_component(nodes, refs)],
            tensor_index=_analysis_tensors(),
            require_direct_packaging=True,
        )


def test_rejects_non_contiguous_partition_axis() -> None:
    nodes, refs = _optional_component()
    nodes[1]["attrs"]["selected_parameter_accesses"][0]["partition_axis"] = 1
    tensors = _analysis_tensors()
    tensors["tensors"]["tensor.projection_bank"]["shape"] = [8, 3]

    with pytest.raises(ModelCompileError, match="non-contiguous selected axis"):
        analyze_resource_residency_components(
            components=[_component(nodes, refs)],
            tensor_index=tensors,
            require_direct_packaging=True,
        )


def test_rejects_tensor_that_is_both_dynamic_and_unconditional() -> None:
    nodes, refs = _optional_component()
    nodes.append(
        {
            "id": "unconditional_reuse",
            "op": "projection",
            "inputs": ["output"],
            "outputs": ["final"],
            "params": ["projection_bank"],
            "attrs": {},
        }
    )

    with pytest.raises(ModelCompileError, match="both selected and unconditionally"):
        analyze_resource_residency_components(
            components=[_component(nodes, refs)],
            tensor_index=_analysis_tensors(),
            require_direct_packaging=True,
        )


def test_reuses_compatible_partitions_across_independent_selectors() -> None:
    first_nodes, refs = _optional_component()
    second_nodes, _ = _optional_component()
    second_nodes[0]["id"] = "second_switch"
    second_nodes[0]["outputs"] = ["second_index"]
    second_nodes[0]["attrs"]["selection_domain"]["id"] = "second_feature"
    second_nodes[0]["attrs"]["selection_domain"]["selection_signal"] = "second_index"
    second_nodes[1]["id"] = "second_projection"
    second_nodes[1]["inputs"] = ["input", "second_index"]
    second_nodes[1]["attrs"]["selected_parameter_accesses"][0]["selection_signal"] = (
        "second_index"
    )

    analysis = analyze_resource_residency_components(
        components=[
            _component(first_nodes, refs, component_id="first"),
            _component(second_nodes, refs, component_id="second"),
        ],
        tensor_index=_analysis_tensors(),
        require_direct_packaging=True,
    )

    assert len(analysis["groups"]) == 2
    assert analysis["dynamic_tensors"] == {
        "tensor.projection_bank": {
            "partition_axis": 0,
            "partition_count": 3,
        }
    }


def test_exact_early_selection_can_control_later_dynamic_parameters() -> None:
    nodes, refs = _independent_component()
    nodes[0]["attrs"]["predictable_dependency"] = {
        "schema": "nerve.predictable_resource_selection.v1",
        "kind": "parameter_table_lookup",
        "key_signal": "input",
        "table_parameter": "router",
        "selection_semantics": "exact",
    }
    nodes.insert(
        1,
        {
            "id": "weight_selected_units",
            "op": "weight_preselected_units",
            "inputs": ["input", "chosen"],
            "outputs": ["weighted"],
            "params": [],
            "attrs": {},
        },
    )
    nodes[2]["inputs"] = ["weighted"]

    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index={
            "tensors": {
                "tensor.router": _tensor(4, [2]),
                "tensor.unit_0_scale": _tensor(2, [2]),
                "tensor.unit_0_weight": _tensor(8, [2, 4]),
                "tensor.unit_1_scale": _tensor(2, [2]),
                "tensor.unit_1_weight": _tensor(8, [2, 4]),
            }
        },
        require_direct_packaging=True,
    )

    assert analysis["groups"][0]["selector_node_id"] == "choose"
    assert analysis["groups"][0]["selection_signal"] == "chosen"
    assert analysis["groups"][0]["resume_node_id"] == "selected_compute"
    assert analysis["groups"][0]["predictable_dependency"] == nodes[0]["attrs"][
        "predictable_dependency"
    ]


def test_rejects_non_exact_or_unbound_predictable_selection_dependencies() -> None:
    for dependency in (
        {
            "schema": "nerve.predictable_resource_selection.v1",
            "kind": "parameter_table_lookup",
            "key_signal": "input",
            "table_parameter": "router",
            "selection_semantics": "advisory",
        },
        {
            "schema": "nerve.predictable_resource_selection.v1",
            "kind": "parameter_table_lookup",
            "key_signal": "not_an_input",
            "table_parameter": "router",
            "selection_semantics": "exact",
        },
    ):
        nodes, refs = _optional_component()
        nodes[0]["params"] = ["always_bias"]
        nodes[0]["attrs"]["predictable_dependency"] = dependency
        with pytest.raises(ModelCompileError, match="predictable dependency"):
            analyze_resource_residency_components(
                components=[_component(nodes, refs)],
                tensor_index=_analysis_tensors(),
                require_direct_packaging=True,
            )


def test_rejects_predictable_dependency_without_selection_domain() -> None:
    nodes, refs = _optional_component()
    dependency = {
        "schema": "nerve.predictable_resource_selection.v1",
        "kind": "parameter_table_lookup",
        "key_signal": "input",
        "table_parameter": "router",
        "selection_semantics": "exact",
    }
    nodes[1]["attrs"]["predictable_dependency"] = dependency

    with pytest.raises(
        ModelCompileError, match="predictable dependency requires a selection domain"
    ):
        analyze_resource_residency_components(
            components=[_component(nodes, refs)],
            tensor_index=_analysis_tensors(),
            require_direct_packaging=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda nodes: nodes[0]["attrs"]["selection_domain"].update(
                {"runtime_device": "gpu0"}
            ),
            "ambiguous selection domain",
        ),
        (
            lambda nodes: nodes[1]["attrs"]["selected_parameter_accesses"][0].update(
                {"prefetch": True}
            ),
            "ambiguous selected parameter access",
        ),
        (
            lambda nodes: nodes[0]["attrs"]["selection_domain"].update(
                {"selection_signal": "not_a_node_output"}
            ),
            "does not produce declared selection signal",
        ),
        (
            lambda nodes: nodes[0]["attrs"]["selection_domain"]["encoding"].update(
                {"index_mask": 1}
            ),
            "invalid selection index encoding",
        ),
    ),
)
def test_rejects_ambiguous_residency_metadata(mutation, message: str) -> None:
    nodes, refs = _optional_component()
    mutation(nodes)

    with pytest.raises(ModelCompileError, match=message):
        analyze_resource_residency_components(
            components=[_component(nodes, refs)],
            tensor_index=_analysis_tensors(),
            require_direct_packaging=True,
        )


def _write_source_tensor(
    root: Path,
    *,
    tensor_name: str,
    shape: list[int],
    payload: bytes,
) -> dict[str, object]:
    header = {
        tensor_name: {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path = root / f"{tensor_name.replace('.', '_')}.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return {
        "dtype": "U8",
        "shape": shape,
        "byte_count": len(payload),
        "parameter_count": len(payload),
        "source_file": str(path),
        "source_header_bytes": len(encoded),
        "data_offsets": [0, len(payload)],
    }


def test_packages_partition_digests_and_builds_compact_dynamic_contract(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tensor_index = {
        "tensors": {
            "tensor.projection_bank": _write_source_tensor(
                source_dir,
                tensor_name="tensor.projection_bank",
                shape=[3, 8],
                payload=bytes(range(24)),
            ),
            "tensor.always_bias": _write_source_tensor(
                source_dir,
                tensor_name="tensor.always_bias",
                shape=[8],
                payload=b"abcdefgh",
            ),
        },
        "totals": {
            "tensor_count": 2,
            "parameter_count": 32,
            "byte_count": 32,
        },
    }
    nodes, refs = _optional_component()
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensor_index,
        require_direct_packaging=True,
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    packaged = copy_tensor_package(
        tensor_index,
        package_dir,
        partition_counts=partition_counts_for_packaging(analysis),
        artifact_affinity_groups=artifact_affinity_groups_for_packaging(analysis),
    )
    manifest = {
        "circuit_graph": {
            "components": [
                {
                    "component_id": "component",
                    "circuit": {"nodes": nodes},
                    "params": {"refs": refs},
                },
                {
                    "component_id": "component_copy",
                    "circuit": {"nodes": nodes},
                    "params": {"refs": refs},
                },
            ]
        },
        "speculative_decoders": [],
    }
    contract = build_planned_resource_residency_contract(
        package_dir=package_dir,
        tensor_index=packaged,
        manifest=manifest,
    )

    partition_metadata = packaged["tensors"]["tensor.projection_bank"][
        "partition_integrity"
    ]
    assert partition_metadata["schema"] == TENSOR_PARTITION_INTEGRITY_SCHEMA
    table = package_dir / partition_metadata["digest_table_path"]
    assert table.stat().st_size == 3 * 32
    assert table.read_bytes() == b"".join(
        sha256(bytes(range(start, start + 8))).digest() for start in (0, 8, 16)
    )
    assert len(contract["resources"]) == 1
    assert len(contract["atomic_groups"]) == 1
    assert len(contract["partition_templates"]) == 1
    template = contract["partition_templates"][0]
    assert template["partition_count"] == 3
    assert len(template["member_templates"]) == 1
    assert len(contract["selectors"]) == 2
    assert [
        (
            selector["selection_signal"],
            selector["encoding"]["element_type"],
            selector["encoding"]["selection_count_per_activation"],
            selector["encoding"]["index_shift"],
            selector["encoding"]["index_mask"],
        )
        for selector in contract["selectors"]
    ] == [
        ("feature_index", "u32", 1, 0, 0xFFFF),
        ("feature_index", "u32", 1, 0, 0xFFFF),
    ]
    assert len(contract["checkpoints"]) == 2
    assert len(contract["bindings"]) == 4
    assert {
        binding["mapping"].get("partition_template_id")
        for binding in contract["bindings"]
        if binding["mapping"]["kind"] == "partition_template_member"
    } == {template["id"]}
    dynamic_binding = next(
        binding
        for binding in contract["bindings"]
        if binding["parameter_id"] == "projection_bank"
    )
    assert dynamic_binding["mapping"]["kind"] == "partition_template_member"
    assert dynamic_binding["mapping"]["partition_template_id"] == template["id"]
    assert dynamic_binding["mapping"]["selection_signal"] == "feature_index"
    spine_binding = next(
        binding
        for binding in contract["bindings"]
        if binding["parameter_id"] == "always_bias"
    )
    assert spine_binding["mapping"]["kind"] == "atomic_group"
    validate_resource_residency_contract(package_dir, contract, manifest)
    forbidden_runtime_policy_fields = {
        "device",
        "device_id",
        "placement",
        "capacity",
        "available_memory",
        "initial_resident",
        "prefetch",
        "eviction",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    assert keys(contract).isdisjoint(forbidden_runtime_policy_fields)


def test_partition_bindings_preserve_physical_parameter_slot_order(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    specs = {
        "tensor.router": ([4, 2], 8),
        "tensor.bank_a": ([4, 8], 32),
        "tensor.scale_a": ([4, 4], 16),
        "tensor.bank_b": ([4, 12], 48),
        "tensor.scale_b": ([4, 4], 16),
    }
    tensor_index = {
        "tensors": {
            name: _write_source_tensor(
                source_dir,
                tensor_name=name,
                shape=shape,
                payload=bytes([index]) * byte_count,
            )
            for index, (name, (shape, byte_count)) in enumerate(
                specs.items(), start=1
            )
        },
        "totals": {
            "tensor_count": len(specs),
            "parameter_count": sum(byte_count for _, byte_count in specs.values()),
            "byte_count": sum(byte_count for _, byte_count in specs.values()),
        },
    }
    nodes, refs = _routed_component()
    nodes[1]["params"] = ["scale_a", "bank_a"]
    nodes[1]["attrs"]["selected_parameter_accesses"][0]["parameter_ids"] = [
        "scale_a",
        "bank_a",
    ]
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensor_index,
        require_direct_packaging=True,
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    packaged = copy_tensor_package(
        tensor_index,
        package_dir,
        partition_counts=partition_counts_for_packaging(analysis),
        artifact_affinity_groups=artifact_affinity_groups_for_packaging(analysis),
    )
    manifest = {
        "circuit_graph": {
            "components": [
                {
                    "component_id": "component",
                    "circuit": {"nodes": nodes},
                    "params": {"refs": refs},
                }
            ]
        },
        "speculative_decoders": [],
    }
    contract = build_planned_resource_residency_contract(
        package_dir=package_dir,
        tensor_index=packaged,
        manifest=manifest,
    )

    first_node_slots = {
        binding["parameter_id"]: binding["mapping"]["parameter_slot"]
        for binding in contract["bindings"]
        if binding["node_id"] == "first_selected_compute"
    }
    assert first_node_slots == {"scale_a": 0, "bank_a": 1}
    validate_resource_residency_contract(package_dir, contract, manifest)

    corrupt = json.loads(json.dumps(contract))
    first_node_bindings = [
        binding
        for binding in corrupt["bindings"]
        if binding["node_id"] == "first_selected_compute"
    ]
    next(
        binding
        for binding in first_node_bindings
        if binding["parameter_id"] == "bank_a"
    )["mapping"]["parameter_slot"] = 0
    with pytest.raises(ModelCompileError, match="contiguous parameter-slot layout"):
        validate_resource_residency_contract(package_dir, corrupt, manifest)


def test_builds_concrete_group_table_for_independent_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tensor_index = {
        "tensors": {
            "tensor.router": _write_source_tensor(
                source_dir,
                tensor_name="tensor.router",
                shape=[2],
                payload=b"rt",
            ),
            **{
                f"tensor.unit_{unit}_{kind}": _write_source_tensor(
                    source_dir,
                    tensor_name=f"tensor.unit_{unit}_{kind}",
                    shape=[2] if kind == "scale" else [2, 4],
                    payload=bytes([16 * unit + offset]) * (2 if kind == "scale" else 8),
                )
                for unit in range(2)
                for offset, kind in enumerate(("scale", "weight"), start=1)
            },
        },
        "totals": {
            "tensor_count": 5,
            "parameter_count": 22,
            "byte_count": 22,
        },
    }
    nodes, refs = _independent_component()
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensor_index,
        require_direct_packaging=True,
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    packaged = copy_tensor_package(
        tensor_index,
        package_dir,
        partition_counts=partition_counts_for_packaging(analysis),
        artifact_affinity_groups=artifact_affinity_groups_for_packaging(analysis),
    )
    manifest = {
        "circuit_graph": {
            "components": [
                {
                    "component_id": "component",
                    "circuit": {"nodes": nodes},
                    "params": {"refs": refs},
                }
            ]
        },
        "speculative_decoders": [],
    }

    resource_builds: Counter[str] = Counter()
    source_header_catalogs: set[int] = set()
    artifact_size_catalogs: set[int] = set()
    original_resource_builder = residency_planning.compiled_immutable_resource

    def count_resource_builds(**kwargs: object) -> dict[str, object]:
        resource_builds[str(kwargs["tensor_name"])] += 1
        source_header_catalogs.add(id(kwargs["source_headers"]))
        artifact_size_catalogs.add(id(kwargs["artifact_byte_counts"]))
        return original_resource_builder(**kwargs)

    monkeypatch.setattr(
        residency_planning,
        "compiled_immutable_resource",
        count_resource_builds,
    )

    contract = build_planned_resource_residency_contract(
        package_dir=package_dir,
        tensor_index=packaged,
        manifest=manifest,
    )

    selector = contract["selectors"][0]
    assert selector["mapping"]["kind"] == "group_table"
    assert len(selector["mapping"]["atomic_group_ids"]) == 2
    assert contract["partition_templates"] == []
    assert len(contract["resources"]) == 5
    assert len(contract["atomic_groups"]) == 3
    dynamic_bindings = [
        binding
        for binding in contract["bindings"]
        if binding["parameter_id"] != "router"
    ]
    assert len(dynamic_bindings) == 4
    assert {
        binding["mapping"]["atomic_group_id"] for binding in dynamic_bindings
    } == set(selector["mapping"]["atomic_group_ids"])
    assert all(
        binding["mapping"]["kind"] == "selected_atomic_group"
        for binding in dynamic_bindings
    )
    dynamic_resource_ids = {
        binding["mapping"]["resource_id"] for binding in dynamic_bindings
    }
    dynamic_ranges = [
        resource["ranges"][0]
        for resource in contract["resources"]
        if resource["id"] in dynamic_resource_ids
    ]
    assert len({range_["artifact_path"] for range_ in dynamic_ranges}) == 1
    ordered_ranges = sorted(dynamic_ranges, key=lambda range_: range_["byte_offset"])
    assert all(
        current["byte_offset"] + current["byte_count"] == following["byte_offset"]
        for current, following in zip(ordered_ranges, ordered_ranges[1:])
    )
    assert sorted(
        (
            binding["mapping"]["selector_index"],
            binding["mapping"]["parameter_slot"],
        )
        for binding in dynamic_bindings
    ) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert resource_builds == Counter({tensor_name: 1 for tensor_name in packaged["tensors"]})
    assert len(source_header_catalogs) == 1
    assert len(artifact_size_catalogs) == 1
    validate_resource_residency_contract(package_dir, contract, manifest)

    corrupt = json.loads(json.dumps(contract))
    corrupt_dynamic = [
        binding
        for binding in corrupt["bindings"]
        if binding["mapping"]["kind"] == "selected_atomic_group"
    ]
    corrupt_dynamic[-1]["mapping"]["parameter_slot"] = 2
    with pytest.raises(ModelCompileError, match="contiguous parameter-slot layout"):
        validate_resource_residency_contract(package_dir, corrupt, manifest)

    corrupt = json.loads(json.dumps(contract))
    corrupt_dynamic = [
        binding
        for binding in corrupt["bindings"]
        if binding["mapping"]["kind"] == "selected_atomic_group"
    ]
    corrupt_dynamic[-1]["mapping"]["selector_index"] = 0
    with pytest.raises(ModelCompileError, match="exactly one group-table selector"):
        validate_resource_residency_contract(package_dir, corrupt, manifest)

    corrupt = json.loads(json.dumps(contract))
    corrupt_dynamic = [
        binding
        for binding in corrupt["bindings"]
        if binding["mapping"]["kind"] == "selected_atomic_group"
    ]
    same_selector = [
        binding
        for binding in corrupt_dynamic
        if binding["mapping"]["selector_index"] == 1
    ]
    same_selector[1]["mapping"]["parameter_slot"] = same_selector[0]["mapping"][
        "parameter_slot"
    ]
    with pytest.raises(ModelCompileError, match="repeat a selector parameter slot"):
        validate_resource_residency_contract(package_dir, corrupt, manifest)


def test_packages_co_selected_resources_contiguously_in_one_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    payloads = {
        "tensor.router": b"rt",
        "tensor.unit_0_scale": b"aa",
        "tensor.unit_0_weight": b"bbbbbbbb",
        "tensor.unit_1_scale": b"cc",
        "tensor.unit_1_weight": b"dddddddd",
    }
    tensor_index = {
        "tensors": {
            name: _write_source_tensor(
                source_dir,
                tensor_name=name,
                shape=[len(payload)],
                payload=payload,
            )
            for name, payload in payloads.items()
        },
        "totals": {
            "tensor_count": len(payloads),
            "parameter_count": sum(map(len, payloads.values())),
            "byte_count": sum(map(len, payloads.values())),
        },
    }
    nodes, refs = _independent_component()
    analysis = analyze_resource_residency_components(
        components=[_component(nodes, refs)],
        tensor_index=tensor_index,
        require_direct_packaging=True,
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    individual_writes: list[str] = []
    original_writer = __import__(
        "nerve.model_package_assets", fromlist=["write_compiled_tensor"]
    ).write_compiled_tensor

    def count_individual_writes(**kwargs: object) -> tuple[int, str, list[bytes]]:
        individual_writes.append(str(kwargs["tensor_name"]))
        return original_writer(**kwargs)

    monkeypatch.setattr(
        "nerve.model_package_assets.write_compiled_tensor",
        count_individual_writes,
    )

    packaged = copy_tensor_package(
        tensor_index,
        package_dir,
        artifact_affinity_groups=artifact_affinity_groups_for_packaging(analysis),
    )

    dynamic_names = artifact_affinity_groups_for_packaging(analysis)[0]
    dynamic_infos = [packaged["tensors"][name] for name in dynamic_names]
    assert len({info["source_file"] for info in dynamic_infos}) == 1
    assert [info["data_offsets"] for info in dynamic_infos] == [
        [0, 2],
        [2, 10],
        [10, 12],
        [12, 20],
    ]
    assert len(packaged["source"]["weights_files"]) == 2
    bank_path = package_dir / dynamic_infos[0]["source_file"]
    bank_payload = bank_path.read_bytes()
    header_bytes = dynamic_infos[0]["safetensors_header_bytes"]
    data_start = 8 + header_bytes
    assert bank_payload[data_start:] == b"aabbbbbbbbccdddddddd"
    assert all(
        info["data_sha256"] == sha256(payloads[name]).hexdigest()
        for name, info in zip(dynamic_names, dynamic_infos, strict=True)
    )
    assert len(list((package_dir / "weights").glob("tensor_*.safetensors"))) == 1
    assert individual_writes == ["tensor.router"]


def test_rejects_overlapping_artifact_affinity_groups(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tensor_index = {
        "tensors": {
            name: _write_source_tensor(
                source_dir,
                tensor_name=name,
                shape=[1],
                payload=payload,
            )
            for name, payload in (("a", b"a"), ("b", b"b"), ("c", b"c"))
        },
        "totals": {"tensor_count": 3, "parameter_count": 3, "byte_count": 3},
    }
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    with pytest.raises(ModelCompileError, match="appears in multiple artifact affinity"):
        copy_tensor_package(
            tensor_index,
            package_dir,
            artifact_affinity_groups=[["a", "b"], ["b", "c"]],
        )


def test_affinity_repack_retains_a_shared_artifact_still_used_elsewhere(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True)
    payloads = {"a": b"aa", "b": b"bbb", "c": b"cccc"}
    offsets: dict[str, list[int]] = {}
    cursor = 0
    header: dict[str, object] = {}
    for name, payload in payloads.items():
        offsets[name] = [cursor, cursor + len(payload)]
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": offsets[name],
        }
        cursor += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    source_path = weights_dir / "shared.safetensors"
    source_path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads.values())
    )
    tensors = {
        name: {
            "dtype": "U8",
            "shape": [len(payload)],
            "byte_count": len(payload),
            "source_file": "weights/shared.safetensors",
            "safetensors_header_bytes": len(encoded),
            "data_offsets": offsets[name],
            "data_sha256": sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }

    records = pack_tensor_artifacts_by_affinity(
        package_dir=package_dir,
        tensors=tensors,
        compiled_sources=[
            {
                "path": "weights/shared.safetensors",
                "safetensors_header_bytes": len(encoded),
            }
        ],
        affinity_groups=[["a", "b"]],
    )

    assert source_path.is_file()
    assert tensors["c"]["source_file"] == "weights/shared.safetensors"
    assert tensors["a"]["source_file"] == tensors["b"]["source_file"]
    assert tensors["a"]["source_file"] != tensors["c"]["source_file"]
    assert {record["path"] for record in records} == {
        "weights/shared.safetensors",
        tensors["a"]["source_file"],
    }


def test_composite_packaging_hashes_output_partitions_across_source_parts(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_parts = []
    payload = bytes(range(24))
    cursor = 0
    for index, part_bytes in enumerate((3, 5, 4, 7, 5)):
        part_name = f"part.{index}"
        part_payload = payload[cursor : cursor + part_bytes]
        metadata = _write_source_tensor(
            source_dir,
            tensor_name=part_name,
            shape=[part_bytes],
            payload=part_payload,
        )
        source_parts.append(
            {
                "tensor": part_name,
                "source_file": metadata["source_file"],
                "source_header_bytes": metadata["source_header_bytes"],
                "data_offsets": metadata["data_offsets"],
                "byte_count": part_bytes,
            }
        )
        cursor += part_bytes
    tensor_index = {
        "tensors": {
            "tensor.composite_bank": {
                "dtype": "U8",
                "shape": [3, 8],
                "byte_count": 24,
                "parameter_count": 24,
                "data_offsets": [0, 24],
                "layout_hint": "row_major",
                "source_parts": source_parts,
            }
        },
        "totals": {
            "tensor_count": 1,
            "parameter_count": 24,
            "byte_count": 24,
        },
    }
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    packaged = copy_tensor_package(
        tensor_index,
        package_dir,
        partition_counts={"tensor.composite_bank": 3},
    )

    integrity = packaged["tensors"]["tensor.composite_bank"]["partition_integrity"]
    table = (package_dir / integrity["digest_table_path"]).read_bytes()
    assert table == b"".join(
        sha256(payload[start : start + 8]).digest() for start in (0, 8, 16)
    )


@pytest.mark.parametrize(
    "forbidden",
    (
        "qwen",
        "mixture_of_experts",
        "routed_experts",
        "sparse_moe",
    ),
)
def test_planner_implementation_has_no_model_or_operator_special_cases(
    forbidden: str,
) -> None:
    source = (
        Path(__file__).parents[1] / "nerve" / "resource_residency_planning.py"
    ).read_text()
    assert forbidden not in source.lower()


@pytest.mark.parametrize("forbidden", ("deepseek", "qwen", "gemma", "llama", "lfm"))
def test_artifact_layout_has_no_model_family_switches(forbidden: str) -> None:
    root = Path(__file__).parents[1] / "nerve"
    source = "\n".join(
        (root / filename).read_text()
        for filename in (
            "model_package_artifact_layout.py",
            "model_package_assets.py",
            "resource_residency_planning.py",
        )
    )
    assert forbidden not in source.lower()
