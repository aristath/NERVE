from __future__ import annotations

import json
import struct
from hashlib import sha256
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.model_package_assets import copy_tensor_package
from nerve.resource_residency import (
    validate_resource_residency_contract,
)
from nerve.resource_residency_planning import (
    RESIDENCY_ANALYSIS_SCHEMA,
    TENSOR_PARTITION_INTEGRITY_SCHEMA,
    analyze_resource_residency_components,
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


def test_builds_concrete_group_table_for_independent_resources(
    tmp_path: Path,
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
        binding["mapping"]["kind"] == "atomic_group" for binding in dynamic_bindings
    )
    validate_resource_residency_contract(package_dir, contract, manifest)


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
