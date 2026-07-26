from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer import (
    ContractValidationError,
    canonical_json_bytes,
    optimization_scope_catalog_id,
    validate_contract,
)
from nerve.representation_optimizer.scope_enumeration import (
    SemanticDependencyGraph,
    enumerate_optimization_scope_catalog,
)
from nerve.representation_optimizer.scope_enumeration.graph import component_view


def module(
    module_id: str,
    role: str,
    *,
    parent_id: str | None,
    nodes: list[str] | None = None,
    children: list[str] | None = None,
    virtual: bool = False,
    params: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": module_id,
        "role": role,
        "responsibility": f"fixture responsibility for {role}",
        "parent_id": parent_id,
        "source_node_ids": nodes or [],
        "parameter_ref_ids": params or [],
        "owned_state_port_ids": [],
        "virtual": virtual,
        "attrs": {},
        "child_ids": children or [],
        "input_signals": [],
        "output_signals": [],
    }


def layer_component(
    component_id: str,
    family: str,
    *,
    state_type: str | None = None,
    sparse_moe: bool = False,
) -> object:
    state_ports = (
        [
            {
                "id": "memory",
                "type": state_type,
                "shape": [4, 8],
                "dtype": "F32",
            }
        ]
        if state_type is not None
        else []
    )
    mixer = {
        "id": "mix",
        "op": family,
        "inputs": ["normalized"],
        "outputs": ["mixed"],
        "params": ["mix_weight"],
        "state_reads": ["memory"] if state_type else [],
        "state_writes": ["memory"] if state_type else [],
    }
    nodes = [
        {
            "id": "normalize",
            "op": "normalize",
            "inputs": ["input_frame"],
            "outputs": ["normalized"],
            "params": ["norm_weight"],
            "state_reads": [],
            "state_writes": [],
        },
        mixer,
        {
            "id": "feature_transform",
            "op": "dense_mlp" if not sparse_moe else "sparse_moe",
            "inputs": ["mixed"],
            "outputs": ["output_frame"],
            "params": ["feature_weight"],
            "state_reads": [],
            "state_writes": [],
        },
    ]
    feature_children = ["layer.feature_transform.compute"]
    modules = [
        module(
            "layer",
            "layer",
            parent_id=None,
            children=["layer.token_mixer", "layer.feature_transform"],
        ),
        module(
            "layer.token_mixer",
            "token_mixer",
            parent_id="layer",
            children=[
                "layer.token_mixer.normalization",
                "layer.token_mixer.compute",
            ],
        ),
        module(
            "layer.token_mixer.normalization",
            "normalization",
            parent_id="layer.token_mixer",
            nodes=["normalize"],
        ),
        module(
            "layer.token_mixer.compute",
            family,
            parent_id="layer.token_mixer",
            nodes=["mix"],
        ),
        module(
            "layer.feature_transform",
            "feature_transform",
            parent_id="layer",
            children=feature_children,
        ),
        module(
            "layer.feature_transform.compute",
            "sparse_moe" if sparse_moe else "dense_mlp",
            parent_id="layer.feature_transform",
            nodes=["feature_transform"],
        ),
    ]
    if sparse_moe:
        feature_children.append("layer.feature_transform.expert_000")
        modules.append(
            module(
                "layer.feature_transform.expert_000",
                "expert",
                parent_id="layer.feature_transform",
                virtual=True,
                params=["feature_weight"],
            )
        )
    circuit = {
        "schema": "nerve.stream_circuit.v1",
        "id": f"{component_id}_circuit",
        "source": {
            "component_id": component_id,
            "source_layer_index": 0,
            "source_operator_type": family,
        },
        "runtime_role": "signal_processor",
        "implementation": "fixture_exact",
        "boundary": {
            "inputs": [
                {
                    "id": "input_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "input",
                }
            ],
            "outputs": [
                {
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "output",
                }
            ],
            "controls": [],
        },
        "state_ports": state_ports,
        "parameters": {
            "layout": "fixture",
            "storage": "source_tensor_refs",
            "refs": {
                "norm_weight": {"tensor": f"{component_id}.norm"},
                "mix_weight": {"tensor": f"{component_id}.mix"},
                "feature_weight": {"tensor": f"{component_id}.feature"},
            },
        },
        "nodes": nodes,
        "semantic_execution_nodes": nodes,
        "semantic_module_tree": {
            "schema": "nerve.semantic_module_tree.v1",
            "root_module_id": "layer",
            "modules": modules,
        },
    }
    return component_view(
        component_id=component_id,
        operator_type=family,
        runtime_role="signal_processor",
        implementation="fixture_exact",
        artifact_ref=f"lowered/{component_id}/circuit.json",
        circuit=circuit,
    )


def simple_component(
    component_id: str,
    role: str,
    *,
    input_signal: str,
    output_signal: str,
    op: str,
    randomness: bool = False,
) -> object:
    inputs = [
        {
            "id": input_signal,
            "signal": input_signal,
            "shape": [1],
            "component_port": "randomness" if randomness else "input",
        }
    ]
    node_inputs = [input_signal]
    if role == "sampler":
        inputs = [
            {
                "id": "input_logits",
                "signal": "logits",
                "shape": [8],
                "component_port": "logits",
            },
            {
                "id": "random_seed",
                "signal": "random_seed",
                "shape": [1],
                "component_port": "randomness",
            },
        ]
        node_inputs = ["input_logits", "random_seed"]
    params = {} if role == "sampler" else {"weight": {"tensor": f"{component_id}.weight"}}
    node = {
        "id": op,
        "op": op,
        "inputs": node_inputs,
        "outputs": [output_signal],
        "params": list(params),
        "state_reads": [],
        "state_writes": [],
        "attrs": {"randomness": "seed_and_tick"} if role == "sampler" else {},
    }
    circuit = {
        "schema": "nerve.stream_circuit.v1",
        "id": f"{component_id}_circuit",
        "source": {
            "component_id": component_id,
            "source_layer_index": None,
            "source_operator_type": role,
        },
        "runtime_role": role,
        "implementation": "fixture_exact",
        "boundary": {
            "inputs": inputs,
            "outputs": [
                {
                    "id": output_signal,
                    "signal": output_signal,
                    "shape": [1],
                    "component_port": "output",
                }
            ],
            "controls": [],
        },
        "state_ports": [],
        "parameters": {
            "layout": "fixture",
            "storage": "source_tensor_refs",
            "refs": params,
        },
        "nodes": [node],
    }
    return component_view(
        component_id=component_id,
        operator_type=role,
        runtime_role=role,
        implementation="fixture_exact",
        artifact_ref=f"lowered/{component_id}/circuit.json",
        circuit=circuit,
    )


def graph_for_components(
    components: list[object],
    edges: list[dict[str, object]] | None = None,
    *,
    public_outputs: list[dict[str, object]] | None = None,
    graph_artifact_ref: str | None = None,
) -> SemanticDependencyGraph:
    return SemanticDependencyGraph.from_documents(
        components=components,
        edges=edges or [],
        public_outputs=public_outputs or [],
        graph_artifact_ref=graph_artifact_ref,
    )


@pytest.mark.parametrize(
    ("family", "state_type", "sparse_moe"),
    [
        ("full_attention", "append_only_attention_memory", False),
        ("gated_recurrent", "bounded_recurrent_state", False),
        ("temporal_convolution", "rolling_channel_memory", False),
        ("identity_token_mixer", None, False),
        ("expert_routing", None, True),
        ("multimodal_projection", None, False),
    ],
)
def test_scope_enumeration_is_architecture_neutral(
    family: str,
    state_type: str | None,
    sparse_moe: bool,
) -> None:
    component = layer_component(
        "layer_00",
        family,
        state_type=state_type,
        sparse_moe=sparse_moe,
    )
    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_package",
        graph=graph_for_components([component]),
    )
    counts = catalog["summary"]["classification_counts"]

    assert counts["operator"] == 3
    assert counts["semantic_leaf_module"] == 3
    assert counts["coupled_region"] == 3
    assert counts["coupled_sibling_operations"] == 2
    assert counts["feature_transform_region"] == 1
    assert counts["token_mixer_region"] == 1
    assert counts["layer"] == 1
    if state_type is not None:
        assert counts["stateful_system"] == 1
    if sparse_moe:
        assert catalog["summary"]["rejected_scope_count"] == 1
        assert "virtual semantic module" in catalog["diagnostics"][0]["reason"]
    else:
        assert catalog["diagnostics"] == []


def test_scope_boundaries_capture_signals_parameters_state_and_randomness() -> None:
    layer = layer_component(
        "layer_00",
        "full_attention",
        state_type="append_only_attention_memory",
    )
    sampler = simple_component(
        "sampler",
        "sampler",
        input_signal="input_logits",
        output_signal="sampled_token",
        op="sample",
    )
    layer_catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_layer",
        graph=graph_for_components([layer]),
    )
    layer_scope = next(
        scope
        for scope in layer_catalog["scopes"]
        if "layer" in scope["extensions"]["classifications"]
    )
    assert [item["signal_id"] for item in layer_scope["boundary"]["inputs"]] == [
        "input_frame"
    ]
    assert [item["signal_id"] for item in layer_scope["boundary"]["outputs"]] == [
        "output_frame"
    ]
    assert {
        item["parameter_ref_id"]
        for item in layer_scope["boundary"]["parameters"]
    } == {"feature_weight", "mix_weight", "norm_weight"}
    assert layer_scope["boundary"]["states"][0]["access"] == ["read", "write"]

    sampler_catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_sampler",
        graph=graph_for_components(
            [sampler],
            public_outputs=[
                {
                    "id": "model_output",
                    "endpoint": {
                        "component_id": "sampler",
                        "port_id": "sampled_token",
                    },
                }
            ],
        ),
    )
    sampler_scope = next(
        scope
        for scope in sampler_catalog["scopes"]
        if "sampler" in scope["extensions"]["classifications"]
    )
    assert [item["signal_id"] for item in sampler_scope["boundary"]["inputs"]] == [
        "input_logits"
    ]
    assert [item["signal_id"] for item in sampler_scope["boundary"]["randomness"]] == [
        "random_seed"
    ]
    assert sampler_scope["boundary"]["randomness"][0]["semantics"] == [
        "seed_and_tick"
    ]
    assert sampler_scope["boundary"]["outputs"][0]["public"] is True


def test_scope_boundaries_capture_controls_without_treating_them_as_data() -> None:
    component = layer_component("layer_00", "gated_recurrent")
    component.circuit["boundary"]["controls"] = [
        {
            "id": "reset_state",
            "signal": "reset_state",
            "shape": [1],
            "component_port": "control",
        }
    ]
    component.circuit["nodes"][1]["inputs"].append("reset_state")
    component = component_view(
        component_id=component.component_id,
        operator_type=component.operator_type,
        runtime_role=component.runtime_role,
        implementation=component.implementation,
        artifact_ref=component.artifact_ref,
        circuit=component.circuit,
    )

    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_control",
        graph=graph_for_components([component]),
    )
    layer_scope = next(
        scope
        for scope in catalog["scopes"]
        if "layer" in scope["extensions"]["classifications"]
    )
    assert [item["signal_id"] for item in layer_scope["boundary"]["controls"]] == [
        "reset_state"
    ]
    assert "reset_state" not in {
        item["signal_id"] for item in layer_scope["boundary"]["inputs"]
    }


def test_state_scope_couples_distinct_writer_and_reader_nodes() -> None:
    component = layer_component(
        "layer_00",
        "gated_recurrent",
        state_type="bounded_recurrent_state",
    )
    component.circuit["nodes"][1]["state_reads"] = []
    component.circuit["nodes"][2]["state_reads"] = ["memory"]
    component = component_view(
        component_id=component.component_id,
        operator_type=component.operator_type,
        runtime_role=component.runtime_role,
        implementation=component.implementation,
        artifact_ref=component.artifact_ref,
        circuit=component.circuit,
    )

    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_state_system",
        graph=graph_for_components([component]),
    )
    state_scope = next(
        scope
        for scope in catalog["scopes"]
        if "state_writer_reader_system" in scope["extensions"]["classifications"]
    )
    assert state_scope["members"]["source_node_ids"] == [
        "layer_00/mix",
        "layer_00/feature_transform",
    ]
    assert state_scope["boundary"]["states"][0]["access"] == ["read", "write"]


def test_adjacent_component_islands_hide_internal_edges_and_feedback_is_explicit() -> None:
    input_transducer = simple_component(
        "input_transducer",
        "input_transducer",
        input_signal="input_token",
        output_signal="output_frame",
        op="embedding",
    )
    output_transducer = simple_component(
        "output_transducer",
        "output_transducer",
        input_signal="input_frame",
        output_signal="output_logits",
        op="projection",
    )
    sampler = simple_component(
        "sampler",
        "sampler",
        input_signal="input_logits",
        output_signal="sampled_token",
        op="sample",
    )
    edges = [
        {
            "id": "forward_0",
            "connection": {"kind": "forward"},
            "source": {
                "component_id": "input_transducer",
                "port_id": "output_frame",
            },
            "destination": {
                "component_id": "output_transducer",
                "port_id": "input_frame",
            },
        },
        {
            "id": "forward_1",
            "connection": {"kind": "forward"},
            "source": {
                "component_id": "output_transducer",
                "port_id": "output_logits",
            },
            "destination": {
                "component_id": "sampler",
                "port_id": "input_logits",
            },
        },
        {
            "id": "feedback",
            "connection": {"kind": "temporal_feedback", "delay_activations": 1},
            "source": {
                "component_id": "sampler",
                "port_id": "sampled_token",
            },
            "destination": {
                "component_id": "input_transducer",
                "port_id": "input_token",
            },
        },
    ]
    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_generation",
        graph=graph_for_components(
            [input_transducer, output_transducer, sampler],
            edges,
            public_outputs=[
                {
                    "id": "model_output",
                    "endpoint": {
                        "component_id": "sampler",
                        "port_id": "sampled_token",
                    },
                }
            ],
        ),
    )
    counts = catalog["summary"]["classification_counts"]
    assert counts["input_transducer"] == 1
    assert counts["output_transducer"] == 1
    assert counts["sampler"] == 1
    assert counts["feedback_transducer"] == 1
    assert counts["representation_island"] == 2
    feedback = next(
        scope
        for scope in catalog["scopes"]
        if "feedback_transducer" in scope["extensions"]["classifications"]
    )
    assert feedback["boundary"]["dependencies"] == [
        {
            "edge_id": "feedback",
            "connection": {
                "kind": "temporal_feedback",
                "delay_activations": 1,
            },
            "source": {
                "component_id": "sampler",
                "port_id": "sampled_token",
            },
            "destination": {
                "component_id": "input_transducer",
                "port_id": "input_token",
            },
            "covered_consumer_node_ids": ["input_transducer/embedding"],
        }
    ]

    input_output_island = next(
        scope
        for scope in catalog["scopes"]
        if scope["members"]["component_ids"]
        == ["input_transducer", "output_transducer"]
    )
    assert [item["signal_id"] for item in input_output_island["boundary"]["inputs"]] == [
        "input_token"
    ]
    assert [item["signal_id"] for item in input_output_island["boundary"]["outputs"]] == [
        "output_logits"
    ]


def test_cross_component_island_contains_only_boundary_producer_and_consumer() -> None:
    first = layer_component("layer_00", "full_attention")
    second = layer_component("layer_01", "temporal_convolution")
    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_exact_island",
        graph=graph_for_components(
            [first, second],
            [
                {
                    "id": "layer_edge",
                    "connection": {"kind": "forward"},
                    "source": {
                        "component_id": "layer_00",
                        "port_id": "output_frame",
                    },
                    "destination": {
                        "component_id": "layer_01",
                        "port_id": "input_frame",
                    },
                }
            ],
            graph_artifact_ref="lowered/execution_graph.circuits.json",
        ),
    )
    island = next(
        scope
        for scope in catalog["scopes"]
        if scope["members"]["component_ids"] == ["layer_00", "layer_01"]
        and "representation_island" in scope["extensions"]["classifications"]
    )

    assert island["members"]["source_node_ids"] == [
        "layer_00/feature_transform",
        "layer_01/normalize",
    ]
    assert [item["signal_id"] for item in island["boundary"]["inputs"]] == [
        "mixed",
    ]
    assert [item["signal_id"] for item in island["boundary"]["outputs"]] == [
        "normalized",
    ]
    assert island["boundary"]["dependencies"][0]["edge_id"] == "layer_edge"
    assert island["boundary"]["dependencies"][0]["connection"] == {
        "kind": "forward"
    }
    source_contract = next(
        contract
        for contract in catalog["source_contracts"]
        if contract["scope_id"] == island["scope_id"]
    )
    assert source_contract["exact_reference"]["artifact_refs"] == [
        "lowered/execution_graph.circuits.json",
        "lowered/layer_00/circuit.json",
        "lowered/layer_01/circuit.json",
    ]


def test_repeated_corresponding_modules_form_cross_layer_scope_once() -> None:
    first = layer_component("layer_00", "full_attention")
    second = layer_component("layer_01", "full_attention")
    graph = graph_for_components(
        [first, second],
        [
            {
                "id": "layer_edge",
                "connection": {"kind": "forward"},
                "source": {
                    "component_id": "layer_00",
                    "port_id": "output_frame",
                },
                "destination": {
                    "component_id": "layer_01",
                    "port_id": "input_frame",
                },
            }
        ],
    )
    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_repetition",
        graph=graph,
    )
    cross_layer = [
        scope
        for scope in catalog["scopes"]
        if "cross_layer_group" in scope["extensions"]["classifications"]
    ]

    assert cross_layer
    normalization_group = next(
        scope
        for scope in cross_layer
        if scope["members"]["semantic_module_ids"]
        == [
            "layer_00/layer.token_mixer.normalization",
            "layer_01/layer.token_mixer.normalization",
        ]
    )
    assert normalization_group["members"]["source_node_ids"] == [
        "layer_00/normalize",
        "layer_01/normalize",
    ]
    assert len(
        {
            scope["extensions"]["region_id"]
            for scope in catalog["scopes"]
        }
    ) == catalog["summary"]["scope_count"]


def test_scope_catalog_is_deterministic_and_rejects_linkage_drift() -> None:
    graph = graph_for_components(
        [layer_component("layer_00", "temporal_convolution")]
    )
    first = enumerate_optimization_scope_catalog(
        package_id="fixture_determinism",
        graph=graph,
    )
    second = enumerate_optimization_scope_catalog(
        package_id="fixture_determinism",
        graph=graph,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)

    drifted = deepcopy(first)
    drifted["source_contracts"][0]["semantic_role"] = "mutated"
    drifted["catalog_id"] = optimization_scope_catalog_id(drifted)
    with pytest.raises(
        ContractValidationError,
        match="source behavior contract digest",
    ):
        validate_contract(drifted)


def test_scope_catalog_rejects_dependency_drift_from_source_contract() -> None:
    first = layer_component("layer_00", "full_attention")
    second = layer_component("layer_01", "temporal_convolution")
    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_dependency_link",
        graph=graph_for_components(
            [first, second],
            [
                {
                    "id": "layer_edge",
                    "connection": {"kind": "forward"},
                    "source": {
                        "component_id": "layer_00",
                        "port_id": "output_frame",
                    },
                    "destination": {
                        "component_id": "layer_01",
                        "port_id": "input_frame",
                    },
                }
            ],
        ),
    )
    drifted = deepcopy(catalog)
    island = next(
        scope
        for scope in drifted["scopes"]
        if scope["boundary"]["dependencies"]
    )
    island["boundary"]["dependencies"][0]["connection"]["kind"] = "temporal_feedback"
    drifted["catalog_id"] = optimization_scope_catalog_id(drifted)

    with pytest.raises(
        ContractValidationError,
        match="boundary does not match",
    ):
        validate_contract(drifted)


def test_dependency_graph_rejects_ambiguous_signal_producers() -> None:
    component = layer_component("layer_00", "full_attention")
    duplicate = deepcopy(component.circuit["nodes"][0])
    duplicate["id"] = "duplicate_normalize"
    duplicate["inputs"] = ["input_frame"]
    component.circuit["nodes"].append(duplicate)
    component = component_view(
        component_id=component.component_id,
        operator_type=component.operator_type,
        runtime_role=component.runtime_role,
        implementation=component.implementation,
        artifact_ref=component.artifact_ref,
        circuit=component.circuit,
    )

    with pytest.raises(ModelCompileError, match="multiple source producers"):
        graph_for_components([component])


def test_dependency_graph_rejects_multiple_edges_to_one_input() -> None:
    first = simple_component(
        "first",
        "input_transducer",
        input_signal="first_input",
        output_signal="frame",
        op="first_source",
    )
    second = simple_component(
        "second",
        "input_transducer",
        input_signal="second_input",
        output_signal="frame",
        op="second_source",
    )
    destination = simple_component(
        "destination",
        "output_transducer",
        input_signal="input_frame",
        output_signal="logits",
        op="consume",
    )
    edges = [
        {
            "id": source.component_id,
            "connection": {"kind": "forward"},
            "source": {
                "component_id": source.component_id,
                "port_id": "frame",
            },
            "destination": {
                "component_id": "destination",
                "port_id": "input_frame",
            },
        }
        for source in (first, second)
    ]

    with pytest.raises(ModelCompileError, match="multiple graph producers"):
        graph_for_components([first, second, destination], edges)


def test_semantic_scopes_reject_ambiguous_leaf_ownership() -> None:
    component = layer_component("layer_00", "full_attention")
    feature_module = next(
        module
        for module in component.circuit["semantic_module_tree"]["modules"]
        if module["id"] == "layer.feature_transform.compute"
    )
    feature_module["source_node_ids"].append("normalize")
    component = component_view(
        component_id=component.component_id,
        operator_type=component.operator_type,
        runtime_role=component.runtime_role,
        implementation=component.implementation,
        artifact_ref=component.artifact_ref,
        circuit=component.circuit,
    )

    catalog = enumerate_optimization_scope_catalog(
        package_id="fixture_ambiguous_ownership",
        graph=graph_for_components([component]),
    )

    assert any(
        "unambiguous leaf-module owner" in diagnostic["reason"]
        or "ownership is ambiguous" in diagnostic["reason"]
        for diagnostic in catalog["diagnostics"]
    )
    assert not any(
        "layer" in scope["extensions"]["classifications"]
        for scope in catalog["scopes"]
    )
