from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def exact_representation_graph(
    *,
    candidate_id: str = "candidate_fixture",
    scope_ids: tuple[str, ...] = ("scope_a", "scope_b"),
    source_contract_digests: tuple[str, ...] | None = None,
    evidence_ref: str = "evidence_fixture",
) -> Json:
    if not scope_ids:
        raise ValueError("fixture requires at least one scope")
    ordered_scopes = tuple(sorted(scope_ids))
    digests = source_contract_digests or tuple(
        f"nerve.optimizer.canonical_json_sha256.v1:{index + 1:064x}"
        for index in range(len(ordered_scopes))
    )
    if len(digests) != len(ordered_scopes):
        raise ValueError("fixture scope and digest counts differ")
    first_scope = ordered_scopes[0]
    second_scope = ordered_scopes[-1]

    def provenance(*scopes: str, source_nodes: tuple[str, ...] = ()) -> Json:
        return {
            "scope_ids": sorted(set(scopes)),
            "source_node_ids": sorted(source_nodes),
            "evidence_refs": [evidence_ref],
            "transform_refs": [],
        }

    graph = {
        "schema": REPRESENTATION_GRAPH_SCHEMA,
        "graph_id": "",
        "candidate_id": candidate_id,
        "scope_ids": list(ordered_scopes),
        "source_contract_digests": dict(
            sorted(zip(ordered_scopes, digests, strict=True))
        ),
        "logical_contracts": [
            {
                "id": "logical.coefficients",
                "signal": "coefficients",
                "shape": [8, 8],
                "dtype": "BF16",
            },
            {
                "id": "logical.hidden",
                "signal": "hidden",
                "shape": [8],
                "dtype": "BF16",
            },
            {
                "id": "logical.state",
                "signal": "temporal_state",
                "shape": [4],
                "dtype": "BF16",
            },
            {
                "id": "logical.topology",
                "signal": "event_topology",
                "shape": [8],
                "dtype": "U32",
            },
        ],
        "physical_representations": [
            {
                "id": "repr.parameter.dense",
                "kind": "dense_tensor",
                "domain": "parameter",
                "physical_shape": [8, 8],
                "encoding": {"dtype": "BF16"},
                "storage": {"layout": "row_major"},
            },
            {
                "id": "repr.parameter.sparse",
                "kind": "sparse_event_weights",
                "domain": "parameter",
                "physical_shape": [16],
                "encoding": {"index_dtype": "U16", "value_dtype": "BF16"},
                "storage": {"layout": "compressed_rows"},
            },
            {
                "id": "repr.signal.dense",
                "kind": "dense_tensor",
                "domain": "signal",
                "physical_shape": [8],
                "encoding": {"dtype": "BF16"},
                "storage": {"layout": "contiguous"},
            },
            {
                "id": "repr.signal.spectral",
                "kind": "spectral_coefficients",
                "domain": "signal",
                "physical_shape": [5],
                "encoding": {"basis": "real_fft", "dtype": "BF16"},
                "storage": {"layout": "packed_frequency"},
            },
            {
                "id": "repr.state.compact",
                "kind": "bounded_recurrent_state",
                "domain": "state",
                "physical_shape": [2],
                "encoding": {"dtype": "BF16", "reconstruction": "exact"},
                "storage": {"layout": "packed"},
            },
            {
                "id": "repr.topology.events",
                "kind": "sparse_event_graph",
                "domain": "topology",
                "physical_shape": [8],
                "encoding": {"index_dtype": "U16"},
                "storage": {"layout": "csr"},
            },
        ],
        "signals": [
            {
                "id": "signal.between",
                "logical_contract_id": "logical.hidden",
                "physical_representation_id": "repr.signal.spectral",
                "provenance": provenance(
                    first_scope,
                    second_scope,
                    source_nodes=("source.mix_a", "source.mix_b"),
                ),
            },
            {
                "id": "signal.input",
                "logical_contract_id": "logical.hidden",
                "physical_representation_id": "repr.signal.dense",
                "provenance": provenance(
                    first_scope, source_nodes=("source.input",)
                ),
            },
            {
                "id": "signal.native_input",
                "logical_contract_id": "logical.hidden",
                "physical_representation_id": "repr.signal.spectral",
                "provenance": provenance(
                    first_scope, source_nodes=("source.mix_a",)
                ),
            },
            {
                "id": "signal.native_output",
                "logical_contract_id": "logical.hidden",
                "physical_representation_id": "repr.signal.spectral",
                "provenance": provenance(
                    second_scope, source_nodes=("source.mix_b",)
                ),
            },
            {
                "id": "signal.output",
                "logical_contract_id": "logical.hidden",
                "physical_representation_id": "repr.signal.dense",
                "provenance": provenance(
                    second_scope, source_nodes=("source.output",)
                ),
            },
        ],
        "resources": [
            {
                "id": "resource.parameter",
                "kind": "parameter",
                "logical_contract_id": "logical.coefficients",
                "physical_representation_id": "repr.parameter.sparse",
                "artifact": {
                    "path": "parameters/sparse_weights.bin",
                    "format": "sparse_event_weights.v1",
                },
                "provenance": provenance(
                    first_scope, source_nodes=("source.weight",)
                ),
            },
            {
                "id": "resource.state",
                "kind": "state",
                "logical_contract_id": "logical.state",
                "physical_representation_id": "repr.state.compact",
                "artifact": {
                    "path": "state/compact_layout.json",
                    "format": "bounded_state.v1",
                },
                "provenance": provenance(
                    first_scope, source_nodes=("source.state",)
                ),
            },
            {
                "id": "resource.topology",
                "kind": "topology",
                "logical_contract_id": "logical.topology",
                "physical_representation_id": "repr.topology.events",
                "artifact": {
                    "path": "topology/events.bin",
                    "format": "sparse_event_graph.v1",
                },
                "provenance": provenance(
                    second_scope, source_nodes=("source.routes",)
                ),
            },
        ],
        "nodes": [
            {
                "id": "node.decode_input",
                "kind": "transducer",
                "operation": "dense_to_spectral",
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": "signal.input",
                        "physical_representation_id": "repr.signal.dense",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": "signal.native_input",
                        "physical_representation_id": "repr.signal.spectral",
                    }
                ],
                "resource_ids": [],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "measured",
                    "metrics": {"latency_ns": 20.0, "transferred_bytes": 16.0},
                },
                "provenance": provenance(
                    first_scope, source_nodes=("source.input",)
                ),
            },
            {
                "id": "node.decode_output",
                "kind": "transducer",
                "operation": "spectral_to_dense",
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": "signal.native_output",
                        "physical_representation_id": "repr.signal.spectral",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": "signal.output",
                        "physical_representation_id": "repr.signal.dense",
                    }
                ],
                "resource_ids": [],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {"latency_ns": 24.0, "transferred_bytes": 16.0},
                },
                "provenance": provenance(
                    second_scope, source_nodes=("source.output",)
                ),
            },
            {
                "id": "node.scope_a",
                "kind": "operator",
                "operation": "spectral_state_transition",
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": "signal.native_input",
                        "physical_representation_id": "repr.signal.spectral",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": "signal.between",
                        "physical_representation_id": "repr.signal.spectral",
                    }
                ],
                "resource_ids": ["resource.parameter"],
                "state_read_ids": ["resource.state"],
                "state_write_ids": ["resource.state"],
                "cost": None,
                "provenance": provenance(
                    first_scope, source_nodes=("source.mix_a",)
                ),
            },
            {
                "id": "node.scope_b",
                "kind": "operator",
                "operation": "event_graph_projection",
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": "signal.between",
                        "physical_representation_id": "repr.signal.spectral",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": "signal.native_output",
                        "physical_representation_id": "repr.signal.spectral",
                    }
                ],
                "resource_ids": ["resource.topology"],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": None,
                "provenance": provenance(
                    second_scope, source_nodes=("source.mix_b",)
                ),
            },
        ],
        "connections": [
            {
                "id": "connection.cross_scope",
                "producer": {"node_id": "node.scope_a", "port_id": "output"},
                "consumer": {"node_id": "node.scope_b", "port_id": "input"},
                "signal_id": "signal.between",
                "materializes_source": False,
            },
            {
                "id": "connection.decode_output",
                "producer": {"node_id": "node.scope_b", "port_id": "output"},
                "consumer": {"node_id": "node.decode_output", "port_id": "input"},
                "signal_id": "signal.native_output",
                "materializes_source": False,
            },
            {
                "id": "connection.encode_input",
                "producer": {"node_id": "node.decode_input", "port_id": "output"},
                "consumer": {"node_id": "node.scope_a", "port_id": "input"},
                "signal_id": "signal.native_input",
                "materializes_source": False,
            },
        ],
        "public_ports": [
            {
                "id": "port.input",
                "direction": "input",
                "logical_contract_id": "logical.hidden",
                "signal_id": "signal.input",
                "node_id": "node.decode_input",
                "node_port_id": "input",
            },
            {
                "id": "port.output",
                "direction": "output",
                "logical_contract_id": "logical.hidden",
                "signal_id": "signal.output",
                "node_id": "node.decode_output",
                "node_port_id": "output",
            },
        ],
        "islands": [
            {
                "id": "island.native",
                "scope_ids": list(ordered_scopes),
                "node_ids": ["node.scope_a", "node.scope_b"],
                "connection_ids": ["connection.cross_scope"],
                "representation_ids": ["repr.signal.spectral"],
                "boundary_port_ids": ["port.input", "port.output"],
            }
        ],
        "absorbed_transforms": [
            {
                "id": "transform.parameter_basis",
                "kind": "basis_change",
                "source_representation_id": "repr.parameter.dense",
                "target_representation_id": "repr.parameter.sparse",
                "adjacent_node_ids": ["node.scope_a"],
                "parameter_resource_ids": ["resource.parameter"],
                "proof_ref": "proof.exact_parameter_reconstruction",
                "evidence_refs": [evidence_ref],
                "provenance": provenance(
                    first_scope, source_nodes=("source.weight",)
                ),
            }
        ],
        "physical_kernels": [
            {
                "id": "kernel.native_island",
                "node_ids": ["node.scope_a", "node.scope_b"],
                "artifact": {
                    "path": "kernels/native_island.spv",
                    "format": "spirv",
                },
                "target_predicate": {"capability_class": "fixture.gpu"},
                "cost": {
                    "status": "measured",
                    "metrics": {"latency_ns": 80.0},
                },
                "provenance": provenance(
                    first_scope,
                    second_scope,
                    source_nodes=("source.mix_a", "source.mix_b"),
                ),
            }
        ],
        "confidence": {
            "mode": "exact",
            "score": 1.0,
            "basis": "algebraic reconstruction proof",
            "evidence_refs": [evidence_ref],
        },
        "unresolved": [],
        "correction_requests": [],
    }
    return finalize_representation_graph(graph)


def mutable_exact_representation_graph(**kwargs) -> Json:
    return deepcopy(exact_representation_graph(**kwargs))
