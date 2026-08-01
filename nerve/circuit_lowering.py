from __future__ import annotations

from nerve.circuit_lowering_common import *
from nerve.circuit_lowering_helpers import *
from nerve.circuit_lowering_nodes import *
from nerve.circuit_lowering_operators import *
from nerve.circuit_lowering_system import *


def lower_component(component_path: Path, out_dir: Path) -> Json:
    component = read_json(component_path)
    circuit = build_component_circuit(component, component_path)
    validation = validate_circuit_against_component(circuit, component)
    validation.raise_for_errors()

    out_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = out_dir / "circuit.json"
    params_path = out_dir / "params.json"
    state_path = out_dir / "state.json"
    write_json(circuit_path, circuit)
    write_json(params_path, build_params_artifact(circuit))
    write_json(state_path, build_state_artifact(circuit))

    return {
        "component": component,
        "circuit": circuit,
        "validation": validation.to_json(),
        "circuit_path": circuit_path,
        "params_path": params_path,
        "state_path": state_path,
    }


def build_component_circuit(component: Json, component_path: Path) -> Json:
    operator_type = component.get("operator_type")
    if operator_type == "conv":
        return build_conv_circuit(component, component_path)
    if operator_type == "full_attention":
        return build_attention_circuit(component, component_path)
    if operator_type == "latent_sparse_attention":
        return build_latent_sparse_attention_circuit(component, component_path)
    if operator_type == "gated_delta":
        return build_gated_delta_circuit(component, component_path)
    if operator_type == "rg_lru":
        return build_rg_lru_circuit(component, component_path)
    raise ValueError(f"unsupported component operator type {operator_type!r}")


def lower_execution_graph(
    execution_graph_dir: Path,
    out_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    model = read_json(execution_graph_dir / "model.json")
    source_components = model["graph"]["execution_graph"]["components"]
    source_drafts = model["graph"].get("draft_execution_graphs", [])
    draft_source_components = [
        component
        for draft in source_drafts
        for component in draft["execution_graph"]["components"]
    ]

    lowered: list[Json] = []
    operator_counts: Counter[str] = Counter()
    total = len(source_components) + len(draft_source_components)
    for current, source_component in enumerate(source_components, start=1):
        check_compile_cancelled(cancel_requested)
        if progress is not None:
            progress(current, total, source_component["id"])
        component_path = execution_graph_dir / source_component["file"]
        component_out_dir = out_dir / source_component["id"]
        result = lower_component(component_path, component_out_dir)
        circuit_rel = result["circuit_path"].relative_to(out_dir)
        params_rel = result["params_path"].relative_to(out_dir)
        state_rel = result["state_path"].relative_to(out_dir)
        operator_counts[source_component["operator_type"]] += 1
        lowered.append(
            {
                "id": source_component["id"],
                "operator_type": source_component["operator_type"],
                "runtime_role": result["circuit"]["runtime_role"],
                "circuit": str(circuit_rel),
                "params": str(params_rel),
                "state": str(state_rel),
                "implementation": result["circuit"]["implementation"],
                "behavioral_role": result["circuit"]["behavioral_role"],
            }
        )

    draft_execution_graphs: list[Json] = []
    lowered_count = len(source_components)
    for draft in source_drafts:
        draft_refs: list[Json] = []
        for source_component in draft["execution_graph"]["components"]:
            check_compile_cancelled(cancel_requested)
            lowered_count += 1
            if progress is not None:
                progress(lowered_count, total, source_component["id"])
            component_path = execution_graph_dir / source_component["file"]
            component_out_dir = (
                out_dir / "drafts" / draft["id"] / source_component["id"]
            )
            result = lower_component(component_path, component_out_dir)
            operator_counts[source_component["operator_type"]] += 1
            draft_refs.append(
                {
                    "id": source_component["id"],
                    "operator_type": source_component["operator_type"],
                    "runtime_role": result["circuit"]["runtime_role"],
                    "circuit": str(result["circuit_path"].relative_to(out_dir)),
                    "params": str(result["params_path"].relative_to(out_dir)),
                    "state": str(result["state_path"].relative_to(out_dir)),
                    "implementation": result["circuit"]["implementation"],
                    "behavioral_role": result["circuit"]["behavioral_role"],
                }
            )
        lowered_draft = lower_draft_execution_graph(model, draft, draft_refs, out_dir)
        operator_counts["draft_input_adapter"] += 1
        operator_counts["draft_output_transducer"] += 1
        draft_execution_graphs.append(lowered_draft)

    if not lowered:
        raise ValueError("cannot lower an empty execution graph")

    system_circuits = build_system_circuits(model)
    system_circuit_sequence = [
        system_circuits["input_transducer"],
        *system_circuits["pre_processors"],
        *system_circuits["post_processors"],
        system_circuits["output_transducer"],
        system_circuits["sampler"],
    ]
    system_refs: dict[str, Json] = {}
    for circuit in system_circuit_sequence:
        circuit_id = circuit["source"]["component_id"]
        circuit_out_dir = out_dir / circuit_id
        circuit_out_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_circuit(circuit)
        validation.raise_for_errors()
        circuit_path = circuit_out_dir / "circuit.json"
        params_path = circuit_out_dir / "params.json"
        state_path = circuit_out_dir / "state.json"
        write_json(circuit_path, circuit)
        write_json(params_path, build_params_artifact(circuit))
        write_json(state_path, build_state_artifact(circuit))
        operator_counts[circuit["source"]["source_operator_type"]] += 1
        system_refs[circuit_id] = {
            "id": circuit_id,
            "operator_type": circuit["source"]["source_operator_type"],
            "runtime_role": circuit["runtime_role"],
            "circuit": str(circuit_path.relative_to(out_dir)),
            "params": str(params_path.relative_to(out_dir)),
            "state": str(state_path.relative_to(out_dir)),
            "implementation": circuit["implementation"],
            "behavioral_role": circuit["behavioral_role"],
        }

    input_ref = system_refs["input_transducer"]
    pre_processor_refs = [
        system_refs[circuit["source"]["component_id"]]
        for circuit in system_circuits["pre_processors"]
    ]
    post_processor_refs = [
        system_refs[circuit["source"]["component_id"]]
        for circuit in system_circuits["post_processors"]
    ]
    output_ref = system_refs["output_transducer"]
    sampler_ref = system_refs["sampler"]
    all_circuits = [
        input_ref,
        *pre_processor_refs,
        *lowered,
        *post_processor_refs,
        output_ref,
        sampler_ref,
    ]
    forward_chain = all_circuits

    index = {
        "schema": "nerve.lowered_execution_graph.v1",
        "source": {
            "format": "nerve.compiled_execution_graph_artifact.v1",
            "artifact_root": ".",
        },
        "architecture": model["architecture"],
        "dimensions": model["dimensions"],
        "numerics": model["numerics"],
        "token_ids": model["token_ids"],
        "graph": {
            "topology": "explicit_graph",
            "circuits": all_circuits,
            "edges": [
                {
                    "id": f"edge_{index:04d}",
                    "connection": {"kind": "forward"},
                    "source": {
                        "component_id": source["id"],
                        "port_id": _canonical_output_port(source["runtime_role"]),
                    },
                    "destination": {
                        "component_id": destination["id"],
                        "port_id": _canonical_input_port(destination["runtime_role"]),
                    },
                }
                for index, (source, destination) in enumerate(
                    zip(forward_chain, forward_chain[1:])
                )
            ]
            + [
                {
                    "id": "generation_feedback",
                    "connection": {
                        "kind": "temporal_feedback",
                        "delay_activations": 1,
                    },
                    "source": {
                        "component_id": sampler_ref["id"],
                        "port_id": "sampled_token",
                    },
                    "destination": {
                        "component_id": input_ref["id"],
                        "port_id": "input_token",
                    },
                }
            ],
            "boundary": {
                "external_inputs": [
                    {
                        "id": "user_input",
                        "endpoint": {
                            "component_id": input_ref["id"],
                            "port_id": "input_token",
                        },
                    },
                    {
                        "id": "random_seed",
                        "endpoint": {
                            "component_id": sampler_ref["id"],
                            "port_id": "random_seed",
                        },
                    },
                ],
                "public_outputs": [
                    {
                        "id": "model_output",
                        "endpoint": {
                            "component_id": sampler_ref["id"],
                            "port_id": "sampled_token",
                        },
                    }
                ],
            },
            "input_transducer": model["graph"]["input_transducer"],
            "output_transducer": model["graph"]["output_transducer"],
        },
        "draft_execution_graphs": draft_execution_graphs,
        "summary": {
            "circuit_count": len(all_circuits)
            + sum(len(draft["circuits"]) for draft in draft_execution_graphs),
            "generation_circuit_count": len(all_circuits),
            "draft_execution_graph_count": len(draft_execution_graphs),
            "operator_counts": dict(sorted(operator_counts.items())),
        },
        "notes": [
            "This index maps the source execution graph to stream-circuit artifacts.",
            "The artifacts preserve component boundaries for now; a backend may later fuse or replace connected regions.",
            "No layer receives privileged treatment; every component is addressed through the same boundary contract.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "execution_graph.circuits.json"
    write_json(index_path, index)
    return {
        "index": index,
        "index_path": index_path,
        "circuits": lowered,
        "draft_execution_graphs": draft_execution_graphs,
    }


def _canonical_input_port(runtime_role: str) -> str:
    return {
        "input_transducer": "input_token",
        "signal_processor": "input_frame",
        "output_transducer": "input_frame",
        "sampler": "input_logits",
    }[runtime_role]


def _canonical_output_port(runtime_role: str) -> str:
    return {
        "input_transducer": "output_frame",
        "signal_processor": "output_frame",
        "output_transducer": "output_logits",
        "sampler": "sampled_token",
    }[runtime_role]


def lower_draft_execution_graph(
    model: Json,
    draft: Json,
    layer_refs: list[Json],
    out_dir: Path,
) -> Json:
    if not layer_refs:
        raise ValueError(
            f"draft execution graph {draft['id']!r} contains no layer components"
        )
    system_circuits = build_draft_system_circuits(model, draft)
    system_refs = []
    for circuit in system_circuits:
        circuit_id = circuit["source"]["component_id"]
        circuit_out_dir = out_dir / "drafts" / draft["id"] / circuit_id
        circuit_out_dir.mkdir(parents=True, exist_ok=True)
        validate_circuit(circuit).raise_for_errors()
        circuit_path = circuit_out_dir / "circuit.json"
        params_path = circuit_out_dir / "params.json"
        state_path = circuit_out_dir / "state.json"
        write_json(circuit_path, circuit)
        write_json(params_path, build_params_artifact(circuit))
        write_json(state_path, build_state_artifact(circuit))
        system_refs.append(
            {
                "id": circuit_id,
                "operator_type": circuit["source"]["source_operator_type"],
                "runtime_role": circuit["runtime_role"],
                "circuit": str(circuit_path.relative_to(out_dir)),
                "params": str(params_path.relative_to(out_dir)),
                "state": str(state_path.relative_to(out_dir)),
                "implementation": circuit["implementation"],
                "behavioral_role": circuit["behavioral_role"],
            }
        )

    input_ref, output_ref = system_refs
    if draft["type"] == "parallel_backbone_markov":
        return lower_parallel_markov_draft_graph(
            draft,
            layer_refs=layer_refs,
            input_ref=input_ref,
            output_ref=output_ref,
        )
    forward_chain = [input_ref, *layer_refs, output_ref]
    return {
        "id": draft["id"],
        "type": draft["type"],
        "source_prefix": draft["source_prefix"],
        "topology": "explicit_graph",
        "circuits": forward_chain,
        "edges": [
            {
                "id": f"{draft['id']}_edge_{index:04d}",
                "connection": {"kind": "forward"},
                "source": {
                    "component_id": source["id"],
                    "port_id": (
                        "output_frame"
                        if source["runtime_role"] != "draft_output_transducer"
                        else "output_hidden"
                    ),
                },
                "destination": {
                    "component_id": destination["id"],
                    "port_id": "input_frame",
                },
            }
            for index, (source, destination) in enumerate(
                zip(forward_chain, forward_chain[1:])
            )
        ],
        "boundary": {
            "external_inputs": [
                {
                    "id": "token_embedding",
                    "endpoint": {
                        "component_id": input_ref["id"],
                        "port_id": "token_embedding",
                    },
                },
                {
                    "id": "target_hidden",
                    "endpoint": {
                        "component_id": input_ref["id"],
                        "port_id": "target_hidden",
                    },
                },
            ],
            "public_outputs": [
                {
                    "id": "draft_hidden",
                    "endpoint": {
                        "component_id": output_ref["id"],
                        "port_id": "output_hidden",
                    },
                },
                {
                    "id": "draft_logits",
                    "endpoint": {
                        "component_id": output_ref["id"],
                        "port_id": "output_logits",
                    },
                },
            ],
        },
        "execution_contract": {
            "mode": "autoregressive_feedback",
            "processor_schedule": "one_token_per_tick",
            "output_schedule": "dedicated_token_transducer",
        },
        "state_contract": dict(draft["state_contract"]),
    }


def lower_parallel_markov_draft_graph(
    draft: Json,
    *,
    layer_refs: list[Json],
    input_ref: Json,
    output_ref: Json,
) -> Json:
    block_width = int(draft["proposal_contract"]["default_draft_tokens"])
    query_chain = [*layer_refs, output_ref]
    query_edges = []
    for index, destination in enumerate(query_chain):
        source = input_ref if index == 0 else layer_refs[index - 1]
        connection = (
            {"kind": "parallel_block_scatter", "width": block_width}
            if source is input_ref
            else {"kind": "parallel_block_gather", "width": block_width}
            if destination is output_ref
            else {"kind": "forward"}
        )
        query_edges.append(
            {
                "id": f"{draft['id']}_query_edge_{index:04d}",
                "connection": connection,
                "source": {
                    "component_id": source["id"],
                    "port_id": "query_frames" if index == 0 else "output_frame",
                },
                "destination": {
                    "component_id": destination["id"],
                    "port_id": (
                        "input_frames"
                        if destination["runtime_role"] == "draft_output_transducer"
                        else "input_frame"
                    ),
                },
            }
        )
    context_edges = [
        {
            "id": f"{draft['id']}_context_edge_{index:04d}",
            "connection": {
                "kind": "shared_context",
                "state_update": "committed_target_only",
            },
            "source": {
                "component_id": input_ref["id"],
                "port_id": "main_context",
            },
            "destination": {
                "component_id": layer_ref["id"],
                "port_id": "main_context",
            },
        }
        for index, layer_ref in enumerate(layer_refs)
    ]
    anchor_edge = {
        "id": f"{draft['id']}_anchor_passthrough",
        "connection": {"kind": "forward"},
        "source": {
            "component_id": input_ref["id"],
            "port_id": "anchor_token_passthrough",
        },
        "destination": {
            "component_id": output_ref["id"],
            "port_id": "anchor_token_id",
        },
    }
    target_inputs = draft["input_adapter"]["inputs"]
    return {
        "id": draft["id"],
        "type": draft["type"],
        "source_prefix": draft["source_prefix"],
        "topology": "explicit_graph",
        "circuits": [input_ref, *layer_refs, output_ref],
        "edges": [*query_edges, *context_edges, anchor_edge],
        "boundary": {
            "external_inputs": [
                {
                    "id": "anchor_token_id",
                    "endpoint": {
                        "component_id": input_ref["id"],
                        "port_id": "anchor_token_id",
                    },
                },
                *[
                    {
                        "id": target_input["id"],
                        "source_layer_index": target_input["source_layer_index"],
                        "lane_reduction": draft["target_features"]["lane_reduction"],
                        "endpoint": {
                            "component_id": input_ref["id"],
                            "port_id": target_input["id"],
                        },
                    }
                    for target_input in target_inputs
                ],
            ],
            "public_outputs": [
                {
                    "id": output_id,
                    "endpoint": {
                        "component_id": output_ref["id"],
                        "port_id": output_id,
                    },
                }
                for output_id in (
                    "draft_token_ids",
                    "confidence_logits",
                )
            ],
        },
        "execution_contract": {
            "mode": "parallel_block",
            "block_width": block_width,
            "processor_schedule": "parallel_lanes",
            "output_schedule": "compiled_component_graph",
        },
        "proposal_contract": dict(draft["proposal_contract"]),
        "state_contract": dict(draft["state_contract"]),
    }
