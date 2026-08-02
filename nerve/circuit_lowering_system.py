from nerve.circuit_lowering_common import *
from nerve.circuit_lowering_helpers import *


def build_system_circuits(model: Json) -> Json:
    dimensions = model["dimensions"]
    hidden_size = dimensions["hidden_size"]
    vocab_size = dimensions["vocab_size"]
    input_component = model["graph"]["input_transducer"]
    output_components = model["graph"]["output_transducer"].get("components", [])
    if not output_components:
        raise ValueError("output transducer must contain at least one component")

    input_params = {
        name: _system_param_ref(ref, f"input_transducer.{name}")
        for name, ref in input_component.get("params", {}).items()
    }
    input_attrs = dict(input_component.get("attrs", {}))
    input_shape = [
        int(value) for value in input_attrs.get("output_shape", [hidden_size])
    ]
    stream_expansion = input_attrs.get("stream_expansion")
    input_nodes: list[Json] = [
        {
            "id": input_component.get("id", "token_embedding"),
            "op": input_component["type"],
            "inputs": ["input_token"],
            "outputs": ["output_frame"],
            "params": list(input_params),
            "state_reads": [],
            "state_writes": [],
            "attrs": {
                **input_attrs,
                "output_shape": [hidden_size],
                "stream_expansion": None,
            },
        }
    ]
    input_circuit = _system_circuit(
        component_id="input_transducer",
        operator_type="input_transducer",
        runtime_role="input_transducer",
        implementation="compiled_input_transducer_v1",
        inputs=[_system_port("input_token", "token_id", [1], "token")],
        outputs=[
            _system_port(
                "output_frame",
                "frame",
                [hidden_size],
                "frame",
                source="output_frame",
            )
        ],
        parameters=input_params,
        nodes=input_nodes,
    )

    pre_processors = []
    if stream_expansion is not None:
        if stream_expansion.get("type") != "repeat":
            raise ValueError(
                f"unsupported input stream expansion {stream_expansion.get('type')!r}"
            )
        multiplicity = int(stream_expansion["multiplicity"])
        if input_shape != [multiplicity, hidden_size]:
            raise ValueError(
                "input stream expansion shape does not match its repeat contract"
            )
        pre_processors.append(
            _system_circuit(
                component_id="input_stream_adapter",
                operator_type="stream_adapter",
                runtime_role="signal_processor",
                implementation="compiled_repeat_stream_adapter_v1",
                inputs=[_system_port("input_frame", "frame", [hidden_size], "frame")],
                outputs=[
                    _system_port(
                        "output_frame",
                        "frame",
                        input_shape,
                        "frame",
                        source="output_frame",
                    )
                ],
                parameters={},
                nodes=[
                    {
                        "id": "stream_expansion",
                        "op": "repeat_stream_lanes",
                        "inputs": ["input_frame"],
                        "outputs": ["output_frame"],
                        "params": [],
                        "state_reads": [],
                        "state_writes": [],
                        "attrs": {
                            "multiplicity": multiplicity,
                            "hidden_size": hidden_size,
                            "input_shape": [hidden_size],
                            "output_shape": input_shape,
                        },
                    }
                ],
            )
        )

    if len(output_components) < 2 or [
        component["type"] for component in output_components[-2:]
    ] != ["rms_norm", "linear_projection"]:
        raise ValueError(
            "output transducer must end in RMS normalization and linear projection"
        )
    output_adapter_components = output_components[:-2]
    output_components = output_components[-2:]
    post_processors = []
    if output_adapter_components:
        adapter_params: Json = {}
        adapter_nodes: list[Json] = []
        adapter_input_shape = [
            int(value)
            for value in output_adapter_components[0]
            .get("attrs", {})
            .get("input_shape", [hidden_size])
        ]
        signal = "input_frame"
        adapter_output_shape = adapter_input_shape
        for component_index, component in enumerate(output_adapter_components):
            component_id = component.get("id", f"component_{component_index}")
            attrs = dict(component.get("attrs", {}))
            adapter_output_shape = [
                int(value) for value in attrs.get("output_shape", [hidden_size])
            ]
            if component["type"] == "sinkhorn_hyper_connection_head":
                multiplicity = int(adapter_input_shape[0])
                semantic_params = (
                    ("function", "head_function"),
                    ("scale", "head_scale"),
                    ("base", "head_base"),
                )
                param_ids = []
                for source_name, parameter_id in semantic_params:
                    adapter_params[parameter_id] = _system_param_ref(
                        component["params"][source_name],
                        f"output_stream_adapter.{parameter_id}",
                    )
                    param_ids.append(parameter_id)
                attrs.update(
                    {
                        "block_width": 1,
                        "multiplicity": multiplicity,
                        "hidden_size": hidden_size,
                        "output_element_bytes": [2],
                    }
                )
            else:
                param_ids = []
                for name, ref in component.get("params", {}).items():
                    parameter_id = f"{component_id}.{name}"
                    adapter_params[parameter_id] = _system_param_ref(
                        ref, f"output_stream_adapter.{parameter_id}"
                    )
                    param_ids.append(parameter_id)
            output_signal = (
                "output_frame"
                if component_index + 1 == len(output_adapter_components)
                else f"{component_id}_output"
            )
            adapter_nodes.append(
                {
                    "id": component_id,
                    "op": component["type"],
                    "inputs": [signal],
                    "outputs": [output_signal],
                    "params": param_ids,
                    "state_reads": [],
                    "state_writes": [],
                    "attrs": attrs,
                }
            )
            signal = output_signal
        post_processors.append(
            _system_circuit(
                component_id="output_stream_adapter",
                operator_type="stream_adapter",
                runtime_role="signal_processor",
                implementation="compiled_output_stream_adapter_v1",
                inputs=[
                    _system_port("input_frame", "frame", adapter_input_shape, "frame")
                ],
                outputs=[
                    _system_port(
                        "output_frame",
                        "frame",
                        adapter_output_shape,
                        "frame",
                        source="output_frame",
                    )
                ],
                parameters=adapter_params,
                nodes=adapter_nodes,
            )
        )

    output_params: Json = {}
    output_nodes: list[Json] = []
    signal = "input_frame"
    for component_index, component in enumerate(output_components):
        component_id = component.get("id", f"component_{component_index}")
        param_ids = []
        for name, ref in component.get("params", {}).items():
            param_id = f"{component_id}.{name}"
            output_params[param_id] = _system_param_ref(
                ref, f"output_transducer.{param_id}"
            )
            param_ids.append(param_id)
        output_signal = (
            "output_logits"
            if component_index + 1 == len(output_components)
            else f"{component_id}_output"
        )
        output_nodes.append(
            {
                "id": component_id,
                "op": component["type"],
                "inputs": [signal],
                "outputs": [output_signal],
                "params": param_ids,
                "state_reads": [],
                "state_writes": [],
                "attrs": dict(component.get("attrs", {})),
            }
        )
        signal = output_signal
    output_circuit = _system_circuit(
        component_id="output_transducer",
        operator_type="output_transducer",
        runtime_role="output_transducer",
        implementation="compiled_output_transducer_v1",
        inputs=[
            _system_port(
                "input_frame",
                "frame",
                [
                    int(value)
                    for value in output_components[0]
                    .get("attrs", {})
                    .get("input_shape", [hidden_size])
                ],
                "frame",
            )
        ],
        outputs=[
            _system_port(
                "output_logits",
                "logits",
                [vocab_size],
                "logits",
                source="output_logits",
            )
        ],
        parameters=output_params,
        nodes=output_nodes,
    )

    sampling = model["sampling"]
    sampler_method = sampling["method"]
    sampler_presence_penalty = sampling["presence_penalty"]
    sampler_repetition_penalty = sampling["repetition_penalty"]
    if sampler_method == "greedy":
        sampler_temperature = 1.0
        sampler_top_k = 1
        sampler_top_p = 1.0
        sampler_min_p = 0.0
    else:
        sampler_temperature = sampling["temperature"]
        sampler_top_k = sampling.get("top_k")
        sampler_top_p = sampling["top_p"]
        sampler_min_p = sampling["min_p"]
    sampler_circuit = _system_circuit(
        component_id="sampler",
        operator_type="sampler",
        runtime_role="sampler",
        implementation="compiled_sampler_v1",
        inputs=[
            _system_port("input_logits", "logits", [vocab_size], "logits"),
            _system_port("random_seed", "random_seed", [1], "randomness"),
        ],
        outputs=[
            _system_port(
                "sampled_token",
                "token_id",
                [1],
                "token",
                source="sampled_token",
            )
        ],
        parameters={},
        nodes=[
            {
                "id": "sample",
                "op": "sample_token",
                "inputs": ["input_logits", "random_seed"],
                "outputs": ["sampled_token"],
                "params": [],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "method": sampler_method,
                    "temperature": sampler_temperature,
                    "top_k": sampler_top_k,
                    "top_p": sampler_top_p,
                    "min_p": sampler_min_p,
                    "presence_penalty": sampler_presence_penalty,
                    "repetition_penalty": sampler_repetition_penalty,
                    "randomness": "seed_and_stream_tick",
                },
            }
        ],
    )
    return {
        "input_transducer": input_circuit,
        "pre_processors": pre_processors,
        "post_processors": post_processors,
        "output_transducer": output_circuit,
        "sampler": sampler_circuit,
    }


def build_draft_system_circuits(model: Json, draft: Json) -> list[Json]:
    if draft.get("type") == "parallel_backbone_markov":
        return build_parallel_markov_draft_system_circuits(model, draft)
    if draft.get("type") != "multi_token_prediction":
        raise ValueError(f"unsupported draft execution type {draft.get('type')!r}")
    hidden_size = int(model["dimensions"]["hidden_size"])
    vocab_size = int(model["dimensions"]["vocab_size"])
    adapter = draft["input_adapter"]
    adapter_id = f"{draft['id']}_input_adapter"
    adapter_params = {
        name: _system_param_ref(ref, f"{adapter_id}.{name}")
        for name, ref in adapter["params"].items()
    }
    norm_attrs = {
        "eps": float(adapter["attrs"]["eps"]),
        "weight_offset": float(adapter["attrs"]["weight_offset"]),
    }
    input_circuit = _system_circuit(
        component_id=adapter_id,
        operator_type="draft_input_adapter",
        runtime_role="draft_input_adapter",
        implementation="compiled_normalized_embedding_hidden_projection_v1",
        inputs=[
            _system_port("token_embedding", "frame", [hidden_size], "token_embedding"),
            _system_port("target_hidden", "frame", [hidden_size], "target_hidden"),
        ],
        outputs=[
            _system_port(
                "output_frame",
                "frame",
                [hidden_size],
                "output_frame",
                source="output_frame",
            )
        ],
        parameters=adapter_params,
        nodes=[
            {
                "id": "embedding_norm",
                "op": "rms_norm",
                "inputs": ["token_embedding"],
                "outputs": ["normalized_embedding"],
                "params": ["embedding_norm"],
                "state_reads": [],
                "state_writes": [],
                "attrs": norm_attrs,
            },
            {
                "id": "hidden_norm",
                "op": "rms_norm",
                "inputs": ["target_hidden"],
                "outputs": ["normalized_hidden"],
                "params": ["hidden_norm"],
                "state_reads": [],
                "state_writes": [],
                "attrs": norm_attrs,
            },
            {
                "id": "embedding_hidden_concat",
                "op": "concatenate",
                "inputs": ["normalized_embedding", "normalized_hidden"],
                "outputs": ["combined_frame"],
                "params": [],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "axis": "channel",
                    "part_widths": [hidden_size, hidden_size],
                },
            },
            {
                "id": "input_projection",
                "op": "linear",
                "inputs": ["combined_frame"],
                "outputs": ["output_frame"],
                "params": _linear_params("input_projection", adapter_params),
                "state_reads": [],
                "state_writes": [],
                "attrs": {},
            },
        ],
    )

    output = draft["output_transducer"]
    output_id = f"{draft['id']}_output_transducer"
    output_params = {
        name: _system_param_ref(ref, f"{output_id}.{name}")
        for name, ref in output["params"].items()
    }
    output_circuit = _system_circuit(
        component_id=output_id,
        operator_type="draft_output_transducer",
        runtime_role="draft_output_transducer",
        implementation="compiled_draft_output_transducer_v1",
        inputs=[_system_port("input_frame", "frame", [hidden_size], "input_frame")],
        outputs=[
            _system_port(
                "output_hidden",
                "frame",
                [hidden_size],
                "output_hidden",
                source="output_hidden",
            ),
            _system_port(
                "output_logits",
                "logits",
                [vocab_size],
                "output_logits",
                source="output_logits",
            ),
        ],
        parameters=output_params,
        nodes=[
            {
                "id": "output_norm",
                "op": "rms_norm",
                "inputs": ["input_frame"],
                "outputs": ["output_hidden"],
                "params": ["norm"],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "eps": float(output["attrs"]["eps"]),
                    "weight_offset": float(output["attrs"]["weight_offset"]),
                },
            },
            {
                "id": "output_projection",
                "op": "linear_projection",
                "inputs": ["output_hidden"],
                "outputs": ["output_logits"],
                "params": ["projection"],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "scale": float(output["attrs"]["scale"]),
                    "soft_cap": output["attrs"].get("soft_cap"),
                },
            },
        ],
    )
    return [input_circuit, output_circuit]


def build_parallel_markov_draft_system_circuits(model: Json, draft: Json) -> list[Json]:
    hidden_size = int(model["dimensions"]["hidden_size"])
    vocab_size = int(model["dimensions"]["vocab_size"])
    stream_mixer = dict(draft["output_transducer"]["attrs"]["stream_mixer"])
    stream_multiplicity = int(stream_mixer["multiplicity"])
    adapter = draft["input_adapter"]
    adapter_id = f"{draft['id']}_input_adapter"
    adapter_params = {
        name: _system_param_ref(ref, f"{adapter_id}.{name}")
        for name, ref in adapter["params"].items()
    }
    target_inputs = list(adapter["inputs"])
    if not target_inputs:
        raise ValueError("parallel Markov draft must consume target features")
    target_signal_ids = [str(item["id"]) for item in target_inputs]
    lane_reduction = draft["target_features"]["lane_reduction"]
    if lane_reduction != "mean":
        raise ValueError(
            f"unsupported parallel Markov target-feature reduction {lane_reduction!r}"
        )
    reduced_target_signal_ids = [
        f"{signal_id}_reduced" for signal_id in target_signal_ids
    ]
    reduction_nodes = [
        {
            "id": f"{signal_id}_lane_mean",
            "op": "mean_stream_lanes",
            "inputs": [signal_id],
            "outputs": [reduced_signal_id],
            "params": [],
            "state_reads": [],
            "state_writes": [],
            "attrs": {
                "multiplicity": stream_multiplicity,
                "hidden_size": hidden_size,
                "input_shape": [stream_multiplicity, hidden_size],
                "output_shape": [hidden_size],
                "output_element_bytes": [2],
            },
        }
        for signal_id, reduced_signal_id in zip(
            target_signal_ids, reduced_target_signal_ids
        )
    ]
    minimum_block_size = int(draft["proposal_contract"]["minimum_draft_tokens"])
    block_size = int(draft["proposal_contract"]["default_draft_tokens"])
    noise_token_id = int(draft["proposal_contract"]["noise_token_id"])
    norm_attrs = {
        "eps": float(adapter["attrs"]["eps"]),
        "weight_offset": float(adapter["attrs"]["weight_offset"]),
    }
    concatenation_nodes = []
    combined_target_signal = reduced_target_signal_ids[0]
    combined_target_width = hidden_size
    for index, target_signal_id in enumerate(reduced_target_signal_ids[1:], start=1):
        output_signal = (
            "combined_target_features"
            if index == len(target_signal_ids) - 1
            else f"combined_target_features_{index:02d}"
        )
        concatenation_nodes.append(
            {
                "id": f"target_feature_concat_{index:02d}",
                "op": "concatenate",
                "inputs": [combined_target_signal, target_signal_id],
                "outputs": [output_signal],
                "params": [],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "axis": "channel",
                    "part_widths": [combined_target_width, hidden_size],
                },
            }
        )
        combined_target_signal = output_signal
        combined_target_width += hidden_size
    input_circuit = _system_circuit(
        component_id=adapter_id,
        operator_type="draft_input_adapter",
        runtime_role="draft_input_adapter",
        implementation="compiled_parallel_markov_input_adapter_v1",
        inputs=[
            _system_port("anchor_token_id", "token_id", [1], "anchor_token"),
            *[
                _system_port(
                    signal_id,
                    "frame",
                    [stream_multiplicity, hidden_size],
                    signal_id,
                )
                for signal_id in target_signal_ids
            ],
        ],
        outputs=[
            _system_port(
                "query_frames",
                "stream_frame_block",
                [block_size, stream_multiplicity, hidden_size],
                "query_frames",
                source="query_frames",
            ),
            _system_port(
                "main_context",
                "frame",
                [hidden_size],
                "main_context",
                source="main_context",
            ),
            _system_port(
                "anchor_token_passthrough",
                "token_id",
                [1],
                "anchor_token_passthrough",
                source="anchor_token_id",
            ),
        ],
        parameters=adapter_params,
        nodes=[
            *reduction_nodes,
            *concatenation_nodes,
            {
                "id": "target_projection",
                "op": "linear",
                "inputs": [combined_target_signal],
                "outputs": ["projected_target_features"],
                "params": _linear_params("target_projection", adapter_params),
                "state_reads": [],
                "state_writes": [],
                "attrs": {},
            },
            {
                "id": "target_norm",
                "op": "rms_norm",
                "inputs": ["projected_target_features"],
                "outputs": ["main_context"],
                "params": ["target_norm"],
                "state_reads": [],
                "state_writes": [],
                "attrs": norm_attrs,
            },
            {
                "id": "query_embedding_block",
                "op": "anchor_noise_embedding_block",
                "inputs": ["anchor_token_id"],
                "outputs": ["query_frames"],
                "params": ["token_embedding"],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "minimum_block_size": minimum_block_size,
                    "block_size": block_size,
                    "noise_token_id": noise_token_id,
                    "anchor_position": 0,
                    "runtime_extensible": False,
                    "runtime_selectable_prefix": True,
                    "hidden_size": hidden_size,
                    "stream_multiplicity": stream_multiplicity,
                    "output_layout": "block_stream_hidden",
                },
            },
        ],
    )

    output = draft["output_transducer"]
    output_id = f"{draft['id']}_output_transducer"
    output_params = {
        name: _system_param_ref(ref, f"{output_id}.{name}")
        for name, ref in output["params"].items()
    }
    markov_rank = int(output["attrs"]["markov_rank"])
    vocabulary_tile_width = 256
    candidate_count = (vocab_size + vocabulary_tile_width - 1) // vocabulary_tile_width
    markov_nodes = []
    draft_token_signals = []
    markov_embedding_signals = []
    previous_token_signal = "anchor_token_id"
    for position in range(block_size):
        candidate_signal = f"markov_candidates_{position:02d}"
        embedding_signal = f"markov_embedding_{position:02d}"
        token_signal = f"draft_token_{position:02d}"
        markov_nodes.extend(
            [
                {
                    "id": f"markov_argmax_partials_{position:02d}",
                    "op": "markov_argmax_partials",
                    "inputs": ["base_logits", previous_token_signal],
                    "outputs": [candidate_signal, embedding_signal],
                    "params": ["markov_embedding", "markov_projection"],
                    "state_reads": [],
                    "state_writes": [],
                    "attrs": {
                        "rank": markov_rank,
                        "sampling": "greedy",
                        "dependency": "previous_sampled_token",
                        "position": position,
                        "block_width": block_size,
                        "vocabulary_size": vocab_size,
                        "vocabulary_tile_width": vocabulary_tile_width,
                        "output_element_bytes": [4, 2],
                    },
                },
                {
                    "id": f"markov_argmax_reduce_{position:02d}",
                    "op": "argmax_candidate_reduce",
                    "inputs": [candidate_signal],
                    "outputs": [token_signal],
                    "params": [],
                    "state_reads": [],
                    "state_writes": [],
                    "attrs": {
                        "candidate_count": candidate_count,
                        "tie_break": "lowest_token_id",
                        "output_element_bytes": [4],
                    },
                },
            ]
        )
        draft_token_signals.append(token_signal)
        markov_embedding_signals.append(embedding_signal)
        previous_token_signal = token_signal
    output_circuit = _system_circuit(
        component_id=output_id,
        operator_type="draft_output_transducer",
        runtime_role="draft_output_transducer",
        implementation="compiled_parallel_markov_output_transducer_v1",
        inputs=[
            _system_port(
                "input_frames",
                "stream_frame_block",
                [block_size, int(stream_mixer["multiplicity"]), hidden_size],
                "input_frames",
            ),
            _system_port("anchor_token_id", "token_id", [1], "anchor_token"),
        ],
        outputs=[
            _system_port(
                "draft_token_ids",
                "token_id_block",
                [block_size],
                "draft_token_ids",
                source="draft_token_ids",
            ),
            _system_port(
                "confidence_logits",
                "scalar_block",
                [block_size],
                "confidence_logits",
                source="confidence_logits",
            ),
        ],
        parameters=output_params,
        nodes=[
            {
                "id": "stream_head",
                "op": "sinkhorn_hyper_connection_head",
                "inputs": ["input_frames"],
                "outputs": ["head_hidden"],
                "params": ["head_function", "head_scale", "head_base"],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "multiplicity": int(stream_mixer["multiplicity"]),
                    "epsilon": float(stream_mixer["epsilon"]),
                    "normalization": "root_mean_square",
                    "activation": "sigmoid",
                    "block_width": block_size,
                    "hidden_size": hidden_size,
                    "output_element_bytes": [2],
                },
            },
            {
                "id": "output_norm",
                "op": "rms_norm",
                "inputs": ["head_hidden"],
                "outputs": ["normalized_hidden"],
                "params": ["norm"],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "eps": float(output["attrs"]["eps"]),
                    "weight_offset": float(output["attrs"]["weight_offset"]),
                    "block_width": block_size,
                    "hidden_size": hidden_size,
                    "output_element_bytes": [2],
                },
            },
            {
                "id": "base_projection",
                "op": "linear_projection",
                "inputs": ["normalized_hidden"],
                "outputs": ["base_logits"],
                "params": _linear_params("projection", output_params),
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "block_width": block_size,
                    "input_size": hidden_size,
                    "output_size": vocab_size,
                    "output_element_bytes": [4],
                },
            },
            *markov_nodes,
            {
                "id": "confidence_projection",
                "op": "confidence_projection_block",
                "inputs": ["head_hidden", *markov_embedding_signals],
                "outputs": ["confidence_logits"],
                "params": ["confidence_projection"],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "output_activation": None,
                    "block_width": block_size,
                    "input_size": hidden_size + markov_rank,
                    "output_element_bytes": [4],
                },
            },
            {
                "id": "draft_token_pack",
                "op": "pack_token_block",
                "inputs": draft_token_signals,
                "outputs": ["draft_token_ids"],
                "params": [],
                "state_reads": [],
                "state_writes": [],
                "attrs": {
                    "block_width": block_size,
                    "output_element_bytes": [4],
                },
            },
        ],
    )
    return [input_circuit, output_circuit]


def _system_port(
    port_id: str,
    signal: str,
    shape: list[int],
    component_port: str,
    *,
    source: str | None = None,
) -> Json:
    port = {
        "id": port_id,
        "signal": signal,
        "shape": shape,
        "component_port": component_port,
    }
    if source is not None:
        port["source"] = source
    return port


def _system_param_ref(reference: Json, role: str) -> Json:
    return {"tensor": reference["tensor"], "role": role}


def _system_circuit(
    *,
    component_id: str,
    operator_type: str,
    runtime_role: str,
    implementation: str,
    inputs: list[Json],
    outputs: list[Json],
    parameters: Json,
    nodes: list[Json],
) -> Json:
    return {
        "schema": "nerve.stream_circuit.v1",
        "id": f"{component_id}_circuit_v1",
        "source": {
            "component_id": component_id,
            "source_layer_index": None,
            "source_operator_type": operator_type,
        },
        "runtime_role": runtime_role,
        "behavioral_role": "stream_generation_circuit",
        "implementation": implementation,
        "boundary": {"inputs": inputs, "outputs": outputs, "controls": []},
        "state_ports": [],
        "parameters": {
            "layout": "source_tensor_refs",
            "storage": "safetensors",
            "refs": parameters,
        },
        "nodes": nodes,
        "behavioral_error_contract": {
            "mode": "exact_source_operation",
            "reference": operator_type,
        },
        "lowering_notes": [
            "This stream entity is part of the editable execution graph contract.",
            "Its optimized Vulkan implementation is a backend lowering, not a host-side exception.",
        ],
    }


__all__ = [name for name in globals() if not name.startswith("__")]
