from nerve.model_transpiler_types import *
from nerve.model_transpiler_tensor_index import *
from nerve.model_transpiler_quantization import *


def make_layer(
    structure: ModelStructure,
    layer: LayerStructure,
    *,
    component_id: str | None = None,
    runtime_role: str = "signal_processor",
    execution_contract: Json | None = None,
) -> Json:
    hidden_size = structure.hidden_size
    component_id = component_id or f"layer_{layer.index:02d}"
    tensor_refs = list(layer.tensors.values())
    operator = (
        make_conv_operator(structure, layer)
        if layer.operator_type == "conv"
        else make_attention_operator(structure, layer)
        if layer.operator_type == "full_attention"
        else make_latent_sparse_attention_operator(structure, layer)
        if layer.operator_type == "latent_sparse_attention"
        else make_gated_delta_operator(structure, layer)
        if layer.operator_type == "gated_delta"
        else make_rg_lru_operator(structure, layer)
    )

    return {
        "schema": "nerve.node_instance.v1",
        "id": component_id,
        "source_layer_index": layer.index,
        "type": "node_instance",
        "runtime_role": runtime_role,
        "component_class": make_component_class(structure, layer),
        "operator_type": layer.operator_type,
        "operator": deepcopy(operator),
        "feed_forward": make_feed_forward_descriptor(structure, layer),
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": layer.rope_theta,
            "rope_type": layer.rope_type,
            "rope_scaling": deepcopy(layer.rope_scaling),
            "rope_interleaved": structure.rope_interleaved,
            "rotary_width": layer.rotary_width,
            "rms_norm_weight_offset": structure.rms_norm_weight_offset,
            "attention_output_gate": structure.attention_output_gate,
            "attention_gate_activation": layer.attention_gate_activation,
            "attention_gate_per_head": layer.attention_gate_per_head,
            "attention_key_equals_value": layer.attention_key_equals_value,
            "residual_scale": structure.residual_scale,
            "attention_scale": layer.attention_scale,
            "attention_window_size": layer.attention_window_size,
            "max_position_embeddings": structure.max_position_embeddings,
            "value_head_norm": layer.value_head_norm,
            "per_layer_input_width": layer.per_layer_input_width,
            "per_layer_input_layer_index": layer.index,
            "per_layer_input_layer_count": structure.num_hidden_layers,
            "per_layer_embedding_chunk_count": len(
                [
                    name
                    for name in layer.tensors
                    if name.startswith("per_layer_embedding_chunk_")
                ]
            )
            or None,
            "per_layer_embedding_chunk_rows": (
                MAX_SHADER_PARAMETER_CHUNK_BYTES
                // (structure.num_hidden_layers * layer.per_layer_input_width * 2)
                if layer.per_layer_input_width is not None
                else None
            ),
            "token_embedding_scale": structure.embedding_scale,
            "per_layer_embedding_scale": (
                round_float_to_bf16(math.sqrt(layer.per_layer_input_width))
                if layer.per_layer_input_width is not None
                else None
            ),
            "per_layer_model_projection_scale": hidden_size**-0.5,
            "per_layer_input_scale": 2.0**-0.5,
            "activation_quantization": deepcopy(
                (structure.quantization or {}).get("activation")
            ),
        },
        "ports": {
            "inputs": [
                {"id": "input", "signal": "frame", "shape": list(layer.boundary_shape)},
                *(
                    [
                        {
                            "id": "main_context",
                            "signal": "frame",
                            "shape": [hidden_size],
                        }
                    ]
                    if execution_contract is not None
                    and execution_contract.get("type")
                    == "parallel_query_with_external_kv_context"
                    else []
                ),
            ],
            "outputs": [
                {"id": "output", "signal": "frame", "shape": list(layer.boundary_shape)}
            ],
            "controls": (
                [
                    {
                        "id": "token_id",
                        "signal": "token_id",
                        "shape": [],
                        "dtype": "U32",
                        "runtime_source": "input_token_id",
                    }
                ]
                if layer.feed_forward_attributes.get("routing", {}).get("selection")
                == "token_id_table"
                else []
            ),
        },
        "residual_mixer": deepcopy(layer.residual_mixer),
        "execution_contract": deepcopy(execution_contract),
        "state_ports": make_state_ports(structure, layer),
        "parameter_block": make_parameter_block(
            layer.operator_type, layer.feed_forward_type, layer.tensors
        ),
        "transition_contract": {
            "type": "stateful_frame_transform",
            "equation": "(output_frame, next_state, events) = component(input_frame, state, params, control)",
            "reference_behavior": f"source_checkpoint_entity:{layer.prefix}",
            "behavioral_error_contract": "not_defined_yet",
        },
        "runtime_boundary": {
            "opaque_to_execution_graph": True,
            "compiler_may_fuse_internal_operations": True,
            "compiler_may_replace_reference_decomposition": True,
        },
        "reference_decomposition": make_reference_decomposition(
            structure, layer, operator
        ),
        "tensor_refs": tensor_refs,
    }


def make_model_graph(
    structure: ModelStructure, output_dir: Path, tensor_index: Json
) -> Json:
    components = [
        {
            "id": f"layer_{layer.index:02d}",
            "type": "node_instance",
            "component_class": make_component_class(structure, layer),
            "operator_type": layer.operator_type,
            "file": f"layers/layer_{layer.index:02d}.json",
        }
        for layer in structure.layers
    ]

    output_projection = {
        "id": "output_projection",
        "type": "linear_projection",
        "attrs": {
            "scale": 1.0 / structure.logits_scale,
            "soft_cap": structure.logits_soft_cap,
        },
        "params": {"weight": tensor_ref(structure.tensors["output_projection"])},
    }
    if structure.tensors["output_projection"] == structure.tensors["token_embedding"]:
        output_projection["sharing"] = "same_parameter_object_as_token_embedding"

    stream_collapse = (
        [
            {
                "id": "stream_collapse",
                "type": "sinkhorn_hyper_connection_head",
                "attrs": {
                    "input_shape": list(structure.stream_shape),
                    "output_shape": [structure.hidden_size],
                    "normalization": "root_mean_square",
                    "activation": "sigmoid",
                    "epsilon": float(structure.stream_mixer["epsilon"]),
                },
                "params": {
                    role: tensor_ref(tensor)
                    for role, tensor in structure.stream_mixer["head"].items()
                },
            }
        ]
        if structure.stream_mixer is not None
        else []
    )

    return {
        "schema": "nerve.model_graph.v1",
        "source": tensor_index["source"],
        "architecture": {
            "family": "decoder_only_transformer",
            "model_type": structure.model_type,
            "architectures": list(structure.architectures),
            "dtype": structure.dtype,
        },
        "dimensions": {
            "hidden_size": structure.hidden_size,
            "intermediate_sizes": [
                layer.intermediate_size for layer in structure.layers
            ],
            "num_hidden_layers": structure.num_hidden_layers,
            "num_attention_heads": structure.num_attention_heads,
            "num_key_value_heads": structure.num_key_value_heads,
            "head_width": structure.head_width,
            "rotary_width": structure.rotary_width,
            "attention_window_size": structure.attention_window_size,
            "conv_l_cache": structure.conv_l_cache,
            "vocab_size": structure.vocab_size,
            "max_position_embeddings": structure.max_position_embeddings,
            "num_experts": structure.num_experts,
            "experts_per_token": structure.experts_per_token,
            "attention_layer_shapes": [
                {
                    "layer": layer.index,
                    "query_heads": layer.num_attention_heads,
                    "key_value_heads": layer.num_key_value_heads,
                    "head_width": layer.head_width,
                    "rotary_width": layer.rotary_width,
                    "rope_theta": layer.rope_theta,
                    "rope_type": layer.rope_type,
                    "shared_kv_source_layer": layer.shared_kv_source_layer,
                }
                for layer in structure.layers
                if layer.operator_type == "full_attention"
            ],
        },
        "numerics": {
            "rms_norm_eps": structure.norm_eps,
            "rope_theta": structure.rope_theta,
            "rope_interleaved": structure.rope_interleaved,
            "rms_norm_weight_offset": structure.rms_norm_weight_offset,
            "embedding_scale": structure.embedding_scale,
            "residual_scale": structure.residual_scale,
            "attention_scale": structure.attention_scale,
            "logits_scale": structure.logits_scale,
            "logits_soft_cap": structure.logits_soft_cap,
        },
        "quantization": structure.quantization,
        "sampling": structure.sampling,
        "token_ids": structure.token_ids,
        "files": {
            "tensor_index": "tensors.json",
            "components_dir": "layers/",
        },
        "graph": {
            "input_transducer": {
                "id": "token_embedding",
                "type": "embedding_lookup",
                "output": "stream_frame",
                "attrs": {
                    "scale": structure.embedding_scale,
                    "output_shape": list(structure.stream_shape),
                    "stream_expansion": (
                        {
                            "type": "repeat",
                            "multiplicity": int(structure.stream_mixer["multiplicity"]),
                        }
                        if structure.stream_mixer is not None
                        else None
                    ),
                },
                "params": {"weight": tensor_ref(structure.tensors["token_embedding"])},
            },
            "execution_graph": {
                "topology": "series",
                "components": components,
            },
            "output_transducer": {
                "components": [
                    *stream_collapse,
                    {
                        "id": "output_norm",
                        "type": "rms_norm",
                        "attrs": {
                            "eps": structure.norm_eps,
                            "weight_offset": structure.rms_norm_weight_offset,
                        },
                        "params": {
                            "weight": tensor_ref(structure.tensors["output_norm"])
                        },
                    },
                    output_projection,
                ]
            },
            "draft_execution_graphs": [
                make_draft_execution_graph_descriptor(structure, draft)
                for draft in structure.draft_execution_graphs
            ],
        },
        "component_templates": {
            "shortconv_layer": "opaque layer component with fixed rolling temporal state",
            "rg_lru_layer": "opaque recurrent layer component with fixed convolution and recurrent state",
            "gqa_attention_layer": "opaque layer component with append-only KV state",
            "swiglu_feed_forward": "dense gated feed-forward operator",
            "rms_norm": "stateless normalization operator",
            "residual_add": "stateless signal mixer",
        },
        "output_dir": ".",
    }


def make_draft_execution_graph_descriptor(
    structure: ModelStructure,
    draft: DraftExecutionGraphStructure,
) -> Json:
    if draft.draft_type == "parallel_backbone_markov":
        return make_parallel_markov_draft_descriptor(structure, draft)
    if draft.draft_type != "multi_token_prediction":
        raise ModelTranspileError(
            f"unsupported draft execution type {draft.draft_type!r}"
        )
    hidden_size = structure.hidden_size
    adapter_params = {
        name: tensor_ref(tensor)
        for name, tensor in draft.tensors.items()
        if name not in {"output_norm", "output_projection"}
    }
    return make_standard_mtp_draft_descriptor(structure, draft, adapter_params)


def make_parallel_markov_draft_descriptor(
    structure: ModelStructure,
    draft: DraftExecutionGraphStructure,
) -> Json:
    hidden_size = structure.hidden_size
    target_features = deepcopy(draft.attributes["target_features"])
    target_inputs = [
        {
            "id": f"target_hidden_{index}",
            "signal": "frame",
            "shape": [hidden_size],
            "source_component_id": f"layer_{layer_index:02d}",
        }
        for index, layer_index in enumerate(target_features["layer_indices"])
    ]
    adapter_param_ids = {
        "target_projection",
        "target_projection_scale",
        "target_projection_scale_inv",
        "target_norm",
    }
    output_param_ids = set(draft.tensors) - adapter_param_ids
    return {
        "id": draft.id,
        "type": draft.draft_type,
        "source_prefix": draft.prefix,
        "target_features": target_features,
        "proposal_contract": deepcopy(draft.attributes["proposal_contract"]),
        "input_adapter": {
            "type": "target_feature_projection",
            "inputs": target_inputs,
            "output": {
                "id": "main_context",
                "signal": "frame",
                "shape": [hidden_size],
            },
            "attrs": {
                "eps": structure.norm_eps,
                "weight_offset": structure.rms_norm_weight_offset,
                "concatenation_order": [item["id"] for item in target_inputs],
            },
            "params": {
                "token_embedding": tensor_ref(structure.tensors["token_embedding"]),
                **{
                    name: tensor_ref(tensor)
                    for name, tensor in draft.tensors.items()
                    if name in adapter_param_ids
                },
            },
        },
        "query_block": {
            "type": "anchor_then_noise_embeddings",
            "token_embedding": tensor_ref(structure.tensors["token_embedding"]),
            "block_size": draft.attributes["proposal_contract"]["default_draft_tokens"],
            "noise_token_id": draft.attributes["proposal_contract"]["noise_token_id"],
        },
        "execution_graph": {
            "topology": "series_with_shared_context",
            "shared_input": "main_context",
            "components": [
                {
                    "id": f"{draft.id}_layer_{layer.index:02d}",
                    "type": "node_instance",
                    "component_class": make_component_class(structure, layer),
                    "operator_type": layer.operator_type,
                    "file": (
                        f"drafts/{draft.id}/layers/"
                        f"{draft.id}_layer_{layer.index:02d}.json"
                    ),
                }
                for layer in draft.layers
            ],
        },
        "output_transducer": {
            "type": "hyper_reduced_markov_confidence_projection",
            "inputs": [
                {
                    "id": "input_frames",
                    "signal": "frame_block",
                    "shape": [-1, hidden_size],
                }
            ],
            "outputs": [
                {
                    "id": "draft_logits",
                    "signal": "logits_block",
                    "shape": [-1, structure.vocab_size],
                },
                {
                    "id": "confidence_logits",
                    "signal": "scalar_block",
                    "shape": [-1],
                },
            ],
            "attrs": {
                "eps": structure.norm_eps,
                "weight_offset": structure.rms_norm_weight_offset,
                "markov_rank": draft.attributes["markov_rank"],
                "configured_block_size": draft.attributes["proposal_contract"][
                    "configured_block_size"
                ],
                "default_draft_tokens": draft.attributes["proposal_contract"][
                    "default_draft_tokens"
                ],
                "stream_mixer": deepcopy(draft.attributes["stream_mixer"]),
            },
            "params": {
                name: tensor_ref(tensor)
                for name, tensor in draft.tensors.items()
                if name in output_param_ids
            },
        },
        "state_contract": {
            "ownership": "per_stream_per_node_instance",
            "draft_updates": "tentative",
            "acceptance": "commit_accepted_prefix",
            "rejection": "restore_last_committed_state",
            "context_updates": "target_feature_projection_per_committed_token",
        },
    }


def make_standard_mtp_draft_descriptor(
    structure: ModelStructure,
    draft: DraftExecutionGraphStructure,
    adapter_params: Json,
) -> Json:
    hidden_size = structure.hidden_size
    return {
        "id": draft.id,
        "type": "multi_token_prediction",
        "source_prefix": draft.prefix,
        "input_adapter": {
            "type": "normalized_embedding_hidden_projection",
            "inputs": [
                {"id": "token_embedding", "signal": "frame", "shape": [hidden_size]},
                {"id": "target_hidden", "signal": "frame", "shape": [hidden_size]},
            ],
            "output": {"id": "output_frame", "signal": "frame", "shape": [hidden_size]},
            "attrs": {
                "eps": structure.norm_eps,
                "weight_offset": structure.rms_norm_weight_offset,
                "concatenation_order": ["token_embedding", "target_hidden"],
            },
            "params": adapter_params,
        },
        "execution_graph": {
            "topology": "series",
            "components": [
                {
                    "id": f"{draft.id}_layer_{layer.index:02d}",
                    "type": "node_instance",
                    "component_class": make_component_class(structure, layer),
                    "operator_type": layer.operator_type,
                    "file": (
                        f"drafts/{draft.id}/layers/"
                        f"{draft.id}_layer_{layer.index:02d}.json"
                    ),
                }
                for layer in draft.layers
            ],
        },
        "output_transducer": {
            "type": "normalized_hidden_projection",
            "inputs": [
                {"id": "input_frame", "signal": "frame", "shape": [hidden_size]}
            ],
            "outputs": [
                {"id": "output_hidden", "signal": "frame", "shape": [hidden_size]},
                {
                    "id": "output_logits",
                    "signal": "logits",
                    "shape": [structure.vocab_size],
                },
            ],
            "attrs": {
                "eps": structure.norm_eps,
                "weight_offset": structure.rms_norm_weight_offset,
                "scale": 1.0 / structure.logits_scale,
                "soft_cap": structure.logits_soft_cap,
            },
            "params": {
                "norm": tensor_ref(draft.tensors["output_norm"]),
                "projection": tensor_ref(draft.tensors["output_projection"]),
            },
        },
        "state_contract": {
            "ownership": "per_stream_per_node_instance",
            "draft_updates": "tentative",
            "acceptance": "commit_accepted_prefix",
            "rejection": "restore_last_committed_state",
        },
    }


def make_feed_forward_descriptor(
    structure: ModelStructure, layer: LayerStructure
) -> Json:
    descriptor: Json = {
        "type": layer.feed_forward_type,
        "hidden_size": structure.hidden_size,
        "intermediate_size": layer.intermediate_size,
        "activation": structure.activation,
    }
    if layer.feed_forward_type == "sparse_moe":
        descriptor.update(
            {
                "num_experts": structure.num_experts,
                "experts_per_token": structure.experts_per_token,
                "routing": structure.moe_routing,
                "shared_intermediate_size": layer.shared_intermediate_size,
            }
        )
        descriptor.update(deepcopy(layer.feed_forward_attributes))
    return descriptor


def make_reference_decomposition(
    structure: ModelStructure,
    layer: LayerStructure,
    operator: Json,
) -> Json:
    if layer.residual_mixer is not None:
        return make_hyper_connected_reference_decomposition(structure, layer, operator)
    hidden_size = structure.hidden_size
    return {
        "source": "source_transformers_layer",
        "topology": [
            {
                "id": "operator_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "input",
                "output": "operator_norm.output",
                "params": {"weight": tensor_ref(layer.tensors["operator_norm"])},
            },
            operator,
            {
                "id": "operator_residual",
                "type": "residual_add",
                "circuit_template": f"add_h{hidden_size}_v1",
                "inputs": ["input", "operator.output"],
                "output": "operator_residual.output",
            },
            {
                "id": "ffn_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "operator_residual.output",
                "output": "ffn_norm.output",
                "params": {"weight": tensor_ref(layer.tensors["ffn_norm"])},
            },
            make_ffn_component(structure, layer),
            {
                "id": "ffn_residual",
                "type": "residual_add",
                "circuit_template": f"add_h{hidden_size}_v1",
                "inputs": ["operator_residual.output", "ffn.output"],
                "output": "output",
            },
        ],
    }


def make_hyper_connected_reference_decomposition(
    structure: ModelStructure,
    layer: LayerStructure,
    operator: Json,
) -> Json:
    mixer = layer.residual_mixer
    if mixer is None or mixer.get("type") != "sinkhorn_hyper_connection":
        raise ModelTranspileError(
            "hyper-connected decomposition has no supported mixer"
        )
    hidden_size = structure.hidden_size
    mixer_attrs = {
        "multiplicity": int(mixer["multiplicity"]),
        "sinkhorn_iterations": int(mixer["sinkhorn_iterations"]),
        "epsilon": float(mixer["epsilon"]),
    }

    def stage_prefix(stage: str, input_signal: str, output_signal: str) -> list[Json]:
        return [
            {
                "id": f"hyper_{stage}_function",
                "type": "normalized_linear",
                "input": input_signal,
                "output": f"hyper_{stage}.mixes",
                "attrs": {
                    "normalization": "root_mean_square",
                    **mixer_attrs,
                },
                "params": {
                    "weight": tensor_ref(layer.tensors[f"hyper_{stage}_function"])
                },
            },
            {
                "id": f"hyper_{stage}_sinkhorn",
                "type": "sinkhorn_hyper_connection",
                "input": f"hyper_{stage}.mixes",
                "outputs": [
                    f"hyper_{stage}.pre",
                    f"hyper_{stage}.post",
                    f"hyper_{stage}.combination",
                ],
                "attrs": mixer_attrs,
                "params": {
                    "base": tensor_ref(layer.tensors[f"hyper_{stage}_base"]),
                    "scale": tensor_ref(layer.tensors[f"hyper_{stage}_scale"]),
                },
            },
            {
                "id": f"hyper_{stage}_reduce",
                "type": "sinkhorn_hyper_connection_reduce",
                "inputs": [input_signal, f"hyper_{stage}.pre"],
                "output": output_signal,
                "attrs": mixer_attrs,
            },
        ]

    return {
        "source": "source_transformers_layer",
        "topology": [
            *stage_prefix("attention", "input", "hyper_attention.input"),
            {
                "id": "operator_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "hyper_attention.input",
                "output": "operator_norm.output",
                "params": {"weight": tensor_ref(layer.tensors["operator_norm"])},
            },
            operator,
            {
                "id": "operator_residual",
                "type": "sinkhorn_hyper_connection_post",
                "inputs": [
                    "operator.output",
                    "input",
                    "hyper_attention.post",
                    "hyper_attention.combination",
                ],
                "output": "operator_residual.output",
                "attrs": mixer_attrs,
            },
            *stage_prefix(
                "feed_forward",
                "operator_residual.output",
                "hyper_feed_forward.input",
            ),
            {
                "id": "ffn_norm",
                "type": "rms_norm",
                "circuit_template": f"rms_norm_h{hidden_size}_v1",
                "input": "hyper_feed_forward.input",
                "output": "ffn_norm.output",
                "params": {"weight": tensor_ref(layer.tensors["ffn_norm"])},
            },
            make_ffn_component(structure, layer),
            {
                "id": "ffn_residual",
                "type": "sinkhorn_hyper_connection_post",
                "inputs": [
                    "ffn.output",
                    "operator_residual.output",
                    "hyper_feed_forward.post",
                    "hyper_feed_forward.combination",
                ],
                "output": "output",
                "attrs": mixer_attrs,
            },
        ],
    }


def make_ffn_component(structure: ModelStructure, layer: LayerStructure) -> Json:
    if layer.feed_forward_type == "sparse_moe":
        if (
            layer.feed_forward_attributes.get("expert_storage")
            == "independent_resources"
        ):
            return {
                "id": "feed_forward",
                "type": "independently_addressed_sparse_moe_feed_forward",
                "input": "ffn_norm.output",
                "output": "ffn.output",
                "dimensions": make_feed_forward_descriptor(structure, layer),
                "params": {
                    name: tensor_ref(tensor)
                    for name, tensor in layer.tensors.items()
                    if name == "moe_router"
                    or name == "moe_route_table"
                    or name == "moe_router_selection_bias"
                    or name.startswith("routed_expert_")
                    or name.startswith("shared_expert_")
                },
            }
        params = {
            "router": tensor_ref(layer.tensors["moe_router"]),
            "input": tensor_ref(layer.tensors["moe_input"]),
            "output": tensor_ref(layer.tensors["moe_output"]),
        }
        if "moe_router_correction_bias" in layer.tensors:
            params["router_correction_bias"] = tensor_ref(
                layer.tensors["moe_router_correction_bias"]
            )
        if layer.shared_intermediate_size is not None:
            params.update(
                {
                    "shared_input": tensor_ref(layer.tensors["shared_mlp_input"]),
                    "shared_output": tensor_ref(layer.tensors["shared_mlp_output"]),
                }
            )
            if "shared_mlp_gate" in layer.tensors:
                params["shared_gate"] = tensor_ref(layer.tensors["shared_mlp_gate"])
        return {
            "id": "feed_forward",
            "type": "sparse_moe_feed_forward",
            "input": "ffn_norm.output",
            "output": "ffn.output",
            "dimensions": make_feed_forward_descriptor(structure, layer),
            "params": params,
        }
    if "ffn_gate_up" in layer.tensors:
        params = {
            "gate_up": tensor_ref(layer.tensors["ffn_gate_up"]),
            "down": tensor_ref(layer.tensors["ffn_down"]),
        }
        if "ffn_gate_up_bias" in layer.tensors:
            params["gate_up_bias"] = tensor_ref(layer.tensors["ffn_gate_up_bias"])
    else:
        params = {
            "gate": tensor_ref(layer.tensors["ffn_gate"]),
            "down": tensor_ref(layer.tensors["ffn_down"]),
            "up": tensor_ref(layer.tensors["ffn_up"]),
        }
    for source_id, target_id in (
        ("ffn_gate_bias", "gate_bias"),
        ("ffn_down_bias", "down_bias"),
        ("ffn_up_bias", "up_bias"),
    ):
        if source_id in layer.tensors:
            params[target_id] = tensor_ref(layer.tensors[source_id])
    return {
        "id": "feed_forward",
        "type": "swiglu_feed_forward",
        "circuit_template": (
            f"swiglu_ffn_{structure.hidden_size}_{layer.intermediate_size}_v1"
        ),
        "input": "ffn_norm.output",
        "output": "ffn.output",
        "activation": structure.activation,
        "params": params,
    }


def make_conv_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    return {
        "id": "operator",
        "type": "short_conv_operator",
        "circuit_template": f"short_conv_h{structure.hidden_size}_k{structure.conv_l_cache}_v1",
        "input": "operator_norm.output",
        "output": "operator.output",
        "state_ports": make_state_ports(structure, layer),
        "params": {
            "in_projection": tensor_ref(layer.tensors["conv_in_projection"]),
            "depthwise_kernel": tensor_ref(layer.tensors["conv_depthwise_kernel"]),
            "out_projection": tensor_ref(layer.tensors["conv_out_projection"]),
        },
        "internal_components": [
            {"id": "in_projection", "type": "linear"},
            {"id": "split_b_c_x", "type": "split", "parts": ["b", "c", "x"]},
            {"id": "input_gate", "type": "multiply", "expression": "b * x"},
            {"id": "temporal_memory", "type": "stateful_delay_line"},
            {"id": "depthwise_conv", "type": "depthwise_temporal_convolution"},
            {"id": "output_gate", "type": "multiply", "expression": "c * conv_out"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_attention_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    head_width = layer.head_width
    heads = {
        "query_heads": layer.num_attention_heads,
        "key_value_heads": layer.num_key_value_heads,
        "head_width": head_width,
        "query_groups_per_kv_head": layer.num_attention_heads
        // layer.num_key_value_heads,
    }
    if "qkv_projection" in layer.tensors:
        params = {
            "qkv_projection": tensor_ref(layer.tensors["qkv_projection"]),
            "out_projection": tensor_ref(layer.tensors["attention_out_projection"]),
        }
        if "qkv_projection_bias" in layer.tensors:
            params["qkv_projection_bias"] = tensor_ref(
                layer.tensors["qkv_projection_bias"]
            )
    else:
        params = {
            "q_projection": tensor_ref(layer.tensors["q_projection"]),
            "out_projection": tensor_ref(layer.tensors["attention_out_projection"]),
        }
        if layer.shared_kv_source_layer is None:
            params["k_projection"] = tensor_ref(layer.tensors["k_projection"])
            if "v_projection" in layer.tensors:
                params["v_projection"] = tensor_ref(layer.tensors["v_projection"])
    for source_id, target_id in (
        ("q_projection_bias", "q_projection_bias"),
        ("k_projection_bias", "k_projection_bias"),
        ("v_projection_bias", "v_projection_bias"),
        ("attention_out_projection_bias", "out_projection_bias"),
        ("attention_gate_projection_bias", "attention_gate_projection_bias"),
    ):
        if source_id in layer.tensors:
            params[target_id] = tensor_ref(layer.tensors[source_id])
    internal_components = (
        [
            {"id": "qkv_projection", "type": "linear"},
            {"id": "qkv_split", "type": "split"},
        ]
        if "qkv_projection" in layer.tensors
        else [
            {"id": "q_projection", "type": "linear"},
            *(
                [
                    {"id": "k_projection", "type": "linear"},
                    *(
                        [{"id": "v_projection", "type": "linear"}]
                        if "v_projection" in layer.tensors
                        else []
                    ),
                ]
                if layer.shared_kv_source_layer is None
                else []
            ),
        ]
    )
    if structure.attention_output_gate:
        internal_components.append({"id": "q_gate_split", "type": "split"})
    if "attention_gate_projection" in layer.tensors:
        params["attention_gate_projection"] = tensor_ref(
            layer.tensors["attention_gate_projection"]
        )
        internal_components.append(
            {"id": "attention_gate_projection", "type": "linear"}
        )
    if "q_norm" in layer.tensors:
        params["q_norm"] = tensor_ref(layer.tensors["q_norm"])
        internal_components.append({"id": "q_norm", "type": "rms_norm_per_head"})
    if "k_norm" in layer.tensors:
        params["k_norm"] = tensor_ref(layer.tensors["k_norm"])
        internal_components.append({"id": "k_norm", "type": "rms_norm_per_head"})
    if "attention_sinks" in layer.tensors:
        params["attention_sinks"] = tensor_ref(layer.tensors["attention_sinks"])
    internal_components.extend(
        [
            {"id": "rope", "type": "rotary_position_embedding"},
            {
                "id": "kv_memory",
                "type": (
                    "shared_state_read"
                    if layer.shared_kv_source_layer is not None
                    else "stateful_append_memory"
                ),
            },
            {"id": "attention_read", "type": "scaled_dot_product_attention"},
            *(
                [
                    {
                        "id": "attention_output_gate",
                        "type": (
                            "sigmoid_multiply"
                            if structure.attention_output_gate
                            else f"{layer.attention_gate_activation}_multiply"
                        ),
                    }
                ]
                if structure.attention_output_gate
                or layer.attention_gate_activation is not None
                else []
            ),
            {"id": "out_projection", "type": "linear"},
        ]
    )
    return {
        "id": "operator",
        "type": "gqa_attention_operator",
        "circuit_template": (
            "gqa_attention_"
            f"h{structure.hidden_size}_q{layer.num_attention_heads}_"
            f"kv{layer.num_key_value_heads}_d{head_width}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "heads": heads,
        "rotary_width": layer.rotary_width,
        "rope_type": layer.rope_type,
        "output_gate": structure.attention_output_gate,
        "attention_gate": (
            {
                "activation": layer.attention_gate_activation,
                "per_head": layer.attention_gate_per_head,
            }
            if layer.attention_gate_activation is not None
            else None
        ),
        "window_size": layer.attention_window_size,
        "shared_kv_source_layer": layer.shared_kv_source_layer,
        "state_ports": make_state_ports(structure, layer),
        "params": params,
        "internal_components": internal_components,
    }


def make_latent_sparse_attention_operator(
    structure: ModelStructure, layer: LayerStructure
) -> Json:
    attributes = deepcopy(layer.operator_attributes)
    parameter_names = {
        "q_input_projection",
        "q_input_norm",
        "q_head_projection",
        "kv_projection",
        "kv_norm",
        "attention_sinks",
        "out_group_projection",
        "attention_out_projection",
    }
    return {
        "id": "operator",
        "type": "latent_sparse_attention_operator",
        "circuit_template": (
            "latent_sparse_attention_"
            f"h{structure.hidden_size}_q{layer.num_attention_heads}_"
            f"d{layer.head_width}_r{attributes['query_rank']}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "heads": {
            "query_heads": layer.num_attention_heads,
            "key_value_heads": layer.num_key_value_heads,
            "head_width": layer.head_width,
        },
        "attributes": attributes,
        "state_ports": make_state_ports(structure, layer),
        "params": {
            name: tensor_ref(tensor)
            for name, tensor in layer.tensors.items()
            if name in parameter_names
            or name.startswith("compressor_")
            or name.startswith("indexer_")
        },
        "internal_components": [
            {"id": "query_low_rank_projection", "type": "linear"},
            {"id": "query_low_rank_norm", "type": "rms_norm"},
            {"id": "query_head_projection", "type": "linear"},
            {"id": "key_value_projection", "type": "linear"},
            {"id": "local_temporal_memory", "type": "rolling_attention_memory"},
            *(
                [
                    {"id": "memory_compressor", "type": "learned_gated_pooling"},
                    {
                        "id": "compressed_temporal_memory",
                        "type": "append_only_memory",
                    },
                ]
                if attributes["compression"] is not None
                else []
            ),
            *(
                [{"id": "memory_indexer", "type": "learned_topk_indexer"}]
                if attributes["indexer"]["selection"] == "learned_topk"
                else []
            ),
            {"id": "sparse_attention_read", "type": "indexed_attention"},
            {"id": "grouped_output_projection", "type": "grouped_linear"},
            {"id": "output_projection", "type": "linear"},
        ],
    }


def make_gated_delta_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    mixer = structure.recurrent_mixer
    if mixer is None:
        raise ModelTranspileError("gated-delta layer has no recurrent mixer dimensions")
    return {
        "id": "operator",
        "type": "gated_delta_operator",
        "circuit_template": (
            f"gated_delta_k{mixer['key_heads']}x{mixer['key_head_width']}_"
            f"v{mixer['value_heads']}x{mixer['value_head_width']}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "dimensions": mixer,
        "state_ports": make_state_ports(structure, layer),
        "params": {
            "qkv_projection": tensor_ref(layer.tensors["delta_qkv_projection"]),
            "z_projection": tensor_ref(layer.tensors["delta_z_projection"]),
            "b_projection": tensor_ref(layer.tensors["delta_b_projection"]),
            "a_projection": tensor_ref(layer.tensors["delta_a_projection"]),
            "conv_kernel": tensor_ref(layer.tensors["delta_conv_kernel"]),
            "a_log": tensor_ref(layer.tensors["delta_a_log"]),
            "dt_bias": tensor_ref(layer.tensors["delta_dt_bias"]),
            "norm": tensor_ref(layer.tensors["delta_norm"]),
            "out_projection": tensor_ref(layer.tensors["delta_out_projection"]),
        },
        "internal_components": [
            {"id": "qkv_projection", "type": "linear"},
            {"id": "z_projection", "type": "linear"},
            {"id": "b_projection", "type": "linear"},
            {"id": "a_projection", "type": "linear"},
            {"id": "causal_conv", "type": "stateful_depthwise_convolution"},
            {"id": "delta_update", "type": "gated_delta_recurrence"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_rg_lru_operator(structure: ModelStructure, layer: LayerStructure) -> Json:
    mixer = structure.recurrent_mixer
    if mixer is None or mixer.get("type") != "rg_lru":
        raise ModelTranspileError("RG-LRU layer has no recurrent mixer dimensions")
    params = {
        name: tensor_ref(layer.tensors[name])
        for name in (
            "rg_lru_x_projection",
            "rg_lru_y_projection",
            "rg_lru_out_projection",
            "rg_lru_conv_kernel",
            "rg_lru_input_gate_weight",
            "rg_lru_input_gate_bias",
            "rg_lru_recurrent_gate_weight",
            "rg_lru_recurrent_gate_bias",
            "rg_lru_recurrent_param",
        )
    }
    for name in (
        "rg_lru_x_projection_bias",
        "rg_lru_y_projection_bias",
        "rg_lru_out_projection_bias",
        "rg_lru_conv_bias",
    ):
        if name in layer.tensors:
            params[name] = tensor_ref(layer.tensors[name])
    return {
        "id": "operator",
        "type": "rg_lru_operator",
        "circuit_template": (
            f"rg_lru_h{structure.hidden_size}_b{mixer['heads']}x{mixer['block_width']}"
            f"_k{mixer['conv_kernel_width']}_v1"
        ),
        "input": "operator_norm.output",
        "output": "operator.output",
        "dimensions": mixer,
        "activation": structure.activation,
        "state_ports": make_state_ports(structure, layer),
        "params": params,
        "internal_components": [
            {"id": "x_projection", "type": "linear"},
            {"id": "y_projection", "type": "linear"},
            {"id": "y_activation", "type": structure.activation},
            {"id": "depthwise_convolution", "type": "stateful_depthwise_convolution"},
            {"id": "real_gated_recurrence", "type": "rg_lru_recurrence"},
            {"id": "output_gate", "type": "multiply"},
            {"id": "out_projection", "type": "linear"},
        ],
    }


def make_parameter_block(
    operator_type: str, feed_forward_type: str, tensors: dict[str, str]
) -> Json:
    if operator_type == "conv":
        layout = "shortconv_layer_params_v1"
    elif operator_type == "full_attention":
        layout = "gqa_attention_layer_params_v1"
    elif operator_type == "latent_sparse_attention":
        layout = "latent_sparse_attention_layer_params_v1"
    elif operator_type == "gated_delta":
        layout = "gated_delta_layer_params_v1"
    elif operator_type == "rg_lru":
        layout = "rg_lru_layer_params_v1"
    else:
        raise ModelTranspileError(
            f"unsupported parameter layout for operator {operator_type!r}"
        )
    return {
        "layout": f"{layout}_{feed_forward_type}",
        "storage": "source_tensor_refs",
        "params": {name: tensor_ref(tensor) for name, tensor in tensors.items()},
        "tensor_refs": list(tensors.values()),
    }


def make_state_ports(
    structure: ModelStructure,
    layer: LayerStructure,
) -> list[Json]:
    operator_type = layer.operator_type
    if operator_type == "conv":
        return [
            {
                "id": "temporal_memory",
                "type": "rolling_frame_memory",
                "shape": [structure.conv_l_cache, structure.hidden_size],
                "dtype": "BF16",
                "update": "shift_append",
                "sharing": "per_stream_per_node_instance",
            }
        ]

    if operator_type == "full_attention":
        head_width = layer.head_width
        sharing = (
            f"shared_from:layer_{layer.shared_kv_source_layer:02d}.kv_memory"
            if layer.shared_kv_source_layer is not None
            else "per_stream_per_node_instance"
        )
        return [
            {
                "id": "kv_memory",
                "type": "append_only_attention_memory",
                "query_heads": layer.num_attention_heads,
                "key_shape_per_token": [layer.num_key_value_heads, head_width],
                "value_shape_per_token": [layer.num_key_value_heads, head_width],
                "dtype": "BF16",
                "growth": "per_activation",
                "max_dynamic_activations": layer.attention_window_size,
                "sharing": sharing,
            }
        ]

    if operator_type == "latent_sparse_attention":
        attributes = layer.operator_attributes
        states: list[Json] = [
            {
                "id": "local_kv_memory",
                "type": "rolling_attention_memory",
                "shape_per_token": [layer.head_width],
                "capacity": int(attributes["window_size"]),
                "dtype": "BF16",
                "update": "ring_append",
                "sharing": "per_stream_per_node_instance",
            }
        ]
        compression = attributes["compression"]
        if compression is not None:
            ratio = int(compression["ratio"])
            coefficient = int(compression["lane_coefficient"])
            states.extend(
                [
                    {
                        "id": "compressed_kv_memory",
                        "type": "append_only_attention_memory",
                        "shape_per_token": [layer.head_width],
                        "dtype": "BF16",
                        "growth": f"one_per_{ratio}_activations",
                        "sharing": "per_stream_per_node_instance",
                    },
                    {
                        "id": "compressor_accumulator",
                        "type": "gated_pooling_memory",
                        "shape": [
                            2,
                            coefficient * ratio,
                            coefficient * layer.head_width,
                        ],
                        "dtype": "F32",
                        "update": "position_biased_softmax_pool",
                        "sharing": "per_stream_per_node_instance",
                    },
                ]
            )
        if attributes["indexer"]["selection"] == "learned_topk":
            indexer = attributes["indexer"]
            states.extend(
                [
                    {
                        "id": "indexer_compressor_accumulator",
                        "type": "gated_pooling_memory",
                        "shape": [
                            2,
                            int(indexer["compressor_lane_coefficient"])
                            * int(compression["ratio"]),
                            int(indexer["compressor_lane_coefficient"])
                            * int(indexer["head_width"]),
                        ],
                        "dtype": "F32",
                        "update": "position_biased_softmax_pool",
                        "sharing": "per_stream_per_node_instance",
                    },
                    {
                        "id": "indexer_kv_memory",
                        "type": "append_only_index_memory",
                        "shape_per_token": [int(indexer["head_width"])],
                        "dtype": "BF16",
                        "growth": "with_compressed_kv_memory",
                        "sharing": "per_stream_per_node_instance",
                    },
                ]
            )
        return states

    if operator_type == "gated_delta":
        mixer = structure.recurrent_mixer
        if mixer is None:
            raise ModelTranspileError(
                "gated-delta layer has no recurrent mixer dimensions"
            )
        key_width = int(mixer["key_heads"]) * int(mixer["key_head_width"])
        value_width = int(mixer["value_heads"]) * int(mixer["value_head_width"])
        conv_width = key_width * 2 + value_width
        return [
            {
                "id": "conv_state",
                "type": "rolling_channel_memory",
                "shape": [conv_width, int(mixer["conv_kernel_width"])],
                "dtype": "BF16",
                "update": "shift_append",
                "sharing": "per_stream_per_node_instance",
            },
            {
                "id": "recurrent_state",
                "type": "gated_delta_matrix_memory",
                "shape": [
                    int(mixer["value_heads"]),
                    int(mixer["key_head_width"]),
                    int(mixer["value_head_width"]),
                ],
                "dtype": mixer["state_dtype"],
                "update": "decay_delta_outer_product",
                "sharing": "per_stream_per_node_instance",
            },
        ]

    if operator_type == "rg_lru":
        mixer = structure.recurrent_mixer
        if mixer is None or mixer.get("type") != "rg_lru":
            raise ModelTranspileError("RG-LRU layer has no recurrent mixer dimensions")
        return [
            {
                "id": "conv_state",
                "type": "rolling_channel_memory",
                "shape": [
                    int(mixer["width"]),
                    int(mixer["conv_kernel_width"]),
                ],
                "dtype": "BF16",
                "update": "shift_append",
                "sharing": "per_stream_per_node_instance",
            },
            {
                "id": "recurrent_state",
                "type": "diagonal_recurrent_memory",
                "shape": [int(mixer["width"])],
                "dtype": str(mixer["state_dtype"]),
                "update": "real_gated_linear_recurrence",
                "sharing": "per_stream_per_node_instance",
            },
        ]

    raise ModelTranspileError(f"unsupported state ports for operator {operator_type!r}")


def make_component_class(structure: ModelStructure, layer: LayerStructure) -> str:
    operator_type = layer.operator_type
    feed_forward = (
        f"moe{structure.num_experts}x{structure.experts_per_token}i{layer.intermediate_size}"
        if layer.feed_forward_type == "sparse_moe"
        else f"ffn{layer.intermediate_size}"
    )
    if operator_type == "conv":
        return (
            f"shortconv_layer_h{structure.hidden_size}_"
            f"k{structure.conv_l_cache}_{feed_forward}_v1"
        )

    if operator_type == "rg_lru":
        mixer = structure.recurrent_mixer
        if mixer is None or mixer.get("type") != "rg_lru":
            raise ModelTranspileError("RG-LRU layer has no recurrent mixer dimensions")
        return (
            "rg_lru_layer_"
            f"h{structure.hidden_size}_b{mixer['heads']}x{mixer['block_width']}_"
            f"k{mixer['conv_kernel_width']}_{feed_forward}_v1"
        )

    if operator_type == "full_attention":
        head_width = layer.head_width
        return (
            "gqa_attention_layer_"
            f"h{structure.hidden_size}_q{layer.num_attention_heads}_"
            f"kv{layer.num_key_value_heads}_d{head_width}_"
            f"{feed_forward}_v1"
        )

    if operator_type == "latent_sparse_attention":
        attributes = layer.operator_attributes
        compression = attributes["compression"]
        compression_class = (
            f"c{compression['ratio']}" if compression is not None else "window_only"
        )
        index_class = (
            "indexed"
            if attributes["indexer"]["selection"] == "learned_topk"
            else "chronological"
        )
        return (
            "latent_sparse_attention_layer_"
            f"h{structure.hidden_size}_q{layer.num_attention_heads}_"
            f"d{layer.head_width}_{compression_class}_{index_class}_{feed_forward}_v1"
        )

    if operator_type == "gated_delta":
        mixer = structure.recurrent_mixer
        if mixer is None:
            raise ModelTranspileError(
                "gated-delta layer has no recurrent mixer dimensions"
            )
        return (
            "gated_delta_layer_"
            f"h{structure.hidden_size}_k{mixer['key_heads']}x{mixer['key_head_width']}_"
            f"v{mixer['value_heads']}x{mixer['value_head_width']}_"
            f"{feed_forward}_v1"
        )

    raise ModelTranspileError(
        f"unsupported component class for operator {operator_type!r}"
    )
