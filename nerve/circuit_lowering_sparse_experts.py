from __future__ import annotations

from nerve.circuit_lowering_helpers import _linear_params
from nerve.model_transpiler_types import Json


def independent_sparse_moe_body(*, feed_forward: Json, parameters: Json) -> list[Json]:
    routing = feed_forward["routing"]
    expert_ids = [int(value) for value in feed_forward["expert_ids"]]
    route_inputs = ["moe_router_logits"]
    route_params: list[str] = []
    predictable_preselection = routing["selection"] == "token_id_table"
    if predictable_preselection:
        route_inputs.append("moe_resource_routes")
    else:
        route_params.append("moe_router_selection_bias")

    def expert_params(expert: int, projections: tuple[str, ...]) -> list[str]:
        return [
            parameter
            for projection in projections
            for parameter in _linear_params(
                f"routed_expert_{expert:03d}_{projection}", parameters
            )
        ]

    selected_gate_up = [
        {"selector": expert, "parameter_ids": expert_params(expert, ("w1", "w3"))}
        for expert in expert_ids
    ]
    selected_down = [
        {"selector": expert, "parameter_ids": expert_params(expert, ("w2",))}
        for expert in expert_ids
    ]
    shared_resource = len(expert_ids)
    selected_gate_up.append(
        {
            "selector": shared_resource,
            "parameter_ids": [
                *_linear_params("shared_expert_w1", parameters),
                *_linear_params("shared_expert_w3", parameters),
            ],
        }
    )
    selected_down.append(
        {
            "selector": shared_resource,
            "parameter_ids": _linear_params("shared_expert_w2", parameters),
        }
    )
    width = int(feed_forward["intermediate_size"])
    limit = float(feed_forward.get("swiglu_limit", 0.0))
    routed_selection_count = int(feed_forward["experts_per_token"])
    selected_resource_count = routed_selection_count + 1
    resource_count = len(selected_gate_up)
    selection_domain = {
        "id": "experts",
        "resource_count": resource_count,
        "selection_signal": (
            "moe_resource_routes" if predictable_preselection else "moe_routes"
        ),
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": selected_resource_count,
            "index_shift": 0,
            "index_mask": (1 << (resource_count - 1).bit_length()) - 1,
            "calibration_word_base": 0,
        },
    }
    resource_selection_signal = selection_domain["selection_signal"]
    body = []
    if predictable_preselection:
        body.append(
            {
                "id": "moe_resource_preselection",
                "op": "parameter_table_resource_preselection",
                "inputs": ["token_id"],
                "outputs": ["moe_resource_routes"],
                "params": ["moe_route_table"],
                "attrs": {
                    "experts_per_token": selected_resource_count,
                    "routed_resource_count": len(expert_ids),
                    "routed_selection_count": routed_selection_count,
                    "always_selected_resources": [
                        {"resource_index": shared_resource, "weight": 1.0}
                    ],
                    "predictable_dependency": {
                        "schema": "nerve.predictable_resource_selection.v1",
                        "kind": "parameter_table_lookup",
                        "key_signal": "token_id",
                        "table_parameter": "moe_route_table",
                        "selection_semantics": "exact",
                    },
                    "selection_domain": selection_domain,
                },
            }
        )
    body.extend([
        {
            "id": "moe_router_projection",
            "op": "linear",
            "inputs": ["ffn_norm_out"],
            "outputs": ["moe_router_logits"],
            "params": _linear_params("moe_router", parameters),
        },
        {
            "id": "moe_topk",
            "op": "moe_route",
            "inputs": route_inputs,
            "outputs": ["moe_routes"],
            "params": route_params,
            "attrs": {
                "experts_per_token": selected_resource_count,
                "routed_resource_count": len(expert_ids),
                "routed_selection_count": routed_selection_count,
                "always_selected_resources": [
                    {"resource_index": shared_resource, "weight": 1.0}
                ],
                **(
                    {**routing, "selection": "preselected_resource_indices"}
                    if predictable_preselection
                    else routing
                ),
                **({} if predictable_preselection else {"selection_domain": selection_domain}),
            },
        },
        {
            "id": "sparse_moe_gate_up",
            "op": "independent_sparse_moe_gate_up",
            "inputs": ["ffn_norm_out", "moe_routes"],
            "outputs": ["moe_expert_intermediates"],
            "params": [
                parameter
                for entry in selected_gate_up
                for parameter in entry["parameter_ids"]
            ],
            "attrs": {
                "hidden_size": int(feed_forward["hidden_size"]),
                "intermediate_size": width,
                "experts_per_token": selected_resource_count,
                "swiglu_limit": limit,
                "selected_parameter_accesses": [
                    {
                        "selection_signal": resource_selection_signal,
                        "execution_signal": "moe_routes",
                        "execution_calibration_word_base": 0x3F800000,
                        "mapping": selected_gate_up,
                    }
                ],
            },
        },
        {
            "id": "sparse_moe_down",
            "op": "independent_sparse_moe_down",
            "inputs": ["moe_expert_intermediates", "moe_routes"],
            "outputs": ["moe_expert_outputs"],
            "params": [
                parameter
                for entry in selected_down
                for parameter in entry["parameter_ids"]
            ],
            "attrs": {
                "hidden_size": int(feed_forward["hidden_size"]),
                "intermediate_size": width,
                "experts_per_token": selected_resource_count,
                "selected_parameter_accesses": [
                    {
                        "selection_signal": resource_selection_signal,
                        "execution_signal": "moe_routes",
                        "execution_calibration_word_base": 0x3F800000,
                        "mapping": selected_down,
                    }
                ],
            },
        },
        {
            "id": "moe_reduce",
            "op": "moe_reduce",
            "inputs": ["moe_expert_outputs"],
            "outputs": ["ffn_out"],
            "attrs": {
                "hidden_size": int(feed_forward["hidden_size"]),
                "experts_per_token": selected_resource_count,
                "routed_scaling_factor": 1.0,
                "routing_weights_already_scaled": True,
            },
        },
    ])
    return body
