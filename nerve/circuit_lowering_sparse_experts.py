from __future__ import annotations

from nerve.circuit_lowering_helpers import _linear_params
from nerve.model_transpiler_types import Json


def independent_sparse_moe_body(*, feed_forward: Json, parameters: Json) -> list[Json]:
    routing = feed_forward["routing"]
    expert_ids = [int(value) for value in feed_forward["expert_ids"]]
    route_inputs = ["moe_router_logits"]
    route_params: list[str] = []
    if routing["selection"] == "token_id_table":
        route_inputs.append("token_id")
        route_params.append("moe_route_table")
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
    width = int(feed_forward["intermediate_size"])
    limit = float(feed_forward.get("swiglu_limit", 0.0))
    return [
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
                "num_experts": int(feed_forward["num_experts"]),
                "experts_per_token": int(feed_forward["experts_per_token"]),
                **routing,
                "selection_domain": {
                    "id": "routed_experts",
                    "resource_count": len(expert_ids),
                    "selection_signal": "moe_routes",
                    "resource_granularity": "expert",
                },
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
                "experts_per_token": int(feed_forward["experts_per_token"]),
                "swiglu_limit": limit,
                "selected_parameter_accesses": [
                    {"selection_signal": "moe_routes", "mapping": selected_gate_up}
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
                "experts_per_token": int(feed_forward["experts_per_token"]),
                "selected_parameter_accesses": [
                    {"selection_signal": "moe_routes", "mapping": selected_down}
                ],
            },
        },
        {
            "id": "moe_reduce",
            "op": "moe_reduce",
            "inputs": ["moe_expert_outputs"],
            "outputs": ["moe_out"],
            "attrs": {
                "hidden_size": int(feed_forward["hidden_size"]),
                "experts_per_token": int(feed_forward["experts_per_token"]),
                "routed_scaling_factor": 1.0,
                "routing_weights_already_scaled": True,
            },
        },
        {
            "id": "shared_mlp_gate_projection",
            "op": "linear",
            "inputs": ["ffn_norm_out"],
            "outputs": ["shared_gate"],
            "params": _linear_params("shared_expert_w1", parameters),
        },
        {
            "id": "shared_mlp_up_projection",
            "op": "linear",
            "inputs": ["ffn_norm_out"],
            "outputs": ["shared_up"],
            "params": _linear_params("shared_expert_w3", parameters),
        },
        {
            "id": "shared_mlp_activation",
            "op": "bounded_silu_multiply" if limit > 0.0 else "silu_multiply",
            "inputs": ["shared_gate", "shared_up"],
            "outputs": ["shared_hidden"],
            "attrs": {"element_count": width, "limit": limit},
        },
        {
            "id": "shared_mlp_output_projection",
            "op": "linear",
            "inputs": ["shared_hidden"],
            "outputs": ["shared_out"],
            "params": _linear_params("shared_expert_w2", parameters),
        },
        {
            "id": "shared_and_sparse_expert_add",
            "op": "residual_add",
            "inputs": ["moe_out", "shared_out"],
            "outputs": ["ffn_out"],
        },
    ]
