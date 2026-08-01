from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from nerve.model_package import (
    ModelCompileError,
    ROW_MAJOR_LAYOUT,
    compile_shader_artifacts,
    copy_shader_templates,
    local_size_x_for_shader_file,
    shader_file_for_node,
    workgroup_count_x_for_node,
)


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    nodes = [
        {
            "id": "remember",
            "op": "rolling_state_update",
            "inputs": ["current_kv", "local_kv_memory"],
            "outputs": ["local_kv_values"],
            "state_reads": ["local_kv_memory"],
            "state_writes": ["local_kv_memory"],
            "attrs": {"update": "ring_append", "capacity": 4},
        },
        {
            "id": "derotate",
            "op": "inverse_rotary_position_embedding",
            "inputs": ["attention_heads"],
            "outputs": ["attention_unpositioned"],
            "attrs": {
                "position_source": "stream_tick",
                "position_offset": 1,
                "theta": 10_000.0,
                "rope_type": "default",
                "scaling": None,
                "interleaved": False,
                "rotary_width": 32,
                "head_count": 2,
                "head_width": 64,
            },
        },
        {
            "id": "group_project",
            "op": "grouped_linear",
            "inputs": ["attention_unpositioned"],
            "outputs": ["attention_ranked"],
            "params": ["group_weight", "group_scale"],
            "attrs": {"groups": 2, "rank_per_group": 32},
        },
        {
            "id": "bounded_activation",
            "op": "bounded_silu_multiply",
            "inputs": ["gate", "up"],
            "outputs": ["hidden"],
            "attrs": {"element_count": 128, "limit": 10.0},
        },
    ]
    circuit = {
        "id": "latent_primitives",
        "boundary": {"controls": []},
        "state_ports": [
            {
                "id": "local_kv_memory",
                "type": "rolling_attention_memory",
                "shape_per_token": [128],
                "capacity": 4,
                "dtype": "BF16",
            }
        ],
        "parameters": {
            "refs": {
                "group_weight": {"tensor": "group.weight"},
                "group_scale": {"tensor": "group.scale"},
            }
        },
        "nodes": nodes,
    }
    tensor_index = {
        "tensors": {
            "group.weight": {
                "dtype": "F8_E4M3",
                "shape": [64, 128],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "group.scale": {
                "dtype": "F8_E8M0",
                "shape": [1, 1],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }
    return circuit, tensor_index


def test_compiles_latent_attention_primitives(tmp_path: Path) -> None:
    circuit, tensor_index = _fixture()
    shaders = {
        node["id"]: shader_file_for_node(
            circuit, node, tensor_index, {"hidden_size": 128}
        )
        for node in circuit["nodes"]
    }

    assert shaders == {
        "remember": "rolling_state_update_bf16_4x128.comp",
        "derotate": (
            "inverse_rotary_bf16_2x64_r32_theta10000_half_po1__sc2.comp"
        ),
        "group_project": (
            "grouped_linear_fp8_e4m3_se8m0_b64x128_g2_256x64.comp"
        ),
        "bounded_activation": "bounded_silu_multiply_bf16_128_limit10.comp",
    }
    group_node = circuit["nodes"][2]
    assert workgroup_count_x_for_node(circuit, group_node, tensor_index) == 4
    assert local_size_x_for_shader_file(shaders["group_project"], group_node) == 1024

    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, set(shaders.values()))
    rendered = {
        name: (tmp_path / shader).read_text() for name, shader in shaders.items()
    }
    assert "const float ROPE_DIRECTION = -1.0;" in rendered["derotate"]
    assert "const int POSITION_OFFSET = 1;" in rendered["derotate"]
    assert "uint offset = (group * INPUT_SIZE + column) >> 1u;" in rendered[
        "group_project"
    ]
    assert "gate = min(gate, LIMIT);" in rendered["bounded_activation"]
    assert all("{{" not in source for source in rendered.values())
    compile_shader_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.spv"))) == 4


@pytest.mark.parametrize(
    ("node_id", "mutation", "message"),
    [
        (
            "remember",
            lambda circuit, _tensors: circuit["state_ports"][0].update(capacity=3),
            "incompatible state geometry",
        ),
        (
            "derotate",
            lambda circuit, _tensors: circuit["nodes"][1]["attrs"].update(
                rotary_width=65
            ),
            "invalid contract",
        ),
        (
            "group_project",
            lambda circuit, _tensors: circuit["nodes"][2]["attrs"].update(
                rank_per_group=31
            ),
            "invalid contract",
        ),
        (
            "group_project",
            lambda _circuit, tensors: tensors["tensors"]["group.scale"].update(
                shape=[1, 2]
            ),
            "requires 128-column blocks",
        ),
        (
            "bounded_activation",
            lambda circuit, _tensors: circuit["nodes"][3]["attrs"].update(limit=0.0),
            "invalid contract",
        ),
    ],
)
def test_rejects_malformed_latent_attention_primitives(
    node_id: str, mutation, message: str
) -> None:
    circuit, tensor_index = _fixture()
    mutation(circuit, tensor_index)
    node = next(node for node in circuit["nodes"] if node["id"] == node_id)

    with pytest.raises(ModelCompileError, match=message):
        shader_file_for_node(circuit, node, tensor_index, {"hidden_size": 128})


def test_forward_rope_keeps_zero_offset_filename_stable() -> None:
    circuit, tensor_index = _fixture()
    node = deepcopy(circuit["nodes"][1])
    node["id"] = "rotate"
    node["op"] = "rotary_position_embedding"
    node["attrs"].pop("position_offset")

    assert shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 128}
    ) == "rotary_bf16_2x64_r32_theta10000_half__sc2.comp"
