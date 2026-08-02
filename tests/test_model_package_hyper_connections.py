from __future__ import annotations

from pathlib import Path

import pytest

from nerve.circuit_optimizer import optimize_circuit_for_vulkan
from nerve.model_package import (
    ModelCompileError,
    ROW_MAJOR_LAYOUT,
    compile_shader_artifacts,
    copy_shader_templates,
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.model_package_shader_selection import local_size_x_for_shader_file


def _circuit() -> tuple[dict[str, object], dict[str, object]]:
    refs = {
        "attention_function_weight": {"tensor": "attention.function"},
        "attention_scale": {"tensor": "attention.scale"},
        "attention_base": {"tensor": "attention.base"},
        "feed_forward_function_weight": {"tensor": "feed_forward.function"},
        "feed_forward_scale": {"tensor": "feed_forward.scale"},
        "feed_forward_base": {"tensor": "feed_forward.base"},
    }
    pre = [
        {
            "id": "attention_function",
            "op": "normalized_linear",
            "inputs": ["input_frame"],
            "outputs": ["attention_mixes"],
            "params": ["attention_function_weight"],
            "attrs": {
                "normalization": "root_mean_square",
                "normalization_epsilon": 1e-6,
                "multiplicity": 4,
                "output_element_bytes": [4],
            },
        },
        {
            "id": "attention_sinkhorn",
            "op": "hyper_connection_sinkhorn",
            "inputs": ["attention_mixes"],
            "outputs": ["attention_pre", "attention_post", "attention_comb"],
            "params": ["attention_scale", "attention_base"],
            "attrs": {
                "multiplicity": 4,
                "sinkhorn_iterations": 20,
                "epsilon": 1e-6,
                "output_element_bytes": [4, 4, 4],
            },
        },
        {
            "id": "attention_reduce",
            "op": "hyper_connection_reduce",
            "inputs": ["input_frame", "attention_pre"],
            "outputs": ["operator_input"],
            "attrs": {"multiplicity": 4, "output_element_bytes": [2]},
        },
    ]
    post = {
        "id": "attention_post",
        "op": "hyper_connection_post",
        "inputs": [
            "operator_out",
            "input_frame",
            "attention_post",
            "attention_comb",
        ],
        "outputs": ["attention_residual"],
        "attrs": {
            "multiplicity": 4,
            "sinkhorn_iterations": 20,
            "epsilon": 1e-6,
            "output_element_bytes": [2],
        },
    }
    feed_forward = []
    for node in pre:
        clone = {
            **node,
            "id": node["id"].replace("attention", "feed_forward"),
            "inputs": [
                signal.replace("attention_", "feed_forward_")
                for signal in node["inputs"]
            ],
            "outputs": [
                signal.replace("attention_", "feed_forward_").replace(
                    "operator_input", "ffn_input"
                )
                for signal in node["outputs"]
            ],
            "params": [
                parameter.replace("attention_", "feed_forward_")
                for parameter in node.get("params", [])
            ],
            "attrs": dict(node["attrs"]),
        }
        feed_forward.append(clone)
    feed_forward[0]["inputs"] = ["attention_residual"]
    feed_forward[2]["inputs"] = ["attention_residual", "feed_forward_pre"]
    final_post = {
        "id": "feed_forward_post",
        "op": "hyper_connection_post",
        "inputs": [
            "ffn_out",
            "attention_residual",
            "feed_forward_post",
            "feed_forward_comb",
        ],
        "outputs": ["output_frame"],
        "attrs": {
            "multiplicity": 4,
            "sinkhorn_iterations": 20,
            "epsilon": 1e-6,
            "output_element_bytes": [2],
        },
    }
    circuit = {
        "parameters": {"refs": refs},
        "nodes": [*pre, {"id": "operator", "op": "operator"}, post, *feed_forward, final_post],
        "boundary": {"outputs": [{"id": "output_frame", "source": "output_frame"}]},
    }
    tensor_index = {
        "tensors": {
            "attention.function": {
                "dtype": "F32",
                "shape": [24, 32],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "attention.scale": {
                "dtype": "F32",
                "shape": [3],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "attention.base": {
                "dtype": "F32",
                "shape": [24],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "feed_forward.function": {
                "dtype": "F32",
                "shape": [24, 32],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "feed_forward.scale": {
                "dtype": "F32",
                "shape": [3],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "feed_forward.base": {
                "dtype": "F32",
                "shape": [24],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }
    return circuit, tensor_index


def test_compiles_fused_hyper_connection_kernels(tmp_path: Path) -> None:
    circuit, tensor_index = _circuit()
    optimized = optimize_circuit_for_vulkan(circuit)
    nodes = [
        node
        for node in optimized["nodes"]
        if node["op"].startswith("hyper_connection_")
    ]

    shader_files = {
        shader_file_for_node(
            optimized,
            node,
            tensor_index,
            {"hidden_size": 8},
        )
        for node in nodes
    }

    assert shader_files == {
        "hyper_connection_pre_m4_h8_i20_neps1e-06_heps1e-06.comp",
        "hyper_connection_post_pre_m4_h8_i20_neps1e-06_heps1e-06.comp",
        "hyper_connection_post_m4_h8.comp",
    }
    assert all(
        workgroup_count_x_for_node(
            optimized,
            node,
            tensor_index,
            dimensions={"hidden_size": 8},
        )
        == 1
        for node in nodes
    )
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, shader_files)
    sources = [(tmp_path / shader_file).read_text() for shader_file in shader_files]
    assert all("{{" not in source for source in sources)
    assert any("SINKHORN_ITERATIONS = 20u" in source for source in sources)
    assert any("prior_combination.values" in source for source in sources)
    compile_shader_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.spv"))) == 3


def test_hyper_connection_pre_parallelizes_rows_without_changing_reduction_order(
    tmp_path: Path,
) -> None:
    circuit, tensor_index = _circuit()
    optimized = optimize_circuit_for_vulkan(circuit)
    pre = next(
        node for node in optimized["nodes"] if node["op"] == "hyper_connection_pre"
    )
    shader_file = shader_file_for_node(
        optimized,
        pre,
        tensor_index,
        {"hidden_size": 8},
    )
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})
    source = (tmp_path / shader_file).read_text()

    assert local_size_x_for_shader_file(shader_file, pre) == 1024
    assert "layout(local_size_x = 1024" in source
    assert "const uint REDUCTION_WIDTH = 64u;" in source
    assert "const uint ROWS_PER_REDUCTION_BATCH = 16u;" in source
    assert "uint logical_lane = lane % REDUCTION_WIDTH;" in source
    assert "uint row_slot = lane / REDUCTION_WIDTH;" in source
    assert "for (uint row_base = 0u; row_base < MIX_COUNT;" in source
    assert "for (uint column = logical_lane; column < HYPER_SIZE;" in source
    assert "column += REDUCTION_WIDTH" in source
    assert "reduction[lane] += reduction[lane + stride];" in source


def test_hyper_connection_post_dispatch_covers_every_output_stream_word() -> None:
    circuit, tensor_index = _circuit()
    optimized = optimize_circuit_for_vulkan(circuit)
    pre = next(
        node for node in optimized["nodes"] if node["op"] == "hyper_connection_pre"
    )
    post = next(
        node for node in optimized["nodes"] if node["op"] == "hyper_connection_post"
    )

    # The pre kernel deliberately uses one cooperative workgroup and loops over
    # the complete hyper-state internally. The post kernel maps one invocation
    # to one packed BF16 output word, so its dispatch must cover M * H / 2
    # words. A 4x4096 hyper-state therefore requires 8192 invocations, or 128
    # workgroups at the kernel's fixed local size of 64.
    dimensions = {"hidden_size": 4096}
    assert (
        workgroup_count_x_for_node(
            optimized,
            pre,
            tensor_index,
            dimensions=dimensions,
        )
        == 1
    )
    assert (
        workgroup_count_x_for_node(
            optimized,
            post,
            tensor_index,
            dimensions=dimensions,
        )
        == 128
    )


def test_hyper_connection_post_reads_source_to_output_coefficients(
    tmp_path: Path,
) -> None:
    circuit, tensor_index = _circuit()
    optimized = optimize_circuit_for_vulkan(circuit)
    nodes = [
        node
        for node in optimized["nodes"]
        if node["op"] in {"hyper_connection_post_pre", "hyper_connection_post"}
    ]
    shader_files = {
        shader_file_for_node(
            optimized,
            node,
            tensor_index,
            {"hidden_size": 8},
        )
        for node in nodes
    }
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    fused = (
        tmp_path
        / "hyper_connection_post_pre_m4_h8_i20_neps1e-06_heps1e-06.comp"
    ).read_text()
    post = (tmp_path / "hyper_connection_post_m4_h8.comp").read_text()

    # DeepSeek mHC consumes comb transposed: output k is the sum over source j
    # of comb[j, k] * residual[j]. A non-symmetric doubly-stochastic matrix is
    # valid, so reading comb[k, j] is a behavior-changing error.
    assert "source_stream * MULTIPLICITY + stream_index" in fused
    assert "source_stream * MULTIPLICITY + output_stream" in post


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda circuit, _tensor_index: circuit["nodes"][0]["attrs"].update(
                normalization_epsilon=0.0
            ),
            "invalid contract",
        ),
            (
                lambda circuit, _tensor_index: next(
                    node
                    for node in circuit["nodes"]
                    if node["op"] == "hyper_connection_pre"
                )["attrs"].update(output_element_bytes=[2, 2, 2]),
                "invalid contract",
            ),
        (
            lambda _circuit, tensor_index: tensor_index["tensors"][
                "attention.function"
            ].update(shape=[24, 31]),
                "incompatible parameters",
        ),
    ],
)
def test_rejects_malformed_hyper_connection_contracts(mutation, message: str) -> None:
    circuit, tensor_index = _circuit()
    optimized = optimize_circuit_for_vulkan(circuit)
    mutation(optimized, tensor_index)
    node = next(
        node for node in optimized["nodes"] if node["op"] == "hyper_connection_pre"
    )

    with pytest.raises(ModelCompileError, match=message):
        shader_file_for_node(optimized, node, tensor_index, {"hidden_size": 8})
