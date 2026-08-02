import pytest

from nerve.compilation import ModelCompileError
from nerve.model_package_assets import (
    stream_control_binding_from_artifact_path,
    stream_control_binding_for_node,
)
from nerve.model_package_validation import (
    validate_kernel_stream_control_contract,
)


def circuit_and_node():
    node = {
        "id": "temporal",
        "op": "architecture_defined_operation",
        "inputs": ["frame"],
        "outputs": ["updated"],
        "params": [],
        "state_reads": ["memory"],
        "state_writes": ["memory"],
        "attrs": {},
    }
    return {"nodes": [node]}, node


def test_shader_artifact_exposes_its_stream_control_binding():
    assert stream_control_binding_from_artifact_path("temporal__sc6.comp") == 6
    assert stream_control_binding_from_artifact_path("temporal__sc6.spv") == 6


def test_non_temporal_shader_has_no_stream_control_binding():
    assert stream_control_binding_from_artifact_path("plain.comp") is None


def test_package_validation_rejects_an_unbound_stream_control_shader():
    circuit, node = circuit_and_node()
    binding = stream_control_binding_for_node(circuit, node)
    kernel = {
        "node_id": node["id"],
        "shader_path": f"shaders/temporal__sc{binding}.spv",
        "stream_control_binding": None,
    }

    with pytest.raises(ModelCompileError, match="contract disagrees"):
        validate_kernel_stream_control_contract(circuit, node, kernel)


def test_package_validation_requires_an_explicit_stream_control_contract():
    circuit, node = circuit_and_node()
    kernel = {
        "node_id": node["id"],
        "shader_path": "shaders/plain.spv",
    }

    with pytest.raises(ModelCompileError, match="no explicit stream-control binding contract"):
        validate_kernel_stream_control_contract(circuit, node, kernel)


@pytest.mark.parametrize("value", [False, True, -1, "6", [], {}])
def test_package_validation_rejects_an_invalid_stream_control_binding(value):
    circuit, node = circuit_and_node()
    kernel = {
        "node_id": node["id"],
        "shader_path": "shaders/plain.spv",
        "stream_control_binding": value,
    }

    with pytest.raises(ModelCompileError, match="invalid stream-control binding contract"):
        validate_kernel_stream_control_contract(circuit, node, kernel)


def test_package_validation_rejects_the_wrong_stream_control_binding():
    circuit, node = circuit_and_node()
    binding = stream_control_binding_for_node(circuit, node)
    kernel = {
        "node_id": node["id"],
        "shader_path": f"shaders/temporal__sc{binding + 1}.spv",
        "stream_control_binding": binding + 1,
    }

    with pytest.raises(ModelCompileError, match="expected"):
        validate_kernel_stream_control_contract(circuit, node, kernel)
