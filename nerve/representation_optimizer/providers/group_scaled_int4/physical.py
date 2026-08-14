from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_manifest import component_kernel_spec
from nerve.model_package_shader_selection import (
    local_size_x_for_shader_file,
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.model_package_spirv_requirements import (
    spirv_vulkan_requirements_from_payloads,
)
from nerve.physical_execution_contracts import (
    build_kernel_physical_execution_contracts,
)
from nerve.representation_optimizer.providers.group_scaled_int4.artifacts import (
    kernel_artifact_path,
)
from nerve.representation_optimizer.providers.group_scaled_int4.discovery import (
    GroupScaledInt4Opportunity,
)


@dataclass(frozen=True)
class ShaderArtifact:
    artifact_path: str
    template_name: str


@dataclass(frozen=True)
class PreparedInt4Region:
    circuit: Json
    tensor_index: Json
    source_node: Json
    replacement_node: Json
    source_kernel: Json
    replacement_kernel: Json
    source_parameter_refs: Json
    replacement_parameter_refs: Json
    shader_artifacts: tuple[ShaderArtifact, ...]


def candidate_tensor_metadata(opportunity: GroupScaledInt4Opportunity) -> Json:
    return {
        opportunity.candidate_weight_name: {
            "dtype": "I32",
            "shape": list(opportunity.packed_shape),
            "logical_shape": [
                opportunity.output_features,
                opportunity.input_features,
            ],
            "byte_count": (
                opportunity.output_features * opportunity.input_features // 2
            ),
            "layout": "row_major",
            "quantization": {
                "format": "compressed_tensors_pack_quantized",
                "bits": 4,
                "group_size": opportunity.group_size,
                "symmetric": True,
                "signed_offset": 8,
                "scales": opportunity.candidate_scale_name,
            },
        },
        opportunity.candidate_scale_name: {
            "dtype": "BF16",
            "shape": list(opportunity.scale_shape),
            "byte_count": (
                opportunity.output_features
                * (opportunity.input_features // opportunity.group_size)
                * 2
            ),
            "layout": "row_major",
        },
    }


def prepare_group_scaled_int4_component_from_documents(
    *,
    opportunity: GroupScaledInt4Opportunity,
    manifest: Json,
    tensor_index: Json,
) -> PreparedInt4Region:
    component = _unique(
        manifest.get("circuit_graph", {}).get("components"),
        "component_id",
        opportunity.component_id,
    )
    execution = _unique(
        manifest.get("component_executions"),
        "component_id",
        opportunity.component_id,
    )
    source_node = deepcopy(
        _unique(
            component.get("circuit", {}).get("nodes"),
            "id",
            opportunity.node_id,
        )
    )
    source_kernel = deepcopy(
        _unique(
            execution.get("kernels"),
            "node_id",
            opportunity.node_id,
        )
    )
    circuit = deepcopy(component["circuit"])
    candidate_tensors = candidate_tensor_metadata(opportunity)
    transformed_tensor_index = deepcopy(tensor_index)
    transformed_tensor_index.setdefault("tensors", {}).update(
        deepcopy(candidate_tensors)
    )

    source_ref_id = opportunity.source_weight_ref_id
    source_ref = circuit["parameters"]["refs"].get(source_ref_id)
    if source_ref != opportunity.source_weight_ref:
        raise ModelCompileError(
            f"component {opportunity.component_id!r} source weight binding drifted"
        )
    uses = [
        str(node["id"])
        for node in circuit["nodes"]
        if source_ref_id in node.get("params", [])
    ]
    if uses != [opportunity.node_id]:
        raise ModelCompileError(
            f"component {opportunity.component_id!r} INT4 source parameter "
            "is not private"
        )
    source_parameter_refs = {source_ref_id: deepcopy(source_ref)}
    replacement_parameter_refs = {
        opportunity.replacement_weight_ref_id: {
            "tensor": opportunity.candidate_weight_name,
            "role": (
                f"{opportunity.component_id}.{opportunity.node_id}."
                "group_scaled_int4_weight"
            ),
        },
        opportunity.replacement_scale_ref_id: {
            "tensor": opportunity.candidate_scale_name,
            "role": (
                f"{opportunity.component_id}.{opportunity.node_id}."
                "group_scaled_int4_scale"
            ),
        },
    }
    del circuit["parameters"]["refs"][source_ref_id]
    circuit["parameters"]["refs"].update(deepcopy(replacement_parameter_refs))
    replacement_node = deepcopy(source_node)
    replacement_node["params"] = [
        opportunity.replacement_weight_ref_id,
        opportunity.replacement_scale_ref_id,
    ]
    replacement_node["attrs"] = {
        **replacement_node.get("attrs", {}),
        "parameter_representation": {
            "kind": "group_scaled_signed_int4",
            "source_tensor": opportunity.source_weight.tensor_name,
            "group_size": opportunity.group_size,
            "scale_dtype": "BF16",
            "packing": "eight_input_columns_per_i32",
        },
    }
    circuit["nodes"] = [
        deepcopy(replacement_node) if node.get("id") == opportunity.node_id else node
        for node in circuit["nodes"]
    ]
    shader_file = shader_file_for_node(
        circuit,
        replacement_node,
        transformed_tensor_index,
        {"hidden_size": opportunity.input_features},
        compiler_target={"devices": [opportunity.compiler_device]},
    )
    local_size_x = local_size_x_for_shader_file(shader_file, replacement_node)
    workgroup_count_x = workgroup_count_x_for_node(
        circuit,
        replacement_node,
        transformed_tensor_index,
        dimensions={"hidden_size": opportunity.input_features},
    )
    replacement_kernel = component_kernel_spec(
        execution_index=int(source_kernel["execution_index"]),
        node=replacement_node,
        circuit=circuit,
        shader_file=shader_file,
        local_size_x=local_size_x,
        workgroup_count_x=workgroup_count_x,
        tensor_index=transformed_tensor_index,
    )
    shader_artifacts: dict[str, ShaderArtifact] = {}
    _rewrite_shader_paths(replacement_kernel, shader_artifacts)
    return PreparedInt4Region(
        circuit=circuit,
        tensor_index=transformed_tensor_index,
        source_node=source_node,
        replacement_node=replacement_node,
        source_kernel=source_kernel,
        replacement_kernel=replacement_kernel,
        source_parameter_refs=source_parameter_refs,
        replacement_parameter_refs=replacement_parameter_refs,
        shader_artifacts=tuple(
            shader_artifacts[path] for path in sorted(shader_artifacts)
        ),
    )


def finalize_group_scaled_int4_kernel(
    source_kernel: Json,
    *,
    prepared: PreparedInt4Region,
    artifact_payloads: dict[str, bytes],
) -> Json:
    kernel = deepcopy(source_kernel)
    for implementation in kernel.get("batch_implementations", []):
        paths = {
            str(stage["shader_path"])
            for stage in implementation.get("stages", [])
        }
        _require_artifacts(paths, artifact_payloads)
        features, subgroup_operations = spirv_vulkan_requirements_from_payloads(
            {path: artifact_payloads[path] for path in paths}
        )
        requirements = implementation["device_requirements"]
        requirements["vulkan_features"] = features
        requirements["subgroup_operations"] = subgroup_operations
    paths = {str(kernel["shader_path"])}
    paths.update(
        str(implementation["shader_path"])
        for implementation in kernel.get("physical_implementations", [])
    )
    _require_artifacts(paths, artifact_payloads)
    kernel["physical_execution_contracts"] = (
        build_kernel_physical_execution_contracts(
            node=prepared.replacement_node,
            circuit=prepared.circuit,
            tensor_index=prepared.tensor_index,
            kernel=kernel,
            package_dir=Path("."),
            artifact_payloads=artifact_payloads,
        )
    )
    kernel.pop("physical_implementations", None)
    return kernel


def _rewrite_shader_paths(
    kernel: Json,
    artifacts: dict[str, ShaderArtifact],
) -> None:
    def rewrite(record: Json) -> None:
        source_path = str(record.get("shader_path", ""))
        if not source_path.startswith("shaders/") or not source_path.endswith(".comp"):
            raise ModelCompileError(
                f"group-scaled INT4 kernel has an invalid source shader {source_path!r}"
            )
        template_name = source_path.removeprefix("shaders/")
        artifact_path = kernel_artifact_path(template_name)
        artifact = ShaderArtifact(artifact_path, template_name)
        existing = artifacts.get(artifact_path)
        if existing is not None and existing != artifact:
            raise ModelCompileError(
                f"group-scaled INT4 shader artifact {artifact_path!r} is ambiguous"
            )
        artifacts[artifact_path] = artifact
        record["shader_path"] = artifact_path

    rewrite(kernel)
    for implementation in kernel.get("batch_implementations", []):
        for stage in implementation.get("stages", []):
            rewrite(stage)
    for implementation in kernel.get("physical_implementations", []):
        rewrite(implementation)


def _require_artifacts(paths: set[str], payloads: dict[str, bytes]) -> None:
    missing = sorted(paths.difference(payloads))
    if missing:
        raise ModelCompileError(
            f"group-scaled INT4 physical contracts lack shader artifacts {missing}"
        )


def _unique(records: object, field: str, value: str) -> Json:
    if not isinstance(records, list):
        raise ModelCompileError(f"group-scaled INT4 source {field} records are missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get(field) == value
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            f"group-scaled INT4 source has no unique {field}={value!r}"
        )
    return matches[0]


__all__ = [
    "PreparedInt4Region",
    "ShaderArtifact",
    "candidate_tensor_metadata",
    "finalize_group_scaled_int4_kernel",
    "prepare_group_scaled_int4_component_from_documents",
]
