from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_manifest import component_kernel_spec
from nerve.model_package_shader_selection import (
    local_size_x_for_shader_file,
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.artifacts import (
    kernel_artifact_path,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.discovery import (
    ParallelProjectionFusionOpportunity,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.transformation import (
    FusedComponent,
    fuse_component,
)
from nerve.representation_optimizer.providers.types import ProviderContext


@dataclass(frozen=True)
class ShaderArtifact:
    artifact_path: str
    template_name: str


@dataclass(frozen=True)
class PreparedFusedComponent:
    transformed: FusedComponent
    source_kernels: tuple[Json, ...]
    replacement_kernels: tuple[Json, ...]
    shader_artifacts: tuple[ShaderArtifact, ...]


def prepare_fused_component(
    context: ProviderContext,
    opportunity: ParallelProjectionFusionOpportunity,
) -> PreparedFusedComponent:
    key = (
        "parallel_projection_fusion.prepared.v1:"
        f"{opportunity.component_id}:{opportunity.performance_signature}:"
        f"{','.join(opportunity.scope_ids)}:"
        f"{context.hardware_profile['capability_class']}"
    )
    return context.memoized(
        key,
        lambda: _prepare(context, opportunity),
    )  # type: ignore[return-value]


def _prepare(
    context: ProviderContext,
    opportunity: ParallelProjectionFusionOpportunity,
) -> PreparedFusedComponent:
    context.checkpoint()
    manifest = _source_json(context, opportunity.manifest_ref)
    tensor_index = _source_json(context, opportunity.tensor_index_ref)
    source_circuit = _source_json(context, opportunity.circuit_ref)
    return prepare_fused_component_from_documents(
        opportunity=opportunity,
        manifest=manifest,
        tensor_index=tensor_index,
        source_circuit=source_circuit,
    )


def prepare_fused_component_from_documents(
    *,
    opportunity: ParallelProjectionFusionOpportunity,
    manifest: Json,
    tensor_index: Json,
    source_circuit: Json,
) -> PreparedFusedComponent:
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
    transformed = fuse_component(
        opportunity=opportunity,
        source_circuit=source_circuit,
        compiled_circuit=component["circuit"],
        tensor_index=tensor_index,
    )
    source_ids = set(opportunity.region.source_node_ids)
    source_kernels = tuple(
        deepcopy(kernel)
        for kernel in execution["kernels"]
        if kernel.get("node_id") in source_ids
    )
    if {kernel["node_id"] for kernel in source_kernels} != source_ids:
        raise ModelCompileError(
            f"component {opportunity.component_id!r} fusion kernel source is incomplete"
        )
    source_execution_indices = {
        str(kernel["node_id"]): int(kernel["execution_index"])
        for kernel in source_kernels
    }
    replacement_kernels = [
        deepcopy(kernel)
        for kernel in source_kernels
        if kernel["node_id"] == opportunity.region.quantizer_node_id
    ]
    if len(replacement_kernels) != 1:
        raise ModelCompileError(
            f"component {opportunity.component_id!r} has no unique shared quantizer kernel"
        )
    shader_artifacts: dict[str, ShaderArtifact] = {}
    for node in transformed.replacement_nodes:
        if node["id"] == opportunity.region.quantizer_node_id:
            continue
        shader_file = shader_file_for_node(
            transformed.circuit,
            node,
            tensor_index,
            {"hidden_size": opportunity.hidden_size},
            compiler_target={"devices": [opportunity.compiler_device]},
        )
        local_size_x = local_size_x_for_shader_file(shader_file, node)
        workgroup_count_x = workgroup_count_x_for_node(
            transformed.circuit,
            node,
            tensor_index,
            dimensions={"hidden_size": opportunity.hidden_size},
        )
        kernel = component_kernel_spec(
            execution_index=min(
                source_execution_indices[node_id]
                for node_id in opportunity.region.linear_node_ids
            ),
            node=node,
            circuit=transformed.circuit,
            shader_file=shader_file,
            local_size_x=local_size_x,
            workgroup_count_x=workgroup_count_x,
            tensor_index=tensor_index,
        )
        _rewrite_shader_paths(kernel, shader_artifacts)
        replacement_kernels.append(kernel)
    replacement_kernels.sort(key=lambda kernel: int(kernel["execution_index"]))
    return PreparedFusedComponent(
        transformed=transformed,
        source_kernels=source_kernels,
        replacement_kernels=tuple(replacement_kernels),
        shader_artifacts=tuple(
            shader_artifacts[path] for path in sorted(shader_artifacts)
        ),
    )


def _rewrite_shader_paths(
    kernel: Json,
    artifacts: dict[str, ShaderArtifact],
) -> None:
    def rewrite(record: Json) -> None:
        source_path = str(record.get("shader_path", ""))
        if not source_path.startswith("shaders/") or not source_path.endswith(".comp"):
            raise ModelCompileError(
                f"fused parallel projection kernel has an invalid source shader {source_path!r}"
            )
        template_name = source_path.removeprefix("shaders/")
        artifact_path = kernel_artifact_path(template_name)
        existing = artifacts.get(artifact_path)
        artifact = ShaderArtifact(artifact_path, template_name)
        if existing is not None and existing != artifact:
            raise ModelCompileError(
                f"fused parallel projection shader artifact {artifact_path!r} is ambiguous"
            )
        artifacts[artifact_path] = artifact
        record["shader_path"] = artifact_path

    rewrite(kernel)
    for implementation in kernel.get("batch_implementations", []):
        for stage in implementation.get("stages", []):
            rewrite(stage)
    for implementation in kernel.get("physical_implementations", []):
        rewrite(implementation)


def _source_json(context: ProviderContext, path: str) -> Json:
    key = f"parallel_projection_fusion.source_json.v1:{path}"

    def load() -> Json:
        try:
            value = json.loads(context.source_artifacts.read_path(path))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelCompileError(f"{path} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ModelCompileError(f"{path} must contain a JSON object")
        return value

    return context.memoized(key, load)  # type: ignore[return-value]


def _unique(records: object, field: str, value: str) -> Json:
    if not isinstance(records, list):
        raise ModelCompileError(f"fusion source {field} records are missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get(field) == value
    ]
    if len(matches) != 1:
        raise ModelCompileError(f"fusion source has no unique {field}={value!r}")
    return matches[0]
