from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from nerve.model_package_physical_kernels import (
    local_output_shard_intermediates_for_node,
)


Json = dict[str, Any]

PHYSICAL_EXECUTION_CONTRACT_SCHEMA = "nerve.physical_execution_contract.v1"

ExecutionPhase = Literal["decode", "prefill"]
ExecutionStrategy = Literal[
    "single_device",
    "tensor_parallel",
    "expert_parallel",
    "tensor_parallel_expert",
]
ExecutionForm = Literal[
    "local",
    "replicated_input_partitioned_output",
    "partitioned_input_partial_output",
    "whole_expert_ownership",
]
ParameterPartitionKind = Literal["contiguous", "block_cyclic", "expert_range"]
WorkgroupXMapping = Literal["proportional", "repeated"]
PartitionOrigin = Literal["local_zero", "push_constant_u32"]
InputDistribution = Literal["replicated", "sharded", "routed", "local"]
OutputCollection = Literal[
    "local", "concatenated", "reduced", "routed", "retained"
]
ReductionOperation = Literal["sum_f32"]
ResourceKind = Literal[
    "persistent_parameter",
    "transient",
    "kv_state",
    "recurrent_state",
    "lazy_resource",
    "control",
]
ResidencyRequirement = Literal["permanent", "transaction", "demand"]
ResourceAccess = Literal["read", "write", "read_write"]
EquivalenceKind = Literal["bit_exact", "absolute_relative_tolerance"]


class ArtifactIdentity(TypedDict):
    path: str
    sha256: str
    entry_point: str


class PhysicalFormats(TypedDict):
    storage: str
    compute: str
    accumulation: str


class ExecutionGeometry(TypedDict):
    shape_class: str
    dimensions: dict[str, int]
    dynamic_dimensions: list[str]


class ParameterPartition(TypedDict):
    binding: int
    resource: str
    dimension: int
    kind: ParameterPartitionKind
    alignment_elements: int
    logical_elements_per_index: int


class PartitionExtent(TypedDict):
    dimension_name: str
    elements: int
    alignment_elements: int


class PartitionLaunch(TypedDict, total=False):
    workgroup_x: WorkgroupXMapping
    origin: PartitionOrigin
    origin_push_constant: str
    count_push_constant: str


class InputContract(TypedDict, total=False):
    binding: int
    distribution: InputDistribution
    dimension: int
    alignment_elements: int


class OutputContract(TypedDict, total=False):
    binding: int
    collection: OutputCollection
    dimension: int
    alignment_elements: int
    reduction: ReductionContract


class ReductionContract(TypedDict):
    operation: ReductionOperation
    dimension_name: str
    finalization: ReductionFinalization


class ReductionFinalization(TypedDict, total=False):
    kind: Literal["store_f32", "add_bf16_residual_to_bf16"]
    residual_binding: int


class LocalIntermediateContract(TypedDict):
    signal: str
    producer_binding: int
    consumer_binding: int
    format: str


class ResourceRequirement(TypedDict, total=False):
    resource: str
    kind: ResourceKind
    residency: ResidencyRequirement
    access: ResourceAccess
    binding: int
    atomic_group: str


class EquivalenceRequirement(TypedDict, total=False):
    output: EquivalenceKind
    state: EquivalenceKind
    absolute_tolerance: float
    relative_tolerance: float


class PhysicalExecutionContract(TypedDict, total=False):
    schema: str
    contract_id: str
    operation_family: str
    region_family: str
    member_node_ids: list[str]
    artifacts: list[ArtifactIdentity]
    implementation_digest: str
    phases: list[ExecutionPhase]
    formats: PhysicalFormats
    geometry: ExecutionGeometry
    strategy: ExecutionStrategy
    execution_form: ExecutionForm
    partition_extent: PartitionExtent
    partition_launch: PartitionLaunch
    parameter_partitions: list[ParameterPartition]
    inputs: list[InputContract]
    outputs: list[OutputContract]
    local_intermediates: list[LocalIntermediateContract]
    resources: list[ResourceRequirement]
    equivalence: EquivalenceRequirement


_PHASES = {"decode", "prefill"}
_STRATEGIES = {
    "single_device",
    "tensor_parallel",
    "expert_parallel",
    "tensor_parallel_expert",
}
_EXECUTION_FORMS = {
    "local",
    "replicated_input_partitioned_output",
    "partitioned_input_partial_output",
    "whole_expert_ownership",
}
_PARTITION_KINDS = {"contiguous", "block_cyclic", "expert_range"}
_WORKGROUP_X_MAPPINGS = {"proportional", "repeated"}
_PARTITION_ORIGINS = {"local_zero", "push_constant_u32"}
_INPUT_DISTRIBUTIONS = {"replicated", "sharded", "routed", "local"}
_OUTPUT_COLLECTIONS = {"local", "concatenated", "reduced", "routed", "retained"}
_REDUCTION_OPERATIONS = {"sum_f32"}
_RESOURCE_KINDS = {
    "persistent_parameter",
    "transient",
    "kv_state",
    "recurrent_state",
    "lazy_resource",
    "control",
}
_RESIDENCY_REQUIREMENTS = {"permanent", "transaction", "demand"}
_RESOURCE_ACCESSES = {"read", "write", "read_write"}
_EQUIVALENCE_KINDS = {"bit_exact", "absolute_relative_tolerance"}

_TOP_LEVEL_KEYS = {
    "schema",
    "contract_id",
    "operation_family",
    "region_family",
    "member_node_ids",
    "artifacts",
    "implementation_digest",
    "phases",
    "formats",
    "geometry",
    "strategy",
    "execution_form",
    "partition_extent",
    "partition_launch",
    "parameter_partitions",
    "inputs",
    "outputs",
    "local_intermediates",
    "resources",
    "equivalence",
}
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS - {
    "region_family",
    "local_intermediates",
    "partition_extent",
    "partition_launch",
}


class PhysicalExecutionContractError(ValueError):
    pass


def artifact_sha256(payload: bytes) -> str:
    return f"sha256:{sha256(payload).hexdigest()}"


def implementation_digest(
    *,
    artifacts: list[ArtifactIdentity],
    phases: list[ExecutionPhase],
    formats: PhysicalFormats,
    geometry: ExecutionGeometry,
    strategy: ExecutionStrategy,
    execution_form: ExecutionForm,
    partition_extent: PartitionExtent | None,
    partition_launch: PartitionLaunch | None,
    parameter_partitions: list[ParameterPartition],
    inputs: list[InputContract],
    outputs: list[OutputContract],
    local_intermediates: list[LocalIntermediateContract],
) -> str:
    return artifact_sha256(
        _canonical_json_bytes(
            {
                "artifacts": artifacts,
                "phases": phases,
                "formats": formats,
                "geometry": geometry,
                "strategy": strategy,
                "execution_form": execution_form,
                "partition_extent": partition_extent,
                "partition_launch": partition_launch,
                "parameter_partitions": parameter_partitions,
                "inputs": inputs,
                "outputs": outputs,
                "local_intermediates": local_intermediates,
            }
        )
    )


def build_kernel_physical_execution_contracts(
    *,
    node: Json,
    circuit: Json,
    tensor_index: Json,
    kernel: Json,
    package_dir: Path,
) -> list[PhysicalExecutionContract]:
    """Describe compiled implementations without assigning physical devices."""
    node = deepcopy(node)
    node["semantic_module_ids"] = list(kernel.get("semantic_module_ids", []))
    node["_physical_contract_member_node_ids"] = list(
        dict.fromkeys(
            [
                _non_empty_string(node.get("id"), "node.id"),
                *[
                    _non_empty_string(source_node_id, "kernel.source_node_ids")
                    for source_node_id in kernel.get("source_node_ids", [])
                ],
            ]
        )
    )
    scalar_artifacts = [_artifact_identity(package_dir, kernel["shader_path"])]
    scalar_geometry = _kernel_geometry(
        node,
        circuit,
        tensor_index,
        local_size_x=int(kernel["local_size_x"]),
        workgroup_count_x=int(kernel["workgroup_count_x"]),
    )
    scalar_formats = _kernel_formats(node, circuit, tensor_index, scalar_artifacts)
    contracts = [
        _build_contract(
            node=node,
            circuit=circuit,
            tensor_index=tensor_index,
            artifacts=scalar_artifacts,
            phases=["decode"],
            formats=scalar_formats,
            geometry=scalar_geometry,
            strategy="single_device",
            execution_form="local",
            partition_extent=None,
            partition_launch=None,
            parameter_partitions=[],
            inputs=_local_inputs(node),
            outputs=_local_outputs(node),
            local_intermediates=[],
        )
    ]
    distributed = _distributed_kernel_contract(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        artifacts=scalar_artifacts,
        phases=["decode"],
        formats=scalar_formats,
        geometry=scalar_geometry,
        workgroup_count_x=int(kernel["workgroup_count_x"]),
        local_intermediates=local_output_shard_intermediates_for_node(
            circuit, node, tensor_index
        ),
    )
    if distributed is not None:
        contracts.append(distributed)
    contracts.extend(
        _physical_implementation_contract(
            node=node,
            circuit=circuit,
            tensor_index=tensor_index,
            implementation=implementation,
            package_dir=package_dir,
        )
        for implementation in kernel.get("physical_implementations", [])
    )

    for implementation in kernel.get("batch_implementations", []):
        stages = implementation.get("stages", [])
        if not stages:
            _invalid(
                f"batch implementation for {node.get('id')!r} has no artifact stages"
            )
        artifacts = [
            _artifact_identity(package_dir, stage["shader_path"]) for stage in stages
        ]
        phases = _execution_domain_phases(implementation["execution_domain"])
        lane_tile_width = _positive_int(
            implementation["lane_tile_width"], "batch lane_tile_width"
        )
        geometry = _kernel_geometry(
            node,
            circuit,
            tensor_index,
            local_size_x=max(int(stage["local_size_x"]) for stage in stages),
            workgroup_count_x=max(int(stage["workgroup_count_x"]) for stage in stages),
            batch_width=lane_tile_width,
        )
        batch_formats = _kernel_formats(node, circuit, tensor_index, artifacts)
        contracts.append(
            _build_contract(
                node=node,
                circuit=circuit,
                tensor_index=tensor_index,
                artifacts=artifacts,
                phases=phases,
                formats=batch_formats,
                geometry=geometry,
                strategy="single_device",
                execution_form="local",
                partition_extent=None,
                partition_launch=None,
                parameter_partitions=[],
                inputs=_local_inputs(node),
                outputs=_local_outputs(node),
                local_intermediates=[],
            )
        )
        distributed_phases = [phase for phase in phases if phase == "prefill"]
        if len(stages) == 1 and distributed_phases:
            distributed_batch = _distributed_kernel_contract(
                node=node,
                circuit=circuit,
                tensor_index=tensor_index,
                artifacts=artifacts,
                phases=distributed_phases,
                formats=batch_formats,
                geometry=geometry,
                workgroup_count_x=int(stages[0]["workgroup_count_x"]),
                local_intermediates=local_output_shard_intermediates_for_node(
                    circuit, node, tensor_index
                ),
            )
            if distributed_batch is not None:
                contracts.append(distributed_batch)

    by_id = {contract["contract_id"]: contract for contract in contracts}
    if len(by_id) != len(contracts):
        _invalid(f"kernel {node.get('id')!r} emitted duplicate physical contracts")
    return list(by_id.values())


def _build_contract(
    *,
    node: Json,
    circuit: Json,
    tensor_index: Json,
    artifacts: list[ArtifactIdentity],
    phases: list[ExecutionPhase],
    formats: PhysicalFormats,
    geometry: ExecutionGeometry,
    strategy: ExecutionStrategy,
    execution_form: ExecutionForm,
    partition_extent: PartitionExtent | None,
    partition_launch: PartitionLaunch | None,
    parameter_partitions: list[ParameterPartition],
    inputs: list[InputContract],
    outputs: list[OutputContract],
    local_intermediates: list[LocalIntermediateContract],
    resources: list[ResourceRequirement] | None = None,
    equivalence: EquivalenceRequirement | None = None,
) -> PhysicalExecutionContract:
    resources = (
        _kernel_resources(node, circuit, tensor_index)
        if resources is None
        else deepcopy(resources)
    )
    contract: Json = {
        "schema": PHYSICAL_EXECUTION_CONTRACT_SCHEMA,
        "operation_family": _non_empty_string(node.get("op"), "node.op"),
        "member_node_ids": node["_physical_contract_member_node_ids"],
        "artifacts": artifacts,
        "implementation_digest": implementation_digest(
            artifacts=artifacts,
            phases=phases,
            formats=formats,
            geometry=geometry,
            strategy=strategy,
            execution_form=execution_form,
            partition_extent=partition_extent,
            partition_launch=partition_launch,
            parameter_partitions=parameter_partitions,
            inputs=inputs,
            outputs=outputs,
            local_intermediates=local_intermediates,
        ),
        "phases": phases,
        "formats": formats,
        "geometry": geometry,
        "strategy": strategy,
        "execution_form": execution_form,
        "parameter_partitions": parameter_partitions,
        "inputs": inputs,
        "outputs": outputs,
        "local_intermediates": local_intermediates,
        "resources": resources,
        "equivalence": deepcopy(equivalence)
        if equivalence is not None
        else {"output": "bit_exact", "state": "bit_exact"},
    }
    if partition_extent is not None:
        contract["partition_extent"] = partition_extent
    if partition_launch is not None:
        contract["partition_launch"] = partition_launch
    semantic_modules = node.get("semantic_module_ids")
    if isinstance(semantic_modules, list) and semantic_modules:
        contract["region_family"] = "+".join(map(str, semantic_modules))
    return seal_physical_execution_contract(contract)


def _physical_implementation_contract(
    *,
    node: Json,
    circuit: Json,
    tensor_index: Json,
    implementation: Json,
    package_dir: Path,
) -> PhysicalExecutionContract:
    local_size_x = _positive_int(
        implementation.get("local_size_x"), "physical implementation local_size_x"
    )
    workgroup_count_x = _positive_int(
        implementation.get("workgroup_count_x"),
        "physical implementation workgroup_count_x",
    )
    dimensions = {
        "local_size_x": local_size_x,
        "workgroup_count_x": workgroup_count_x,
        **{
            _non_empty_string(name, "physical implementation dimension name"): _positive_int(
                value, f"physical implementation dimension {name}"
            )
            for name, value in _mapping(
                implementation.get("geometry_dimensions"),
                "physical implementation geometry_dimensions",
            ).items()
        },
    }
    geometry: ExecutionGeometry = {
        "shape_class": (
            f"shape:{sha256(_canonical_json_bytes(dimensions)).hexdigest()[:24]}"
        ),
        "dimensions": dimensions,
        "dynamic_dimensions": [],
    }
    return _build_contract(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        artifacts=[_artifact_identity(package_dir, implementation["shader_path"])],
        phases=list(implementation["phases"]),
        formats=deepcopy(implementation["formats"]),
        geometry=geometry,
        strategy=implementation["strategy"],
        execution_form=implementation["execution_form"],
        partition_extent=deepcopy(implementation.get("partition_extent")),
        partition_launch=deepcopy(implementation.get("partition_launch")),
        parameter_partitions=deepcopy(implementation["parameter_partitions"]),
        inputs=deepcopy(implementation["inputs"]),
        outputs=deepcopy(implementation["outputs"]),
        local_intermediates=deepcopy(implementation.get("local_intermediates", [])),
        resources=deepcopy(implementation["resources"]),
        equivalence=deepcopy(implementation["equivalence"]),
    )


def _distributed_kernel_contract(
    *,
    node: Json,
    circuit: Json,
    tensor_index: Json,
    artifacts: list[ArtifactIdentity],
    phases: list[ExecutionPhase],
    formats: PhysicalFormats,
    geometry: ExecutionGeometry,
    workgroup_count_x: int,
    local_intermediates: list[LocalIntermediateContract],
) -> PhysicalExecutionContract | None:
    op = node.get("op")
    parameter_metadata = _node_parameter_metadata(node, circuit, tensor_index)
    if not parameter_metadata:
        return None
    parameter_bindings = _direct_parameter_bindings(node)
    input_count = len(node.get("inputs", []))
    output_count = len(node.get("outputs", []))
    if output_count != 1:
        return None
    output_binding = input_count
    output_rows = int(parameter_metadata[0][1].get("shape", [0])[0])
    if output_rows <= 0 or workgroup_count_x <= 0:
        return None

    if op == "parallel_linear_silu_multiply":
        if output_rows % workgroup_count_x:
            return None
        artifact_alignment = output_rows // workgroup_count_x
        dtypes = [metadata.get("dtype") for _, metadata in parameter_metadata]
        if dtypes == ["BF16", "BF16"]:
            partition_layouts = [
                (artifact_alignment, 1),
                (artifact_alignment, 1),
            ]
        elif dtypes == ["F8_E4M3", "BF16", "F8_E4M3", "BF16"]:
            block_rows = _block_row_alignment(parameter_metadata)
            partition_layouts = [
                (max(artifact_alignment, block_rows), 1),
                (max(1, artifact_alignment // block_rows), block_rows),
                (max(artifact_alignment, block_rows), 1),
                (max(1, artifact_alignment // block_rows), block_rows),
            ]
        else:
            return None
        logical_alignment = max(
            alignment * logical_elements_per_index
            for alignment, logical_elements_per_index in partition_layouts
        )
        return _build_contract(
            node=node,
            circuit=circuit,
            tensor_index=tensor_index,
            artifacts=artifacts,
            phases=phases,
            formats=formats,
            geometry=geometry,
            strategy="tensor_parallel",
            execution_form="replicated_input_partitioned_output",
            partition_extent={
                "dimension_name": "parameter_0_dimension_0",
                "elements": output_rows,
                "alignment_elements": logical_alignment,
            },
            partition_launch={"workgroup_x": "proportional", "origin": "local_zero"},
            parameter_partitions=[
                {
                    "binding": binding,
                    "resource": tensor,
                    "dimension": 0,
                    "kind": "contiguous",
                    "alignment_elements": alignment,
                    "logical_elements_per_index": logical_elements_per_index,
                }
                for binding, (tensor, _), (alignment, logical_elements_per_index) in zip(
                    parameter_bindings,
                    parameter_metadata,
                    partition_layouts,
                    strict=True,
                )
            ],
            inputs=[
                {"binding": binding, "distribution": "replicated"}
                for binding in range(input_count)
            ],
            outputs=[
                {
                    "binding": output_binding,
                    "collection": "concatenated",
                    "dimension": 0,
                    "alignment_elements": artifact_alignment,
                }
            ],
            local_intermediates=local_intermediates,
        )

    if op == "linear_residual":
        if output_rows % workgroup_count_x:
            return None
        artifact_alignment = output_rows // workgroup_count_x
        dtypes = [metadata.get("dtype") for _, metadata in parameter_metadata]
        if dtypes != ["F8_E4M3", "BF16"] or input_count != 3:
            return None
        block_rows = _block_row_alignment(parameter_metadata)
        return _build_contract(
            node=node,
            circuit=circuit,
            tensor_index=tensor_index,
            artifacts=artifacts,
            phases=phases,
            formats=formats,
            geometry=geometry,
            strategy="tensor_parallel",
            execution_form="replicated_input_partitioned_output",
            partition_extent={
                "dimension_name": "parameter_0_dimension_0",
                "elements": output_rows,
                "alignment_elements": max(artifact_alignment, block_rows),
            },
            partition_launch={"workgroup_x": "proportional", "origin": "local_zero"},
            parameter_partitions=[
                {
                    "binding": parameter_bindings[0],
                    "resource": parameter_metadata[0][0],
                    "dimension": 0,
                    "kind": "contiguous",
                    "alignment_elements": max(artifact_alignment, block_rows),
                    "logical_elements_per_index": 1,
                },
                {
                    "binding": parameter_bindings[1],
                    "resource": parameter_metadata[1][0],
                    "dimension": 0,
                    "kind": "contiguous",
                    "alignment_elements": max(1, artifact_alignment // block_rows),
                    "logical_elements_per_index": block_rows,
                },
            ],
            inputs=[
                {"binding": 0, "distribution": "replicated"},
                {"binding": 1, "distribution": "replicated"},
                {
                    "binding": 2,
                    "distribution": "sharded",
                    "dimension": 0,
                    "alignment_elements": artifact_alignment,
                },
            ],
            outputs=[
                {
                    "binding": output_binding,
                    "collection": "concatenated",
                    "dimension": 0,
                    "alignment_elements": artifact_alignment,
                }
            ],
            local_intermediates=[],
        )

    if op in {"sparse_moe_gate_up", "sparse_moe_down"}:
        expert_count = int(parameter_metadata[0][1].get("shape", [0])[0])
        if expert_count <= 0 or any(
            int(metadata.get("shape", [0])[0]) != expert_count
            for _, metadata in parameter_metadata
        ):
            return None
        inputs: list[InputContract] = [
            {"binding": 0, "distribution": "replicated"}
        ]
        inputs.extend(
            {
                "binding": binding,
                "distribution": "routed",
                "dimension": 0,
                "alignment_elements": 1,
            }
            for binding in range(1, input_count)
        )
        return _build_contract(
            node=node,
            circuit=circuit,
            tensor_index=tensor_index,
            artifacts=artifacts,
            phases=phases,
            formats=formats,
            geometry=geometry,
            strategy="expert_parallel",
            execution_form="whole_expert_ownership",
            partition_extent={
                "dimension_name": "parameter_0_dimension_0",
                "elements": expert_count,
                "alignment_elements": 1,
            },
            partition_launch={
                "workgroup_x": "repeated",
                "origin": "push_constant_u32",
                "origin_push_constant": "expert_start",
                "count_push_constant": "expert_count",
            },
            parameter_partitions=[
                {
                    "binding": binding,
                    "resource": tensor,
                    "dimension": 0,
                    "kind": "expert_range",
                    "alignment_elements": 1,
                    "logical_elements_per_index": 1,
                }
                for binding, (tensor, _) in zip(
                    parameter_bindings, parameter_metadata, strict=True
                )
            ],
            inputs=inputs,
            outputs=[
                {
                    "binding": output_binding,
                    "collection": "routed",
                    "dimension": 0,
                    "alignment_elements": 1,
                }
            ],
            local_intermediates=[],
        )
    return None


def _artifact_identity(package_dir: Path, relative_path: object) -> ArtifactIdentity:
    path = _non_empty_string(relative_path, "artifact path")
    artifact_path = package_dir / path
    try:
        payload = artifact_path.read_bytes()
    except OSError as error:
        _invalid(f"could not read compiled artifact {path!r}: {error}")
    return {"path": path, "sha256": artifact_sha256(payload), "entry_point": "main"}


def _kernel_geometry(
    node: Json,
    circuit: Json,
    tensor_index: Json,
    *,
    local_size_x: int,
    workgroup_count_x: int,
    batch_width: int | None = None,
) -> ExecutionGeometry:
    dimensions: dict[str, int] = {
        "local_size_x": _positive_int(local_size_x, "local_size_x"),
        "workgroup_count_x": _positive_int(workgroup_count_x, "workgroup_count_x"),
    }
    for parameter_index, (_, metadata) in enumerate(
        _node_parameter_metadata(node, circuit, tensor_index)
    ):
        for dimension_index, dimension in enumerate(metadata.get("shape", [])):
            dimensions[f"parameter_{parameter_index}_dimension_{dimension_index}"] = (
                _positive_int(dimension, "parameter shape")
            )
    dynamic_dimensions: list[str] = []
    if batch_width is not None:
        dimensions["batch_width"] = _positive_int(batch_width, "batch_width")
        dynamic_dimensions.append("batch_width")
    shape_payload = _canonical_json_bytes(dimensions)
    return {
        "shape_class": f"shape:{sha256(shape_payload).hexdigest()[:24]}",
        "dimensions": dimensions,
        "dynamic_dimensions": dynamic_dimensions,
    }


def _kernel_formats(
    node: Json,
    circuit: Json,
    tensor_index: Json,
    artifacts: list[ArtifactIdentity],
) -> PhysicalFormats:
    storage_formats = sorted(
        {
            ":".join(
                filter(
                    None,
                    (str(metadata.get("dtype", "unknown")).lower(), metadata.get("layout")),
                )
            )
            for _, metadata in _node_parameter_metadata(node, circuit, tensor_index)
        }
    )
    storage = "+".join(storage_formats) if storage_formats else "activation_only"
    artifact_names = " ".join(artifact["path"].lower() for artifact in artifacts)
    compute = next(
        (
            format_name
            for token, format_name in (
                ("mxfp4_e2m1", "mxfp4_e2m1"),
                ("nvfp4", "nvfp4"),
                ("int4", "int4"),
                ("fp8_e4m3", "fp8_e4m3"),
                ("q8_0", "int8"),
                ("int8", "int8"),
                ("bf16", "bf16"),
                ("f16", "f16"),
                ("f32", "f32"),
            )
            if token in artifact_names
        ),
        "f32",
    )
    return {"storage": storage, "compute": compute, "accumulation": "f32"}


def _node_parameter_metadata(
    node: Json, circuit: Json, tensor_index: Json
) -> list[tuple[str, Json]]:
    refs = circuit.get("parameters", {}).get("refs", {})
    tensors = tensor_index.get("tensors", {})
    result = []
    for parameter_id in node.get("params", []):
        ref = refs.get(parameter_id, {})
        tensor_name = ref.get("tensor")
        metadata = tensors.get(tensor_name)
        if not isinstance(tensor_name, str) or not isinstance(metadata, dict):
            _invalid(
                f"node {node.get('id')!r} parameter {parameter_id!r} has no tensor metadata"
            )
        result.append((tensor_name, metadata))
    return result


def _direct_parameter_bindings(node: Json) -> list[int]:
    base = len(node.get("inputs", [])) + len(node.get("outputs", []))
    return list(range(base, base + len(node.get("params", []))))


def _kernel_resources(node: Json, circuit: Json, tensor_index: Json) -> list[ResourceRequirement]:
    resources: list[ResourceRequirement] = []
    direct_bindings = _direct_parameter_bindings(node)
    lazy = str(node.get("op", "")).startswith("independent_sparse_moe_")
    for index, (tensor, _) in enumerate(
        _node_parameter_metadata(node, circuit, tensor_index)
    ):
        resource: ResourceRequirement = {
            "resource": tensor,
            "kind": "lazy_resource" if lazy else "persistent_parameter",
            "residency": "demand" if lazy else "permanent",
            "access": "read",
        }
        if not lazy:
            resource["binding"] = direct_bindings[index]
        resources.append(resource)
    state_access: dict[str, str] = {}
    for state_id in node.get("state_reads", []):
        state_access[str(state_id)] = "read"
    for state_id in node.get("state_writes", []):
        state_access[str(state_id)] = (
            "read_write" if str(state_id) in state_access else "write"
        )
    resources.extend(
        {
            "resource": state_id,
            "kind": "recurrent_state",
            "residency": "transaction",
            "access": access,
        }
        for state_id, access in sorted(state_access.items())
    )
    return resources


def _local_inputs(node: Json) -> list[InputContract]:
    return [
        {"binding": binding, "distribution": "local"}
        for binding in range(len(node.get("inputs", [])))
    ]


def _local_outputs(node: Json) -> list[OutputContract]:
    input_count = len(node.get("inputs", []))
    return [
        {"binding": input_count + index, "collection": "local"}
        for index, _ in enumerate(node.get("outputs", []))
    ]


def _execution_domain_phases(value: object) -> list[ExecutionPhase]:
    return {
        "decode": ["decode"],
        "prefill": ["prefill"],
        "decode_and_prefill": ["decode", "prefill"],
    }.get(_non_empty_string(value, "execution_domain")) or _invalid(
        f"unsupported execution domain {value!r}"
    )


def _block_row_alignment(parameter_metadata: list[tuple[str, Json]]) -> int:
    weight_shape = parameter_metadata[0][1].get("shape", [])
    scale_shape = parameter_metadata[1][1].get("shape", [])
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        _invalid("block-scaled projection tensors must be rank two")
    weight_rows = _positive_int(weight_shape[0], "weight rows")
    scale_rows = _positive_int(scale_shape[0], "scale rows")
    if weight_rows % scale_rows:
        _invalid("block-scaled projection rows do not align with scale rows")
    return weight_rows // scale_rows


def seal_physical_execution_contract(contract: Json) -> PhysicalExecutionContract:
    sealed = deepcopy(contract)
    sealed.pop("contract_id", None)
    sealed["contract_id"] = artifact_sha256(_canonical_json_bytes(sealed))
    validate_physical_execution_contract(sealed)
    return sealed  # type: ignore[return-value]


def validate_physical_execution_contract(value: object) -> None:
    contract = _mapping(value, "contract")
    _keys(contract, _REQUIRED_TOP_LEVEL_KEYS, _TOP_LEVEL_KEYS, "contract")
    if contract["schema"] != PHYSICAL_EXECUTION_CONTRACT_SCHEMA:
        _invalid(f"unsupported physical execution contract schema {contract['schema']!r}")
    _digest(contract["contract_id"], "contract.contract_id")
    _non_empty_string(contract["operation_family"], "contract.operation_family")
    if "region_family" in contract:
        _non_empty_string(contract["region_family"], "contract.region_family")
    _non_empty_unique_strings(contract["member_node_ids"], "contract.member_node_ids")

    artifacts = _list(contract["artifacts"], "contract.artifacts")
    if not artifacts:
        _invalid("contract.artifacts must not be empty")
    artifact_paths: set[str] = set()
    for index, value in enumerate(artifacts):
        path = f"contract.artifacts[{index}]"
        artifact = _mapping(value, path)
        _keys(
            artifact,
            {"path", "sha256", "entry_point"},
            {"path", "sha256", "entry_point"},
            path,
        )
        artifact_path = _non_empty_string(artifact["path"], f"{path}.path")
        if artifact_path in artifact_paths:
            _invalid("contract artifact paths must be unique")
        artifact_paths.add(artifact_path)
        _digest(artifact["sha256"], f"{path}.sha256")
        _non_empty_string(artifact["entry_point"], f"{path}.entry_point")
    _digest(contract["implementation_digest"], "contract.implementation_digest")

    phases = _list(contract["phases"], "contract.phases")
    _enum_list(phases, _PHASES, "contract.phases")

    formats = _mapping(contract["formats"], "contract.formats")
    _keys(
        formats,
        {"storage", "compute", "accumulation"},
        {"storage", "compute", "accumulation"},
        "contract.formats",
    )
    for field in ("storage", "compute", "accumulation"):
        _non_empty_string(formats[field], f"contract.formats.{field}")

    geometry = _mapping(contract["geometry"], "contract.geometry")
    _keys(
        geometry,
        {"shape_class", "dimensions", "dynamic_dimensions"},
        {"shape_class", "dimensions", "dynamic_dimensions"},
        "contract.geometry",
    )
    _non_empty_string(geometry["shape_class"], "contract.geometry.shape_class")
    dimensions = _mapping(geometry["dimensions"], "contract.geometry.dimensions")
    if not dimensions:
        _invalid("contract.geometry.dimensions must not be empty")
    for name, dimension in dimensions.items():
        _non_empty_string(name, "contract.geometry.dimensions key")
        _positive_int(dimension, f"contract.geometry.dimensions.{name}")
    dynamic_dimensions = _list(
        geometry["dynamic_dimensions"], "contract.geometry.dynamic_dimensions"
    )
    _non_empty_unique_strings(
        dynamic_dimensions, "contract.geometry.dynamic_dimensions", allow_empty=True
    )
    if any(name not in dimensions for name in dynamic_dimensions):
        _invalid("dynamic dimensions must name declared geometry dimensions")

    strategy = _enum(contract["strategy"], _STRATEGIES, "contract.strategy")
    execution_form = _enum(
        contract["execution_form"], _EXECUTION_FORMS, "contract.execution_form"
    )
    partition_extent = contract.get("partition_extent")
    if partition_extent is not None:
        extent = _mapping(partition_extent, "contract.partition_extent")
        _keys(
            extent,
            {"dimension_name", "elements", "alignment_elements"},
            {"dimension_name", "elements", "alignment_elements"},
            "contract.partition_extent",
        )
        dimension_name = _non_empty_string(
            extent["dimension_name"], "contract.partition_extent.dimension_name"
        )
        elements = _positive_int(
            extent["elements"], "contract.partition_extent.elements"
        )
        _positive_int(
            extent["alignment_elements"],
            "contract.partition_extent.alignment_elements",
        )
        if dimensions.get(dimension_name) != elements:
            _invalid("partition extent must match its declared geometry dimension")
        if elements % extent["alignment_elements"]:
            _invalid("partition extent must be divisible by its alignment")
    partition_launch = contract.get("partition_launch")
    if partition_launch is not None:
        launch = _mapping(partition_launch, "contract.partition_launch")
        _keys(
            launch,
            {"workgroup_x", "origin"},
            {
                "workgroup_x",
                "origin",
                "origin_push_constant",
                "count_push_constant",
            },
            "contract.partition_launch",
        )
        workgroup_x = _enum(
            launch["workgroup_x"],
            _WORKGROUP_X_MAPPINGS,
            "contract.partition_launch.workgroup_x",
        )
        origin = _enum(
            launch["origin"],
            _PARTITION_ORIGINS,
            "contract.partition_launch.origin",
        )
        has_origin_push_constant = "origin_push_constant" in launch
        has_count_push_constant = "count_push_constant" in launch
        expected_count_push_constant = (
            origin == "push_constant_u32" and workgroup_x == "repeated"
        )
        if (origin == "push_constant_u32") != has_origin_push_constant or (
            expected_count_push_constant != has_count_push_constant
        ):
            _invalid(
                "partition launch push constants must exactly match its origin and workgroup mapping"
            )
        if has_origin_push_constant:
            origin_name = _non_empty_string(
                launch["origin_push_constant"],
                "contract.partition_launch.origin_push_constant",
            )
        else:
            origin_name = None
        if has_count_push_constant:
            count_name = _non_empty_string(
                launch["count_push_constant"],
                "contract.partition_launch.count_push_constant",
            )
            if count_name == origin_name:
                _invalid("partition launch origin and count push constants must differ")
    resources = _list(contract["resources"], "contract.resources")
    _validate_resources(resources)
    partitions = _list(
        contract["parameter_partitions"], "contract.parameter_partitions"
    )
    _validate_parameter_partitions(partitions, partition_extent, resources)
    inputs = _list(contract["inputs"], "contract.inputs")
    outputs = _list(contract["outputs"], "contract.outputs")
    _validate_inputs(inputs)
    _validate_outputs(outputs, inputs, dimensions, formats["accumulation"])
    intermediates = _list(
        contract.get("local_intermediates", []), "contract.local_intermediates"
    )
    _validate_intermediates(intermediates)
    _validate_strategy(
        strategy,
        execution_form,
        partition_extent,
        partition_launch,
        partitions,
        inputs,
        outputs,
        intermediates,
    )
    _validate_equivalence(contract["equivalence"])


def _validate_parameter_partitions(
    values: list[object],
    partition_extent: object | None,
    resources: list[object],
) -> None:
    extent = (
        _mapping(partition_extent, "contract.partition_extent")
        if partition_extent is not None
        else None
    )
    bindings: set[int] = set()
    for index, value in enumerate(values):
        path = f"contract.parameter_partitions[{index}]"
        item = _mapping(value, path)
        _keys(
            item,
            {
                "binding",
                "resource",
                "dimension",
                "kind",
                "alignment_elements",
                "logical_elements_per_index",
            },
            {
                "binding",
                "resource",
                "dimension",
                "kind",
                "alignment_elements",
                "logical_elements_per_index",
            },
            path,
        )
        binding = _non_negative_int(item["binding"], f"{path}.binding")
        if binding in bindings:
            _invalid("parameter partition bindings must be unique")
        bindings.add(binding)
        resource_name = _non_empty_string(item["resource"], f"{path}.resource")
        matches = [
            _mapping(resource, "contract resource")
            for resource in resources
            if _mapping(resource, "contract resource").get("resource") == resource_name
        ]
        if len(matches) != 1:
            _invalid(
                "each parameter partition must name exactly one declared parameter resource"
            )
        resource = matches[0]
        if resource.get("kind") not in {"persistent_parameter", "lazy_resource"}:
            _invalid("parameter partition resources must be parameters")
        if "binding" in resource and resource["binding"] != binding:
            _invalid(
                "parameter partition resource binding must match its partition binding"
            )
        _non_negative_int(item["dimension"], f"{path}.dimension")
        _enum(item["kind"], _PARTITION_KINDS, f"{path}.kind")
        alignment = _positive_int(
            item["alignment_elements"], f"{path}.alignment_elements"
        )
        logical_elements_per_index = _positive_int(
            item["logical_elements_per_index"],
            f"{path}.logical_elements_per_index",
        )
        if extent is not None:
            logical_alignment = alignment * logical_elements_per_index
            if (
                extent["elements"] % logical_elements_per_index
                or extent["alignment_elements"] % logical_alignment
            ):
                _invalid(
                    "parameter partitions must divide the logical extent and its alignment"
                )


def _validate_inputs(values: list[object]) -> None:
    if not values:
        _invalid("contract.inputs must not be empty")
    bindings: set[int] = set()
    for index, value in enumerate(values):
        path = f"contract.inputs[{index}]"
        item = _mapping(value, path)
        _keys(item, {"binding", "distribution"}, {"binding", "distribution", "dimension", "alignment_elements"}, path)
        binding = _non_negative_int(item["binding"], f"{path}.binding")
        if binding in bindings:
            _invalid("input bindings must be unique")
        bindings.add(binding)
        distribution = _enum(
            item["distribution"], _INPUT_DISTRIBUTIONS, f"{path}.distribution"
        )
        partitioned = distribution in {"sharded", "routed"}
        if partitioned != ("dimension" in item) or partitioned != (
            "alignment_elements" in item
        ):
            _invalid(
                "sharded and routed inputs require dimension and alignment; other inputs forbid them"
            )
        if partitioned:
            _non_negative_int(item["dimension"], f"{path}.dimension")
            _positive_int(item["alignment_elements"], f"{path}.alignment_elements")


def _validate_outputs(
    values: list[object],
    inputs: list[object],
    dimensions: dict[str, object],
    accumulation: object,
) -> None:
    if not values:
        _invalid("contract.outputs must not be empty")
    bindings: set[int] = set()
    for index, value in enumerate(values):
        path = f"contract.outputs[{index}]"
        item = _mapping(value, path)
        _keys(item, {"binding", "collection"}, {"binding", "collection", "dimension", "alignment_elements", "reduction"}, path)
        binding = _non_negative_int(item["binding"], f"{path}.binding")
        if binding in bindings:
            _invalid("output bindings must be unique")
        bindings.add(binding)
        collection = _enum(
            item["collection"], _OUTPUT_COLLECTIONS, f"{path}.collection"
        )
        partitioned = collection in {"concatenated", "routed"}
        if partitioned != ("dimension" in item) or partitioned != (
            "alignment_elements" in item
        ):
            _invalid(
                "concatenated and routed outputs require dimension and alignment; other outputs forbid them"
            )
        if partitioned:
            _non_negative_int(item["dimension"], f"{path}.dimension")
            _positive_int(item["alignment_elements"], f"{path}.alignment_elements")
        if (collection == "reduced") != ("reduction" in item):
            _invalid("only reduced outputs require a reduction operation")
        if "reduction" in item:
            reduction = _mapping(item["reduction"], f"{path}.reduction")
            _keys(
                reduction,
                {"operation", "dimension_name", "finalization"},
                {"operation", "dimension_name", "finalization"},
                f"{path}.reduction",
            )
            operation = _enum(
                reduction["operation"],
                _REDUCTION_OPERATIONS,
                f"{path}.reduction.operation",
            )
            dimension_name = _non_empty_string(
                reduction["dimension_name"],
                f"{path}.reduction.dimension_name",
            )
            if dimension_name not in dimensions:
                _invalid(
                    "reduced output dimension must name a declared geometry dimension"
                )
            if operation == "sum_f32" and accumulation != "f32":
                _invalid("sum_f32 reduction requires f32 accumulation")
            finalization = _mapping(
                reduction["finalization"], f"{path}.reduction.finalization"
            )
            kind = _enum(
                finalization.get("kind"),
                {"store_f32", "add_bf16_residual_to_bf16"},
                f"{path}.reduction.finalization.kind",
            )
            required = (
                {"kind", "residual_binding"}
                if kind == "add_bf16_residual_to_bf16"
                else {"kind"}
            )
            _keys(finalization, required, required, f"{path}.reduction.finalization")
            if kind == "add_bf16_residual_to_bf16":
                if int(dimensions[dimension_name]) % 2:
                    _invalid(
                        "BF16 residual finalization requires an even element count"
                    )
                residual_binding = _non_negative_int(
                    finalization["residual_binding"],
                    f"{path}.reduction.finalization.residual_binding",
                )
                residual = next(
                    (
                        _mapping(value, "contract input")
                        for value in inputs
                        if _mapping(value, "contract input").get("binding")
                        == residual_binding
                    ),
                    None,
                )
                if residual is None:
                    _invalid(
                        "BF16 residual finalization binding must name a contract input"
                    )
                if residual["distribution"] != "replicated":
                    _invalid("BF16 residual finalization input must be replicated")


def _validate_intermediates(values: list[object]) -> None:
    signals: set[str] = set()
    for index, value in enumerate(values):
        path = f"contract.local_intermediates[{index}]"
        item = _mapping(value, path)
        _keys(
            item,
            {"signal", "producer_binding", "consumer_binding", "format"},
            {"signal", "producer_binding", "consumer_binding", "format"},
            path,
        )
        signal = _non_empty_string(item["signal"], f"{path}.signal")
        if signal in signals:
            _invalid("local intermediate signals must be unique")
        signals.add(signal)
        _non_negative_int(item["producer_binding"], f"{path}.producer_binding")
        _non_negative_int(item["consumer_binding"], f"{path}.consumer_binding")
        _non_empty_string(item["format"], f"{path}.format")


def _validate_strategy(
    strategy: str,
    execution_form: str,
    partition_extent: object | None,
    partition_launch: object | None,
    partitions: list[object],
    inputs: list[object],
    outputs: list[object],
    intermediates: list[object],
) -> None:
    if strategy == "single_device":
        if (
            execution_form != "local"
            or partition_extent is not None
            or partition_launch is not None
            or partitions
            or intermediates
        ):
            _invalid("single-device contracts cannot declare distributed flow")
        if any(
            _mapping(item, "input")["distribution"] not in {"local", "replicated"}
            for item in inputs
        ) or any(
            _mapping(item, "output")["collection"] not in {"local", "retained"}
            for item in outputs
        ):
            _invalid("single-device contracts cannot declare distributed flow")
        return
    if execution_form == "local":
        _invalid("distributed contracts require a distributed execution form")
    has_reduced_output = any(
        _mapping(item, "output")["collection"] == "reduced" for item in outputs
    )
    if (execution_form == "partitioned_input_partial_output") != has_reduced_output:
        _invalid(
            "partitioned-input partial-output execution requires a reduced output and reduced outputs require that execution form"
        )
    if execution_form == "partitioned_input_partial_output" and not any(
        _mapping(item, "input")["distribution"] == "sharded" for item in inputs
    ):
        _invalid("partitioned-input execution requires a sharded input")
    if partition_extent is None:
        _invalid("distributed contracts require an explicit partition extent")
    if partition_launch is None:
        _invalid("distributed contracts require an explicit partition launch")
    workgroup_x = _mapping(
        partition_launch, "contract.partition_launch"
    )["workgroup_x"]
    mapping_matches_form = (
        execution_form == "replicated_input_partitioned_output"
        and workgroup_x == "proportional"
    ) or (
        execution_form
        in {"partitioned_input_partial_output", "whole_expert_ownership"}
        and workgroup_x == "repeated"
    )
    if not mapping_matches_form:
        _invalid("partition workgroup mapping disagrees with the execution form")
    strategy_matches_form = (
        strategy in {"tensor_parallel", "tensor_parallel_expert"}
        and execution_form
        in {
            "replicated_input_partitioned_output",
            "partitioned_input_partial_output",
        }
    ) or (
        strategy == "expert_parallel"
        and execution_form == "whole_expert_ownership"
    )
    if not strategy_matches_form:
        _invalid("distributed execution strategy disagrees with its execution form")
    if execution_form == "replicated_input_partitioned_output" and not any(
        _mapping(item, "output")["collection"] == "concatenated"
        for item in outputs
    ):
        _invalid("partitioned-output execution requires a concatenated output")
    if execution_form == "whole_expert_ownership" and (
        not any(
            _mapping(item, "parameter partition")["kind"] == "expert_range"
            for item in partitions
        )
        or not any(
            _mapping(item, "input")["distribution"] == "routed"
            for item in inputs
        )
        or not any(
            _mapping(item, "output")["collection"] == "routed"
            for item in outputs
        )
    ):
        _invalid(
            "whole-expert execution requires expert-range parameters and routed input and output"
        )
    if not partitions:
        _invalid("distributed contracts require an explicit parameter partition")
    extent = _mapping(partition_extent, "contract.partition_extent")
    for flow_kind, flows in (("input", inputs), ("output", outputs)):
        for flow in flows:
            alignment = _mapping(flow, flow_kind).get("alignment_elements")
            if alignment is not None and extent["alignment_elements"] % alignment:
                _invalid(
                    "distributed input and output alignment must divide partition alignment"
                )
    if not any(
        _mapping(item, "output")["collection"]
        in {"concatenated", "reduced", "routed", "retained"}
        for item in outputs
    ):
        _invalid("distributed contracts require an explicit output collection")


def _validate_resources(values: list[object]) -> None:
    identities: set[tuple[str, int | None]] = set()
    for index, value in enumerate(values):
        path = f"contract.resources[{index}]"
        item = _mapping(value, path)
        _keys(
            item,
            {"resource", "kind", "residency", "access"},
            {"resource", "kind", "residency", "access", "binding", "atomic_group"},
            path,
        )
        resource = _non_empty_string(item["resource"], f"{path}.resource")
        kind = _enum(item["kind"], _RESOURCE_KINDS, f"{path}.kind")
        residency = _enum(
            item["residency"], _RESIDENCY_REQUIREMENTS, f"{path}.residency"
        )
        _enum(item["access"], _RESOURCE_ACCESSES, f"{path}.access")
        binding = (
            _non_negative_int(item["binding"], f"{path}.binding")
            if "binding" in item
            else None
        )
        if "atomic_group" in item:
            _non_empty_string(item["atomic_group"], f"{path}.atomic_group")
        if (resource, binding) in identities:
            _invalid("resource requirements must be unique by name and binding")
        identities.add((resource, binding))
        if kind == "lazy_resource" and residency != "demand":
            _invalid("lazy resources must be demand resident")


def _validate_equivalence(value: object) -> None:
    item = _mapping(value, "contract.equivalence")
    _keys(
        item,
        {"output", "state"},
        {"output", "state", "absolute_tolerance", "relative_tolerance"},
        "contract.equivalence",
    )
    output = _enum(item["output"], _EQUIVALENCE_KINDS, "contract.equivalence.output")
    state = _enum(item["state"], _EQUIVALENCE_KINDS, "contract.equivalence.state")
    tolerant = "absolute_relative_tolerance" in {output, state}
    has_tolerances = "absolute_tolerance" in item and "relative_tolerance" in item
    if tolerant != has_tolerances:
        _invalid("tolerant equivalence requires both tolerances; bit-exact equivalence forbids them")
    if tolerant:
        for field in ("absolute_tolerance", "relative_tolerance"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _invalid(f"contract.equivalence.{field} must be numeric")
            numeric = float(value)
            if not 0.0 <= numeric < float("inf"):
                _invalid(f"contract.equivalence.{field} must be finite and non-negative")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _mapping(value: object, path: str) -> Json:
    if not isinstance(value, dict):
        _invalid(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _invalid(f"{path} must be an array")
    return value


def _keys(value: Json, required: set[str], allowed: set[str], path: str) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing:
        _invalid(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        _invalid(f"{path} has unknown fields: {', '.join(sorted(unknown))}")


def _enum(value: object, allowed: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _invalid(f"{path} must be one of: {', '.join(sorted(allowed))}")
    return value


def _enum_list(values: list[object], allowed: set[str], path: str) -> None:
    if not values:
        _invalid(f"{path} must not be empty")
    normalized = [_enum(value, allowed, f"{path}[]") for value in values]
    if len(set(normalized)) != len(normalized):
        _invalid(f"{path} must not contain duplicates")


def _non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{path} must be a non-empty string")
    return value


def _non_empty_unique_strings(
    value: object, path: str, *, allow_empty: bool = False
) -> list[str]:
    values = _list(value, path)
    if not values and not allow_empty:
        _invalid(f"{path} must not be empty")
    normalized = [_non_empty_string(item, f"{path}[]") for item in values]
    if len(set(normalized)) != len(normalized):
        _invalid(f"{path} must not contain duplicates")
    return normalized


def _non_negative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid(f"{path} must be a non-negative integer")
    return value


def _positive_int(value: object, path: str) -> int:
    number = _non_negative_int(value, path)
    if number == 0:
        _invalid(f"{path} must be positive")
    return number


def _digest(value: object, path: str) -> str:
    digest = _non_empty_string(value, path)
    prefix = "sha256:"
    suffix = digest.removeprefix(prefix)
    if not digest.startswith(prefix) or len(suffix) != 64 or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        _invalid(f"{path} must use sha256:<64 lowercase hex>")
    return digest


def _invalid(message: str) -> None:
    raise PhysicalExecutionContractError(message)
