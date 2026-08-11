from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Literal, TypedDict


Json = dict[str, Any]

PHYSICAL_EXECUTION_CONTRACT_SCHEMA = "nerve.physical_execution_contract.v1"

ExecutionPhase = Literal["decode", "prefill"]
ExecutionStrategy = Literal[
    "single_device",
    "tensor_parallel",
    "expert_parallel",
    "tensor_parallel_expert",
]
ParameterPartitionKind = Literal["contiguous", "block_cyclic", "expert_range"]
InputDistribution = Literal["replicated", "sharded", "routed", "local"]
OutputCollection = Literal[
    "local", "concatenated", "reduced", "routed", "retained"
]
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
    dimension: int
    kind: ParameterPartitionKind
    alignment_elements: int


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
    reduction: str


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
    artifact: ArtifactIdentity
    implementation_digest: str
    phases: list[ExecutionPhase]
    formats: PhysicalFormats
    geometry: ExecutionGeometry
    strategy: ExecutionStrategy
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
_PARTITION_KINDS = {"contiguous", "block_cyclic", "expert_range"}
_INPUT_DISTRIBUTIONS = {"replicated", "sharded", "routed", "local"}
_OUTPUT_COLLECTIONS = {"local", "concatenated", "reduced", "routed", "retained"}
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
    "artifact",
    "implementation_digest",
    "phases",
    "formats",
    "geometry",
    "strategy",
    "parameter_partitions",
    "inputs",
    "outputs",
    "local_intermediates",
    "resources",
    "equivalence",
}
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS - {"region_family", "local_intermediates"}


class PhysicalExecutionContractError(ValueError):
    pass


def artifact_sha256(payload: bytes) -> str:
    return f"sha256:{sha256(payload).hexdigest()}"


def implementation_digest(
    *,
    artifact: ArtifactIdentity,
    phases: list[ExecutionPhase],
    formats: PhysicalFormats,
    geometry: ExecutionGeometry,
    strategy: ExecutionStrategy,
) -> str:
    return artifact_sha256(
        _canonical_json_bytes(
            {
                "artifact": artifact,
                "phases": phases,
                "formats": formats,
                "geometry": geometry,
                "strategy": strategy,
            }
        )
    )


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

    artifact = _mapping(contract["artifact"], "contract.artifact")
    _keys(
        artifact,
        {"path", "sha256", "entry_point"},
        {"path", "sha256", "entry_point"},
        "contract.artifact",
    )
    _non_empty_string(artifact["path"], "contract.artifact.path")
    _digest(artifact["sha256"], "contract.artifact.sha256")
    _non_empty_string(artifact["entry_point"], "contract.artifact.entry_point")
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
    partitions = _list(
        contract["parameter_partitions"], "contract.parameter_partitions"
    )
    _validate_parameter_partitions(partitions)
    inputs = _list(contract["inputs"], "contract.inputs")
    outputs = _list(contract["outputs"], "contract.outputs")
    _validate_inputs(inputs)
    _validate_outputs(outputs)
    intermediates = _list(
        contract.get("local_intermediates", []), "contract.local_intermediates"
    )
    _validate_intermediates(intermediates)
    _validate_strategy(strategy, partitions, inputs, outputs, intermediates)
    _validate_resources(_list(contract["resources"], "contract.resources"))
    _validate_equivalence(contract["equivalence"])


def _validate_parameter_partitions(values: list[object]) -> None:
    bindings: set[int] = set()
    for index, value in enumerate(values):
        path = f"contract.parameter_partitions[{index}]"
        item = _mapping(value, path)
        _keys(
            item,
            {"binding", "dimension", "kind", "alignment_elements"},
            {"binding", "dimension", "kind", "alignment_elements"},
            path,
        )
        binding = _non_negative_int(item["binding"], f"{path}.binding")
        if binding in bindings:
            _invalid("parameter partition bindings must be unique")
        bindings.add(binding)
        _non_negative_int(item["dimension"], f"{path}.dimension")
        _enum(item["kind"], _PARTITION_KINDS, f"{path}.kind")
        _positive_int(item["alignment_elements"], f"{path}.alignment_elements")


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


def _validate_outputs(values: list[object]) -> None:
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
            _non_empty_string(item["reduction"], f"{path}.reduction")


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
    partitions: list[object],
    inputs: list[object],
    outputs: list[object],
    intermediates: list[object],
) -> None:
    if strategy == "single_device":
        if partitions or intermediates:
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
    if not partitions:
        _invalid("distributed contracts require an explicit parameter partition")
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
