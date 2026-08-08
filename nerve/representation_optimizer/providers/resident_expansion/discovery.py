from __future__ import annotations

import json
import re
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    adaptive_shader_artifact_path,
)
from nerve.representation_optimizer.providers.types import ProviderContext
from nerve.resident_representations import (
    MXFP4_TO_FP8_REQUIRED_FEATURES,
    mxfp4_to_fp8_resident_derivation,
)


_EXPERT_ROLES = (
    "independent_sparse_moe_down",
    "independent_sparse_moe_gate_up",
)
_CONTROL_SUFFIX = re.compile(r"__(?:pbc|sc)\d+$")


@dataclass(frozen=True)
class ResidentShaderReplacement:
    node_id: str
    source_path: str
    artifact_path: str
    template_name: str
    execution_kind: str

    def to_json(self) -> Json:
        return {
            "node_id": self.node_id,
            "source_path": self.source_path,
            "artifact_path": self.artifact_path,
            "template_name": self.template_name,
            "execution_kind": self.execution_kind,
        }


@dataclass(frozen=True)
class ResidentWeightDerivation:
    node_id: str
    parameter_id: str
    tensor_name: str
    source_resource_id: str
    source_byte_count: int
    derivation: Json

    def to_json(self) -> Json:
        return {
            "node_id": self.node_id,
            "parameter_id": self.parameter_id,
            "tensor_name": self.tensor_name,
            "source_resource_id": self.source_resource_id,
            "source_byte_count": self.source_byte_count,
            "derivation": self.derivation,
        }


@dataclass(frozen=True)
class ResidentExpansionOpportunity:
    scope_ids: tuple[str, str]
    source_contract_digests: tuple[str, str]
    component_id: str
    node_ids: tuple[str, str]
    evidence_ids: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    manifest_ref: str
    hidden_size: int
    intermediate_size: int
    expert_count: int
    experts_per_token: int
    max_context_activations: int
    weight_derivations: tuple[ResidentWeightDerivation, ...]
    shader_replacements: tuple[ResidentShaderReplacement, ...]

    @property
    def source_weight_bytes(self) -> int:
        return sum(item.source_byte_count for item in self.weight_derivations)

    @property
    def resident_weight_bytes(self) -> int:
        return sum(
            int(item.derivation["resident_byte_count"])
            for item in self.weight_derivations
        )

    @property
    def shader_artifact_paths(self) -> tuple[str, ...]:
        return tuple(sorted({item.artifact_path for item in self.shader_replacements}))


@dataclass(frozen=True)
class DiscoveryResult:
    opportunities: tuple[ResidentExpansionOpportunity, ...]
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def is_resident_expansion_scope(scope: Json, source_contract: Json) -> bool:
    return (
        scope.get("kind") == "operator"
        and len(scope.get("members", {}).get("component_ids", [])) == 1
        and source_contract.get("semantic_role") in _EXPERT_ROLES
    )


def discover_resident_expansions(
    context: ProviderContext,
) -> tuple[ResidentExpansionOpportunity, ...]:
    key = "resident_expansion.v1:" + ",".join(context.scope_ids)
    return context.memoized(
        key,
        lambda: _discover(context).opportunities,
    )  # type: ignore[return-value]


def discovery_result(context: ProviderContext) -> DiscoveryResult:
    key = "resident_expansion.result.v1:" + ",".join(context.scope_ids)
    return context.memoized(key, lambda: _discover(context))  # type: ignore[return-value]


def require_resident_expansion(
    context: ProviderContext,
    scope_ids: tuple[str, ...],
) -> ResidentExpansionOpportunity:
    matches = [
        opportunity
        for opportunity in discover_resident_expansions(context)
        if opportunity.scope_ids == scope_ids
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            "resident parameter expansion requires one exact component "
            f"opportunity for scopes {scope_ids!r}, found {len(matches)}"
        )
    return matches[0]


def source_inputs(
    context: ProviderContext,
    opportunity: ResidentExpansionOpportunity,
) -> list[Json]:
    return [
        context.source_artifacts.resolve_path(path).source_input()
        for path in opportunity.source_artifact_refs
    ]


def _discover(context: ProviderContext) -> DiscoveryResult:
    context.checkpoint()
    capability_reason = _capability_reason(context.hardware_profile)
    if capability_reason is not None:
        return DiscoveryResult((), (capability_reason,))

    contracts = {
        str(contract["scope_id"]): contract for contract in context.source_contracts
    }
    eligible = [
        (scope, contracts[str(scope["scope_id"])])
        for scope in context.scopes
        if is_resident_expansion_scope(
            scope,
            contracts[str(scope["scope_id"])],
        )
    ]
    if not eligible:
        return DiscoveryResult(
            (),
            ("no independently addressable sparse expert projection scopes",),
        )
    evidence_by_scope: dict[str, tuple[str, ...]] = {}
    for scope, _contract in eligible:
        scope_id = str(scope["scope_id"])
        evidence_by_scope[scope_id] = tuple(
            sorted(
                str(record["evidence_id"])
                for record in context.evidence
                if record["scope_id"] == scope_id
                and any(
                    claim.get("status") == "supported"
                    for claim in record.get("claims", [])
                )
            )
        )

    resolver = context.source_artifacts
    manifest_ref = "vulkan_resident_package.json"
    manifest = _json_object(resolver.read_path(manifest_ref), manifest_ref)
    tensor_index_ref = str(manifest.get("tensor_index_path", ""))
    if tensor_index_ref != "tensors.json":
        return DiscoveryResult(
            (),
            ("compiled package has no canonical tensor index",),
        )
    tensor_index = _json_object(
        resolver.read_path(tensor_index_ref),
        tensor_index_ref,
    )
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict):
        return DiscoveryResult(
            (),
            ("compiled package tensor index has no tensor map",),
        )
    components = _unique_by_id(
        manifest.get("circuit_graph", {}).get("components"),
        "component_id",
        "resident components",
    )
    executions = _unique_by_id(
        manifest.get("component_executions"),
        "component_id",
        "component executions",
    )
    residency = manifest.get("resource_residency")
    if not isinstance(residency, dict):
        return DiscoveryResult(
            (),
            ("compiled package has no resource-residency graph",),
        )
    bindings = _binding_index(residency.get("bindings"))
    resources = _unique_by_id(
        residency.get("resources"),
        "id",
        "resident resources",
    )
    resource_consumers = _resource_consumers(residency.get("bindings"))
    compiler_target = _compiler_target_for_profile(context.hardware_profile)

    by_component: dict[str, dict[str, tuple[Json, Json]]] = {}
    for scope, contract in eligible:
        component_id = str(scope["members"]["component_ids"][0])
        by_component.setdefault(component_id, {})[str(contract["semantic_role"])] = (
            scope,
            contract,
        )

    opportunities = []
    rejection_reasons = []
    for component_id in sorted(by_component):
        context.checkpoint()
        pair = by_component[component_id]
        if set(pair) != set(_EXPERT_ROLES):
            rejection_reasons.append(
                f"component {component_id!r} has an incomplete sparse expert pair"
            )
            continue
        ordered = tuple(pair[role] for role in _EXPERT_ROLES)
        scope_ids = tuple(str(scope["scope_id"]) for scope, _ in ordered)
        if any(not evidence_by_scope[scope_id] for scope_id in scope_ids):
            rejection_reasons.append(
                f"component {component_id!r} lacks supported scope evidence"
            )
            continue
        try:
            opportunity = _component_opportunity(
                component_id=component_id,
                ordered_scopes=ordered,
                evidence_ids=tuple(
                    sorted(
                        {
                            evidence_id
                            for scope_id in scope_ids
                            for evidence_id in evidence_by_scope[scope_id]
                        }
                    )
                ),
                manifest=manifest,
                components=components,
                executions=executions,
                tensors=tensors,
                bindings=bindings,
                resources=resources,
                resource_consumers=resource_consumers,
                compiler_target=compiler_target,
            )
        except ModelCompileError as error:
            rejection_reasons.append(str(error))
            continue
        opportunities.append(opportunity)
    evidence_ids = tuple(
        sorted(
            {
                evidence_id
                for opportunity in opportunities
                for evidence_id in opportunity.evidence_ids
            }
        )
    )
    if opportunities:
        return DiscoveryResult(
            tuple(opportunities),
            (
                f"discovered {len(opportunities)} exact component-local "
                "compact-to-resident sparse expert alternatives",
            ),
            evidence_ids,
        )
    return DiscoveryResult(
        (),
        tuple(rejection_reasons[:8])
        or ("no structurally valid resident expert alternative",),
    )


def _component_opportunity(
    *,
    component_id: str,
    ordered_scopes: tuple[tuple[Json, Json], ...],
    evidence_ids: tuple[str, ...],
    manifest: Json,
    components: dict[str, Json],
    executions: dict[str, Json],
    tensors: Json,
    bindings: dict[tuple[str, str, str], Json],
    resources: dict[str, Json],
    resource_consumers: dict[str, set[str]],
    compiler_target: Json,
) -> ResidentExpansionOpportunity:
    component = components.get(component_id)
    execution = executions.get(component_id)
    if component is None or execution is None:
        raise ModelCompileError(
            f"component {component_id!r} has no complete resident execution"
        )
    nodes = _unique_by_id(
        component.get("circuit", {}).get("nodes"),
        "id",
        f"component {component_id!r} nodes",
    )
    kernels = _unique_by_id(
        execution.get("kernels"),
        "node_id",
        f"component {component_id!r} kernels",
    )
    component_refs = component.get("params", {}).get("refs")
    if not isinstance(component_refs, dict):
        raise ModelCompileError(
            f"component {component_id!r} has no parameter references"
        )

    node_ids = []
    weight_derivations = []
    shader_replacements = []
    geometry: tuple[int, int, int, int] | None = None
    for role, (scope, contract) in zip(
        _EXPERT_ROLES,
        ordered_scopes,
        strict=True,
    ):
        matches = [node for node in nodes.values() if node.get("op") == role]
        if len(matches) != 1:
            raise ModelCompileError(
                f"component {component_id!r} has no unique {role!r} node"
            )
        node = matches[0]
        node_id = str(node["id"])
        if f"{component_id}/{node_id}" not in scope["members"]["source_node_ids"]:
            raise ModelCompileError(
                f"component {component_id!r} scope does not own node {node_id!r}"
            )
        node_ids.append(node_id)
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            raise ModelCompileError(
                f"component {component_id!r} node {node_id!r} has no attributes"
            )
        node_geometry = (
            _positive_int(attrs.get("hidden_size"), "hidden size"),
            _positive_int(attrs.get("intermediate_size"), "intermediate size"),
            _expert_count(attrs),
            _positive_int(attrs.get("experts_per_token"), "experts per token"),
        )
        if geometry is None:
            geometry = node_geometry
        elif geometry != node_geometry:
            raise ModelCompileError(
                f"component {component_id!r} expert nodes disagree on geometry"
            )
        hidden_size, intermediate_size, expert_count, experts_per_token = node_geometry
        if (
            hidden_size % 128
            or intermediate_size % 128
            or not experts_per_token <= expert_count
        ):
            raise ModelCompileError(
                f"component {component_id!r} expert geometry is unsupported"
            )
        parameter_stride = 2 if role.endswith("down") else 4
        mapping = _selector_mapping(node, parameter_stride, expert_count)
        expected_parameters = [
            parameter_id
            for record in mapping
            for parameter_id in record["parameter_ids"]
        ]
        if node.get("params") != expected_parameters:
            raise ModelCompileError(
                f"component {component_id!r} node {node_id!r} parameter order drifted"
            )
        contract_parameters = {
            str(parameter["parameter_ref_id"]): parameter["definition"]
            for parameter in contract["interface"]["parameters"]
        }
        if set(contract_parameters) != set(expected_parameters):
            raise ModelCompileError(
                f"component {component_id!r} node {node_id!r} scope boundary drifted"
            )
        for record in mapping:
            parameter_ids = record["parameter_ids"]
            for offset in range(0, len(parameter_ids), 2):
                weight_id = parameter_ids[offset]
                scale_id = parameter_ids[offset + 1]
                weight_ref = component_refs.get(weight_id)
                scale_ref = component_refs.get(scale_id)
                if (
                    not isinstance(weight_ref, dict)
                    or not isinstance(scale_ref, dict)
                    or contract_parameters[weight_id] != weight_ref
                    or contract_parameters[scale_id] != scale_ref
                ):
                    raise ModelCompileError(
                        f"component {component_id!r} parameter bindings drifted"
                    )
                weight_name = weight_ref.get("tensor")
                scale_name = scale_ref.get("tensor")
                weight = tensors.get(weight_name)
                scale = tensors.get(scale_name)
                _validate_tensor_pair(
                    weight,
                    scale,
                    weight_name=weight_name,
                    scale_name=scale_name,
                    output_size=(
                        hidden_size if role.endswith("down") else intermediate_size
                    ),
                    input_size=(
                        intermediate_size if role.endswith("down") else hidden_size
                    ),
                )
                derivation = mxfp4_to_fp8_resident_derivation(
                    weight,
                    compiler_target,
                )
                if derivation is None:
                    raise ModelCompileError(
                        f"component {component_id!r} weight {weight_name!r} "
                        "has no exact native resident expansion"
                    )
                binding = bindings.get((component_id, node_id, weight_id))
                if binding is None:
                    raise ModelCompileError(
                        f"component {component_id!r} weight {weight_id!r} "
                        "has no resident binding"
                    )
                binding_mapping = binding.get("mapping")
                if (
                    not isinstance(binding_mapping, dict)
                    or binding_mapping.get("kind") != "selected_atomic_group"
                    or binding_mapping.get("selector_index") != record["selector"]
                    or binding_mapping.get("parameter_slot") != offset
                ):
                    raise ModelCompileError(
                        f"component {component_id!r} weight {weight_id!r} "
                        "is not independently selector-addressable"
                    )
                resource_id = str(binding_mapping.get("resource_id", ""))
                resource = resources.get(resource_id)
                if (
                    resource is None
                    or resource.get("lifetime") != "dynamic"
                    or "resident_derivation" in resource
                    or _resource_byte_count(resource) != weight["byte_count"]
                    or resource_consumers.get(resource_id) != {component_id}
                ):
                    raise ModelCompileError(
                        f"component {component_id!r} weight {weight_id!r} "
                        "does not own one exact compact dynamic resource"
                    )
                weight_derivations.append(
                    ResidentWeightDerivation(
                        node_id=node_id,
                        parameter_id=weight_id,
                        tensor_name=str(weight_name),
                        source_resource_id=resource_id,
                        source_byte_count=int(weight["byte_count"]),
                        derivation=derivation,
                    )
                )
        kernel = kernels.get(node_id)
        if kernel is None:
            raise ModelCompileError(
                f"component {component_id!r} node {node_id!r} has no kernel"
            )
        scalar_path = str(kernel.get("shader_path", ""))
        shader_replacements.append(_shader_replacement(node_id, scalar_path, "scalar"))
        batches = kernel.get("batch_implementations")
        if not isinstance(batches, list) or not batches:
            raise ModelCompileError(
                f"component {component_id!r} node {node_id!r} has no batch path"
            )
        for batch in batches:
            stages = batch.get("stages") if isinstance(batch, dict) else None
            if not isinstance(stages, list):
                raise ModelCompileError(
                    f"component {component_id!r} node {node_id!r} has invalid batch stages"
                )
            expert_stages = [
                stage
                for stage in stages
                if isinstance(stage, dict)
                and "_mxfp4_e2m1_" in str(stage.get("shader_path", ""))
            ]
            if len(expert_stages) != 1:
                raise ModelCompileError(
                    f"component {component_id!r} node {node_id!r} batch path "
                    "has no unique compact expert stage"
                )
            shader_replacements.append(
                _shader_replacement(
                    node_id,
                    str(expert_stages[0]["shader_path"]),
                    "batch",
                )
            )
    assert geometry is not None
    shader_replacements = sorted(
        set(shader_replacements),
        key=lambda item: (item.node_id, item.source_path, item.execution_kind),
    )
    artifacts_by_source: dict[str, str] = {}
    source_by_artifact: dict[str, str] = {}
    for replacement in shader_replacements:
        previous_artifact = artifacts_by_source.setdefault(
            replacement.source_path,
            replacement.artifact_path,
        )
        previous_source = source_by_artifact.setdefault(
            replacement.artifact_path,
            replacement.source_path,
        )
        if (
            previous_artifact != replacement.artifact_path
            or previous_source != replacement.source_path
        ):
            raise ModelCompileError(
                f"component {component_id!r} shader identity is ambiguous"
            )
    scope_ids = tuple(
        sorted(str(scope["scope_id"]) for scope, _contract in ordered_scopes)
    )
    digest_by_scope = {
        str(scope["scope_id"]): str(contract["contract_digest"])
        for scope, contract in ordered_scopes
    }
    exact_refs = tuple(
        sorted(
            {
                str(path)
                for _scope, contract in ordered_scopes
                for path in contract["exact_reference"]["artifact_refs"]
            }
        )
    )
    if not exact_refs:
        raise ModelCompileError(
            f"component {component_id!r} has no exact source artifacts"
        )
    return ResidentExpansionOpportunity(
        scope_ids=(scope_ids[0], scope_ids[1]),
        source_contract_digests=(
            digest_by_scope[scope_ids[0]],
            digest_by_scope[scope_ids[1]],
        ),
        component_id=component_id,
        node_ids=tuple(sorted(node_ids)),  # type: ignore[arg-type]
        evidence_ids=evidence_ids,
        source_artifact_refs=("tensors.json", "vulkan_resident_package.json"),
        manifest_ref="vulkan_resident_package.json",
        hidden_size=geometry[0],
        intermediate_size=geometry[1],
        expert_count=geometry[2],
        experts_per_token=geometry[3],
        max_context_activations=_positive_int(
            manifest.get("max_context_activations"),
            "maximum context activations",
        ),
        weight_derivations=tuple(
            sorted(
                weight_derivations,
                key=lambda item: (item.node_id, item.parameter_id),
            )
        ),
        shader_replacements=tuple(shader_replacements),
    )


def _capability_reason(profile: Json) -> str | None:
    if (
        profile.get("hardware_identity", {}).get("device_kind") != "gpu"
        or profile.get("provenance", {}).get("api") != "vulkan"
    ):
        return "exact resident parameter expansion currently requires a Vulkan GPU"
    packed = [
        process
        for process in profile.get("processes", [])
        if process.get("name") == "packed_dot_product"
        and process.get("availability") == "available"
        and process.get("programmability") != "none"
        and "f8_e4m3" in process.get("numeric_formats", [])
    ]
    features = set(
        profile.get("capability_extensions", {})
        .get("vulkan_compiler_capabilities", {})
        .get("shader_features", [])
    )
    if not packed or not set(MXFP4_TO_FP8_REQUIRED_FEATURES) <= features:
        return "target has no programmable native F8 E4M3 resident path"
    return None


def _compiler_target_for_profile(profile: Json) -> Json:
    features = profile["capability_extensions"]["vulkan_compiler_capabilities"][
        "shader_features"
    ]
    return {"devices": [{"shader_features": list(features)}]}


def _expert_count(attrs: Json) -> int:
    accesses = attrs.get("selected_parameter_accesses")
    if not isinstance(accesses, list) or len(accesses) != 1:
        raise ModelCompileError("expert node has no unique selected parameter access")
    mapping = accesses[0].get("mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ModelCompileError("expert node has no selector mapping")
    return len(mapping)


def _selector_mapping(node: Json, stride: int, expert_count: int) -> list[Json]:
    accesses = node["attrs"]["selected_parameter_accesses"]
    mapping = accesses[0]["mapping"]
    if (
        any(
            not isinstance(record, dict)
            or record.get("selector") != index
            or not isinstance(record.get("parameter_ids"), list)
            or len(record["parameter_ids"]) != stride
            or any(
                not isinstance(parameter_id, str) or not parameter_id
                for parameter_id in record["parameter_ids"]
            )
            for index, record in enumerate(mapping)
        )
        or len(mapping) != expert_count
    ):
        raise ModelCompileError(
            f"expert node {node.get('id')!r} has an invalid selector mapping"
        )
    return mapping


def _validate_tensor_pair(
    weight: object,
    scale: object,
    *,
    weight_name: object,
    scale_name: object,
    output_size: int,
    input_size: int,
) -> None:
    if not isinstance(weight_name, str) or not isinstance(scale_name, str):
        raise ModelCompileError("expert parameter has no tensor binding")
    if not isinstance(weight, dict) or not isinstance(scale, dict):
        raise ModelCompileError(
            f"expert tensor pair {weight_name!r}/{scale_name!r} is missing"
        )
    quantization = weight.get("quantization")
    expected_weight_bytes = output_size * input_size // 2
    if (
        weight.get("dtype") != "I8"
        or weight.get("shape") != [output_size, input_size // 2]
        or weight.get("logical_shape") != [output_size, input_size]
        or weight.get("byte_count") != expected_weight_bytes
        or weight.get("layout") != "row_major"
        or not isinstance(quantization, dict)
        or quantization.get("format") != "mxfp4_e2m1"
        or quantization.get("bits") != 4
        or quantization.get("element_type") != "float"
        or quantization.get("values_per_byte") != 2
        or quantization.get("packing_axis") != 1
        or quantization.get("packing_order") != "low_nibble_then_high_nibble_along_k"
        or quantization.get("group_size") != 32
        or quantization.get("scales") != scale_name
        or quantization.get("scale_dtype") != "F8_E8M0"
        or scale.get("dtype") != "F8_E8M0"
        or scale.get("shape") != [output_size, input_size // 32]
        or scale.get("byte_count") != output_size * input_size // 32
        or scale.get("layout") != "row_major"
    ):
        raise ModelCompileError(
            f"expert tensor pair {weight_name!r}/{scale_name!r} has an "
            "unsupported compact numeric contract"
        )


def _shader_replacement(
    node_id: str,
    source_path: str,
    execution_kind: str,
) -> ResidentShaderReplacement:
    artifact_path = adaptive_shader_artifact_path(source_path)
    stem = artifact_path.rsplit("/", 1)[-1][:-4]
    template_name = f"{_CONTROL_SUFFIX.sub('', stem)}.comp"
    if execution_kind == "batch" and "_batch1_" not in template_name:
        raise ModelCompileError(
            f"resident expansion has no compiler template for {source_path!r}"
        )
    return ResidentShaderReplacement(
        node_id=node_id,
        source_path=source_path,
        artifact_path=artifact_path,
        template_name=template_name,
        execution_kind=execution_kind,
    )


def _binding_index(value: object) -> dict[tuple[str, str, str], Json]:
    if not isinstance(value, list):
        raise ModelCompileError("resource residency bindings must be a list")
    result = {}
    for binding in value:
        if not isinstance(binding, dict):
            raise ModelCompileError("resource residency binding is malformed")
        key = (
            str(binding.get("component_id", "")),
            str(binding.get("node_id", "")),
            str(binding.get("parameter_id", "")),
        )
        if not all(key) or key in result:
            raise ModelCompileError("resource residency bindings are not unique")
        result[key] = binding
    return result


def _resource_consumers(value: object) -> dict[str, set[str]]:
    if not isinstance(value, list):
        raise ModelCompileError("resource residency bindings must be a list")
    result: dict[str, set[str]] = {}
    for binding in value:
        mapping = binding.get("mapping") if isinstance(binding, dict) else None
        resource_id = mapping.get("resource_id") if isinstance(mapping, dict) else None
        component_id = (
            binding.get("component_id") if isinstance(binding, dict) else None
        )
        if isinstance(resource_id, str) and isinstance(component_id, str):
            result.setdefault(resource_id, set()).add(component_id)
    return result


def _resource_byte_count(resource: Json) -> int:
    ranges = resource.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return -1
    total = 0
    for record in ranges:
        byte_count = record.get("byte_count") if isinstance(record, dict) else None
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            return -1
        total += byte_count
    return total


def _unique_by_id(
    value: object,
    key: str,
    label: str,
) -> dict[str, Json]:
    if not isinstance(value, list):
        raise ModelCompileError(f"{label} must be a list")
    result = {}
    for record in value:
        identity = record.get(key) if isinstance(record, dict) else None
        if not isinstance(identity, str) or not identity or identity in result:
            raise ModelCompileError(f"{label} have invalid identities")
        result[identity] = record
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelCompileError(f"{label} must be a positive integer")
    return value


def _json_object(payload: bytes, label: str) -> Json:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must contain a JSON object")
    return value
