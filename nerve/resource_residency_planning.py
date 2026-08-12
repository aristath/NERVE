from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError, read_json
from nerve.resource_residency import (
    RESOURCE_IDENTITY_ALGORITHM,
    RESOURCE_RESIDENCY_SCHEMA,
    RESOURCE_STATE_MACHINE_SCHEMA,
    SUPPORTED_RESIDENCY_POLICIES,
    atomic_group_identity,
    checkpoint_identity,
    compiled_immutable_resource,
    compiled_parameter_bindings,
    partition_group_identity_seed,
    partition_template_identity,
    residency_content_id,
    selector_identity,
    validate_resource_residency_contract,
    compiled_resource_artifact_metadata,
)


RESIDENCY_ANALYSIS_SCHEMA = "nerve.compiler_resource_residency_analysis.v1"
TENSOR_PARTITION_INTEGRITY_SCHEMA = "nerve.tensor_partition_integrity.v1"
SOURCE_INTEGRITY_PARTITION_COUNT_FIELD = "source_integrity_partition_count"
SELECTED_PARAMETER_ACCESSES_ATTRIBUTE = "selected_parameter_accesses"
ROW_MAJOR_LAYOUT = "row_major"

_SELECTION_DOMAIN_FIELDS = frozenset(
    ("id", "resource_count", "selection_signal", "encoding")
)
_SELECTION_ENCODING_FIELDS = frozenset(
    (
        "element_type",
        "selection_count_per_activation",
        "index_shift",
        "index_mask",
        "calibration_word_base",
    )
)
_PREDICTABLE_DEPENDENCY_FIELDS = frozenset(
    (
        "schema",
        "kind",
        "key_signal",
        "table_parameter",
        "selection_semantics",
    )
)
_PREDICTABLE_DEPENDENCY_SCHEMA = "nerve.predictable_resource_selection.v1"
_PARTITIONED_ACCESS_FIELDS = frozenset(
    (
        "selection_signal",
        "execution_signal",
        "execution_calibration_word_base",
        "partition_axis",
        "parameter_ids",
    )
)
_INDEPENDENT_ACCESS_FIELDS = frozenset(
    (
        "selection_signal",
        "execution_signal",
        "execution_calibration_word_base",
        "mapping",
    )
)
_INDEPENDENT_MAPPING_FIELDS = frozenset(("selector", "parameter_ids"))
_COMPATIBILITY = {
    "device_api": "vulkan",
    "storage_class": "storage_buffer",
    "read_only": True,
    "required_features": [],
}


def analyze_lowered_resource_residency(
    *,
    lowered_index: Json,
    lowered_dir: Path,
    tensor_index: Json,
) -> Json:
    """Discover dynamic tensor partitions before package assets are written."""

    components: list[Json] = []
    for circuit_ref in lowered_index["graph"]["circuits"]:
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        components.append(
            _analysis_component(
                execution_scope="target",
                component_id=circuit_ref["id"],
                nodes=circuit["nodes"],
                parameter_refs=circuit["parameters"]["refs"],
            )
        )
    for draft in lowered_index.get("draft_execution_graphs", []):
        draft_id = _non_empty_string(draft.get("id"), "draft execution graph id")
        for circuit_ref in draft["circuits"]:
            circuit = read_json(lowered_dir / circuit_ref["circuit"])
            components.append(
                _analysis_component(
                    execution_scope=f"draft:{draft_id}",
                    component_id=circuit_ref["id"],
                    nodes=circuit["nodes"],
                    parameter_refs=circuit["parameters"]["refs"],
                )
            )
    return analyze_resource_residency_components(
        components=components,
        tensor_index=tensor_index,
        require_direct_packaging=True,
    )


def analyze_manifest_resource_residency(
    *,
    manifest: Json,
    tensor_index: Json,
) -> Json:
    """Re-analyze the compiled physical graph after circuit optimization."""

    components: list[Json] = []

    def collect(execution_scope: str, graph: Any) -> None:
        if not isinstance(graph, dict) or not isinstance(graph.get("components"), list):
            raise ModelCompileError(
                f"{execution_scope} compiled circuit graph is invalid"
            )
        for component in graph["components"]:
            if not isinstance(component, dict):
                raise ModelCompileError(
                    f"{execution_scope} compiled component is invalid"
                )
            circuit = component.get("circuit")
            params = component.get("params")
            if not isinstance(circuit, dict) or not isinstance(params, dict):
                raise ModelCompileError(
                    f"{execution_scope} compiled component is incomplete"
                )
            components.append(
                _analysis_component(
                    execution_scope=execution_scope,
                    component_id=component.get("component_id"),
                    nodes=circuit.get("nodes"),
                    parameter_refs=params.get("refs"),
                )
            )

    collect("target", manifest.get("circuit_graph"))
    decoders = manifest.get("speculative_decoders", [])
    if not isinstance(decoders, list):
        raise ModelCompileError("compiled speculative decoders are invalid")
    for decoder in decoders:
        if not isinstance(decoder, dict):
            raise ModelCompileError("compiled speculative decoder is invalid")
        decoder_id = _non_empty_string(
            decoder.get("id"), "compiled speculative decoder id"
        )
        collect(f"draft:{decoder_id}", decoder.get("circuit_graph"))

    return analyze_resource_residency_components(
        components=components,
        tensor_index=tensor_index,
        require_direct_packaging=False,
    )


def analyze_resource_residency_components(
    *,
    components: list[Json],
    tensor_index: Json,
    require_direct_packaging: bool,
) -> Json:
    """Classify physical parameter accesses without model-family knowledge.

    A selector is discovered solely through ``selection_domain`` metadata.
    A physical consumer opts parameters into that domain through
    ``selected_parameter_accesses``. Nothing is inferred from operation,
    component, parameter, tensor, or architecture names.
    """

    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict):
        raise ModelCompileError("tensor index has no tensor mapping")

    dynamic_semantics: dict[tuple[str, str, str, str], tuple[str, str, str, str]] = {}
    tensor_uses: dict[str, list[tuple[tuple[str, str, str, str], bool]]] = defaultdict(
        list
    )
    groups: list[Json] = []

    for component in components:
        scope = _non_empty_string(component.get("execution_scope"), "execution scope")
        component_id = _non_empty_string(component.get("component_id"), "component id")
        nodes = component.get("nodes")
        refs = component.get("parameter_refs")
        if not isinstance(nodes, list) or any(
            not isinstance(node, dict) for node in nodes
        ):
            raise ModelCompileError(
                f"{scope} component {component_id!r} has invalid nodes"
            )
        if not isinstance(refs, dict):
            raise ModelCompileError(
                f"{scope} component {component_id!r} has invalid parameter refs"
            )

        node_ids: set[str] = set()
        signal_producers: dict[str, tuple[int, Json]] = {}
        selectors: dict[str, Json] = {}
        for node_index, node in enumerate(nodes):
            node_id = _non_empty_string(node.get("id"), "physical node id")
            if node_id in node_ids:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} repeats node {node_id!r}"
                )
            node_ids.add(node_id)
            outputs = node.get("outputs", [])
            if not isinstance(outputs, list):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} node {node_id!r} has invalid outputs"
                )
            for output in outputs:
                output = _non_empty_string(output, "physical output signal")
                if output in signal_producers:
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} has multiple producers "
                        f"for signal {output!r}"
                    )
                signal_producers[output] = (node_index, node)

            attrs = node.get("attrs", {})
            if attrs is None:
                attrs = {}
            if not isinstance(attrs, dict):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} node {node_id!r} has invalid attrs"
                )
            domain = attrs.get("selection_domain")
            if domain is None:
                if attrs.get("predictable_dependency") is not None:
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} selector {node_id!r} "
                        "predictable dependency requires a selection domain"
                    )
                continue
            if not isinstance(domain, dict) or set(domain) != _SELECTION_DOMAIN_FIELDS:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector {node_id!r} "
                    "has an ambiguous selection domain"
                )
            domain_id = _non_empty_string(domain.get("id"), "selection domain id")
            resource_count = _positive_int(
                domain.get("resource_count"), "selection domain resource count"
            )
            selection_signal = _non_empty_string(
                domain.get("selection_signal"), "selection signal"
            )
            if selection_signal not in outputs:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector {node_id!r} "
                    f"does not produce declared selection signal {selection_signal!r}"
                )
            encoding = domain.get("encoding")
            if (
                not isinstance(encoding, dict)
                or set(encoding) != _SELECTION_ENCODING_FIELDS
                or encoding.get("element_type") != "u32"
            ):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector {node_id!r} "
                    "has an unsupported selection encoding"
                )
            predictable_dependency = attrs.get("predictable_dependency")
            if predictable_dependency is not None:
                if (
                    not isinstance(predictable_dependency, dict)
                    or set(predictable_dependency) != _PREDICTABLE_DEPENDENCY_FIELDS
                    or predictable_dependency.get("schema")
                    != _PREDICTABLE_DEPENDENCY_SCHEMA
                    or predictable_dependency.get("kind")
                    != "parameter_table_lookup"
                    or predictable_dependency.get("selection_semantics") != "exact"
                ):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} selector {node_id!r} "
                        "has an invalid predictable dependency"
                    )
                key_signal = _non_empty_string(
                    predictable_dependency.get("key_signal"),
                    "predictable dependency key signal",
                )
                table_parameter = _non_empty_string(
                    predictable_dependency.get("table_parameter"),
                    "predictable dependency table parameter",
                )
                inputs = node.get("inputs", [])
                params = node.get("params", [])
                if (
                    not isinstance(inputs, list)
                    or not isinstance(params, list)
                    or key_signal not in inputs
                    or table_parameter not in params
                ):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} selector {node_id!r} "
                        "predictable dependency is not bound by the selector"
                    )
            selection_count_per_activation = _positive_int(
                encoding.get("selection_count_per_activation"),
                "selection count per activation",
            )
            index_shift = _non_negative_int(
                encoding.get("index_shift"), "selection index shift"
            )
            index_mask = _positive_int(
                encoding.get("index_mask"), "selection index mask"
            )
            selection_calibration_word_base = _non_negative_int(
                encoding.get("calibration_word_base"),
                "selection calibration word base",
            )
            if (
                index_shift >= 32
                or index_mask > 0xFFFFFFFF
                or index_mask > 0xFFFFFFFF >> index_shift
                or index_mask & (index_mask + 1) != 0
                or (resource_count - 1) & index_mask != resource_count - 1
                or selection_count_per_activation > resource_count
                or selection_calibration_word_base > 0xFFFFFFFF
                or selection_calibration_word_base & (index_mask << index_shift)
            ):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector {node_id!r} "
                    "has an invalid selection index encoding"
                )
            selectors[node_id] = {
                "node_index": node_index,
                "node_id": node_id,
                "domain_id": domain_id,
                "resource_count": resource_count,
                "selection_signal": selection_signal,
                "encoding": {
                    "element_type": "u32",
                    "selection_count_per_activation": selection_count_per_activation,
                    "index_shift": index_shift,
                    "index_mask": index_mask,
                    "calibration_word_base": selection_calibration_word_base,
                },
                "outputs": set(outputs),
                "predictable_dependency": deepcopy(predictable_dependency),
                "accesses": [],
            }

        covered_node_parameters: set[tuple[str, str]] = set()
        for node_index, node in enumerate(nodes):
            node_id = str(node["id"])
            inputs = node.get("inputs", [])
            params = node.get("params", [])
            attrs = node.get("attrs", {}) or {}
            if not isinstance(inputs, list) or not isinstance(params, list):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} node {node_id!r} "
                    "has invalid inputs or parameters"
                )
            parameter_ids = [
                _non_empty_string(parameter, "physical parameter id")
                for parameter in params
            ]
            if len(parameter_ids) != len(set(parameter_ids)):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} node {node_id!r} "
                    "repeats a parameter"
                )
            accesses = attrs.get(SELECTED_PARAMETER_ACCESSES_ATTRIBUTE, [])
            if not isinstance(accesses, list) or any(
                not isinstance(access, dict) for access in accesses
            ):
                raise ModelCompileError(
                    f"{scope} component {component_id!r} node {node_id!r} "
                    "has invalid selected parameter accesses"
                )
            seen_signals: set[str] = set()
            for access in accesses:
                access_fields = set(access)
                if access_fields not in {
                    _PARTITIONED_ACCESS_FIELDS,
                    _INDEPENDENT_ACCESS_FIELDS,
                }:
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        "has an ambiguous selected parameter access"
                    )
                selection_signal = _non_empty_string(
                    access.get("selection_signal"), "selection signal"
                )
                execution_signal = _non_empty_string(
                    access.get("execution_signal"), "selected execution signal"
                )
                execution_calibration_word_base = _non_negative_int(
                    access.get("execution_calibration_word_base"),
                    "selected execution calibration word base",
                )
                if selection_signal in seen_signals:
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        f"repeats access for selection signal {selection_signal!r}"
                    )
                seen_signals.add(selection_signal)
                producer = signal_producers.get(selection_signal)
                selector = (
                    None if producer is None else selectors.get(str(producer[1]["id"]))
                )
                if (
                    producer is None
                    or selector is None
                    or producer[0] >= node_index
                    or selection_signal != selector["selection_signal"]
                ):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        f"access signal {selection_signal!r} is not produced by "
                        "an earlier resource selector"
                    )
                if execution_signal not in inputs:
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        f"does not consume selected execution signal {execution_signal!r}"
                    )
                index_field_mask = (
                    selector["encoding"]["index_mask"]
                    << selector["encoding"]["index_shift"]
                )
                if (
                    execution_calibration_word_base > 0xFFFFFFFF
                    or execution_calibration_word_base & index_field_mask
                ):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        "has an invalid selected calibration word base"
                    )
                group_key = (
                    scope,
                    component_id,
                    selector["node_id"],
                    selection_signal,
                )
                if access_fields == _PARTITIONED_ACCESS_FIELDS:
                    partition_axis = _non_negative_int(
                        access.get("partition_axis"), "selected partition axis"
                    )
                    selected_parameters = _selected_parameter_ids(
                        access.get("parameter_ids"),
                        scope=scope,
                        component_id=component_id,
                        node_id=node_id,
                    )
                    _reject_unknown_selected_parameters(
                        selected_parameters,
                        parameter_ids=parameter_ids,
                        scope=scope,
                        component_id=component_id,
                        node_id=node_id,
                    )
                    for parameter_slot, parameter_id in enumerate(
                        selected_parameters
                    ):
                        node_parameter = (node_id, parameter_id)
                        if node_parameter in covered_node_parameters:
                            raise ModelCompileError(
                                f"{scope} component {component_id!r} node {node_id!r} "
                                f"selects parameter {parameter_id!r} more than once"
                            )
                        covered_node_parameters.add(node_parameter)
                        parameter = refs.get(parameter_id)
                        if not isinstance(parameter, dict):
                            raise ModelCompileError(
                                f"{scope} component {component_id!r} node {node_id!r} "
                                f"references unknown parameter {parameter_id!r}"
                            )
                        tensor_name = _non_empty_string(
                            parameter.get("tensor"), "parameter tensor"
                        )
                        _validate_partitioned_tensor(
                            tensor_name=tensor_name,
                            metadata=tensors.get(tensor_name),
                            partition_axis=partition_axis,
                            partition_count=selector["resource_count"],
                            require_direct_packaging=require_direct_packaging,
                        )
                        semantic_key = (scope, component_id, node_id, parameter_id)
                        dynamic_semantics[semantic_key] = group_key
                        tensor_uses[tensor_name].append((semantic_key, True))
                        selector["accesses"].append(
                            {
                                "kind": "partitioned_tensor",
                                "node_index": node_index,
                                "node_id": node_id,
                                "parameter_id": parameter_id,
                                "tensor": tensor_name,
                                "partition_axis": partition_axis,
                                "parameter_slot": parameter_slot,
                                "selection_signal": selection_signal,
                                "execution_signal": execution_signal,
                                "execution_calibration_word_base": execution_calibration_word_base,
                            }
                        )
                    continue

                mapping = access.get("mapping")
                if (
                    not isinstance(mapping, list)
                    or len(mapping) != selector["resource_count"]
                ):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        "independent selected access must map every selector"
                    )
                mapped_selectors: list[int] = []
                for entry in mapping:
                    if not isinstance(entry, dict) or set(entry) != (
                        _INDEPENDENT_MAPPING_FIELDS
                    ):
                        raise ModelCompileError(
                            f"{scope} component {component_id!r} node {node_id!r} "
                            "has an ambiguous independent resource mapping"
                        )
                    selector_index = _non_negative_int(
                        entry.get("selector"), "independent resource selector"
                    )
                    mapped_selectors.append(selector_index)
                    selected_parameters = _selected_parameter_ids(
                        entry.get("parameter_ids"),
                        scope=scope,
                        component_id=component_id,
                        node_id=node_id,
                    )
                    _reject_unknown_selected_parameters(
                        selected_parameters,
                        parameter_ids=parameter_ids,
                        scope=scope,
                        component_id=component_id,
                        node_id=node_id,
                    )
                    for parameter_slot, parameter_id in enumerate(selected_parameters):
                        node_parameter = (node_id, parameter_id)
                        if node_parameter in covered_node_parameters:
                            raise ModelCompileError(
                                f"{scope} component {component_id!r} node {node_id!r} "
                                f"selects parameter {parameter_id!r} more than once"
                            )
                        covered_node_parameters.add(node_parameter)
                        parameter = refs.get(parameter_id)
                        if not isinstance(parameter, dict):
                            raise ModelCompileError(
                                f"{scope} component {component_id!r} node {node_id!r} "
                                f"references unknown parameter {parameter_id!r}"
                            )
                        tensor_name = _non_empty_string(
                            parameter.get("tensor"), "parameter tensor"
                        )
                        _validate_independent_tensor(
                            tensor_name=tensor_name,
                            metadata=tensors.get(tensor_name),
                            require_direct_packaging=require_direct_packaging,
                        )
                        semantic_key = (scope, component_id, node_id, parameter_id)
                        dynamic_semantics[semantic_key] = group_key
                        tensor_uses[tensor_name].append((semantic_key, True))
                        selector["accesses"].append(
                            {
                                "kind": "independent_resource",
                                "node_index": node_index,
                                "node_id": node_id,
                                "parameter_id": parameter_id,
                                "tensor": tensor_name,
                                "selector": selector_index,
                                "parameter_slot": parameter_slot,
                                "selection_signal": selection_signal,
                                "execution_signal": execution_signal,
                                "execution_calibration_word_base": execution_calibration_word_base,
                            }
                        )
                if mapped_selectors != list(range(selector["resource_count"])):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        "independent resource selectors must be contiguous and ordered"
                    )

            for parameter_id in parameter_ids:
                parameter = refs.get(parameter_id)
                if not isinstance(parameter, dict):
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} node {node_id!r} "
                        f"references unknown parameter {parameter_id!r}"
                    )
                semantic_key = (scope, component_id, node_id, parameter_id)
                if semantic_key in dynamic_semantics:
                    continue
                tensor_name = _non_empty_string(
                    parameter.get("tensor"), "parameter tensor"
                )
                if tensor_name not in tensors:
                    raise ModelCompileError(
                        f"compiled parameter references missing tensor {tensor_name!r}"
                    )
                tensor_uses[tensor_name].append((semantic_key, False))

        for selector in selectors.values():
            if not selector["accesses"]:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector "
                    f"{selector['node_id']!r} does not select any physical parameter access"
                )
            accesses = sorted(
                selector["accesses"],
                key=lambda access: (
                    access["node_index"],
                    access["node_id"],
                    access["parameter_id"],
                ),
            )
            access_kinds = {access["kind"] for access in accesses}
            selected_signals = {access["selection_signal"] for access in accesses}
            execution_signals = {access["execution_signal"] for access in accesses}
            execution_calibration_word_bases = {
                access["execution_calibration_word_base"] for access in accesses
            }
            if len(access_kinds) != 1:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector "
                    f"{selector['node_id']!r} mixes incompatible resource storage"
                )
            if len(selected_signals) != 1:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector "
                    f"{selector['node_id']!r} maps multiple selection signals"
                )
            if len(execution_signals) != 1 or len(execution_calibration_word_bases) != 1:
                raise ModelCompileError(
                    f"{scope} component {component_id!r} selector "
                    f"{selector['node_id']!r} maps incompatible execution records"
                )
            group = {
                "execution_scope": scope,
                "component_id": component_id,
                "selector_node_id": selector["node_id"],
                "selection_signal": next(iter(selected_signals)),
                "execution_signal": next(iter(execution_signals)),
                "execution_calibration_word_base": next(
                    iter(execution_calibration_word_bases)
                ),
                "encoding": deepcopy(selector["encoding"]),
                "domain_id": selector["domain_id"],
                "partition_count": selector["resource_count"],
                "resume_node_id": accesses[0]["node_id"],
            }
            if selector["predictable_dependency"] is not None:
                group["predictable_dependency"] = deepcopy(
                    selector["predictable_dependency"]
                )
            if access_kinds == {"partitioned_tensor"}:
                axes = {access["partition_axis"] for access in accesses}
                if len(axes) != 1:
                    raise ModelCompileError(
                        f"{scope} component {component_id!r} selector "
                        f"{selector['node_id']!r} mixes incompatible partition axes"
                    )
                group["storage"] = "partitioned_tensors"
                group["partition_axis"] = next(iter(axes))
                group["accesses"] = [
                    {
                        key: access[key]
                        for key in (
                            "node_id",
                            "parameter_id",
                            "tensor",
                            "partition_axis",
                            "parameter_slot",
                        )
                    }
                    for access in accesses
                ]
            else:
                group["storage"] = "independent_resources"
                group["accesses"] = [
                    {
                        key: access[key]
                        for key in (
                            "node_id",
                            "parameter_id",
                            "tensor",
                            "selector",
                            "parameter_slot",
                        )
                    }
                    for access in accesses
                ]
            groups.append(group)

    groups.sort(
        key=lambda group: (
            group["execution_scope"],
            group["component_id"],
            group["selector_node_id"],
            group["selection_signal"],
        )
    )
    groups_by_key = {
        (
            group["execution_scope"],
            group["component_id"],
            group["selector_node_id"],
            group["selection_signal"],
        ): group
        for group in groups
    }
    if len(groups_by_key) != len(groups):
        raise ModelCompileError(
            "compiler residency groups are not uniquely addressable"
        )

    dynamic_tensors: dict[str, Json] = {}
    spine_tensors: list[str] = []
    for tensor_name, uses in sorted(tensor_uses.items()):
        selected = [semantic for semantic, is_dynamic in uses if is_dynamic]
        unconditional = [semantic for semantic, is_dynamic in uses if not is_dynamic]
        if selected and unconditional:
            raise ModelCompileError(
                f"tensor {tensor_name!r} is both selected and unconditionally accessed"
            )
        if not selected:
            spine_tensors.append(tensor_name)
            continue
        owning_groups = {dynamic_semantics[semantic] for semantic in selected}
        storage_kinds = {
            groups_by_key[group_key]["storage"] for group_key in owning_groups
        }
        if len(storage_kinds) != 1:
            raise ModelCompileError(
                f"tensor {tensor_name!r} has incompatible dynamic selection groups"
            )
        if storage_kinds == {"independent_resources"}:
            dynamic_tensors[tensor_name] = {"storage": "independent_resource"}
            source_partition_count = tensors[tensor_name].get(
                SOURCE_INTEGRITY_PARTITION_COUNT_FIELD
            )
            if source_partition_count is not None:
                source_partition_count = _positive_int(
                    source_partition_count,
                    f"tensor {tensor_name!r} source integrity partition count",
                )
                if int(tensors[tensor_name]["byte_count"]) % source_partition_count:
                    raise ModelCompileError(
                        f"tensor {tensor_name!r} cannot be sealed into "
                        f"{source_partition_count} equal source ranges"
                    )
                dynamic_tensors[tensor_name][
                    SOURCE_INTEGRITY_PARTITION_COUNT_FIELD
                ] = source_partition_count
        else:
            partitionings = {
                (
                    groups_by_key[group_key]["partition_axis"],
                    groups_by_key[group_key]["partition_count"],
                )
                for group_key in owning_groups
            }
            if len(partitionings) != 1:
                raise ModelCompileError(
                    f"tensor {tensor_name!r} has incompatible dynamic selection groups"
                )
            partition_axis, partition_count = next(iter(partitionings))
            dynamic_tensors[tensor_name] = {
                "partition_axis": partition_axis,
                "partition_count": partition_count,
            }

    return {
        "schema": RESIDENCY_ANALYSIS_SCHEMA,
        "groups": groups,
        "dynamic_tensors": dynamic_tensors,
        "spine_tensors": sorted(spine_tensors),
    }


def partition_counts_for_packaging(analysis: Json) -> dict[str, int]:
    if analysis.get("schema") != RESIDENCY_ANALYSIS_SCHEMA:
        raise ModelCompileError("compiler residency analysis schema is invalid")
    dynamic_tensors = analysis.get("dynamic_tensors")
    if not isinstance(dynamic_tensors, dict):
        raise ModelCompileError("compiler residency analysis has no dynamic tensors")
    counts = {}
    for tensor_name, metadata in dynamic_tensors.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(metadata, dict)
        ):
            raise ModelCompileError(
                "compiler residency analysis has an invalid dynamic tensor"
            )
        if metadata.get("storage") == "independent_resource":
            source_partition_count = metadata.get(
                SOURCE_INTEGRITY_PARTITION_COUNT_FIELD
            )
            if source_partition_count is not None:
                counts[tensor_name] = _positive_int(
                    source_partition_count,
                    "independent source integrity partition count",
                )
            continue
        counts[tensor_name] = _positive_int(
            metadata.get("partition_count"), "dynamic tensor partition count"
        )
    return counts


def artifact_affinity_groups_for_packaging(analysis: Json) -> list[list[str]]:
    """Derive disjoint tensor banks from physical co-access topology.

    This is deliberately independent of model and operator names. Independently
    stored resources selected by the same signal are ordered by selector and
    parameter slot so one selected cohort occupies one contiguous byte run.
    Compatible views that share tensors are merged into one bank; the longest
    physical access order wins and remaining tensors are appended
    deterministically. Always-resident tensors form a separate bank when there
    is more than one of them.
    """

    if analysis.get("schema") != RESIDENCY_ANALYSIS_SCHEMA:
        raise ModelCompileError("compiler residency analysis schema is invalid")
    raw_groups = analysis.get("groups")
    spine_tensors = analysis.get("spine_tensors")
    if not isinstance(raw_groups, list) or not isinstance(spine_tensors, list):
        raise ModelCompileError("compiler residency analysis is incomplete")

    candidates: list[tuple[tuple[str, ...], list[str]]] = []
    normalized_spine = _unique_tensor_sequence(
        spine_tensors, "always-resident artifact affinity"
    )
    if len(normalized_spine) > 1:
        candidates.append((("always_resident",), sorted(normalized_spine)))

    for index, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            raise ModelCompileError(
                f"compiler residency group {index} is not an object"
            )
        if group.get("storage") != "independent_resources":
            continue
        accesses = group.get("accesses")
        if not isinstance(accesses, list) or not accesses:
            raise ModelCompileError(
                "independent residency group has no artifact affinity accesses"
            )
        ordered_accesses: list[tuple[int, int, str, str, str]] = []
        for access in accesses:
            if not isinstance(access, dict):
                raise ModelCompileError(
                    "independent artifact affinity access is not an object"
                )
            ordered_accesses.append(
                (
                    _non_negative_int(
                        access.get("selector"), "artifact affinity selector"
                    ),
                    _non_negative_int(
                        access.get("parameter_slot"),
                        "artifact affinity parameter slot",
                    ),
                    _non_empty_string(
                        access.get("node_id"), "artifact affinity node id"
                    ),
                    _non_empty_string(
                        access.get("parameter_id"),
                        "artifact affinity parameter id",
                    ),
                    _non_empty_string(
                        access.get("tensor"), "artifact affinity tensor"
                    ),
                )
            )
        ordered_accesses.sort()
        sequence = _unique_tensor_sequence(
            [access[-1] for access in ordered_accesses],
            "independent artifact affinity",
        )
        if len(sequence) < 2:
            continue
        candidate_key = (
            "selected",
            _non_empty_string(
                group.get("execution_scope"), "artifact affinity execution scope"
            ),
            _non_empty_string(
                group.get("component_id"), "artifact affinity component id"
            ),
            _non_empty_string(
                group.get("selector_node_id"), "artifact affinity selector node"
            ),
            _non_empty_string(
                group.get("selection_signal"), "artifact affinity selection signal"
            ),
        )
        candidates.append((candidate_key, sequence))

    components: list[dict[str, Any]] = []
    for key, sequence in sorted(candidates, key=lambda item: item[0]):
        tensor_set = set(sequence)
        overlapping = [
            component_index
            for component_index, component in enumerate(components)
            if tensor_set.intersection(component["tensors"])
        ]
        if not overlapping:
            components.append(
                {"tensors": tensor_set, "orders": [(key, sequence)]}
            )
            continue
        first = overlapping[0]
        merged_tensors = set(tensor_set)
        merged_orders = [(key, sequence)]
        for component_index in reversed(overlapping):
            component = components.pop(component_index)
            merged_tensors.update(component["tensors"])
            merged_orders.extend(component["orders"])
        components.insert(
            min(first, len(components)),
            {"tensors": merged_tensors, "orders": merged_orders},
        )

    affinity_groups = []
    for component in components:
        orders = sorted(
            component["orders"],
            key=lambda item: (-len(item[1]), tuple(item[1]), item[0]),
        )
        ordered_tensors: list[str] = []
        seen: set[str] = set()
        for _, sequence in orders:
            for tensor_name in sequence:
                if tensor_name not in seen:
                    seen.add(tensor_name)
                    ordered_tensors.append(tensor_name)
        if seen != component["tensors"]:
            raise ModelCompileError("artifact affinity merge lost a tensor")
        affinity_groups.append(ordered_tensors)
    affinity_groups.sort(key=lambda group: tuple(group))
    return affinity_groups


def _unique_tensor_sequence(values: Any, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ModelCompileError(f"{label} must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tensor_name = _non_empty_string(value, f"{label} tensor")
        if tensor_name not in seen:
            seen.add(tensor_name)
            result.append(tensor_name)
    return result


def build_planned_resource_residency_contract(
    *,
    package_dir: Path,
    tensor_index: Json,
    manifest: Json,
) -> Json:
    analysis = analyze_manifest_resource_residency(
        manifest=manifest,
        tensor_index=tensor_index,
    )
    tensors = tensor_index["tensors"]
    parameter_bindings = compiled_parameter_bindings(manifest)
    source_headers, artifact_byte_counts = compiled_resource_artifact_metadata(
        package_dir, tensor_index
    )
    dynamic_tensors = set(analysis["dynamic_tensors"])
    referenced_tensors = set(parameter_bindings)
    if dynamic_tensors | set(analysis["spine_tensors"]) != referenced_tensors:
        raise ModelCompileError(
            "compiler residency analysis does not exactly cover compiled tensors"
        )

    spine_resource_by_tensor = {
        tensor_name: compiled_immutable_resource(
            package_dir=package_dir,
            tensor_index=tensor_index,
            tensor_name=tensor_name,
            lifetime="always_resident",
            source_headers=source_headers,
            artifact_byte_counts=artifact_byte_counts,
        )
        for tensor_name in sorted(referenced_tensors - dynamic_tensors)
    }
    spine_resources = list(spine_resource_by_tensor.values())
    resources_by_id = {resource["id"]: resource for resource in spine_resources}
    if len(resources_by_id) != len(spine_resources):
        # Equal immutable content is intentionally shareable.
        spine_resources = list(resources_by_id.values())
    if not spine_resources:
        raise ModelCompileError(
            "compiled package has no always-resident execution spine"
        )
    spine_group = {
        "id": "",
        "lifetime": "always_resident",
        "resource_ids": sorted(resources_by_id),
        "dependencies": [],
    }
    spine_group["id"] = atomic_group_identity(spine_group)

    atomic_groups_by_id = {spine_group["id"]: spine_group}
    templates_by_id: dict[str, Json] = {}
    selectors: list[Json] = []
    checkpoints: list[Json] = []
    dynamic_binding_mapping: dict[tuple[str, str, str, str], Json] = {}
    dynamic_resource_by_tensor: dict[str, Json] = {}
    seen_dynamic_tensors: set[str] = set()

    def dynamic_resource(tensor_name: str) -> Json:
        resource = dynamic_resource_by_tensor.get(tensor_name)
        if resource is None:
            resource = compiled_immutable_resource(
                package_dir=package_dir,
                tensor_index=tensor_index,
                tensor_name=tensor_name,
                lifetime="dynamic",
                source_headers=source_headers,
                artifact_byte_counts=artifact_byte_counts,
            )
            dynamic_resource_by_tensor[tensor_name] = resource
        return resource

    for group in analysis["groups"]:
        if group["storage"] == "independent_resources":
            accesses_by_selector: list[list[Json]] = [
                [] for _ in range(group["partition_count"])
            ]
            for access in group["accesses"]:
                accesses_by_selector[access["selector"]].append(access)
            atomic_group_ids: list[str] = []
            for selected_accesses in accesses_by_selector:
                if not selected_accesses:
                    raise ModelCompileError(
                        "independent residency group has an empty selector resource"
                    )
                group_resource_ids: set[str] = set()
                for access in selected_accesses:
                    tensor_name = access["tensor"]
                    seen_dynamic_tensors.add(tensor_name)
                    resource = dynamic_resource(tensor_name)
                    resource_id = resource["id"]
                    resources_by_id.setdefault(resource_id, resource)
                    group_resource_ids.add(resource_id)
                atomic_group = {
                    "id": "",
                    "lifetime": "dynamic",
                    "resource_ids": sorted(group_resource_ids),
                    "dependencies": [spine_group["id"]],
                }
                atomic_group["id"] = atomic_group_identity(atomic_group)
                atomic_groups_by_id.setdefault(atomic_group["id"], atomic_group)
                atomic_group_ids.append(atomic_group["id"])
                for access in selected_accesses:
                    semantic_key = (
                        group["execution_scope"],
                        group["component_id"],
                        access["node_id"],
                        access["parameter_id"],
                    )
                    tensor_name = access["tensor"]
                    resource = dynamic_resource(tensor_name)
                    dynamic_binding_mapping[semantic_key] = {
                        "kind": "selected_atomic_group",
                        "atomic_group_id": atomic_group["id"],
                        "resource_id": resource["id"],
                        "selection_signal": group["selection_signal"],
                        "selector_index": access["selector"],
                        "parameter_slot": access["parameter_slot"],
                    }
            selector_mapping = {
                "kind": "group_table",
                "atomic_group_ids": atomic_group_ids,
            }
        elif group["storage"] == "partitioned_tensors":
            members = []
            tensor_to_seed: dict[str, str] = {}
            for tensor_name in sorted(
                {access["tensor"] for access in group["accesses"]}
            ):
                seen_dynamic_tensors.add(tensor_name)
                metadata = tensors.get(tensor_name)
                member = _partition_member_template(
                    tensor_name=tensor_name,
                    metadata=metadata,
                    partition_count=group["partition_count"],
                    partition_axis=group["partition_axis"],
                )
                tensor_to_seed[tensor_name] = member["resource_identity_seed"]
                members.append(member)
            members.sort(key=lambda member: member["resource_identity_seed"])
            group_seed = partition_group_identity_seed(
                group["partition_count"],
                [member["resource_identity_seed"] for member in members],
            )
            template = {
                "id": "",
                "partition_count": group["partition_count"],
                "lifetime": "dynamic",
                "group_identity_seed": group_seed,
                "member_templates": members,
                "dependencies": [spine_group["id"]],
            }
            template["id"] = partition_template_identity(template)
            templates_by_id.setdefault(template["id"], template)
            selector_mapping = {
                "kind": "partition_template",
                "partition_template_id": template["id"],
            }
            for access in group["accesses"]:
                semantic_key = (
                    group["execution_scope"],
                    group["component_id"],
                    access["node_id"],
                    access["parameter_id"],
                )
                dynamic_binding_mapping[semantic_key] = {
                    "kind": "partition_template_member",
                    "partition_template_id": template["id"],
                    "resource_identity_seed": tensor_to_seed[access["tensor"]],
                    "selection_signal": group["selection_signal"],
                    "parameter_slot": access["parameter_slot"],
                }
        else:
            raise ModelCompileError(
                f"unsupported compiler residency storage {group['storage']!r}"
            )

        selector = {
            "id": "",
            "execution_scope": group["execution_scope"],
            "component_id": group["component_id"],
            "node_id": group["selector_node_id"],
            "domain_id": group["domain_id"],
            "resource_count": group["partition_count"],
            "selection_signal": group["selection_signal"],
            "execution_signal": group["execution_signal"],
            "execution_calibration_word_base": group[
                "execution_calibration_word_base"
            ],
            "encoding": deepcopy(group["encoding"]),
            "mapping": selector_mapping,
        }
        selector["id"] = selector_identity(selector)
        selectors.append(selector)
        checkpoint = {
            "id": "",
            "execution_scope": group["execution_scope"],
            "component_id": group["component_id"],
            "after_node_id": group["selector_node_id"],
            "resume_node_id": group["resume_node_id"],
            "selector_ids": [selector["id"]],
        }
        checkpoint["id"] = checkpoint_identity(checkpoint)
        checkpoints.append(checkpoint)

    if seen_dynamic_tensors != dynamic_tensors:
        raise ModelCompileError(
            "dynamic residency templates do not exactly cover selected tensors"
        )

    bindings = []
    for tensor_name, uses in parameter_bindings.items():
        for use in uses:
            semantic_key = (
                use["execution_scope"],
                use["component_id"],
                use["node_id"],
                use["parameter_id"],
            )
            mapping = dynamic_binding_mapping.get(semantic_key)
            if mapping is None:
                mapping = {
                    "kind": "atomic_group",
                    "atomic_group_id": spine_group["id"],
                    "resource_id": spine_resource_by_tensor[tensor_name]["id"],
                }
            bindings.append({**use, "mapping": mapping})
    bindings.sort(key=_binding_sort_key)

    contract = {
        "schema": RESOURCE_RESIDENCY_SCHEMA,
        "identity_algorithm": RESOURCE_IDENTITY_ALGORITHM,
        "state_machine_schema": RESOURCE_STATE_MACHINE_SCHEMA,
        "supported_policies": list(SUPPORTED_RESIDENCY_POLICIES),
        "resources": sorted(
            resources_by_id.values(), key=lambda resource: resource["id"]
        ),
        "atomic_groups": sorted(
            atomic_groups_by_id.values(), key=lambda group: group["id"]
        ),
        "partition_templates": sorted(
            templates_by_id.values(), key=lambda template: template["id"]
        ),
        "bindings": bindings,
        "selectors": sorted(selectors, key=lambda selector: selector["id"]),
        "checkpoints": sorted(checkpoints, key=lambda checkpoint: checkpoint["id"]),
    }
    validate_resource_residency_contract(package_dir, contract, manifest)
    return contract


def _analysis_component(
    *,
    execution_scope: Any,
    component_id: Any,
    nodes: Any,
    parameter_refs: Any,
) -> Json:
    return {
        "execution_scope": execution_scope,
        "component_id": component_id,
        "nodes": deepcopy(nodes),
        "parameter_refs": deepcopy(parameter_refs),
    }


def _selected_parameter_ids(
    value: Any,
    *,
    scope: str,
    component_id: str,
    node_id: str,
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(parameter, str) or not parameter for parameter in value)
        or len(set(value)) != len(value)
    ):
        raise ModelCompileError(
            f"{scope} component {component_id!r} node {node_id!r} selected "
            "parameters must be non-empty and unique"
        )
    return value


def _reject_unknown_selected_parameters(
    selected_parameters: list[str],
    *,
    parameter_ids: list[str],
    scope: str,
    component_id: str,
    node_id: str,
) -> None:
    unknown_parameters = set(selected_parameters).difference(parameter_ids)
    if unknown_parameters:
        raise ModelCompileError(
            f"{scope} component {component_id!r} node {node_id!r} selects "
            f"parameters it does not access: {sorted(unknown_parameters)}"
        )


def _validate_independent_tensor(
    *,
    tensor_name: str,
    metadata: Any,
    require_direct_packaging: bool,
) -> None:
    if not isinstance(metadata, dict):
        raise ModelCompileError(
            f"selected parameter references missing tensor {tensor_name!r}"
        )
    shape = metadata.get("shape")
    byte_count = metadata.get("byte_count")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in shape
        )
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
    ):
        raise ModelCompileError(
            f"independently selected tensor {tensor_name!r} has invalid storage"
        )
    layout = metadata.get("layout")
    if layout is not None and layout != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"selected tensor {tensor_name!r} has unsupported layout {layout!r}"
        )
    if require_direct_packaging and (
        metadata.get("derived") is not None or metadata.get("compile_only") is True
    ):
        raise ModelCompileError(
            f"selected tensor {tensor_name!r} requires a non-atomic packaging transform"
        )


def _validate_partitioned_tensor(
    *,
    tensor_name: str,
    metadata: Any,
    partition_axis: int,
    partition_count: int,
    require_direct_packaging: bool,
) -> None:
    if not isinstance(metadata, dict):
        raise ModelCompileError(
            f"selected parameter references missing tensor {tensor_name!r}"
        )
    shape = metadata.get("shape")
    byte_count = metadata.get("byte_count")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in shape
        )
        or partition_axis >= len(shape)
        or shape[partition_axis] != partition_count
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
        or byte_count % partition_count
    ):
        raise ModelCompileError(
            f"tensor {tensor_name!r} cannot be partitioned into the selected domain"
        )
    if partition_axis != 0:
        raise ModelCompileError(
            f"row-major tensor {tensor_name!r} has a non-contiguous selected axis; "
            "compile a partition-major physical representation first"
        )
    layout = metadata.get("layout")
    if layout is not None and layout != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"selected tensor {tensor_name!r} has unsupported layout {layout!r}"
        )
    if require_direct_packaging and (
        metadata.get("derived") is not None or metadata.get("compile_only") is True
    ):
        raise ModelCompileError(
            f"selected tensor {tensor_name!r} requires a non-atomic packaging transform"
        )


def _partition_member_template(
    *,
    tensor_name: str,
    metadata: Any,
    partition_count: int,
    partition_axis: int,
) -> Json:
    if not isinstance(metadata, dict):
        raise ModelCompileError(
            f"dynamic tensor {tensor_name!r} is absent from the package index"
        )
    _validate_partitioned_tensor(
        tensor_name=tensor_name,
        metadata=metadata,
        partition_axis=partition_axis,
        partition_count=partition_count,
        require_direct_packaging=False,
    )
    integrity = metadata.get("partition_integrity")
    expected_fields = {
        "schema",
        "partition_axis",
        "partition_count",
        "partition_byte_count",
        "digest_table_path",
        "digest_table_byte_offset",
        "digest_stride_bytes",
        "table_sha256",
    }
    if not isinstance(integrity, dict) or set(integrity) != expected_fields:
        raise ModelCompileError(
            f"dynamic tensor {tensor_name!r} has no exact partition integrity contract"
        )
    partition_bytes = int(metadata["byte_count"]) // partition_count
    if (
        integrity.get("schema") != TENSOR_PARTITION_INTEGRITY_SCHEMA
        or integrity.get("partition_axis") != partition_axis
        or integrity.get("partition_count") != partition_count
        or integrity.get("partition_byte_count") != partition_bytes
        or integrity.get("digest_stride_bytes") != 32
        or not _is_sha256(integrity.get("table_sha256"))
    ):
        raise ModelCompileError(
            f"dynamic tensor {tensor_name!r} partition integrity is inconsistent"
        )
    source_file = _safe_relative_path(
        metadata.get("source_file"), f"dynamic tensor {tensor_name!r} source"
    )
    digest_path = _safe_relative_path(
        integrity.get("digest_table_path"),
        f"dynamic tensor {tensor_name!r} digest table",
    )
    header_bytes = metadata.get("safetensors_header_bytes")
    if not isinstance(header_bytes, int) or isinstance(header_bytes, bool):
        raise ModelCompileError(
            f"dynamic tensor {tensor_name!r} has no packaged header size"
        )
    offsets = metadata.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
            for offset in offsets
        )
    ):
        raise ModelCompileError(
            f"dynamic tensor {tensor_name!r} has invalid packaged offsets"
        )
    base = 8 + header_bytes + offsets[0]
    alignment = _largest_common_power_of_two_divisor(base, partition_bytes)
    resource_seed = residency_content_id(
        "partition_resource_seed",
        {
            "data_sha256": metadata.get("data_sha256"),
            "dtype": metadata.get("dtype"),
            "partition_axis": partition_axis,
            "partition_count": partition_count,
            "partition_shape": [
                1,
                *list(metadata["shape"])[1:],
            ],
            "partition_byte_count": partition_bytes,
            "compatibility": _COMPATIBILITY,
        },
    )
    return {
        "resource_identity_seed": resource_seed,
        "range_templates": [
            {
                "artifact_path": source_file,
                "base_byte_offset": base,
                "stride_bytes": partition_bytes,
                "byte_count": partition_bytes,
                "alignment_bytes": alignment,
                "integrity": {
                    "algorithm": "sha256_table",
                    "digest_table_path": digest_path,
                    "digest_table_byte_offset": _non_negative_int(
                        integrity.get("digest_table_byte_offset"),
                        "partition digest table offset",
                    ),
                    "digest_stride_bytes": 32,
                    "table_sha256": integrity["table_sha256"],
                },
            }
        ],
        "compatibility": deepcopy(_COMPATIBILITY),
    }


def _binding_sort_key(binding: Json) -> tuple[str, str, str, str, str]:
    mapping = binding["mapping"]
    mapping_key = "|".join(
        str(mapping.get(field, ""))
        for field in (
            "kind",
            "atomic_group_id",
            "resource_id",
            "selection_signal",
            "selector_index",
            "parameter_slot",
            "partition_template_id",
            "resource_identity_seed",
        )
    )
    return (
        binding["execution_scope"],
        binding["component_id"],
        binding["node_id"],
        binding["parameter_id"],
        mapping_key,
    )


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelCompileError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelCompileError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelCompileError(f"{label} must be a non-negative integer")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    value = _non_empty_string(value, label)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelCompileError(f"{label} must stay inside the compiled package")
    return value


def _largest_common_power_of_two_divisor(first: int, second: int) -> int:
    common = first | second
    if common == 0:
        return 1
    return min(common & -common, 4096)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
