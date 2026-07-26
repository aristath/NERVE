from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    canonical_json_bytes,
    stable_contract_id,
)


REPRESENTATION_GRAPH_SCHEMA = "nerve.optimizer.representation_graph.v1"
_RESOURCE_KINDS = frozenset({"parameter", "state", "topology"})
_NODE_KINDS = frozenset({"operator", "transducer", "correction"})
_CONFIDENCE_MODES = frozenset({"exact", "verified_approximation", "unresolved"})
_PORT_DIRECTIONS = frozenset({"input", "output"})


@dataclass(frozen=True)
class RepresentationGraphDocument:
    """A validated, immutable copy of one alternative physical graph."""

    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> RepresentationGraphDocument:
        normalized = deepcopy(document)
        validate_representation_graph(normalized)
        return cls(normalized)

    @property
    def graph_id(self) -> str:
        return str(self._document["graph_id"])

    def to_json(self) -> Json:
        return deepcopy(self._document)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self._document)


def representation_graph_id(document: Json) -> str:
    unsigned = deepcopy(document)
    unsigned.pop("graph_id", None)
    return stable_contract_id("representation_graph", unsigned)


def finalize_representation_graph(document: Json) -> Json:
    result = deepcopy(document)
    result["graph_id"] = representation_graph_id(result)
    validate_representation_graph(result)
    return result


def validate_representation_graph(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "graph_id",
            "candidate_id",
            "scope_ids",
            "source_contract_digests",
            "logical_contracts",
            "physical_representations",
            "signals",
            "resources",
            "nodes",
            "connections",
            "public_ports",
            "islands",
            "absorbed_transforms",
            "physical_kernels",
            "confidence",
            "unresolved",
            "correction_requests",
        },
        "$",
    )
    if document["schema"] != REPRESENTATION_GRAPH_SCHEMA:
        raise ContractValidationError(
            f"unsupported representation graph schema {document['schema']!r}"
        )
    _string(document["candidate_id"], "candidate_id")
    scope_ids = _sorted_strings(document["scope_ids"], "scope_ids", nonempty=True)
    source_digests = _object(
        document["source_contract_digests"], "source_contract_digests"
    )
    if list(source_digests) != sorted(source_digests):
        raise ContractValidationError("source_contract_digests keys must be sorted")
    if set(source_digests) != set(scope_ids):
        raise ContractValidationError(
            "source_contract_digests must cover exactly the graph scope_ids"
        )
    for scope_id, digest in source_digests.items():
        _string(digest, f"source_contract_digests.{scope_id}")

    logical_contracts = _records_by_id(
        document["logical_contracts"], "logical_contracts"
    )
    for record in logical_contracts.values():
        _fields(record, {"id", "signal", "shape", "dtype"}, "logical_contract")
        _string(record["signal"], f"logical_contracts.{record['id']}.signal")
        _shape(record["shape"], f"logical_contracts.{record['id']}.shape")
        _string(record["dtype"], f"logical_contracts.{record['id']}.dtype")

    representations = _records_by_id(
        document["physical_representations"], "physical_representations"
    )
    for record in representations.values():
        _fields(
            record,
            {"id", "kind", "domain", "physical_shape", "encoding", "storage"},
            "physical_representation",
        )
        _string(record["kind"], f"physical_representations.{record['id']}.kind")
        _string(record["domain"], f"physical_representations.{record['id']}.domain")
        _shape(
            record["physical_shape"],
            f"physical_representations.{record['id']}.physical_shape",
        )
        _object(
            record["encoding"],
            f"physical_representations.{record['id']}.encoding",
            nonempty=True,
        )
        _object(
            record["storage"],
            f"physical_representations.{record['id']}.storage",
            nonempty=True,
        )

    signals = _records_by_id(document["signals"], "signals")
    for record in signals.values():
        _fields(
            record,
            {
                "id",
                "logical_contract_id",
                "physical_representation_id",
                "provenance",
            },
            "signal",
        )
        _reference(
            record["logical_contract_id"],
            logical_contracts,
            f"signals.{record['id']}.logical_contract_id",
        )
        _reference(
            record["physical_representation_id"],
            representations,
            f"signals.{record['id']}.physical_representation_id",
        )
        if representations[record["physical_representation_id"]]["domain"] != "signal":
            raise ContractValidationError(
                f"signals.{record['id']} must use a signal representation"
            )
        _provenance(record["provenance"], scope_ids, f"signals.{record['id']}")

    resources = _records_by_id(document["resources"], "resources")
    for record in resources.values():
        _fields(
            record,
            {
                "id",
                "kind",
                "logical_contract_id",
                "physical_representation_id",
                "artifact",
                "provenance",
            },
            "resource",
        )
        if record["kind"] not in _RESOURCE_KINDS:
            raise ContractValidationError(
                f"resources.{record['id']}.kind is unsupported"
            )
        _reference(
            record["logical_contract_id"],
            logical_contracts,
            f"resources.{record['id']}.logical_contract_id",
        )
        _reference(
            record["physical_representation_id"],
            representations,
            f"resources.{record['id']}.physical_representation_id",
        )
        if (
            representations[record["physical_representation_id"]]["domain"]
            != record["kind"]
        ):
            raise ContractValidationError(
                f"resources.{record['id']} physical domain does not match its kind"
            )
        artifact = _object(
            record["artifact"], f"resources.{record['id']}.artifact", nonempty=True
        )
        _string(artifact.get("path"), f"resources.{record['id']}.artifact.path")
        _provenance(record["provenance"], scope_ids, f"resources.{record['id']}")

    nodes = _records_by_id(document["nodes"], "nodes")
    for record in nodes.values():
        _fields(
            record,
            {
                "id",
                "kind",
                "operation",
                "inputs",
                "outputs",
                "resource_ids",
                "state_read_ids",
                "state_write_ids",
                "cost",
                "provenance",
            },
            "node",
        )
        if record["kind"] not in _NODE_KINDS:
            raise ContractValidationError(f"nodes.{record['id']}.kind is unsupported")
        _string(record["operation"], f"nodes.{record['id']}.operation")
        inputs = _node_ports(
            record["inputs"], signals, representations, f"nodes.{record['id']}.inputs"
        )
        outputs = _node_ports(
            record["outputs"],
            signals,
            representations,
            f"nodes.{record['id']}.outputs",
        )
        if not inputs and not outputs:
            raise ContractValidationError(
                f"nodes.{record['id']} must declare an input or output"
            )
        for field in ("resource_ids", "state_read_ids", "state_write_ids"):
            ids = _sorted_strings(
                record[field], f"nodes.{record['id']}.{field}", nonempty=False
            )
            for resource_id in ids:
                _reference(
                    resource_id,
                    resources,
                    f"nodes.{record['id']}.{field}",
                )
        for state_id in record["state_read_ids"] + record["state_write_ids"]:
            if resources[state_id]["kind"] != "state":
                raise ContractValidationError(
                    f"nodes.{record['id']} references non-state resource "
                    f"{state_id!r} as state"
                )
        cost = record["cost"]
        if record["kind"] == "transducer":
            _cost(cost, f"nodes.{record['id']}.cost", required=True)
            if not inputs or not outputs:
                raise ContractValidationError(
                    f"transducer {record['id']!r} requires input and output"
                )
            input_representations = {
                port["physical_representation_id"] for port in inputs.values()
            }
            output_representations = {
                port["physical_representation_id"] for port in outputs.values()
            }
            if input_representations == output_representations:
                raise ContractValidationError(
                    f"transducer {record['id']!r} does not change representation"
                )
        elif cost is not None:
            _cost(cost, f"nodes.{record['id']}.cost", required=False)
        _provenance(record["provenance"], scope_ids, f"nodes.{record['id']}")

    connections = _records_by_id(document["connections"], "connections")
    connected_inputs: set[tuple[str, str]] = set()
    connected_outputs: set[tuple[str, str]] = set()
    for record in connections.values():
        _fields(
            record,
            {
                "id",
                "producer",
                "consumer",
                "signal_id",
                "materializes_source",
            },
            "connection",
        )
        producer = _endpoint(
            record["producer"], nodes, "outputs", f"connections.{record['id']}.producer"
        )
        consumer = _endpoint(
            record["consumer"], nodes, "inputs", f"connections.{record['id']}.consumer"
        )
        signal_id = _reference(
            record["signal_id"], signals, f"connections.{record['id']}.signal_id"
        )
        producer_port = _ports_by_id(nodes[producer[0]]["outputs"])[producer[1]]
        consumer_port = _ports_by_id(nodes[consumer[0]]["inputs"])[consumer[1]]
        if producer_port["signal_id"] != signal_id or consumer_port["signal_id"] != signal_id:
            raise ContractValidationError(
                f"connection {record['id']!r} endpoints disagree with its signal"
            )
        if (
            producer_port["physical_representation_id"]
            != consumer_port["physical_representation_id"]
        ):
            raise ContractValidationError(
                f"connection {record['id']!r} has incompatible physical "
                "representations; use an explicit transducer node"
            )
        if not isinstance(record["materializes_source"], bool):
            raise ContractValidationError(
                f"connections.{record['id']}.materializes_source must be boolean"
            )
        if consumer in connected_inputs:
            raise ContractValidationError(
                f"consumer port {consumer!r} has more than one producer"
            )
        connected_inputs.add(consumer)
        connected_outputs.add(producer)

    public_ports = _records_by_id(document["public_ports"], "public_ports")
    public_bindings: set[tuple[str, str]] = set()
    for record in public_ports.values():
        _fields(
            record,
            {
                "id",
                "direction",
                "logical_contract_id",
                "signal_id",
                "node_id",
                "node_port_id",
            },
            "public_port",
        )
        if record["direction"] not in _PORT_DIRECTIONS:
            raise ContractValidationError(
                f"public_ports.{record['id']}.direction is unsupported"
            )
        logical_id = _reference(
            record["logical_contract_id"],
            logical_contracts,
            f"public_ports.{record['id']}.logical_contract_id",
        )
        signal_id = _reference(
            record["signal_id"], signals, f"public_ports.{record['id']}.signal_id"
        )
        if signals[signal_id]["logical_contract_id"] != logical_id:
            raise ContractValidationError(
                f"public port {record['id']!r} changes its logical contract"
            )
        node_id = _reference(
            record["node_id"], nodes, f"public_ports.{record['id']}.node_id"
        )
        field = "inputs" if record["direction"] == "input" else "outputs"
        port_id = _string(
            record["node_port_id"], f"public_ports.{record['id']}.node_port_id"
        )
        node_ports = _ports_by_id(nodes[node_id][field])
        if port_id not in node_ports or node_ports[port_id]["signal_id"] != signal_id:
            raise ContractValidationError(
                f"public port {record['id']!r} does not match its node port"
            )
        binding = (node_id, port_id)
        if binding in public_bindings:
            raise ContractValidationError(
                f"node port {binding!r} has duplicate public bindings"
            )
        public_bindings.add(binding)

    _validate_islands(
        document["islands"],
        scope_ids=scope_ids,
        nodes=nodes,
        connections=connections,
        representations=representations,
        public_ports=public_ports,
    )
    _validate_absorbed_transforms(
        document["absorbed_transforms"],
        scope_ids=scope_ids,
        nodes=nodes,
        resources=resources,
        representations=representations,
    )
    _validate_kernels(
        document["physical_kernels"], scope_ids=scope_ids, nodes=nodes
    )
    confidence = _validate_confidence(document["confidence"])
    unresolved = _validate_unresolved(document["unresolved"], scope_ids)
    correction_requests = _validate_corrections(
        document["correction_requests"],
        scope_ids=scope_ids,
        nodes=nodes,
        public_ports=public_ports,
    )
    if confidence == "exact" and (unresolved or correction_requests):
        raise ContractValidationError(
            "an exact representation graph cannot contain unresolved or correction requests"
        )
    if confidence == "verified_approximation" and not correction_requests:
        raise ContractValidationError(
            "verified approximation requires at least one correction request"
        )
    if confidence == "unresolved" and not unresolved:
        raise ContractValidationError(
            "unresolved confidence requires at least one unresolved record"
        )

    expected_id = representation_graph_id(document)
    if document["graph_id"] != expected_id:
        raise ContractValidationError(
            f"representation graph_id must be canonical {expected_id!r}"
        )


def _validate_islands(
    value: Any,
    *,
    scope_ids: list[str],
    nodes: dict[str, Json],
    connections: dict[str, Json],
    representations: dict[str, Json],
    public_ports: dict[str, Json],
) -> dict[str, Json]:
    islands = _records_by_id(value, "islands")
    for record in islands.values():
        _fields(
            record,
            {
                "id",
                "scope_ids",
                "node_ids",
                "connection_ids",
                "representation_ids",
                "boundary_port_ids",
            },
            "island",
        )
        island_scopes = _sorted_strings(
            record["scope_ids"], f"islands.{record['id']}.scope_ids", nonempty=True
        )
        if not set(island_scopes) <= set(scope_ids):
            raise ContractValidationError(
                f"islands.{record['id']}.scope_ids leave the graph scope"
            )
        for field, records in (
            ("node_ids", nodes),
            ("connection_ids", connections),
            ("representation_ids", representations),
            ("boundary_port_ids", public_ports),
        ):
            for item_id in _sorted_strings(
                record[field], f"islands.{record['id']}.{field}", nonempty=True
            ):
                _reference(item_id, records, f"islands.{record['id']}.{field}")
        island_nodes = set(record["node_ids"])
        shared_native_connection = False
        for connection_id in record["connection_ids"]:
            connection = connections[connection_id]
            producer = connection["producer"]["node_id"]
            consumer = connection["consumer"]["node_id"]
            if producer not in island_nodes or consumer not in island_nodes:
                raise ContractValidationError(
                    f"island connection {connection_id!r} leaves its node set"
                )
            signal_id = connection["signal_id"]
            producer_port = _ports_by_id(nodes[producer]["outputs"])[
                connection["producer"]["port_id"]
            ]
            representation_id = producer_port["physical_representation_id"]
            if representation_id not in record["representation_ids"]:
                raise ContractValidationError(
                    f"island connection {connection_id!r} uses undeclared representation"
                )
            producer_scopes = set(nodes[producer]["provenance"]["scope_ids"])
            consumer_scopes = set(nodes[consumer]["provenance"]["scope_ids"])
            if (
                not connection["materializes_source"]
                and producer_scopes != consumer_scopes
                and signal_id
            ):
                shared_native_connection = True
        if len(island_scopes) > 1 and not shared_native_connection:
            raise ContractValidationError(
                f"multi-scope island {record['id']!r} does not retain a native "
                "representation across scopes"
            )
    return islands


def _validate_absorbed_transforms(
    value: Any,
    *,
    scope_ids: list[str],
    nodes: dict[str, Json],
    resources: dict[str, Json],
    representations: dict[str, Json],
) -> dict[str, Json]:
    transforms = _records_by_id(value, "absorbed_transforms")
    for record in transforms.values():
        _fields(
            record,
            {
                "id",
                "kind",
                "source_representation_id",
                "target_representation_id",
                "adjacent_node_ids",
                "parameter_resource_ids",
                "proof_ref",
                "evidence_refs",
                "provenance",
            },
            "absorbed_transform",
        )
        _string(record["kind"], f"absorbed_transforms.{record['id']}.kind")
        source = _reference(
            record["source_representation_id"],
            representations,
            f"absorbed_transforms.{record['id']}.source_representation_id",
        )
        target = _reference(
            record["target_representation_id"],
            representations,
            f"absorbed_transforms.{record['id']}.target_representation_id",
        )
        if source == target:
            raise ContractValidationError(
                f"absorbed transform {record['id']!r} does not change representation"
            )
        for node_id in _sorted_strings(
            record["adjacent_node_ids"],
            f"absorbed_transforms.{record['id']}.adjacent_node_ids",
            nonempty=True,
        ):
            _reference(node_id, nodes, f"absorbed_transforms.{record['id']}")
        for resource_id in _sorted_strings(
            record["parameter_resource_ids"],
            f"absorbed_transforms.{record['id']}.parameter_resource_ids",
            nonempty=True,
        ):
            _reference(resource_id, resources, f"absorbed_transforms.{record['id']}")
            if resources[resource_id]["kind"] != "parameter":
                raise ContractValidationError(
                    f"absorbed transform {record['id']!r} must target parameters"
                )
        _string(record["proof_ref"], f"absorbed_transforms.{record['id']}.proof_ref")
        _sorted_strings(
            record["evidence_refs"],
            f"absorbed_transforms.{record['id']}.evidence_refs",
            nonempty=True,
        )
        _provenance(
            record["provenance"], scope_ids, f"absorbed_transforms.{record['id']}"
        )
    return transforms


def _validate_kernels(
    value: Any, *, scope_ids: list[str], nodes: dict[str, Json]
) -> dict[str, Json]:
    kernels = _records_by_id(value, "physical_kernels")
    for record in kernels.values():
        _fields(
            record,
            {
                "id",
                "node_ids",
                "artifact",
                "target_predicate",
                "cost",
                "provenance",
            },
            "physical_kernel",
        )
        for node_id in _sorted_strings(
            record["node_ids"],
            f"physical_kernels.{record['id']}.node_ids",
            nonempty=True,
        ):
            _reference(node_id, nodes, f"physical_kernels.{record['id']}.node_ids")
        artifact = _object(
            record["artifact"],
            f"physical_kernels.{record['id']}.artifact",
            nonempty=True,
        )
        _string(artifact.get("path"), f"physical_kernels.{record['id']}.artifact.path")
        _object(
            record["target_predicate"],
            f"physical_kernels.{record['id']}.target_predicate",
            nonempty=True,
        )
        _cost(record["cost"], f"physical_kernels.{record['id']}.cost", required=True)
        _provenance(
            record["provenance"], scope_ids, f"physical_kernels.{record['id']}"
        )
    return kernels


def _validate_confidence(value: Any) -> str:
    record = _object(value, "confidence")
    _fields(record, {"mode", "score", "basis", "evidence_refs"}, "confidence")
    mode = record["mode"]
    if mode not in _CONFIDENCE_MODES:
        raise ContractValidationError("confidence.mode is unsupported")
    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ContractValidationError("confidence.score must be numeric")
    if not 0.0 <= float(score) <= 1.0:
        raise ContractValidationError("confidence.score must be between zero and one")
    _string(record["basis"], "confidence.basis")
    _sorted_strings(record["evidence_refs"], "confidence.evidence_refs", nonempty=True)
    if mode == "exact" and float(score) != 1.0:
        raise ContractValidationError("exact confidence must have score 1")
    return str(mode)


def _validate_unresolved(value: Any, scope_ids: list[str]) -> dict[str, Json]:
    records = _records_by_id(value, "unresolved")
    for record in records.values():
        _fields(
            record,
            {"id", "subject_ids", "reason", "evidence_refs", "provenance"},
            "unresolved",
        )
        _sorted_strings(
            record["subject_ids"],
            f"unresolved.{record['id']}.subject_ids",
            nonempty=True,
        )
        _string(record["reason"], f"unresolved.{record['id']}.reason")
        _sorted_strings(
            record["evidence_refs"],
            f"unresolved.{record['id']}.evidence_refs",
            nonempty=True,
        )
        _provenance(record["provenance"], scope_ids, f"unresolved.{record['id']}")
    return records


def _validate_corrections(
    value: Any,
    *,
    scope_ids: list[str],
    nodes: dict[str, Json],
    public_ports: dict[str, Json],
) -> dict[str, Json]:
    records = _records_by_id(value, "correction_requests")
    for record in records.values():
        _fields(
            record,
            {
                "id",
                "trigger",
                "correction_node_id",
                "fallback_scope_ids",
                "output_port_ids",
                "error_contract",
                "provenance",
            },
            "correction_request",
        )
        _object(
            record["trigger"], f"correction_requests.{record['id']}.trigger", nonempty=True
        )
        node_id = _reference(
            record["correction_node_id"],
            nodes,
            f"correction_requests.{record['id']}.correction_node_id",
        )
        if nodes[node_id]["kind"] != "correction":
            raise ContractValidationError(
                f"correction request {record['id']!r} does not reference a "
                "correction node"
            )
        fallbacks = _sorted_strings(
            record["fallback_scope_ids"],
            f"correction_requests.{record['id']}.fallback_scope_ids",
            nonempty=True,
        )
        if not set(fallbacks) <= set(scope_ids):
            raise ContractValidationError(
                f"correction request {record['id']!r} leaves graph scopes"
            )
        for port_id in _sorted_strings(
            record["output_port_ids"],
            f"correction_requests.{record['id']}.output_port_ids",
            nonempty=True,
        ):
            _reference(port_id, public_ports, f"correction_requests.{record['id']}")
        _object(
            record["error_contract"],
            f"correction_requests.{record['id']}.error_contract",
            nonempty=True,
        )
        _provenance(
            record["provenance"], scope_ids, f"correction_requests.{record['id']}"
        )
    return records


def _node_ports(
    value: Any,
    signals: dict[str, Json],
    representations: dict[str, Json],
    path: str,
) -> dict[str, Json]:
    ports = _records_by_id(value, path)
    for record in ports.values():
        _fields(
            record,
            {"id", "signal_id", "physical_representation_id"},
            path,
        )
        signal_id = _reference(record["signal_id"], signals, f"{path}.signal_id")
        representation_id = _reference(
            record["physical_representation_id"],
            representations,
            f"{path}.physical_representation_id",
        )
        if signals[signal_id]["physical_representation_id"] != representation_id:
            raise ContractValidationError(
                f"{path} port {record['id']!r} has an incompatible physical "
                "representation; use an explicit transducer node"
            )
    return ports


def _endpoint(
    value: Any,
    nodes: dict[str, Json],
    field: str,
    path: str,
) -> tuple[str, str]:
    record = _object(value, path)
    _fields(record, {"node_id", "port_id"}, path)
    node_id = _reference(record["node_id"], nodes, f"{path}.node_id")
    port_id = _string(record["port_id"], f"{path}.port_id")
    if port_id not in _ports_by_id(nodes[node_id][field]):
        raise ContractValidationError(
            f"{path} references unknown {field[:-1]} port {port_id!r}"
        )
    return node_id, port_id


def _cost(value: Any, path: str, *, required: bool) -> None:
    if value is None:
        if required:
            raise ContractValidationError(f"{path} is required")
        return
    record = _object(value, path, nonempty=True)
    if "status" not in record:
        raise ContractValidationError(f"{path}.status is required")
    if record["status"] not in {"estimated", "measured"}:
        raise ContractValidationError(f"{path}.status is unsupported")
    metrics = _object(record.get("metrics"), f"{path}.metrics", nonempty=True)
    for name, metric in metrics.items():
        _string(name, f"{path}.metrics key")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ContractValidationError(f"{path}.metrics.{name} must be numeric")


def _provenance(value: Any, graph_scope_ids: list[str], path: str) -> None:
    record = _object(value, f"{path}.provenance")
    _fields(
        record,
        {"scope_ids", "source_node_ids", "evidence_refs", "transform_refs"},
        f"{path}.provenance",
    )
    scope_ids = _sorted_strings(
        record["scope_ids"], f"{path}.provenance.scope_ids", nonempty=True
    )
    if not set(scope_ids) <= set(graph_scope_ids):
        raise ContractValidationError(f"{path}.provenance leaves graph scopes")
    _sorted_strings(
        record["source_node_ids"],
        f"{path}.provenance.source_node_ids",
        nonempty=False,
    )
    _sorted_strings(
        record["evidence_refs"],
        f"{path}.provenance.evidence_refs",
        nonempty=True,
    )
    _sorted_strings(
        record["transform_refs"],
        f"{path}.provenance.transform_refs",
        nonempty=False,
    )


def _records_by_id(value: Any, path: str) -> dict[str, Json]:
    records = _list(value, path)
    result: dict[str, Json] = {}
    previous = ""
    for index, value in enumerate(records):
        record = _object(value, f"{path}[{index}]")
        record_id = _string(record.get("id"), f"{path}[{index}].id")
        if record_id in result:
            raise ContractValidationError(f"{path} contains duplicate id {record_id!r}")
        if previous and record_id < previous:
            raise ContractValidationError(f"{path} must be sorted by id")
        previous = record_id
        result[record_id] = record
    return result


def _ports_by_id(value: Any) -> dict[str, Json]:
    return {str(port["id"]): port for port in value}


def _fields(record: Json, expected: set[str], path: str) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractValidationError(
            f"{path} fields are invalid: missing={missing}, extra={extra}"
        )


def _reference(value: Any, records: dict[str, Json], path: str) -> str:
    record_id = _string(value, path)
    if record_id not in records:
        raise ContractValidationError(f"{path} references unknown id {record_id!r}")
    return record_id


def _object(value: Any, path: str, *, nonempty: bool = False) -> Json:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} must be an object")
    if nonempty and not value:
        raise ContractValidationError(f"{path} must not be empty")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be a list")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value


def _shape(value: Any, path: str) -> list[int]:
    shape = _list(value, path)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape):
        raise ContractValidationError(f"{path} must contain positive integers")
    return shape


def _sorted_strings(value: Any, path: str, *, nonempty: bool) -> list[str]:
    values = _list(value, path)
    if nonempty and not values:
        raise ContractValidationError(f"{path} must not be empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise ContractValidationError(f"{path} must contain non-empty strings")
    if values != sorted(set(values)):
        raise ContractValidationError(f"{path} must be sorted and unique")
    return values
