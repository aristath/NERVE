from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable

from nerve.compilation import Json, ModelCompileError


OPTIMIZATION_SCOPE_SCHEMA = "nerve.optimizer.optimization_scope.v1"
SOURCE_BEHAVIOR_CONTRACT_SCHEMA = "nerve.optimizer.source_behavior_contract.v1"
ALGEBRAIC_EVIDENCE_SCHEMA = "nerve.optimizer.algebraic_evidence.v1"
HARDWARE_PROCESS_PROFILE_SCHEMA = "nerve.optimizer.hardware_process_profile.v1"
REPRESENTATION_DESCRIPTOR_SCHEMA = "nerve.optimizer.representation_descriptor.v1"
REPRESENTATION_CANDIDATE_SCHEMA = "nerve.optimizer.representation_candidate.v1"
CANDIDATE_CONSTRUCTION_SCHEMA = "nerve.optimizer.candidate_construction.v1"
BENCHMARK_RECORD_SCHEMA = "nerve.optimizer.benchmark_record.v1"
VALIDATION_RECORD_SCHEMA = "nerve.optimizer.validation_record.v1"
PROMOTION_DECISION_SCHEMA = "nerve.optimizer.promotion_decision.v1"
RELOWERING_REQUEST_SCHEMA = "nerve.optimizer.relowering_request.v1"
CONTRACT_DIGEST_SCHEMA = "nerve.optimizer.canonical_json_sha256.v1"

OPTIMIZATION_SCOPE_KINDS = frozenset(
    {
        "operator",
        "semantic_module",
        "coupled_region",
        "layer",
        "stateful_system",
        "cross_layer_group",
        "representation_island",
        "input_transducer",
        "output_transducer",
        "sampler",
        "feedback_transducer",
    }
)


class ContractValidationError(ModelCompileError):
    """A versioned optimizer contract is malformed or internally inconsistent."""


Validator = Callable[[Json], None]


@dataclass(frozen=True)
class ContractDocument:
    """Validated JSON contract with copy-in/copy-out immutability."""

    _document: Json

    @classmethod
    def from_json(
        cls,
        document: Json,
        *,
        expected_schema: str | None = None,
    ) -> ContractDocument:
        normalized = deepcopy(document)
        validate_contract(normalized, expected_schema=expected_schema)
        return cls(normalized)

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_schema: str | None = None,
    ) -> ContractDocument:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractValidationError("optimizer contract is not valid JSON") from error
        if not isinstance(document, dict):
            raise ContractValidationError("optimizer contract must be a JSON object")
        return cls.from_json(document, expected_schema=expected_schema)

    @property
    def schema(self) -> str:
        return str(self._document["schema"])

    @property
    def digest(self) -> str:
        return contract_digest(self._document)

    def to_json(self) -> Json:
        return deepcopy(self._document)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self._document)


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value, "$")
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ContractValidationError(
            f"optimizer contract cannot be serialized canonically: {error}"
        ) from error
    return text.encode("utf-8")


def contract_digest(value: Any) -> str:
    return f"{CONTRACT_DIGEST_SCHEMA}:{sha256(canonical_json_bytes(value)).hexdigest()}"


def stable_contract_id(prefix: str, *identity_parts: Any) -> str:
    _require_nonempty_string(prefix, "stable id prefix")
    digest = sha256(canonical_json_bytes(list(identity_parts))).hexdigest()
    return f"{prefix}_{digest[:32]}"


def representation_descriptor_id(document: Json) -> str:
    unsigned = deepcopy(document)
    unsigned.pop("descriptor_id", None)
    return stable_contract_id("representation_descriptor", unsigned)


def validate_contract(document: Json, *, expected_schema: str | None = None) -> None:
    _validate_json_value(document, "$")
    schema = document.get("schema")
    if not isinstance(schema, str):
        raise ContractValidationError("optimizer contract schema must be a string")
    if expected_schema is not None and schema != expected_schema:
        raise ContractValidationError(
            f"expected optimizer schema {expected_schema!r}, found {schema!r}"
        )
    validator = _VALIDATORS.get(schema)
    if validator is None:
        raise ContractValidationError(f"unsupported optimizer contract schema {schema!r}")
    validator(document)


def source_behavior_contract_digest(document: Json) -> str:
    unsigned = deepcopy(document)
    unsigned.pop("contract_digest", None)
    return contract_digest(unsigned)


def _validate_optimization_scope(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "scope_id",
            "package_id",
            "kind",
            "members",
            "boundary",
            "source_contract_digest",
        },
    )
    _require_nonempty_string(document["scope_id"], "scope_id")
    _require_nonempty_string(document["package_id"], "package_id")
    if document["kind"] not in OPTIMIZATION_SCOPE_KINDS:
        raise ContractValidationError(
            f"optimization scope has unsupported kind {document['kind']!r}"
        )
    members = _require_object(document["members"], "members")
    _require_fields(
        members,
        {
            "component_ids",
            "semantic_module_ids",
            "source_node_ids",
        },
    )
    component_ids = _require_unique_strings(
        members["component_ids"], "members.component_ids", nonempty=True
    )
    _require_unique_strings(
        members["semantic_module_ids"], "members.semantic_module_ids"
    )
    _require_unique_strings(members["source_node_ids"], "members.source_node_ids")
    boundary = _require_object(document["boundary"], "boundary")
    _require_fields(
        boundary,
        {
            "inputs",
            "outputs",
            "parameters",
            "states",
            "controls",
            "randomness",
        },
    )
    for field in boundary:
        _require_reference_list(boundary[field], f"boundary.{field}")
    _require_digest(document["source_contract_digest"], "source_contract_digest")
    expected_id = stable_contract_id(
        "scope",
        document["package_id"],
        document["kind"],
        component_ids,
        members["semantic_module_ids"],
        members["source_node_ids"],
    )
    if document["scope_id"] != expected_id:
        raise ContractValidationError(
            f"scope_id must be the stable semantic identity {expected_id!r}"
        )


def _validate_source_behavior_contract(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "scope_id",
            "semantic_role",
            "interface",
            "exact_reference",
            "contract_digest",
        },
    )
    _require_stable_id(document["scope_id"], "scope", "scope_id")
    _require_nonempty_string(document["semantic_role"], "semantic_role")
    interface = _require_object(document["interface"], "interface")
    _require_fields(
        interface,
        {
            "inputs",
            "outputs",
            "parameters",
            "states",
            "controls",
            "randomness",
        },
    )
    for field in interface:
        _require_reference_list(interface[field], f"interface.{field}")
    exact_reference = _require_object(document["exact_reference"], "exact_reference")
    _require_fields(exact_reference, {"implementation_id", "artifact_refs"})
    _require_nonempty_string(
        exact_reference["implementation_id"],
        "exact_reference.implementation_id",
    )
    _require_unique_strings(
        exact_reference["artifact_refs"],
        "exact_reference.artifact_refs",
        nonempty=True,
    )
    _require_digest(document["contract_digest"], "contract_digest")
    expected = source_behavior_contract_digest(document)
    if document["contract_digest"] != expected:
        raise ContractValidationError(
            "source behavior contract digest does not match its canonical content"
        )


def _validate_algebraic_evidence(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "evidence_id",
            "scope_id",
            "source_contract_digest",
            "analyzer",
            "claims",
            "artifacts",
        },
    )
    _require_stable_id(document["evidence_id"], "evidence", "evidence_id")
    _require_stable_id(document["scope_id"], "scope", "scope_id")
    _require_digest(document["source_contract_digest"], "source_contract_digest")
    _require_implementation_identity(document["analyzer"], "analyzer")
    claims = _require_list(document["claims"], "claims")
    for index, claim in enumerate(claims):
        claim = _require_object(claim, f"claims[{index}]")
        _require_fields(claim, {"kind", "status", "exact", "facts"})
        _require_nonempty_string(claim["kind"], f"claims[{index}].kind")
        if claim["status"] not in {"supported", "rejected", "inconclusive"}:
            raise ContractValidationError(
                f"claims[{index}].status has unsupported value {claim['status']!r}"
            )
        if not isinstance(claim["exact"], bool):
            raise ContractValidationError(f"claims[{index}].exact must be boolean")
        _require_object(claim["facts"], f"claims[{index}].facts")
    _require_artifact_refs(document["artifacts"], "artifacts")


def _validate_hardware_process_profile(document: Json) -> None:
    if "extensions" in document:
        raise ContractValidationError(
            "hardware profile extensions must be classified as capability_extensions, "
            "identity_extensions, or runtime_bindings"
        )
    _require_fields(
        document,
        {
            "schema",
            "profile_id",
            "hardware_identity",
            "capability_class",
            "processes",
            "memory_domains",
            "interconnects",
            "measurements",
            "provenance",
            "capability_extensions",
            "identity_extensions",
            "runtime_bindings",
        },
    )
    _require_stable_id(document["profile_id"], "hardware_profile", "profile_id")
    identity = _require_object(document["hardware_identity"], "hardware_identity")
    _require_fields(
        identity,
        {
            "device_kind",
            "vendor_id",
            "device_id",
            "stable_device_id",
            "name",
            "architecture",
            "physical_location",
        },
    )
    if identity["device_kind"] not in {"cpu", "gpu"}:
        raise ContractValidationError("hardware_identity.device_kind must be cpu or gpu")
    for field in (
        "vendor_id",
        "device_id",
        "stable_device_id",
        "name",
        "architecture",
        "physical_location",
    ):
        _require_nonempty_string(identity[field], f"hardware_identity.{field}")
    _require_stable_id(
        document["capability_class"],
        "hardware_capability",
        "capability_class",
    )
    processes = _require_list(document["processes"], "processes")
    if not processes:
        raise ContractValidationError("processes must not be empty")
    process_names = []
    for index, raw_process in enumerate(processes):
        path = f"processes[{index}]"
        process = _require_object(raw_process, path)
        _require_fields(
            process,
            {
                "name",
                "category",
                "availability",
                "programmability",
                "api",
                "operations",
                "numeric_formats",
                "required_extensions",
                "required_features",
                "limits",
                "properties",
            },
        )
        process_names.append(_require_nonempty_string(process["name"], f"{path}.name"))
        if process["category"] not in {
            "arithmetic",
            "control_flow",
            "memory",
            "transfer",
            "synchronization",
            "scheduling",
            "sampling",
            "graphics",
            "ray_traversal",
            "media",
        }:
            raise ContractValidationError(f"{path}.category is unsupported")
        if process["availability"] not in {
            "available",
            "unavailable",
            "opaque",
            "unknown",
        }:
            raise ContractValidationError(f"{path}.availability is unsupported")
        if process["programmability"] not in {"direct", "indirect", "none"}:
            raise ContractValidationError(f"{path}.programmability is unsupported")
        if (
            process["availability"] == "available"
            and process["programmability"] == "none"
        ):
            raise ContractValidationError(
                f"{path} is available but not programmable"
            )
        if (
            process["availability"] == "unavailable"
            and process["programmability"] != "none"
        ):
            raise ContractValidationError(
                f"{path} is unavailable but claims programmability"
            )
        _require_nonempty_string(process["api"], f"{path}.api")
        for field in (
            "operations",
            "numeric_formats",
            "required_extensions",
            "required_features",
        ):
            _require_sorted_unique_strings(process[field], f"{path}.{field}")
        _require_unsigned_integer_map(process["limits"], f"{path}.limits")
        _require_string_map(process["properties"], f"{path}.properties")
    _require_sorted_unique_names(process_names, "processes")

    memory_domains = _require_list(document["memory_domains"], "memory_domains")
    if not memory_domains:
        raise ContractValidationError("memory_domains must not be empty")
    memory_names = []
    for index, raw_domain in enumerate(memory_domains):
        path = f"memory_domains[{index}]"
        domain = _require_object(raw_domain, path)
        _require_fields(
            domain,
            {
                "name",
                "kind",
                "capacity_bytes",
                "host_visible",
                "device_local",
                "coherent",
                "cached",
                "minimum_alignment_bytes",
                "properties",
            },
        )
        memory_names.append(_require_nonempty_string(domain["name"], f"{path}.name"))
        _require_nonempty_string(domain["kind"], f"{path}.kind")
        for field in ("host_visible", "device_local", "coherent", "cached"):
            if not isinstance(domain[field], bool):
                raise ContractValidationError(f"{path}.{field} must be boolean")
        for field in ("capacity_bytes", "minimum_alignment_bytes"):
            _require_positive_integer(domain[field], f"{path}.{field}")
        alignment = int(domain["minimum_alignment_bytes"])
        if alignment & (alignment - 1):
            raise ContractValidationError(
                f"{path}.minimum_alignment_bytes must be a power of two"
            )
        _require_string_map(domain["properties"], f"{path}.properties")
    _require_sorted_unique_names(memory_names, "memory_domains")

    interconnects = _require_list(document["interconnects"], "interconnects")
    interconnect_names = []
    for index, raw_interconnect in enumerate(interconnects):
        path = f"interconnects[{index}]"
        interconnect = _require_object(raw_interconnect, path)
        _require_fields(
            interconnect,
            {
                "name",
                "kind",
                "availability",
                "api",
                "operations",
                "properties",
            },
        )
        interconnect_names.append(
            _require_nonempty_string(interconnect["name"], f"{path}.name")
        )
        _require_nonempty_string(interconnect["kind"], f"{path}.kind")
        _require_nonempty_string(interconnect["api"], f"{path}.api")
        if interconnect["availability"] not in {
            "available",
            "unavailable",
            "opaque",
            "unknown",
        }:
            raise ContractValidationError(f"{path}.availability is unsupported")
        _require_sorted_unique_strings(
            interconnect["operations"], f"{path}.operations"
        )
        _require_string_map(interconnect["properties"], f"{path}.properties")
    _require_sorted_unique_names(interconnect_names, "interconnects")

    measurements = _require_list(document["measurements"], "measurements")
    measurement_names = []
    for index, raw_measurement in enumerate(measurements):
        path = f"measurements[{index}]"
        measurement = _require_object(raw_measurement, path)
        _require_fields(measurement, {"name", "unit", "regime", "samples"})
        measurement_names.append(
            _require_nonempty_string(measurement["name"], f"{path}.name")
        )
        _require_nonempty_string(measurement["unit"], f"{path}.unit")
        _require_string_map(measurement["regime"], f"{path}.regime")
        samples = _require_list(measurement["samples"], f"{path}.samples")
        if not samples:
            raise ContractValidationError(f"{path}.samples must not be empty")
        for sample_index, sample in enumerate(samples):
            _require_nonnegative_integer(
                sample,
                f"{path}.samples[{sample_index}]",
            )
    _require_sorted_unique_names(measurement_names, "measurements")

    provenance = _require_object(document["provenance"], "provenance")
    _require_fields(
        provenance,
        {
            "api",
            "api_version",
            "driver",
            "driver_version",
            "compiler",
            "operating_system",
            "discovery_backend",
        },
    )
    for field in provenance:
        _require_nonempty_string(provenance[field], f"provenance.{field}")
    capability_extensions = _require_object(
        document["capability_extensions"],
        "capability_extensions",
    )
    identity_extensions = _require_object(
        document["identity_extensions"],
        "identity_extensions",
    )
    _require_object(document["runtime_bindings"], "runtime_bindings")
    capability_body = {
        "device_kind": identity["device_kind"],
        "architecture": identity["architecture"],
        "processes": processes,
        "memory_domains": memory_domains,
        "interconnects": interconnects,
        "api": provenance["api"],
        "api_version": provenance["api_version"],
        "capability_extensions": capability_extensions,
    }
    expected_capability = stable_contract_id(
        "hardware_capability",
        capability_body,
    )
    if document["capability_class"] != expected_capability:
        raise ContractValidationError(
            "capability_class does not match canonical hardware capabilities"
        )
    expected_profile = stable_contract_id(
        "hardware_profile",
        [
            identity,
            document["capability_class"],
            provenance,
            identity_extensions,
            measurements,
        ],
    )
    if document["profile_id"] != expected_profile:
        raise ContractValidationError(
            "profile_id does not match canonical hardware profile identity"
        )


def _validate_representation_candidate(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "candidate_id",
            "scope_ids",
            "source_contract_digests",
            "provider",
            "representation",
            "target_predicate",
            "behavioral_contract",
            "artifact_declarations",
        },
    )
    _require_stable_id(document["candidate_id"], "candidate", "candidate_id")
    scope_ids = _require_unique_strings(document["scope_ids"], "scope_ids", nonempty=True)
    for index, scope_id in enumerate(scope_ids):
        _require_stable_id(scope_id, "scope", f"scope_ids[{index}]")
    source_digests = _require_unique_strings(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    if len(scope_ids) != len(source_digests):
        raise ContractValidationError(
            "candidate scope_ids and source_contract_digests must have equal length"
        )
    for index, digest in enumerate(source_digests):
        _require_digest(digest, f"source_contract_digests[{index}]")
    _require_implementation_identity(document["provider"], "provider")
    representation = _require_object(document["representation"], "representation")
    _require_fields(
        representation,
        {
            "kind",
            "signal_formats",
            "parameter_format",
            "state_format",
            "topology",
        },
    )
    _require_nonempty_string(representation["kind"], "representation.kind")
    _require_named_records(representation["signal_formats"], "representation.signal_formats")
    for field in ("parameter_format", "state_format", "topology"):
        _require_object(representation[field], f"representation.{field}")
    _require_object(document["target_predicate"], "target_predicate")
    behavioral = _require_object(document["behavioral_contract"], "behavioral_contract")
    _require_fields(behavioral, {"mode", "proof_obligations", "error_contract"})
    if behavioral["mode"] not in {"exact", "approximate"}:
        raise ContractValidationError("behavioral_contract.mode must be exact or approximate")
    _require_unique_strings(
        behavioral["proof_obligations"],
        "behavioral_contract.proof_obligations",
    )
    if behavioral["mode"] == "exact" and behavioral["error_contract"] is not None:
        raise ContractValidationError(
            "an exact candidate cannot declare an approximation error contract"
        )
    if behavioral["mode"] == "approximate":
        _require_object(
            behavioral["error_contract"],
            "behavioral_contract.error_contract",
        )
    _require_artifact_refs(
        document["artifact_declarations"], "artifact_declarations"
    )


def _validate_representation_descriptor(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "descriptor_id",
            "identity",
            "summary",
            "responsibilities",
            "evidence",
            "representations",
            "execution",
            "hardware",
            "construction",
            "boundaries",
            "behavioral",
            "correction_paths",
            "tags",
        },
    )
    _require_stable_id(
        document["descriptor_id"],
        "representation_descriptor",
        "descriptor_id",
    )
    identity = _require_object(document["identity"], "identity")
    _require_fields(identity, {"namespace", "name", "version"})
    for field in identity:
        _require_nonempty_string(identity[field], f"identity.{field}")
    _require_nonempty_string(document["summary"], "summary")

    responsibilities = _require_object(
        document["responsibilities"], "responsibilities"
    )
    _require_fields(
        responsibilities,
        {"may_express", "composition_scopes"},
    )
    _require_sorted_nonempty_unique_strings(
        responsibilities["may_express"],
        "responsibilities.may_express",
    )
    _require_sorted_nonempty_unique_strings(
        responsibilities["composition_scopes"],
        "responsibilities.composition_scopes",
    )

    evidence = _require_list(document["evidence"], "evidence")
    if not evidence:
        raise ContractValidationError("evidence must not be empty")
    evidence_claims = []
    for index, raw_requirement in enumerate(evidence):
        path = f"evidence[{index}]"
        requirement = _require_object(raw_requirement, path)
        _require_fields(
            requirement,
            {
                "claim_kind",
                "acceptable_statuses",
                "exactness",
                "required_facts",
            },
        )
        evidence_claims.append(
            _require_nonempty_string(
                requirement["claim_kind"],
                f"{path}.claim_kind",
            )
        )
        _require_sorted_nonempty_unique_strings(
            requirement["acceptable_statuses"],
            f"{path}.acceptable_statuses",
        )
        if requirement["exactness"] not in {"exact", "approximate", "either"}:
            raise ContractValidationError(
                f"{path}.exactness must be exact, approximate, or either"
            )
        _require_sorted_unique_strings(
            requirement["required_facts"],
            f"{path}.required_facts",
        )
    _require_sorted_unique_names(evidence_claims, "evidence")

    representations = _require_object(
        document["representations"], "representations"
    )
    _require_fields(representations, {"signals", "parameters", "states"})
    _validate_representation_forms(
        representations["signals"],
        "representations.signals",
        nonempty=True,
    )
    _validate_representation_forms(
        representations["parameters"],
        "representations.parameters",
    )
    _validate_representation_forms(
        representations["states"],
        "representations.states",
    )

    execution = _require_object(document["execution"], "execution")
    _require_fields(execution, {"topologies", "time_models"})
    _require_sorted_nonempty_unique_strings(
        execution["topologies"],
        "execution.topologies",
    )
    _require_sorted_nonempty_unique_strings(
        execution["time_models"],
        "execution.time_models",
    )

    hardware = _require_object(document["hardware"], "hardware")
    _require_fields(hardware, {"compatible_processes", "composition"})
    processes = _require_list(
        hardware["compatible_processes"],
        "hardware.compatible_processes",
    )
    if not processes:
        raise ContractValidationError(
            "hardware.compatible_processes must not be empty"
        )
    process_names = []
    for index, raw_process in enumerate(processes):
        path = f"hardware.compatible_processes[{index}]"
        process = _require_object(raw_process, path)
        _require_fields(
            process,
            {"name", "requirement", "operations", "numeric_formats"},
        )
        process_names.append(
            _require_nonempty_string(process["name"], f"{path}.name")
        )
        if process["requirement"] not in {"required", "alternative", "optional"}:
            raise ContractValidationError(
                f"{path}.requirement must be required, alternative, or optional"
            )
        _require_sorted_nonempty_unique_strings(
            process["operations"],
            f"{path}.operations",
        )
        _require_sorted_unique_strings(
            process["numeric_formats"],
            f"{path}.numeric_formats",
        )
    _require_sorted_unique_names(
        process_names,
        "hardware.compatible_processes",
    )
    if hardware["composition"] not in {"all_required", "one_alternative", "composite"}:
        raise ContractValidationError(
            "hardware.composition must be all_required, one_alternative, or composite"
        )

    construction = _require_object(document["construction"], "construction")
    _require_fields(construction, {"required", "phases", "artifacts"})
    if not isinstance(construction["required"], bool):
        raise ContractValidationError("construction.required must be boolean")
    _require_unique_strings(construction["phases"], "construction.phases")
    artifacts = _require_list(construction["artifacts"], "construction.artifacts")
    artifact_kinds = []
    for index, raw_artifact in enumerate(artifacts):
        path = f"construction.artifacts[{index}]"
        artifact = _require_object(raw_artifact, path)
        _require_fields(artifact, {"kind", "lifetime"})
        artifact_kinds.append(
            _require_nonempty_string(artifact["kind"], f"{path}.kind")
        )
        if artifact["lifetime"] not in {
            "compile",
            "mount",
            "residency",
            "dynamic",
        }:
            raise ContractValidationError(
                f"{path}.lifetime must be compile, mount, residency, or dynamic"
            )
    _require_sorted_unique_names(artifact_kinds, "construction.artifacts")
    if construction["required"] and not construction["phases"]:
        raise ContractValidationError(
            "construction phases are required when construction.required is true"
        )

    boundaries = _require_object(document["boundaries"], "boundaries")
    _require_fields(
        boundaries,
        {
            "accepted_inputs",
            "produced_outputs",
            "cost_terms",
            "island_compatibility",
        },
    )
    _require_sorted_nonempty_unique_strings(
        boundaries["accepted_inputs"],
        "boundaries.accepted_inputs",
    )
    _require_sorted_nonempty_unique_strings(
        boundaries["produced_outputs"],
        "boundaries.produced_outputs",
    )
    cost_terms = _require_list(boundaries["cost_terms"], "boundaries.cost_terms")
    cost_names = []
    for index, raw_term in enumerate(cost_terms):
        path = f"boundaries.cost_terms[{index}]"
        term = _require_object(raw_term, path)
        _require_fields(
            term,
            {"name", "unit", "directions", "measured_phase"},
        )
        cost_names.append(_require_nonempty_string(term["name"], f"{path}.name"))
        _require_nonempty_string(term["unit"], f"{path}.unit")
        directions = _require_sorted_nonempty_unique_strings(
            term["directions"],
            f"{path}.directions",
        )
        if not set(directions) <= {"input", "output", "internal"}:
            raise ContractValidationError(
                f"{path}.directions contains an unsupported boundary direction"
            )
        if term["measured_phase"] not in {
            "construction",
            "mount",
            "steady_state",
            "teardown",
        }:
            raise ContractValidationError(
                f"{path}.measured_phase is unsupported"
            )
    _require_sorted_unique_names(cost_names, "boundaries.cost_terms")
    island = _require_object(
        boundaries["island_compatibility"],
        "boundaries.island_compatibility",
    )
    _require_fields(
        island,
        {
            "can_span_scopes",
            "preserves_native_signals",
            "absorbable_transducers",
        },
    )
    for field in ("can_span_scopes", "preserves_native_signals"):
        if not isinstance(island[field], bool):
            raise ContractValidationError(
                f"boundaries.island_compatibility.{field} must be boolean"
            )
    _require_sorted_unique_strings(
        island["absorbable_transducers"],
        "boundaries.island_compatibility.absorbable_transducers",
    )

    behavioral = _require_object(document["behavioral"], "behavioral")
    _require_fields(
        behavioral,
        {
            "exactness",
            "proof_obligations",
            "error_contract",
            "validity_predicates",
        },
    )
    if behavioral["exactness"] not in {"exact", "approximate"}:
        raise ContractValidationError(
            "behavioral.exactness must be exact or approximate"
        )
    _require_sorted_unique_strings(
        behavioral["proof_obligations"],
        "behavioral.proof_obligations",
    )
    _require_sorted_unique_strings(
        behavioral["validity_predicates"],
        "behavioral.validity_predicates",
    )
    if behavioral["exactness"] == "exact":
        if behavioral["error_contract"] is not None:
            raise ContractValidationError(
                "an exact representation descriptor cannot declare an error contract"
            )
    else:
        if (
            not isinstance(behavioral["error_contract"], dict)
            or not behavioral["error_contract"]
        ):
            raise ContractValidationError(
                "an approximate representation descriptor requires an error contract"
            )

    corrections = _require_list(document["correction_paths"], "correction_paths")
    correction_names = []
    for index, raw_correction in enumerate(corrections):
        path = f"correction_paths[{index}]"
        correction = _require_object(raw_correction, path)
        _require_fields(correction, {"name", "trigger", "action", "guarantee"})
        correction_names.append(
            _require_nonempty_string(correction["name"], f"{path}.name")
        )
        for field in ("trigger", "action", "guarantee"):
            _require_nonempty_string(correction[field], f"{path}.{field}")
    _require_sorted_unique_names(correction_names, "correction_paths")
    if behavioral["exactness"] == "approximate" and not corrections:
        raise ContractValidationError(
            "an approximate representation descriptor requires a correction path"
        )

    _require_sorted_nonempty_unique_strings(document["tags"], "tags")
    expected_id = representation_descriptor_id(document)
    if document["descriptor_id"] != expected_id:
        raise ContractValidationError(
            f"descriptor_id must match canonical descriptor content {expected_id!r}"
        )


def _validate_representation_forms(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
) -> None:
    forms = _require_list(value, path)
    if nonempty and not forms:
        raise ContractValidationError(f"{path} must not be empty")
    names = []
    for index, raw_form in enumerate(forms):
        form_path = f"{path}[{index}]"
        form = _require_object(raw_form, form_path)
        _require_fields(form, {"name", "kind", "properties"})
        names.append(_require_nonempty_string(form["name"], f"{form_path}.name"))
        _require_nonempty_string(form["kind"], f"{form_path}.kind")
        _require_object(form["properties"], f"{form_path}.properties")
    _require_sorted_unique_names(names, path)


def _require_sorted_nonempty_unique_strings(value: Any, path: str) -> list[str]:
    values = _require_sorted_unique_strings(value, path)
    if not values:
        raise ContractValidationError(f"{path} must not be empty")
    return values


def _validate_candidate_construction(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "construction_id",
            "candidate_id",
            "status",
            "staging_identity",
            "artifacts",
            "resource_measurements",
            "diagnostics",
        },
    )
    _require_stable_id(document["construction_id"], "construction", "construction_id")
    _require_nonempty_string(document["candidate_id"], "candidate_id")
    if document["status"] not in {
        "planned",
        "constructing",
        "completed",
        "cancelled",
        "failed",
    }:
        raise ContractValidationError(
            f"construction status is unsupported: {document['status']!r}"
        )
    _require_nonempty_string(document["staging_identity"], "staging_identity")
    _require_artifact_refs(document["artifacts"], "artifacts")
    _require_object(document["resource_measurements"], "resource_measurements")
    _require_string_list(document["diagnostics"], "diagnostics")


def _validate_benchmark_record(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "benchmark_id",
            "candidate_id",
            "reference_implementation_id",
            "workload",
            "matched_conditions_digest",
            "measurements",
            "decision",
        },
    )
    _require_stable_id(document["benchmark_id"], "benchmark", "benchmark_id")
    _require_nonempty_string(document["candidate_id"], "candidate_id")
    _require_nonempty_string(
        document["reference_implementation_id"],
        "reference_implementation_id",
    )
    _require_object(document["workload"], "workload")
    _require_digest(document["matched_conditions_digest"], "matched_conditions_digest")
    measurements = _require_list(document["measurements"], "measurements")
    if not measurements:
        raise ContractValidationError("benchmark measurements must not be empty")
    for index, measurement in enumerate(measurements):
        _require_measurement(measurement, f"measurements[{index}]")
    if document["decision"] not in {
        "materially_faster",
        "not_materially_faster",
        "inconclusive",
        "invalid",
    }:
        raise ContractValidationError(
            f"benchmark decision is unsupported: {document['decision']!r}"
        )


def _validate_validation_record(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "validation_id",
            "candidate_id",
            "source_contract_digests",
            "behavioral_contract",
            "stages",
            "counterexamples",
            "status",
        },
    )
    _require_stable_id(document["validation_id"], "validation", "validation_id")
    _require_nonempty_string(document["candidate_id"], "candidate_id")
    digests = _require_unique_strings(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    for index, digest in enumerate(digests):
        _require_digest(digest, f"source_contract_digests[{index}]")
    _require_object(document["behavioral_contract"], "behavioral_contract")
    stages = _require_list(document["stages"], "stages")
    for index, stage in enumerate(stages):
        stage = _require_object(stage, f"stages[{index}]")
        _require_fields(stage, {"name", "status", "metrics", "artifacts"})
        _require_nonempty_string(stage["name"], f"stages[{index}].name")
        if stage["status"] not in {"passed", "failed", "not_required"}:
            raise ContractValidationError(
                f"stages[{index}].status has unsupported value {stage['status']!r}"
            )
        _require_object(stage["metrics"], f"stages[{index}].metrics")
        _require_artifact_refs(stage["artifacts"], f"stages[{index}].artifacts")
    _require_artifact_refs(document["counterexamples"], "counterexamples")
    if document["status"] not in {"passed", "failed"}:
        raise ContractValidationError("validation status must be passed or failed")


def _validate_promotion_decision(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "promotion_id",
            "candidate_id",
            "benchmark_record_digest",
            "validation_record_digest",
            "runtime_predicate",
            "implementation_id",
            "decision",
            "reason",
        },
    )
    _require_stable_id(document["promotion_id"], "promotion", "promotion_id")
    _require_nonempty_string(document["candidate_id"], "candidate_id")
    _require_digest(document["benchmark_record_digest"], "benchmark_record_digest")
    _require_digest(document["validation_record_digest"], "validation_record_digest")
    _require_object(document["runtime_predicate"], "runtime_predicate")
    _require_nonempty_string(document["implementation_id"], "implementation_id")
    if document["decision"] not in {"promote", "reject"}:
        raise ContractValidationError("promotion decision must be promote or reject")
    _require_nonempty_string(document["reason"], "reason")


def _validate_relowering_request(document: Json) -> None:
    _require_fields(
        document,
        {
            "schema",
            "request_id",
            "candidate_id",
            "scope_ids",
            "representation_digest",
            "required_passes",
            "boundary_contracts",
        },
    )
    _require_stable_id(document["request_id"], "relower", "request_id")
    _require_nonempty_string(document["candidate_id"], "candidate_id")
    scope_ids = _require_unique_strings(document["scope_ids"], "scope_ids", nonempty=True)
    for index, scope_id in enumerate(scope_ids):
        _require_stable_id(scope_id, "scope", f"scope_ids[{index}]")
    _require_digest(document["representation_digest"], "representation_digest")
    _require_unique_strings(
        document["required_passes"], "required_passes", nonempty=True
    )
    _require_named_records(document["boundary_contracts"], "boundary_contracts")


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ContractValidationError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def _require_fields(document: Json, required: set[str]) -> None:
    actual = set(document)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - {"extensions"})
    if missing:
        raise ContractValidationError(f"optimizer contract is missing fields {missing}")
    if unknown:
        raise ContractValidationError(f"optimizer contract has unknown fields {unknown}")
    if "extensions" in document:
        _require_object(document["extensions"], "extensions")


def _require_object(value: Any, path: str) -> Json:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be a list")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value


def _require_string_list(value: Any, path: str) -> list[str]:
    values = _require_list(value, path)
    for index, item in enumerate(values):
        _require_nonempty_string(item, f"{path}[{index}]")
    return values


def _require_unique_strings(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _require_string_list(value, path)
    if nonempty and not values:
        raise ContractValidationError(f"{path} must not be empty")
    if len(values) != len(set(values)):
        raise ContractValidationError(f"{path} must contain unique values")
    return values


def _require_sorted_unique_strings(value: Any, path: str) -> list[str]:
    values = _require_string_list(value, path)
    if values != sorted(set(values)):
        raise ContractValidationError(f"{path} must contain unique sorted values")
    return values


def _require_sorted_unique_names(values: list[str], path: str) -> None:
    if values != sorted(set(values)):
        raise ContractValidationError(f"{path} must have unique sorted names")


def _require_nonnegative_integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractValidationError(f"{path} must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, path: str) -> int:
    parsed = _require_nonnegative_integer(value, path)
    if parsed == 0:
        raise ContractValidationError(f"{path} must be positive")
    return parsed


def _require_unsigned_integer_map(value: Any, path: str) -> None:
    mapping = _require_object(value, path)
    for key, item in mapping.items():
        _require_nonempty_string(key, f"{path} key")
        _require_nonnegative_integer(item, f"{path}.{key}")


def _require_string_map(value: Any, path: str) -> None:
    mapping = _require_object(value, path)
    for key, item in mapping.items():
        _require_nonempty_string(key, f"{path} key")
        if not isinstance(item, str):
            raise ContractValidationError(f"{path}.{key} must be a string")


def _require_reference_list(value: Any, path: str) -> None:
    records = _require_list(value, path)
    identifiers = []
    for index, record in enumerate(records):
        record = _require_object(record, f"{path}[{index}]")
        identifier = _require_nonempty_string(record.get("id"), f"{path}[{index}].id")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ContractValidationError(f"{path} contains duplicate reference ids")


def _require_named_records(value: Any, path: str) -> None:
    records = _require_list(value, path)
    names = []
    for index, record in enumerate(records):
        record = _require_object(record, f"{path}[{index}]")
        name = _require_nonempty_string(record.get("name"), f"{path}[{index}].name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ContractValidationError(f"{path} contains duplicate names")


def _require_artifact_refs(value: Any, path: str) -> None:
    records = _require_list(value, path)
    paths = []
    for index, record in enumerate(records):
        record = _require_object(record, f"{path}[{index}]")
        artifact_path = _require_nonempty_string(
            record.get("path"), f"{path}[{index}].path"
        )
        paths.append(artifact_path)
        if "digest" in record:
            _require_digest(record["digest"], f"{path}[{index}].digest")
    if len(paths) != len(set(paths)):
        raise ContractValidationError(f"{path} contains duplicate artifact paths")


def _require_implementation_identity(value: Any, path: str) -> None:
    identity = _require_object(value, path)
    _require_fields(identity, {"id", "version"})
    _require_nonempty_string(identity["id"], f"{path}.id")
    _require_nonempty_string(identity["version"], f"{path}.version")


def _require_measurement(value: Any, path: str) -> None:
    measurement = _require_object(value, path)
    _require_fields(
        measurement,
        {
            "name",
            "unit",
            "reference_samples",
            "candidate_samples",
            "summary",
        },
    )
    _require_nonempty_string(measurement["name"], f"{path}.name")
    _require_nonempty_string(measurement["unit"], f"{path}.unit")
    for field in ("reference_samples", "candidate_samples"):
        samples = _require_list(measurement[field], f"{path}.{field}")
        if not samples:
            raise ContractValidationError(f"{path}.{field} must not be empty")
        for index, sample in enumerate(samples):
            if (
                not isinstance(sample, (int, float))
                or isinstance(sample, bool)
                or not math.isfinite(float(sample))
            ):
                raise ContractValidationError(
                    f"{path}.{field}[{index}] must be a finite number"
                )
    _require_object(measurement["summary"], f"{path}.summary")


def _require_digest(value: Any, path: str) -> str:
    digest = _require_nonempty_string(value, path)
    prefix = f"{CONTRACT_DIGEST_SCHEMA}:"
    if not digest.startswith(prefix):
        raise ContractValidationError(f"{path} must use {CONTRACT_DIGEST_SCHEMA}")
    hexadecimal = digest[len(prefix) :]
    if len(hexadecimal) != 64 or any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ContractValidationError(f"{path} contains an invalid SHA-256 digest")
    return digest


def _require_stable_id(value: Any, prefix: str, path: str) -> str:
    identifier = _require_nonempty_string(value, path)
    if not identifier.startswith(f"{prefix}_") or len(identifier) != len(prefix) + 33:
        raise ContractValidationError(
            f"{path} must be a stable {prefix!r} identifier"
        )
    hexadecimal = identifier.rsplit("_", 1)[-1]
    if any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ContractValidationError(f"{path} contains an invalid stable identifier")
    return identifier


_VALIDATORS: dict[str, Validator] = {
    OPTIMIZATION_SCOPE_SCHEMA: _validate_optimization_scope,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA: _validate_source_behavior_contract,
    ALGEBRAIC_EVIDENCE_SCHEMA: _validate_algebraic_evidence,
    HARDWARE_PROCESS_PROFILE_SCHEMA: _validate_hardware_process_profile,
    REPRESENTATION_DESCRIPTOR_SCHEMA: _validate_representation_descriptor,
    REPRESENTATION_CANDIDATE_SCHEMA: _validate_representation_candidate,
    CANDIDATE_CONSTRUCTION_SCHEMA: _validate_candidate_construction,
    BENCHMARK_RECORD_SCHEMA: _validate_benchmark_record,
    VALIDATION_RECORD_SCHEMA: _validate_validation_record,
    PROMOTION_DECISION_SCHEMA: _validate_promotion_decision,
    RELOWERING_REQUEST_SCHEMA: _validate_relowering_request,
}
