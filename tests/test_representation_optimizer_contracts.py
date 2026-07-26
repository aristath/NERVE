from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.representation_optimizer.contracts import (
    ALGEBRAIC_EVIDENCE_SCHEMA,
    BENCHMARK_RECORD_SCHEMA,
    CANDIDATE_CONSTRUCTION_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    PROMOTION_DECISION_SCHEMA,
    RELOWERING_REQUEST_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    ContractDocument,
    ContractValidationError,
    canonical_json_bytes,
    contract_digest,
    source_behavior_contract_digest,
    stable_contract_id,
    validate_contract,
)


def digest(label: str) -> str:
    return contract_digest({"label": label})


def fixture_scope_members() -> dict[str, list[str]]:
    return {
        "component_ids": ["component"],
        "semantic_module_ids": ["layer.feature_transform"],
        "source_node_ids": ["gate", "up", "down"],
    }


def fixture_scope_id() -> str:
    members = fixture_scope_members()
    return stable_contract_id(
        "scope",
        "fixture_package",
        "semantic_module",
        members["component_ids"],
        members["semantic_module_ids"],
        members["source_node_ids"],
    )


def source_contract() -> dict[str, object]:
    document: dict[str, object] = {
        "schema": SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
        "scope_id": fixture_scope_id(),
        "semantic_role": "feature transformation",
        "interface": {
            "inputs": [{"id": "input", "shape": [16], "dtype": "BF16"}],
            "outputs": [{"id": "output", "shape": [16], "dtype": "BF16"}],
            "parameters": [{"id": "weight", "shape": [16, 16]}],
            "states": [],
            "controls": [],
            "randomness": [],
        },
        "exact_reference": {
            "implementation_id": "exact_reference",
            "artifact_refs": ["lowered/layer/circuit.json"],
        },
        "contract_digest": "",
    }
    document["contract_digest"] = source_behavior_contract_digest(document)
    return document


def hardware_profile_contract() -> dict[str, object]:
    identity = {
        "device_kind": "gpu",
        "stable_device_id": "vulkan:fixture",
        "name": "fixture GPU",
        "vendor_id": "0x1002",
        "device_id": "0xfixture",
        "architecture": "fixture_architecture",
        "physical_location": "fixture_slot",
    }
    processes = [
        {
            "name": "texture_sampling",
            "category": "sampling",
            "availability": "available",
            "programmability": "direct",
            "api": "vulkan",
            "operations": ["linear_interpolation", "nearest_sampling"],
            "numeric_formats": ["f16", "f32"],
            "required_extensions": [],
            "required_features": [],
            "limits": {"max_image_dimension_2d": 16384},
            "properties": {},
        },
        {
            "name": "vector_fma",
            "category": "arithmetic",
            "availability": "available",
            "programmability": "direct",
            "api": "vulkan",
            "operations": ["fused_multiply_add"],
            "numeric_formats": ["f16", "f32"],
            "required_extensions": [],
            "required_features": ["shader_float16"],
            "limits": {},
            "properties": {},
        },
    ]
    memory_domains = [
        {
            "name": "device_memory",
            "kind": "device_local_heap",
            "capacity_bytes": 1_073_741_824,
            "host_visible": False,
            "device_local": True,
            "coherent": False,
            "cached": False,
            "minimum_alignment_bytes": 256,
            "properties": {},
        }
    ]
    interconnects = [
        {
            "name": "host_staging",
            "kind": "host_staging",
            "availability": "available",
            "api": "vulkan",
            "operations": ["device_to_host", "host_to_device"],
            "properties": {},
        }
    ]
    provenance = {
        "api": "vulkan",
        "api_version": "1.4.0",
        "driver": "fixture",
        "driver_version": "1",
        "compiler": "fixture",
        "operating_system": "linux",
        "discovery_backend": "fixture",
    }
    capability_extensions: dict[str, object] = {}
    identity_extensions: dict[str, object] = {}
    runtime_bindings: dict[str, object] = {}
    capability_class = stable_contract_id(
        "hardware_capability",
        {
            "device_kind": identity["device_kind"],
            "architecture": identity["architecture"],
            "processes": processes,
            "memory_domains": memory_domains,
            "interconnects": interconnects,
            "api": provenance["api"],
            "api_version": provenance["api_version"],
            "capability_extensions": capability_extensions,
        },
    )
    return {
        "schema": HARDWARE_PROCESS_PROFILE_SCHEMA,
        "profile_id": stable_contract_id(
            "hardware_profile",
            [
                identity,
                capability_class,
                provenance,
                identity_extensions,
                [],
            ],
        ),
        "hardware_identity": identity,
        "capability_class": capability_class,
        "processes": processes,
        "memory_domains": memory_domains,
        "interconnects": interconnects,
        "measurements": [],
        "provenance": provenance,
        "capability_extensions": capability_extensions,
        "identity_extensions": identity_extensions,
        "runtime_bindings": runtime_bindings,
    }


def contract_fixtures() -> list[dict[str, object]]:
    source = source_contract()
    source_digest = str(source["contract_digest"])
    scope_members = fixture_scope_members()
    scope_id = fixture_scope_id()
    candidate_id = stable_contract_id("candidate", scope_id, "provider", "field")
    return [
        {
            "schema": OPTIMIZATION_SCOPE_SCHEMA,
            "scope_id": scope_id,
            "package_id": "fixture_package",
            "kind": "semantic_module",
            "members": scope_members,
            "boundary": deepcopy(source["interface"]),
            "source_contract_digest": source_digest,
        },
        source,
        {
            "schema": ALGEBRAIC_EVIDENCE_SCHEMA,
            "evidence_id": stable_contract_id("evidence", scope_id, "analyzer"),
            "scope_id": scope_id,
            "source_contract_digest": source_digest,
            "analyzer": {"id": "spectral_structure", "version": "1"},
            "claims": [
                {
                    "kind": "spectral_concentration",
                    "status": "supported",
                    "exact": False,
                    "facts": {"energy_fraction": 0.999},
                }
            ],
            "artifacts": [{"path": "optimization/evidence/spectral.json"}],
        },
        hardware_profile_contract(),
        {
            "schema": REPRESENTATION_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "scope_ids": [scope_id],
            "source_contract_digests": [source_digest],
            "provider": {"id": "sampled_field", "version": "1"},
            "representation": {
                "kind": "sampled_field",
                "signal_formats": [{"name": "field_coordinate"}],
                "parameter_format": {"kind": "sampled_grid"},
                "state_format": {"kind": "none"},
                "topology": {"kind": "single_scope"},
            },
            "target_predicate": {"process": "texture_sampling"},
            "behavioral_contract": {
                "mode": "approximate",
                "proof_obligations": ["bounded_interpolation_error"],
                "error_contract": {"maximum_absolute_error": 0.001},
            },
            "artifact_declarations": [
                {"path": "optimization/candidates/field/grid.bin"}
            ],
        },
        {
            "schema": CANDIDATE_CONSTRUCTION_SCHEMA,
            "construction_id": stable_contract_id(
                "construction", candidate_id, "fixture-target"
            ),
            "candidate_id": candidate_id,
            "status": "completed",
            "staging_identity": "stage_fixture",
            "artifacts": [
                {
                    "path": "optimization/staging/field/grid.bin",
                    "digest": digest("grid"),
                }
            ],
            "resource_measurements": {
                "construction_time_ns": 100,
                "temporary_bytes": 512,
                "permanent_bytes": 256,
            },
            "diagnostics": [],
        },
        {
            "schema": BENCHMARK_RECORD_SCHEMA,
            "benchmark_id": stable_contract_id(
                "benchmark", candidate_id, "fixture-workload"
            ),
            "candidate_id": candidate_id,
            "reference_implementation_id": "exact_reference",
            "workload": {"id": "fixture-workload", "regime": "decode"},
            "matched_conditions_digest": digest("matched conditions"),
            "measurements": [
                {
                    "name": "latency",
                    "unit": "ns",
                    "reference_samples": [20, 21, 19],
                    "candidate_samples": [10, 11, 9],
                    "summary": {"speedup": 2.0},
                }
            ],
            "decision": "materially_faster",
        },
        {
            "schema": VALIDATION_RECORD_SCHEMA,
            "validation_id": stable_contract_id(
                "validation", candidate_id, "fixture-validation"
            ),
            "candidate_id": candidate_id,
            "source_contract_digests": [source_digest],
            "behavioral_contract": {
                "mode": "approximate",
                "maximum_absolute_error": 0.001,
            },
            "stages": [
                {
                    "name": "component_error",
                    "status": "passed",
                    "metrics": {"maximum_absolute_error": 0.0005},
                    "artifacts": [],
                }
            ],
            "counterexamples": [],
            "status": "passed",
        },
        {
            "schema": PROMOTION_DECISION_SCHEMA,
            "promotion_id": stable_contract_id(
                "promotion", candidate_id, "fixture-target"
            ),
            "candidate_id": candidate_id,
            "benchmark_record_digest": digest("benchmark"),
            "validation_record_digest": digest("validation"),
            "runtime_predicate": {"capability_class": "gpu.fixture"},
            "implementation_id": "field_fixture_target",
            "decision": "promote",
            "reason": "material speedup with validated error contract",
        },
        {
            "schema": RELOWERING_REQUEST_SCHEMA,
            "request_id": stable_contract_id(
                "relower", candidate_id, "representation"
            ),
            "candidate_id": candidate_id,
            "scope_ids": [scope_id],
            "representation_digest": digest("representation"),
            "required_passes": ["representation_boundary_lowering"],
            "boundary_contracts": [
                {"name": "input", "representation": "field_coordinate"},
                {"name": "output", "representation": "BF16"},
            ],
        },
    ]


@pytest.mark.parametrize("document", contract_fixtures())
def test_every_optimizer_contract_schema_round_trips_deterministically(
    document: dict[str, object],
) -> None:
    parsed = ContractDocument.from_json(document)
    reordered = dict(reversed(list(document.items())))

    assert ContractDocument.from_bytes(parsed.to_bytes()).to_json() == document
    assert canonical_json_bytes(reordered) == parsed.to_bytes()
    assert parsed.digest == contract_digest(document)


def test_contract_document_is_copy_in_copy_out_immutable() -> None:
    source = source_contract()
    parsed = ContractDocument.from_json(source)
    source["semantic_role"] = "mutated"
    exported = parsed.to_json()
    exported["semantic_role"] = "also mutated"

    assert parsed.to_json()["semantic_role"] == "feature transformation"


def test_scope_identity_rejects_reordered_or_substituted_members() -> None:
    scope = contract_fixtures()[0]
    scope["members"]["source_node_ids"].reverse()

    with pytest.raises(ContractValidationError, match="stable semantic identity"):
        validate_contract(scope)


def test_source_contract_digest_rejects_behavioral_drift() -> None:
    source = source_contract()
    source["interface"]["outputs"][0]["shape"] = [32]

    with pytest.raises(ContractValidationError, match="digest does not match"):
        validate_contract(source)


def test_contract_validation_rejects_unknown_fields_and_nonfinite_numbers() -> None:
    source = source_contract()
    source["accidental_field"] = True
    with pytest.raises(ContractValidationError, match="unknown fields"):
        validate_contract(source)

    with pytest.raises(ContractValidationError, match="non-finite"):
        canonical_json_bytes({"invalid": float("nan")})


def test_unknown_schema_fails_closed() -> None:
    with pytest.raises(ContractValidationError, match="unsupported"):
        validate_contract({"schema": "nerve.optimizer.future.v99"})


def test_hardware_profile_rejects_capability_drift_and_unsorted_processes() -> None:
    profile = hardware_profile_contract()
    profile["processes"][0]["limits"]["max_image_dimension_2d"] = 8192
    with pytest.raises(ContractValidationError, match="capability_class"):
        validate_contract(profile)

    profile = hardware_profile_contract()
    profile["processes"].reverse()
    with pytest.raises(ContractValidationError, match="unique sorted names"):
        validate_contract(profile)


def test_hardware_profile_rejects_unavailable_programmable_process() -> None:
    profile = hardware_profile_contract()
    profile["processes"][0]["availability"] = "unavailable"
    with pytest.raises(ContractValidationError, match="claims programmability"):
        validate_contract(profile)


def test_hardware_measurements_are_part_of_profile_identity_not_capability_class() -> None:
    profile = hardware_profile_contract()
    old_profile_id = profile["profile_id"]
    capability_class = profile["capability_class"]
    profile["measurements"] = [
        {
            "name": "texture_sampling_throughput",
            "unit": "samples_per_second",
            "regime": {"format": "f16"},
            "samples": [100, 101, 99],
        }
    ]
    profile["profile_id"] = stable_contract_id(
        "hardware_profile",
        [
            profile["hardware_identity"],
            capability_class,
            profile["provenance"],
            profile["identity_extensions"],
            profile["measurements"],
        ],
    )

    validate_contract(profile)
    assert profile["capability_class"] == capability_class
    assert profile["profile_id"] != old_profile_id


def test_runtime_binding_changes_do_not_change_stable_hardware_identity() -> None:
    profile = hardware_profile_contract()
    profile["runtime_bindings"] = {
        "vulkan_runtime_binding": {"physical_device_index": 2}
    }

    validate_contract(profile)
    profile_id = profile["profile_id"]
    profile["runtime_bindings"]["vulkan_runtime_binding"][
        "physical_device_index"
    ] = 4
    validate_contract(profile)

    assert profile["profile_id"] == profile_id


def test_hardware_profile_rejects_unclassified_extensions() -> None:
    profile = hardware_profile_contract()
    profile["extensions"] = {"possibly_capability_relevant": True}

    with pytest.raises(ContractValidationError, match="must be classified"):
        validate_contract(profile)
