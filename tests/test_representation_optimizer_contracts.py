from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.representation_optimizer.benchmarking.contracts import (
    benchmark_record_id,
)
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
    algebraic_evidence_id,
    canonical_json_bytes,
    contract_digest,
    representation_candidate_id,
    source_behavior_contract_digest,
    stable_contract_id,
    validate_contract,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.promotion.contracts import (
    create_runtime_implementation_predicate,
    implementation_id,
    promotion_decision_id,
)
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_FUNNEL_STAGE_NAMES,
    validation_record_id,
)
from nerve.representation_optimizer.validation.planning import (
    create_behavioral_error_contract,
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
            "dependencies": [],
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
    error_contract = create_behavioral_error_contract(
        validity_predicates={"scope": "fixture"},
        metric_limits={
            "maximum_absolute_error": (
                0.001,
                "absolute",
                ("component_output_error",),
            )
        },
        correction_mode="reject",
        correction_trigger_metrics=("maximum_absolute_error",),
        correction_action="reject candidate and retain exact implementation",
    ).to_json()
    candidate = {
        "schema": REPRESENTATION_CANDIDATE_SCHEMA,
        "candidate_id": "",
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
            "error_contract": error_contract,
        },
        "artifact_declarations": [
            {"path": "optimization/candidates/field/grid.bin"}
        ],
    }
    evidence = {
        "schema": ALGEBRAIC_EVIDENCE_SCHEMA,
        "evidence_id": "",
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
    }
    evidence["evidence_id"] = algebraic_evidence_id(evidence)
    candidate["descriptor_id"] = (
        "representation_descriptor_"
        "11111111111111111111111111111111"
    )
    candidate["evidence_refs"] = [evidence["evidence_id"]]
    candidate["candidate_id"] = representation_candidate_id(candidate)
    candidate_id = candidate["candidate_id"]
    benchmark_workload_id = stable_contract_id(
        "benchmark_workload",
        "fixture-workload",
    )
    distribution = {
        "sample_count": 5,
        "minimum": 9,
        "maximum": 11,
        "median": 10,
        "mean": 10,
        "standard_deviation": 1,
        "confidence_interval_low": 9,
        "confidence_interval_high": 11,
        "relative_ci_width_ppm": 200_000,
    }
    role_summary = {
        "latency_ns": deepcopy(distribution),
        "throughput_per_second": deepcopy(distribution),
        "permanent_bytes": 256,
        "peak_transient_bytes": 512,
        "resident_before_bytes": 256,
        "resident_peak_bytes": 512,
        "resident_after_bytes": 256,
        "conversion_bytes": 0,
        "conversion_ns": 0,
        "boundary_count": 0,
        "utilization_ppm": 800_000,
        "synchronization_wait_ns": 10,
        "transport_bytes": 0,
        "transport_ns": 0,
        "queue_wait_ns": 10,
        "timeout_count": 0,
        "useful_units": 640,
        "wasted_units": 0,
    }
    resource_role = {
        "setup_ns": 100,
        "teardown_ns": 50,
        "host_elapsed_ns": 1_000,
        "permanent_bytes": 256,
        "peak_transient_bytes": 512,
        "resident_before_bytes": 256,
        "resident_peak_bytes": 512,
        "resident_after_bytes": 256,
        "conversion_bytes": 0,
        "conversion_ns": 0,
        "boundary_count": 0,
        "device_measurement_ns": 1_000,
        "device_busy_ns": 800,
        "utilization_ppm": 800_000,
        "synchronization_count": 10,
        "synchronization_wait_ns": 10,
        "transport_bytes": 0,
        "transport_ns": 0,
        "queue_wait_count": 0,
        "queue_wait_ns": 10,
        "timeout_count": 0,
        "useful_units": 640,
        "speculative_units": 0,
        "cancelled_units": 0,
        "discarded_units": 0,
        "corrective_units": 0,
    }
    benchmark_record = {
        "schema": BENCHMARK_RECORD_SCHEMA,
        "benchmark_id": "",
        "candidate_id": candidate_id,
        "plan_digest": digest("benchmark plan"),
        "run_digest": digest("benchmark run"),
        "construction_record_digest": digest("construction"),
        "reference_implementation_id": "exact_reference",
        "matched_conditions_digest": digest("matched conditions"),
        "workloads": [
            {
                "workload_id": benchmark_workload_id,
                "decision": "materially_faster",
                "reasons": [],
                "sample_count_per_role": 5,
                "warmup": {
                    "reference": {
                        "sample_count": 8,
                        "maximum_shift_ppm": 1_000,
                        "converged": True,
                    },
                    "candidate": {
                        "sample_count": 8,
                        "maximum_shift_ppm": 1_000,
                        "converged": True,
                    },
                },
                "reference": deepcopy(role_summary),
                "candidate": deepcopy(role_summary),
                "paired": {
                    "speedup_ppm": 100_000,
                    "candidate_is_faster": True,
                },
                "sustained": {
                    "reference_slope_ppm_per_window": 0,
                    "candidate_slope_ppm_per_window": 0,
                    "candidate_regression_ppm": 0,
                    "passed": True,
                },
            }
        ],
        "reproducibility": [
            {
                "workload_id": benchmark_workload_id,
                "role": role,
                "seed": 1,
                "order_index": order,
                "classification": "identical",
                "observation_ids": [
                    stable_contract_id(
                        "benchmark_observation",
                        role,
                        order,
                        repeat,
                    )
                    for repeat in range(2)
                ],
            }
            for role, order in (("candidate", 1), ("reference", 0))
        ],
        "resource_measurements": {
            "construction": {
                "construction_time_ns": 100,
                "peak_temporary_bytes": 512,
                "peak_staging_bytes": 256,
                "final_permanent_bytes": 256,
                "generated_artifact_bytes": 4,
            },
            "roles": {
                "reference": deepcopy(resource_role),
                "candidate": deepcopy(resource_role),
            },
        },
        "raw_evidence": {
            "run_id": stable_contract_id("benchmark_run", "fixture"),
            "observation_count": 20,
            "residency_event_count": 4,
            "host_elapsed_sample_count": 20,
            "trace_artifact_count": 100,
        },
        "decision": "materially_faster",
        "decision_reasons": [],
    }
    benchmark_record["benchmark_id"] = benchmark_record_id(benchmark_record)
    validation_record = {
        "schema": VALIDATION_RECORD_SCHEMA,
        "validation_id": "",
        "candidate_id": candidate_id,
        "source_contract_digests": [source_digest],
        "behavioral_contract": deepcopy(candidate["behavioral_contract"]),
        "validation_plan_digest": digest("validation plan"),
        "construction_record_digest": digest("construction"),
        "prebenchmark_record_digest": digest("prebenchmark"),
        "benchmark_record_digest": contract_digest(benchmark_record),
        "runs": [
            {"stage": "full_local", "run_digest": digest("local run")},
            {"stage": "whole_model", "run_digest": digest("model run")},
        ],
        "stages": [
            {
                "name": name,
                "status": "passed",
                "evidence_digests": [digest(name)],
                "metrics": {},
                "artifacts": [],
                "reason": None,
            }
            for name in VALIDATION_FUNNEL_STAGE_NAMES
        ],
        "counterexamples": [],
        "status": "passed",
    }
    validation_record["validation_id"] = validation_record_id(
        validation_record
    )
    promotion_hardware_profile = hardware_profile_contract()
    runtime_predicate = create_runtime_implementation_predicate(
        capability_classes=(
            promotion_hardware_profile["capability_class"],
        ),
        device_kinds=("gpu",),
        apis=("vulkan",),
        required_processes=("texture_sampling",),
        required_features=("shader_float16",),
        execution_phases=("decode",),
        alternative_execution_phases=("decode",),
        source_retained_execution_phases=(),
        activation_batch_minimum=1,
        activation_batch_maximum=1,
        context_activations_minimum=4096,
        context_activations_maximum=4096,
        state_activations_minimum=4096,
        state_activations_maximum=4096,
        speculative_draft_token_counts=(0,),
        placement_mode="local",
        minimum_device_count=1,
        maximum_device_count=1,
        required_interconnects=(),
    )
    promoted_implementation_id = implementation_id(
        candidate_id,
        runtime_predicate,
    )
    promotion = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "promotion_id": "",
        "candidate_id": candidate_id,
        "implementation_id": promoted_implementation_id,
        "scope_ids": [scope_id],
        "source_contract_digests": [source_digest],
        "candidate_contract_digest": contract_digest(candidate),
        "construction_record_digest": digest("construction"),
        "prebenchmark_record_digest": digest("prebenchmark"),
        "benchmark_record_digest": contract_digest(benchmark_record),
        "validation_record_digest": contract_digest(validation_record),
        "runtime_predicate": runtime_predicate.to_json(),
        "artifact_integrity": {
            "schema": "nerve.optimizer.staged_candidate_integrity.v1",
            "digest": staged_artifact_digest(b"integrity manifest"),
            "file_count": 8,
        },
        "comparison": {
            "exact_implementation_id": "exact_reference",
            "exact_contract_digest": source_digest,
            "benchmark_id": benchmark_record["benchmark_id"],
            "benchmark_decision": "materially_faster",
            "workloads": [
                {
                    "workload_id": benchmark_workload_id,
                    "decision": "materially_faster",
                    "paired": deepcopy(
                        benchmark_record["workloads"][0]["paired"]
                    ),
                }
            ],
            "validation_id": validation_record["validation_id"],
            "validation_status": "passed",
            "behavioral_contract": deepcopy(
                candidate["behavioral_contract"]
            ),
        },
        "provenance": {
            "provider": deepcopy(candidate["provider"]),
            "descriptor_id": candidate["descriptor_id"],
            "evidence_refs": deepcopy(candidate["evidence_refs"]),
            "analysis_runs": [
                {
                    "run_id": stable_contract_id(
                        "analysis_run",
                        "fixture promotion analysis",
                    ),
                    "run_digest": digest("analysis run"),
                    "cited_evidence_ids": deepcopy(
                        candidate["evidence_refs"]
                    ),
                }
            ],
            "hardware_profiles": [
                {
                    "profile_id": promotion_hardware_profile[
                        "profile_id"
                    ],
                    "profile_digest": contract_digest(
                        promotion_hardware_profile
                    ),
                }
            ],
            "representation_graph_digest": digest(
                "representation graph"
            ),
            "target_lowering_digest": digest("target lowering"),
            "relowering_request_digest": digest(
                "relowering request"
            ),
        },
        "decision": "promote",
        "reason": "material speedup with validated error contract",
    }
    promotion["promotion_id"] = promotion_decision_id(promotion)
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
        evidence,
        hardware_profile_contract(),
        candidate,
        {
            "schema": CANDIDATE_CONSTRUCTION_SCHEMA,
            "construction_id": stable_contract_id(
                "construction",
                candidate_id,
                digest("representation graph"),
                digest("target lowering"),
                "stage_fixture",
            ),
            "candidate_id": candidate_id,
            "status": "completed",
            "staging_identity": "stage_fixture",
            "source_seal": {
                "schema": "nerve.optimizer.source_package_seal.v2",
                "package_id": "fixture_package",
                "manifest_digest": staged_artifact_digest(b"manifest"),
                "optimizer_stage_digest": staged_artifact_digest(b"stage"),
                "exact_baseline_digest": digest("baseline"),
                "scope_catalog_digest": digest("scopes"),
                "package_integrity_contract_digest": digest("integrity"),
                "source_inputs": {},
            },
            "representation_graph_digest": digest("representation graph"),
            "target_lowering_digest": digest("target lowering"),
            "relowering_request_digest": digest("relowering request"),
            "phases": [
                {
                    "name": "semantic_construction",
                    "status": "completed",
                    "started_ns": 0,
                    "finished_ns": 30,
                    "duration_ns": 30,
                    "staging_bytes_written": 4,
                    "peak_temporary_bytes": 512,
                    "diagnostics": [],
                },
                {
                    "name": "ordinary_lowering",
                    "status": "completed",
                    "started_ns": 30,
                    "finished_ns": 70,
                    "duration_ns": 40,
                    "staging_bytes_written": 0,
                    "peak_temporary_bytes": 512,
                    "diagnostics": [],
                },
                {
                    "name": "physical_optimization",
                    "status": "completed",
                    "started_ns": 70,
                    "finished_ns": 100,
                    "duration_ns": 30,
                    "staging_bytes_written": 0,
                    "peak_temporary_bytes": 512,
                    "diagnostics": [],
                },
            ],
            "artifacts": [
                {
                    "path": "optimization/staging/field/grid.bin",
                    "digest": staged_artifact_digest(b"grid"),
                    "byte_count": 4,
                    "kind": "sampled_field",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 256,
                    "validation": {
                        "validator_id": "nonempty_binary",
                        "status": "passed",
                        "facts": {"byte_count": 4},
                    },
                }
            ],
            "integrity": {
                "schema": "nerve.optimizer.staged_candidate_integrity.v1",
                "digest": staged_artifact_digest(b"integrity manifest"),
                "file_count": 8,
            },
            "resource_measurements": {
                "construction_time_ns": 100,
                "peak_temporary_bytes": 512,
                "peak_staging_bytes": 256,
                "final_permanent_bytes": 256,
                "generated_artifact_bytes": 4,
            },
            "diagnostics": [],
        },
        benchmark_record,
        validation_record,
        promotion,
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


def test_algebraic_evidence_identity_rejects_claim_drift() -> None:
    evidence = contract_fixtures()[2]
    evidence["claims"][0]["facts"]["energy_fraction"] = 0.5

    with pytest.raises(
        ContractValidationError,
        match="canonical algebraic evidence",
    ):
        validate_contract(evidence)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document["phases"][1].update(
                {"started_ns": 29, "duration_ns": 41}
            ),
            "overlaps",
        ),
        (
            lambda document: document["phases"][0].__setitem__(
                "staging_bytes_written",
                3,
            ),
            "phase staging bytes",
        ),
        (
                lambda document: document["integrity"].__setitem__(
                    "file_count",
                    9,
                ),
            "file_count",
        ),
        (
            lambda document: document["artifacts"][0].__setitem__(
                "path",
                "../escaped.bin",
            ),
            "normalized relative path",
        ),
    ],
)
def test_candidate_construction_rejects_inconsistent_evidence(
    mutate,
    message: str,
) -> None:
    construction = contract_fixtures()[5]
    mutate(construction)

    with pytest.raises(ContractValidationError, match=message):
        validate_contract(construction)


def test_candidate_construction_identity_rejects_rebinding() -> None:
    construction = contract_fixtures()[5]
    construction["target_lowering_digest"] = digest("different target")

    with pytest.raises(ContractValidationError, match="construction_id"):
        validate_contract(construction)


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
