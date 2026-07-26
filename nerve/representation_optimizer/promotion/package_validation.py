from __future__ import annotations

import json
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.analysis.evidence import (
    validate_analysis_run_directory,
)
from nerve.representation_optimizer.benchmarking.storage import (
    load_benchmark_evidence,
)
from nerve.representation_optimizer.contracts import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    ContractDocument,
    contract_digest,
)
from nerve.representation_optimizer.promotion.contracts import (
    ImplementationRegistry,
    PromotionDecision,
)
from nerve.representation_optimizer.staging.integrity import (
    integrity_evidence,
    validate_staged_candidate,
)
from nerve.representation_optimizer.validation.storage import (
    load_prebenchmark_evidence,
    load_validation_evidence,
)


def validate_published_implementation_registry(
    package_dir: Path,
    registry: ImplementationRegistry,
    *,
    scope_catalog: Json | None = None,
) -> None:
    package = package_dir.resolve()
    scope_sources = _scope_sources(scope_catalog)
    for entry in registry.implementations:
        bundle = entry["artifact_bundle"]
        root = _package_path(
            package,
            bundle["root_ref"],
            "implementation artifact root",
        )
        if not root.is_dir() or root.is_symlink():
            raise ModelCompileError(
                "published implementation artifact root is missing"
            )
        candidate_root = root / "candidate"
        integrity = validate_staged_candidate(
            candidate_root,
            expected_candidate_id=entry["candidate_id"],
        )
        if (
            integrity_evidence(integrity)["digest"]
            != bundle["candidate_integrity_digest"]
            or len(integrity["files"]) != bundle["artifact_count"]
        ):
            raise ModelCompileError(
                "published candidate integrity does not match registry"
            )
        evidence = entry["evidence"]
        promotion = _read_object(
            _package_path(
                package,
                evidence["promotion_decision_ref"],
                "promotion decision",
            )
        )
        parsed_promotion = PromotionDecision.from_json(promotion)
        if (
            parsed_promotion.implementation_id
            != entry["implementation_id"]
            or parsed_promotion.candidate_id != entry["candidate_id"]
        ):
            raise ModelCompileError(
                "published promotion decision does not match registry"
            )
        candidate_contract = ContractDocument.from_json(
            _read_object(
                _package_path(
                    package,
                    evidence["candidate_contract_ref"],
                    "candidate contract",
                )
            ),
            expected_schema=REPRESENTATION_CANDIDATE_SCHEMA,
        ).to_json()
        construction_record = ContractDocument.from_json(
            _read_object(
                _package_path(
                    package,
                    evidence["construction_record_ref"],
                    "construction record",
                )
            ),
            expected_schema=CANDIDATE_CONSTRUCTION_SCHEMA,
        ).to_json()
        prebenchmark_path = _package_path(
            package,
            evidence["prebenchmark_record_ref"],
            "prebenchmark record",
        )
        prebenchmark_record = _read_object(prebenchmark_path)
        benchmark_path = _package_path(
            package,
            evidence["benchmark_record_ref"],
            "benchmark record",
        )
        benchmark_record = _read_object(benchmark_path)
        validation_path = _package_path(
            package,
            evidence["validation_record_ref"],
            "validation record",
        )
        validation_record = _read_object(validation_path)
        (
            benchmark_plan,
            _benchmark_run,
            loaded_benchmark_record,
        ) = load_benchmark_evidence(
            benchmark_path.parent.parent.parent,
            str(benchmark_record.get("benchmark_id")),
        )
        (
            validation_plan,
            _prebenchmark_record,
            validation_benchmark_record,
            _validation_runs,
            loaded_validation_record,
        ) = load_validation_evidence(
            validation_path.parent.parent.parent,
            str(validation_record.get("validation_id")),
        )
        (
            loaded_prebenchmark_plan,
            loaded_prebenchmark_record,
            _sanity_run,
        ) = load_prebenchmark_evidence(
            prebenchmark_path.parent.parent.parent,
            str(prebenchmark_record.get("prebenchmark_id")),
        )
        decision = parsed_promotion.to_json()
        comparison = decision["comparison"]
        provenance = decision["provenance"]
        integrity_contract = integrity_evidence(integrity)
        benchmark = loaded_benchmark_record.to_json()
        validation = loaded_validation_record.to_json()
        compared_workloads = [
            {
                "workload_id": workload["workload_id"],
                "decision": workload["decision"],
                "paired": workload["paired"],
            }
            for workload in benchmark["workloads"]
        ]
        scoped_source_contracts = (
            [scope_sources[scope_id] for scope_id in entry["scope_ids"]]
            if scope_sources is not None
            and all(scope_id in scope_sources for scope_id in entry["scope_ids"])
            else None
        )
        analysis_references = evidence["analysis_run_refs"]
        analysis_provenance = provenance["analysis_runs"]
        analysis_reference_by_id = {
            reference["run_id"]: reference
            for reference in analysis_references
        }
        loaded_analysis = {}
        for run_record in analysis_provenance:
            run_id = run_record["run_id"]
            reference = analysis_reference_by_id.get(run_id)
            if reference is None:
                continue
            run = validate_analysis_run_directory(
                _package_path(
                    package,
                    reference["artifact_ref"],
                    "analysis run",
                )
            )
            loaded_analysis[run_id] = run
        analysis_links_valid = (
            [reference["run_id"] for reference in analysis_references]
            == [record["run_id"] for record in analysis_provenance]
            and all(
                run_id in loaded_analysis
                and loaded_analysis[run_id].run_id == run_id
                and contract_digest(loaded_analysis[run_id].document)
                == record["run_digest"]
                and set(record["cited_evidence_ids"]).issubset(
                    {
                        item["evidence_id"]
                        for item in loaded_analysis[run_id].evidence
                    }
                )
                and loaded_analysis[run_id].document["package_id"]
                == registry.to_json()["package_id"]
                and loaded_analysis[run_id].document["scope_id"]
                in entry["scope_ids"]
                and loaded_analysis[run_id].document[
                    "source_contract_digest"
                ]
                in entry["source_contract_digests"]
                for record in analysis_provenance
                for run_id in (record["run_id"],)
            )
        )
        hardware_profile_references = evidence["hardware_profile_refs"]
        hardware_profile_provenance = provenance["hardware_profiles"]
        loaded_profiles = []
        for reference in hardware_profile_references:
            profile = ContractDocument.from_json(
                _read_object(
                    _package_path(
                        package,
                        reference["artifact_ref"],
                        "hardware profile",
                    )
                ),
                expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
            ).to_json()
            loaded_profiles.append(profile)
        predicate = entry["runtime_predicate"]
        hardware_predicate = predicate["hardware"]
        placement_predicate = predicate["placement"]
        hardware_profiles_valid = (
            [reference["profile_id"] for reference in hardware_profile_references]
            == [
                profile["profile_id"]
                for profile in hardware_profile_provenance
            ]
            == [profile["profile_id"] for profile in loaded_profiles]
            and [
                contract_digest(profile) for profile in loaded_profiles
            ]
            == [
                profile["profile_digest"]
                for profile in hardware_profile_provenance
            ]
            and sorted(
                {
                    profile["capability_class"]
                    for profile in loaded_profiles
                }
            )
            == hardware_predicate["capability_classes"]
            and sorted(
                {
                    profile["hardware_identity"]["device_kind"]
                    for profile in loaded_profiles
                }
            )
            == hardware_predicate["device_kinds"]
            and sorted(
                {
                    profile["provenance"]["api"]
                    for profile in loaded_profiles
                }
            )
            == hardware_predicate["apis"]
            and placement_predicate["minimum_device_count"]
            == len(loaded_profiles)
            == placement_predicate["maximum_device_count"]
            and benchmark_plan.matched_conditions["devices"]
            == sorted(
                (
                    {
                        "device_id": profile["hardware_identity"][
                            "stable_device_id"
                        ],
                        "hardware_profile_digest": contract_digest(
                            profile
                        ),
                        "capability_class": profile[
                            "capability_class"
                        ],
                        "api": profile["provenance"]["api"],
                    }
                    for profile in loaded_profiles
                ),
                key=lambda device: device["device_id"],
            )
        )
        checks = (
            (
                candidate_contract.get("candidate_id")
                == entry["candidate_id"],
                "candidate identity",
            ),
            (
                candidate_contract["scope_ids"] == entry["scope_ids"],
                "candidate scopes",
            ),
            (
                candidate_contract["source_contract_digests"]
                == entry["source_contract_digests"],
                "candidate source contracts",
            ),
            (
                candidate_contract["representation"]
                == entry["representation"],
                "candidate representation",
            ),
            (
                candidate_contract["behavioral_contract"]
                == entry["behavioral_contract"],
                "candidate behavioral contract",
            ),
            (
                candidate_contract["provider"] == provenance["provider"],
                "candidate provider provenance",
            ),
            (
                candidate_contract["descriptor_id"]
                == provenance["descriptor_id"],
                "candidate descriptor provenance",
            ),
            (
                candidate_contract["evidence_refs"]
                == provenance["evidence_refs"],
                "candidate analysis provenance",
            ),
            (analysis_links_valid, "analysis run provenance"),
            (hardware_profiles_valid, "hardware profile provenance"),
            (
                construction_record.get("candidate_id")
                == entry["candidate_id"],
                "construction candidate",
            ),
            (
                construction_record["status"] == "completed",
                "construction status",
            ),
            (
                construction_record["construction_id"]
                == integrity["construction_id"],
                "construction identity",
            ),
            (
                construction_record["integrity"] == integrity_contract,
                "construction integrity",
            ),
            (
                construction_record["representation_graph_digest"]
                == provenance["representation_graph_digest"],
                "representation graph provenance",
            ),
            (
                construction_record["target_lowering_digest"]
                == provenance["target_lowering_digest"],
                "target lowering provenance",
            ),
            (
                construction_record["relowering_request_digest"]
                == provenance["relowering_request_digest"],
                "re-lowering provenance",
            ),
            (
                benchmark_record.get("candidate_id")
                == entry["candidate_id"],
                "benchmark candidate",
            ),
            (
                validation_record.get("candidate_id")
                == entry["candidate_id"],
                "validation candidate",
            ),
            (
                contract_digest(candidate_contract)
                == decision["candidate_contract_digest"],
                "candidate decision digest",
            ),
            (
                contract_digest(construction_record)
                == decision["construction_record_digest"],
                "construction decision digest",
            ),
            (
                contract_digest(prebenchmark_record)
                == decision["prebenchmark_record_digest"],
                "prebenchmark decision digest",
            ),
            (
                contract_digest(benchmark_record)
                == decision["benchmark_record_digest"],
                "benchmark decision digest",
            ),
            (
                contract_digest(validation_record)
                == decision["validation_record_digest"],
                "validation decision digest",
            ),
            (benchmark == benchmark_record, "loaded benchmark record"),
            (validation == validation_record, "loaded validation record"),
            (
                loaded_prebenchmark_record.to_json()
                == prebenchmark_record,
                "loaded prebenchmark record",
            ),
            (
                loaded_prebenchmark_plan == validation_plan,
                "prebenchmark validation plan link",
            ),
            (
                _prebenchmark_record == loaded_prebenchmark_record,
                "validation prebenchmark link",
            ),
            (
                validation_benchmark_record == loaded_benchmark_record,
                "validation benchmark link",
            ),
            (
                benchmark_plan.candidate_id == entry["candidate_id"],
                "benchmark plan candidate",
            ),
            (
                validation_plan.candidate_id == entry["candidate_id"],
                "validation plan candidate",
            ),
            (
                benchmark["reference_implementation_id"]
                == comparison["exact_implementation_id"],
                "exact implementation comparison",
            ),
            (
                benchmark_plan.to_json()["implementations"]["reference"][
                    "contract_digest"
                ]
                == comparison["exact_contract_digest"],
                "exact contract comparison",
            ),
            (
                benchmark["benchmark_id"] == comparison["benchmark_id"],
                "benchmark comparison identity",
            ),
            (
                benchmark["decision"]
                == comparison["benchmark_decision"],
                "benchmark comparison decision",
            ),
            (
                compared_workloads == comparison["workloads"],
                "per-regime benchmark comparison",
            ),
            (
                validation["validation_id"]
                == comparison["validation_id"],
                "validation comparison identity",
            ),
            (
                validation["status"]
                == comparison["validation_status"],
                "validation comparison status",
            ),
            (
                validation["behavioral_contract"]
                == comparison["behavioral_contract"],
                "validation comparison contract",
            ),
            (
                entry["scope_ids"] == decision["scope_ids"],
                "registry decision scopes",
            ),
            (
                entry["source_contract_digests"]
                == decision["source_contract_digests"],
                "registry decision source contracts",
            ),
            (
                entry["runtime_predicate"]
                == decision["runtime_predicate"],
                "registry decision runtime predicate",
            ),
            (
                entry["provenance"] == decision["provenance"],
                "registry decision provenance",
            ),
            (
                entry["comparison"] == decision["comparison"],
                "registry decision comparison",
            ),
            (
                entry["decision_reason"] == decision["reason"],
                "registry decision reason",
            ),
            (
                entry["artifact_bundle"]["candidate_integrity_digest"]
                == decision["artifact_integrity"]["digest"],
                "registry decision integrity digest",
            ),
            (
                entry["artifact_bundle"]["artifact_count"]
                == decision["artifact_integrity"]["file_count"],
                "registry decision artifact count",
            ),
            (
                decision["artifact_integrity"] == integrity_contract,
                "promotion decision integrity",
            ),
            (
                scope_sources is None
                or scoped_source_contracts
                == entry["source_contract_digests"],
                "scope catalog source contracts",
            ),
        )
        for passed, label in checks:
            if not passed:
                raise ModelCompileError(
                    "published implementation evidence does not match "
                    f"promotion: {label}"
                )


def _scope_sources(scope_catalog: Json | None) -> dict[str, str] | None:
    if scope_catalog is None:
        return None
    scopes = scope_catalog.get("scopes")
    if not isinstance(scopes, list):
        raise ModelCompileError(
            "optimization scope catalog has no scope list"
        )
    sources: dict[str, str] = {}
    for scope in scopes:
        if not isinstance(scope, dict):
            raise ModelCompileError(
                "optimization scope catalog contains a malformed scope"
            )
        scope_id = scope.get("scope_id")
        source_digest = scope.get("source_contract_digest")
        if (
            not isinstance(scope_id, str)
            or not isinstance(source_digest, str)
            or scope_id in sources
        ):
            raise ModelCompileError(
                "optimization scope catalog contains invalid scope links"
            )
        sources[scope_id] = source_digest
    return sources


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"published implementation artifact is unreadable: {path}"
        ) from error
    if not isinstance(document, dict):
        raise ModelCompileError(
            f"published implementation artifact must be an object: {path}"
        )
    return document


def _package_path(package: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"{label} path must not be empty")
    relative = Path(value)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ModelCompileError(
            f"{label} must be a canonical package-relative path"
        )
    path = package / relative
    try:
        path.resolve().relative_to(package.resolve())
    except ValueError as error:
        raise ModelCompileError(
            f"{label} escapes the compiled package"
        ) from error
    return path
