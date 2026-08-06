from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.analysis.evidence import build_evidence
from nerve.representation_optimizer.contracts import (
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    contract_digest,
    representation_candidate_id,
    source_behavior_contract_digest,
    stable_contract_id,
    validate_contract,
)
from nerve.representation_optimizer.descriptor_registry import (
    load_builtin_representation_descriptors,
)
from nerve.representation_optimizer.providers.builtin import (
    BuiltinCandidateToolchainResolver,
    load_builtin_provider_registry,
)
from nerve.representation_optimizer.providers.registry import ProviderRegistry
from nerve.representation_optimizer.providers.resident_expansion import (
    ExactResidentExpansionProofVerifier,
    ExactResidentExpertExpansionProvider,
    ResidentExpansionToolchainResolver,
)
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    PROOF_PATH,
    component_overlay_path,
    resident_shader_artifact_path,
)
from nerve.representation_optimizer.providers.resident_expansion.contracts import (
    PROOF_VERIFIER_ID,
)
from nerve.representation_optimizer.providers.resident_expansion.discovery import (
    ResidentExpansionOpportunity,
    ResidentShaderReplacement,
    ResidentWeightDerivation,
)
from nerve.representation_optimizer.providers.resident_expansion.proof import (
    _verify_source_coverage,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
    ProviderProblem,
)
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.staging.artifact_validation import (
    ArtifactValidatorRegistry,
)
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
    staged_file_digest,
)
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)
from nerve.representation_optimizer.validation.protocols import ProofRequest
from nerve.resident_representations import (
    MXFP4_TO_FP8_REQUIRED_FEATURES,
    mxfp4_to_fp8_resident_derivation,
)
from tests.test_representation_optimizer_contracts import hardware_profile_contract


_COMPONENT_ID = "sparse_block_alpha"
_HIDDEN_SIZE = 128
_INTERMEDIATE_SIZE = 128
_EXPERT_COUNT = 1
_EXPERTS_PER_TOKEN = 1
_SOURCE_WEIGHT_BYTES = _HIDDEN_SIZE * _INTERMEDIATE_SIZE // 2


def _tensor_pair(weight_name: str, scale_name: str) -> tuple[dict, dict]:
    return (
        {
            "dtype": "I8",
            "shape": [_HIDDEN_SIZE, _INTERMEDIATE_SIZE // 2],
            "logical_shape": [_HIDDEN_SIZE, _INTERMEDIATE_SIZE],
            "byte_count": _SOURCE_WEIGHT_BYTES,
            "layout": "row_major",
            "quantization": {
                "format": "mxfp4_e2m1",
                "bits": 4,
                "element_type": "float",
                "values_per_byte": 2,
                "packing_axis": 1,
                "packing_order": "low_nibble_then_high_nibble_along_k",
                "group_size": 32,
                "scales": scale_name,
                "scale_dtype": "F8_E8M0",
            },
        },
        {
            "dtype": "F8_E8M0",
            "shape": [_HIDDEN_SIZE, _INTERMEDIATE_SIZE // 32],
            "byte_count": _HIDDEN_SIZE * _INTERMEDIATE_SIZE // 32,
            "layout": "row_major",
        },
    )


def _source_package(package: Path) -> tuple[dict, dict]:
    parameter_pairs = (
        ("down_weight", "down_scale"),
        ("gate_weight", "gate_scale"),
        ("up_weight", "up_scale"),
    )
    tensors = {}
    refs = {}
    for weight_name, scale_name in parameter_pairs:
        weight, scale = _tensor_pair(weight_name, scale_name)
        tensors[weight_name] = weight
        tensors[scale_name] = scale
        refs[weight_name] = {"tensor": weight_name, "role": "expert_weight"}
        refs[scale_name] = {"tensor": scale_name, "role": "expert_scale"}

    down_node = {
        "id": "projection_down",
        "op": "independent_sparse_moe_down",
        "params": ["down_weight", "down_scale"],
        "attrs": {
            "hidden_size": _HIDDEN_SIZE,
            "intermediate_size": _INTERMEDIATE_SIZE,
            "experts_per_token": _EXPERTS_PER_TOKEN,
            "selected_parameter_accesses": [
                {
                    "mapping": [
                        {
                            "selector": 0,
                            "parameter_ids": ["down_weight", "down_scale"],
                        }
                    ]
                }
            ],
        },
    }
    gate_node = {
        "id": "projection_gate_up",
        "op": "independent_sparse_moe_gate_up",
        "params": [
            "gate_weight",
            "gate_scale",
            "up_weight",
            "up_scale",
        ],
        "attrs": {
            "hidden_size": _HIDDEN_SIZE,
            "intermediate_size": _INTERMEDIATE_SIZE,
            "experts_per_token": _EXPERTS_PER_TOKEN,
            "selected_parameter_accesses": [
                {
                    "mapping": [
                        {
                            "selector": 0,
                            "parameter_ids": [
                                "gate_weight",
                                "gate_scale",
                                "up_weight",
                                "up_scale",
                            ],
                        }
                    ]
                }
            ],
        },
    }
    shader_paths = {
        "down_scalar": (
            "shaders/independent_sparse_moe_down_mxfp4_e2m1_g32_h128_i128_e1_k1.spv"
        ),
        "down_batch": (
            "shaders/independent_sparse_moe_down_batch1_mxfp4_e2m1_"
            "g32_h128_i128_e1_k1__pbc31.spv"
        ),
        "gate_scalar": (
            "shaders/independent_sparse_moe_gate_up_prequant_mxfp4_e2m1_"
            "g32_h128_i128_e1_k1_limit10.spv"
        ),
        "gate_batch": (
            "shaders/independent_sparse_moe_gate_up_batch1_prequant_mxfp4_e2m1_"
            "g32_h128_i128_e1_k1_limit10__pbc31.spv"
        ),
    }
    execution = {
        "component_id": _COMPONENT_ID,
        "implementation": "compact_source_implementation",
        "kernels": [
            {
                "node_id": down_node["id"],
                "shader_path": shader_paths["down_scalar"],
                "batch_implementations": [
                    {"stages": [{"shader_path": shader_paths["down_batch"]}]}
                ],
            },
            {
                "node_id": gate_node["id"],
                "shader_path": shader_paths["gate_scalar"],
                "batch_implementations": [
                    {"stages": [{"shader_path": shader_paths["gate_batch"]}]}
                ],
            },
        ],
    }
    resources = []
    bindings = []
    for node_id, weight_names in (
        (down_node["id"], ("down_weight",)),
        (gate_node["id"], ("gate_weight", "up_weight")),
    ):
        for slot, weight_name in enumerate(weight_names):
            parameter_slot = slot * 2
            resource_id = f"resource_{weight_name}"
            resources.append(
                {
                    "id": resource_id,
                    "lifetime": "dynamic",
                    "ranges": [{"byte_count": _SOURCE_WEIGHT_BYTES}],
                }
            )
            bindings.append(
                {
                    "component_id": _COMPONENT_ID,
                    "node_id": node_id,
                    "parameter_id": weight_name,
                    "mapping": {
                        "kind": "selected_atomic_group",
                        "selector_index": 0,
                        "parameter_slot": parameter_slot,
                        "resource_id": resource_id,
                    },
                }
            )
    component = {
        "component_id": _COMPONENT_ID,
        "implementation": "compact_source_implementation",
        "circuit": {
            "implementation": "compact_source_implementation",
            "nodes": [down_node, gate_node],
        },
        "params": {"refs": refs},
    }
    tensor_index = {
        "schema": "nerve.tensor_index.v1",
        "source": {"weights_files": []},
        "tensors": tensors,
    }
    manifest = {
        "tensor_index_path": "tensors.json",
        "max_context_activations": 131_072,
        "circuit_graph": {"components": [component]},
        "component_executions": [execution],
        "resource_residency": {
            "resources": resources,
            "bindings": bindings,
        },
    }
    package.mkdir()
    (package / "tensors.json").write_text(json.dumps(tensor_index))
    (package / "vulkan_resident_package.json").write_text(json.dumps(manifest))
    return manifest, tensor_index


def _opportunity(package: Path) -> ResidentExpansionOpportunity:
    manifest, tensor_index = _source_package(package)
    features = list(MXFP4_TO_FP8_REQUIRED_FEATURES)
    derivations = []
    for node_id, parameter_id in (
        ("projection_down", "down_weight"),
        ("projection_gate_up", "gate_weight"),
        ("projection_gate_up", "up_weight"),
    ):
        derivation = mxfp4_to_fp8_resident_derivation(
            tensor_index["tensors"][parameter_id],
            {"devices": [{"shader_features": features}]},
        )
        assert derivation is not None
        derivations.append(
            ResidentWeightDerivation(
                node_id=node_id,
                parameter_id=parameter_id,
                tensor_name=parameter_id,
                source_resource_id=f"resource_{parameter_id}",
                source_byte_count=_SOURCE_WEIGHT_BYTES,
                derivation=derivation,
            )
        )
    execution = manifest["component_executions"][0]
    replacements = []
    for kernel in execution["kernels"]:
        for execution_kind, source_path in (
            ("scalar", kernel["shader_path"]),
            ("batch", kernel["batch_implementations"][0]["stages"][0]["shader_path"]),
        ):
            artifact_path = resident_shader_artifact_path(source_path)
            stem = artifact_path.rsplit("/", 1)[-1][:-4]
            stem = stem.replace("__pbc31", "")
            replacements.append(
                ResidentShaderReplacement(
                    node_id=kernel["node_id"],
                    source_path=source_path,
                    artifact_path=artifact_path,
                    template_name=f"{stem}.comp",
                    execution_kind=execution_kind,
                )
            )
    scope_ids = tuple(
        sorted(
            (
                stable_contract_id("scope", "down"),
                stable_contract_id("scope", "gate"),
            )
        )
    )
    return ResidentExpansionOpportunity(
        scope_ids=scope_ids,
        source_contract_digests=tuple(
            contract_digest({"scope_id": scope_id}) for scope_id in scope_ids
        ),
        component_id=_COMPONENT_ID,
        node_ids=("projection_down", "projection_gate_up"),
        evidence_ids=(stable_contract_id("evidence", "resident"),),
        source_artifact_refs=("tensors.json", "vulkan_resident_package.json"),
        manifest_ref="vulkan_resident_package.json",
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        expert_count=_EXPERT_COUNT,
        experts_per_token=_EXPERTS_PER_TOKEN,
        max_context_activations=131_072,
        weight_derivations=tuple(derivations),
        shader_replacements=tuple(
            sorted(
                replacements,
                key=lambda item: (item.node_id, item.source_path),
            )
        ),
    )


def _provider_products(tmp_path: Path, monkeypatch):
    package = tmp_path / "package"
    opportunity = _opportunity(package)
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.resident_expansion.provider."
        "discover_resident_expansions",
        lambda context: (opportunity,),
    )
    provider = ExactResidentExpertExpansionProvider()
    context = SimpleNamespace(
        hardware_profile={"capability_class": "hardware_capability_fixture"},
        qualification_regime=QualificationRegime(),
        source_artifacts=PackageSourceArtifactResolver(package),
    )
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=opportunity.evidence_ids,
        facts={"exact": True},
        reasons=("fixture evidence",),
    )
    candidate = provider.synthesize_candidates(context, evidence)[0]
    representation = provider.emit_representation_ir(context, candidate)
    lowering = provider.lower_for_target(context, candidate, representation)
    build_plan = CandidateBuildPlan.from_json(
        provider.construction_requirements(context, candidate)
    )
    return (
        provider,
        context,
        opportunity,
        candidate,
        representation,
        lowering,
        build_plan,
    )


def _resident_hardware_profile(*, supports_fp8: bool = True) -> dict:
    profile = deepcopy(hardware_profile_contract())
    if supports_fp8:
        profile["processes"].append(
            {
                "name": "packed_dot_product",
                "category": "arithmetic",
                "availability": "available",
                "programmability": "direct",
                "api": "vulkan",
                "operations": [
                    "mixed_dot_accumulate",
                    "packed_dot_accumulate",
                ],
                "numeric_formats": ["f8_e4m3", "i8"],
                "required_extensions": [],
                "required_features": [
                    "shader_mixed_float_dot_product_float8_acc_float32"
                ],
                "limits": {},
                "properties": {},
            }
        )
        profile["processes"].sort(key=lambda item: item["name"])
        profile["capability_extensions"] = {
            "vulkan_compiler_capabilities": {
                "shader_features": list(MXFP4_TO_FP8_REQUIRED_FEATURES)
            }
        }
    identity = profile["hardware_identity"]
    provenance = profile["provenance"]
    profile["capability_class"] = stable_contract_id(
        "hardware_capability",
        {
            "device_kind": identity["device_kind"],
            "architecture": identity["architecture"],
            "processes": profile["processes"],
            "memory_domains": profile["memory_domains"],
            "interconnects": profile["interconnects"],
            "api": provenance["api"],
            "api_version": provenance["api_version"],
            "capability_extensions": profile["capability_extensions"],
        },
    )
    profile["profile_id"] = stable_contract_id(
        "hardware_profile",
        [
            identity,
            profile["capability_class"],
            provenance,
            profile["identity_extensions"],
            profile["measurements"],
        ],
    )
    validate_contract(
        profile,
        expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
    )
    return profile


def _discovery_problem(
    package: Path,
    *,
    supports_fp8: bool = True,
) -> ProviderProblem:
    manifest, _tensor_index = _source_package(package)
    component = manifest["circuit_graph"]["components"][0]
    refs = component["params"]["refs"]
    scopes = []
    contracts = []
    evidence = []
    package_id = "synthetic_sparse_package"
    for node in component["circuit"]["nodes"]:
        qualified_node_id = f"{_COMPONENT_ID}/{node['id']}"
        scope_id = stable_contract_id(
            "scope",
            package_id,
            "operator",
            [_COMPONENT_ID],
            [],
            [qualified_node_id],
        )
        parameters = [
            {
                "id": f"parameter:{qualified_node_id}/{parameter_id}",
                "component_id": _COMPONENT_ID,
                "parameter_ref_id": parameter_id,
                "definition": refs[parameter_id],
            }
            for parameter_id in node["params"]
        ]
        interface = {
            "inputs": [],
            "outputs": [],
            "parameters": parameters,
            "states": [],
            "controls": [],
            "randomness": [],
            "dependencies": [],
        }
        contract = {
            "schema": SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
            "scope_id": scope_id,
            "semantic_role": node["op"],
            "interface": interface,
            "exact_reference": {
                "implementation_id": "compact_source_implementation",
                "artifact_refs": ["vulkan_resident_package.json"],
            },
            "contract_digest": "",
        }
        contract["contract_digest"] = source_behavior_contract_digest(contract)
        scope = {
            "schema": OPTIMIZATION_SCOPE_SCHEMA,
            "scope_id": scope_id,
            "package_id": package_id,
            "kind": "operator",
            "members": {
                "component_ids": [_COMPONENT_ID],
                "semantic_module_ids": [],
                "source_node_ids": [qualified_node_id],
            },
            "boundary": interface,
            "source_contract_digest": contract["contract_digest"],
        }
        evidence_record, _details = build_evidence(
            scope_id=scope_id,
            source_contract_digest=contract["contract_digest"],
            analyzer_id="synthetic_operator_structure",
            analyzer_version="1",
            claims=(
                {
                    "kind": "operator_structure",
                    "status": "supported",
                    "exact": True,
                    "facts": {"semantic_role": node["op"]},
                },
            ),
            details={},
        )
        scopes.append(scope)
        contracts.append(contract)
        evidence.append(evidence_record)
    return ProviderProblem.from_documents(
        package_id=package_id,
        scopes=scopes,
        source_contracts=contracts,
        evidence=evidence,
        hardware_profile=_resident_hardware_profile(supports_fp8=supports_fp8),
        source_artifacts=PackageSourceArtifactResolver(package),
    )


def _construct(tmp_path: Path, monkeypatch):
    products = _provider_products(tmp_path, monkeypatch)
    (
        provider,
        context,
        _opportunity_value,
        candidate,
        representation,
        lowering,
        build_plan,
    ) = products
    workspace = tmp_path / "workspace"
    root = workspace / "ready" / candidate["candidate_id"]
    root.mkdir(parents=True)
    construction = CandidateConstructionContext(
        package_dir=tmp_path / "package",
        staging_dir=root,
        candidate=candidate,
        representation_graph=representation,
        target_lowering=lowering,
        build_plan=build_plan,
        source_artifacts=context.source_artifacts,
        started_ns=time.monotonic_ns(),
        cancel_requested=None,
    )
    toolchain = ResidentExpansionToolchainResolver().resolve(
        SimpleNamespace(provider=provider.identity, target_lowering=lowering)
    )
    for phase, service in (
        (
            "semantic_construction",
            toolchain.semantic_constructor.construct_semantic_artifacts,
        ),
        ("ordinary_lowering", toolchain.ordinary_relowerer.run_ordinary_lowering),
        (
            "physical_optimization",
            toolchain.physical_optimizer.optimize_physical_artifacts,
        ),
    ):
        construction.begin_phase(phase)
        service(construction)
        construction.end_phase()
    construction.validate_complete()
    construction.write_internal_contract("target_lowering.json", lowering)
    return products, workspace, root


def test_resident_shader_artifacts_reject_unsafe_or_recursive_paths() -> None:
    source = "shaders/independent_sparse_moe_down_mxfp4_e2m1_g32_h128_i128_e1_k1.spv"
    assert resident_shader_artifact_path(source) == (
        "kernels/independent_sparse_moe_down_mxfp4_e2m1_"
        "resident_fp8_e4m3_g32_h128_i128_e1_k1.spv"
    )
    for invalid in (
        "../independent_sparse_moe_down_mxfp4_e2m1.spv",
        "/tmp/independent_sparse_moe_down_mxfp4_e2m1.spv",
        "shaders/already_mxfp4_e2m1_resident_fp8_e4m3_g32.spv",
        "shaders/unrelated.spv",
    ):
        with pytest.raises(ModelCompileError):
            resident_shader_artifact_path(invalid)


def test_registry_discovers_a_generic_sparse_component_without_model_identity(
    tmp_path: Path,
) -> None:
    report = ProviderRegistry.from_providers(
        descriptors=load_builtin_representation_descriptors(),
        providers=(ExactResidentExpertExpansionProvider(),),
    ).run(_discovery_problem(tmp_path / "anonymous_package"))

    assert report.evaluations[0].status == "completed"
    assert report.evaluations[0].error is None
    assert len(report.candidates) == 1
    plan = report.candidates[0]
    assert plan.target_lowering["regions"][0]["source"]["component_id"] == _COMPONENT_ID
    assert len(plan.target_lowering["regions"][0]["resident_derivations"]) == 3
    assert len(plan.target_lowering["regions"][0]["shader_replacements"]) == 4
    builtin_ids = {
        provider.identity.provider_id
        for provider in load_builtin_provider_registry().providers
    }
    assert "nerve.exact_resident_expert_parameter_expansion" in builtin_ids
    toolchain = BuiltinCandidateToolchainResolver().resolve(plan)
    assert toolchain.semantic_constructor is not None
    assert toolchain.ordinary_relowerer is not None
    assert toolchain.physical_optimizer is not None


def test_resident_expansion_requests_only_exact_graph_structure_analysis() -> None:
    provider = ExactResidentExpertExpansionProvider()

    assert provider.required_analyzer_ids({}, {}) == (
        "semantic_graph_structure",
    )


def test_registry_declines_resident_fp8_when_the_target_lacks_exact_capability(
    tmp_path: Path,
) -> None:
    report = ProviderRegistry.from_providers(
        descriptors=load_builtin_representation_descriptors(),
        providers=(ExactResidentExpertExpansionProvider(),),
    ).run(
        _discovery_problem(
            tmp_path / "unsupported_package",
            supports_fp8=False,
        )
    )

    evaluation = report.evaluations[0]
    assert evaluation.status == "declined"
    assert evaluation.error is None
    assert not report.candidates
    assert evaluation.structural_match is not None
    assert evaluation.structural_match.reasons == (
        "target has no programmable native F8 E4M3 resident path",
    )


def test_registry_rejects_a_sparse_weight_shared_across_component_boundaries(
    tmp_path: Path,
) -> None:
    package = tmp_path / "shared_resource_package"
    problem = _discovery_problem(package)
    manifest_path = package / "vulkan_resident_package.json"
    manifest = json.loads(manifest_path.read_text())
    shared = deepcopy(manifest["resource_residency"]["bindings"][0])
    shared["component_id"] = "another_sparse_component"
    manifest["resource_residency"]["bindings"].append(shared)
    manifest_path.write_text(json.dumps(manifest))

    report = ProviderRegistry.from_providers(
        descriptors=load_builtin_representation_descriptors(),
        providers=(ExactResidentExpertExpansionProvider(),),
    ).run(problem)

    evaluation = report.evaluations[0]
    assert evaluation.status == "declined"
    assert evaluation.error is None
    assert not report.candidates
    assert evaluation.structural_match is not None
    assert any(
        "does not own one exact compact dynamic resource" in reason
        for reason in evaluation.structural_match.reasons
    )


def test_provider_plan_is_component_local_exact_and_product_qualified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider, context, opportunity, candidate, representation, lowering, build_plan = (
        _provider_products(tmp_path, monkeypatch)
    )
    validate_contract(candidate)
    assert candidate["scope_ids"] == list(opportunity.scope_ids)
    topology = candidate["representation"]["topology"]
    assert topology["kind"] == "independently_selectable_component_regions"
    assert topology["component_ids"] == [_COMPONENT_ID]
    assert topology["node_ids"] == ["projection_down", "projection_gate_up"]
    assert topology["performance_equivalence_class"].startswith(
        "resident_performance_class_"
    )
    assert representation["confidence"]["mode"] == "exact"
    assert lowering["runtime"] == {
        "max_context_activations": 131_072,
        "required_vulkan_version": "1.4",
        "residency_lifetime": "demand_retained",
    }
    assert set(build_plan.output_paths) == set(
        candidate_path["path"] for candidate_path in candidate["artifact_declarations"]
    )
    declared_artifacts = {
        item["path"] for item in candidate["artifact_declarations"]
    }
    referenced_artifacts = {
        item["artifact"]["path"] for item in representation["resources"]
    } | {
        item["artifact"]["path"] for item in representation["physical_kernels"]
    }
    assert referenced_artifacts <= declared_artifacts

    estimate = provider.estimate_static_cost(
        context,
        candidate,
        representation,
        lowering,
    )
    assert estimate.permanent_bytes == 0
    assert estimate.transient_bytes == opportunity.source_weight_bytes
    assert estimate.steady_state_work["materialization"] == (
        "only_selected_resources_on_demand"
    )
    workloads = provider.benchmark_workloads(context, candidate)
    assert len(workloads) == 5
    assert max(item["useful_work"]["minimum_units"] for item in workloads) == 2
    assert {item["regime"]["mount_mode"] for item in workloads} == {
        "cold",
        "resident_reuse",
    }
    checks = provider.validation_requirements(context, candidate)["checks"]
    whole_model = [item for item in checks if item["stage"] == "whole_model"]
    assert len(whole_model) == 2
    assert all(item["controls"]["enable_thinking"] for item in whole_model)
    assert all(item["controls"]["max_output_tokens"] == 65_536 for item in whole_model)


def test_provider_groups_equivalent_components_into_independently_mountable_regions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "package"
    first = _opportunity(package)
    second_scope_ids = tuple(
        sorted(
            (
                stable_contract_id("scope", "second_down"),
                stable_contract_id("scope", "second_gate"),
            )
        )
    )
    second = replace(
        first,
        component_id="sparse_block_beta",
        scope_ids=second_scope_ids,
        source_contract_digests=tuple(
            contract_digest({"scope_id": scope_id}) for scope_id in second_scope_ids
        ),
        evidence_ids=(stable_contract_id("evidence", "resident_second"),),
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.resident_expansion.provider."
        "discover_resident_expansions",
        lambda context: (first, second),
    )
    provider = ExactResidentExpertExpansionProvider()
    context = SimpleNamespace(
        hardware_profile={"capability_class": "hardware_capability_fixture"},
        qualification_regime=QualificationRegime(),
        source_artifacts=PackageSourceArtifactResolver(package),
    )
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=tuple(sorted((*first.evidence_ids, *second.evidence_ids))),
        facts={"exact": True},
        reasons=("fixture evidence",),
    )

    candidates = provider.synthesize_candidates(context, evidence)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["representation"]["topology"]["component_ids"] == [
        "sparse_block_alpha",
        "sparse_block_beta",
    ]
    representation = provider.emit_representation_ir(context, candidate)
    lowering = provider.lower_for_target(context, candidate, representation)
    assert [region["source"]["component_id"] for region in lowering["regions"]] == [
        "sparse_block_alpha",
        "sparse_block_beta",
    ]
    mount = provider.mount_requirements(context, candidate)
    assert [
        region["replacements"][0]["source_component_id"] for region in mount["regions"]
    ] == ["sparse_block_alpha", "sparse_block_beta"]
    assert len(provider.benchmark_workloads(context, candidate)) == 5


def test_provider_rejects_candidate_topology_not_bound_to_discovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider, context, _opportunity, candidate, *_rest = _provider_products(
        tmp_path,
        monkeypatch,
    )
    candidate["representation"]["topology"]["component_ids"] = ["different_component"]
    candidate["candidate_id"] = representation_candidate_id(candidate)

    with pytest.raises(ModelCompileError, match="topology"):
        provider.emit_representation_ir(context, candidate)


def test_provider_rejects_candidate_source_digest_not_bound_to_discovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider, context, _opportunity, candidate, *_rest = _provider_products(
        tmp_path,
        monkeypatch,
    )
    candidate["source_contract_digests"][0] = contract_digest({"different": "source"})
    candidate["candidate_id"] = representation_candidate_id(candidate)

    with pytest.raises(ModelCompileError, match="source contracts"):
        provider.emit_representation_ir(context, candidate)


def test_provider_never_shares_measurements_across_different_physical_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _opportunity(tmp_path / "package")
    second_scope_ids = tuple(
        sorted(
            (
                stable_contract_id("scope", "different_down"),
                stable_contract_id("scope", "different_gate"),
            )
        )
    )
    different = replace(
        first,
        component_id="different_geometry",
        scope_ids=second_scope_ids,
        source_contract_digests=tuple(
            contract_digest({"scope_id": scope_id}) for scope_id in second_scope_ids
        ),
        evidence_ids=(stable_contract_id("evidence", "different_geometry"),),
        experts_per_token=first.experts_per_token + 1,
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.resident_expansion.provider."
        "discover_resident_expansions",
        lambda context: (first, different),
    )
    provider = ExactResidentExpertExpansionProvider()
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=tuple(sorted((*first.evidence_ids, *different.evidence_ids))),
        facts={"exact": True},
        reasons=("fixture evidence",),
    )

    candidates = provider.synthesize_candidates(
        SimpleNamespace(hardware_profile={"capability_class": "fixture"}),
        evidence,
    )

    assert len(candidates) == 2
    assert {
        tuple(candidate["representation"]["topology"]["component_ids"])
        for candidate in candidates
    } == {(_COMPONENT_ID,), ("different_geometry",)}


def test_toolchain_constructs_and_proves_only_the_declared_component_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    products, workspace, root = _construct(tmp_path, monkeypatch)
    (
        _provider,
        context,
        opportunity,
        candidate,
        _representation,
        lowering,
        build_plan,
    ) = products
    results = ArtifactValidatorRegistry.with_builtin_validators().validate_artifacts(
        root,
        build_plan,
    )
    assert set(results) == set(build_plan.output_paths)

    overlay = json.loads((root / component_overlay_path(_COMPONENT_ID)).read_text())
    assert overlay["source_component_id"] == _COMPONENT_ID
    assert len(overlay["resident_derivations"]) == 3
    assert all(
        item["derivation"]["resident_byte_count"]
        == item["derivation"]["source_byte_count"] * 2
        for item in overlay["resident_derivations"]
    )
    replaced_paths = {
        kernel["shader_path"] for kernel in overlay["execution"]["kernels"]
    }
    assert replaced_paths == {
        item["artifact_path"]
        for item in lowering["regions"][0]["shader_replacements"]
        if item["execution_kind"] == "scalar"
    }

    proof_digest = staged_file_digest(root / PROOF_PATH)
    verifier = ExactResidentExpansionProofVerifier(
        source_artifacts=context.source_artifacts,
        candidate_workspace_root=workspace,
    )
    request = ProofRequest(
        plan_id=stable_contract_id("validation_plan", candidate["candidate_id"]),
        candidate_id=candidate["candidate_id"],
        obligation=candidate["behavioral_contract"]["proof_obligations"][0],
        verifier_id=PROOF_VERIFIER_ID,
        source_contract_digests=opportunity.source_contract_digests,
        construction_record_digest=contract_digest({"construction": "fixture"}),
        reference_implementation={
            "implementation_id": "source",
            "contract_digest": opportunity.source_contract_digests[0],
            "artifact_refs": [],
        },
        candidate_implementation={
            "implementation_id": f"staged-representation:{candidate['candidate_id']}",
            "contract_digest": contract_digest({"implementation": "fixture"}),
            "artifact_refs": [{"path": PROOF_PATH, "digest": proof_digest}],
        },
    )
    result = verifier.verify(request)
    assert result["status"] == "proven"
    assert result["facts"]["code_domain"] == {
        "source_code_count": 16,
        "finite_scale_code_count": 255,
        "region_count": 1,
        "derivation_count": 3,
    }
    assert result["facts"]["resource_boundaries"] == [
        {
            "component_id": _COMPONENT_ID,
            "derived_resource_count": 3,
            "source_component_restored": True,
        }
    ]
    assert result["facts"]["source_coverage"] == [
        {
            "component_id": _COMPONENT_ID,
            "selected_weight_count": 3,
            "execution_path_count": 4,
        }
    ]


def test_proof_rejects_an_internally_consistent_but_incomplete_source_cover(
    tmp_path: Path,
    monkeypatch,
) -> None:
    products = _provider_products(tmp_path, monkeypatch)
    (
        _provider,
        context,
        _opportunity_value,
        _candidate,
        _representation,
        lowering,
        _build_plan,
    ) = products
    incomplete = deepcopy(lowering["regions"][0])
    incomplete["resident_derivations"] = incomplete["resident_derivations"][1:]

    with pytest.raises(
        ModelCompileError,
        match="does not cover every selected source weight",
    ):
        _verify_source_coverage(context.source_artifacts, incomplete)

    incomplete = deepcopy(lowering["regions"][0])
    incomplete["shader_replacements"] = incomplete["shader_replacements"][1:]
    with pytest.raises(
        ModelCompileError,
        match="does not cover every source execution path",
    ):
        _verify_source_coverage(context.source_artifacts, incomplete)


def test_proof_rejects_overlay_changes_outside_declared_shader_and_residency_edits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    products, workspace, root = _construct(tmp_path, monkeypatch)
    (
        _provider,
        context,
        opportunity,
        candidate,
        _representation,
        _lowering,
        _build_plan,
    ) = products
    overlay_path = root / component_overlay_path(_COMPONENT_ID)
    overlay = json.loads(overlay_path.read_text())
    overlay["component"]["circuit"]["nodes"][0]["attrs"]["hidden_size"] += 128
    overlay_path.write_text(json.dumps(overlay))

    proof_digest = staged_file_digest(root / PROOF_PATH)
    verifier = ExactResidentExpansionProofVerifier(
        source_artifacts=context.source_artifacts,
        candidate_workspace_root=workspace,
    )
    result = verifier.verify(
        ProofRequest(
            plan_id=stable_contract_id("validation_plan", candidate["candidate_id"]),
            candidate_id=candidate["candidate_id"],
            obligation=candidate["behavioral_contract"]["proof_obligations"][1],
            verifier_id=PROOF_VERIFIER_ID,
            source_contract_digests=opportunity.source_contract_digests,
            construction_record_digest=contract_digest({"construction": "fixture"}),
            reference_implementation={},
            candidate_implementation={
                "artifact_refs": [{"path": PROOF_PATH, "digest": proof_digest}]
            },
        )
    )
    assert result["status"] == "inconclusive"
    assert result["facts"] == {}
    assert any(
        "outside its declared boundary" in item for item in result["diagnostics"]
    )
