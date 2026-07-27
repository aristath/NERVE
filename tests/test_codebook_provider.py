from __future__ import annotations

import json
import struct
from hashlib import sha256
from pathlib import Path

from nerve.representation_optimizer.analysis.evidence import build_evidence
from nerve.representation_optimizer.contracts import (
    OPTIMIZATION_SCOPE_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    source_behavior_contract_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.descriptor_registry import (
    load_builtin_representation_descriptors,
)
from nerve.representation_optimizer.providers import (
    PackageSourceArtifactResolver,
    ProviderProblem,
    ProviderRegistry,
)
from nerve.representation_optimizer.providers.builtin import (
    BuiltinCandidateToolchainResolver,
    load_builtin_provider_registry,
)
from nerve.representation_optimizer.providers.codebook import (
    ExactCodebookProofVerifier,
    ExactHeadNormCodebookProvider,
)
from nerve.representation_optimizer.providers.codebook.toolchain import (
    CodebookToolchainResolver,
)
from nerve.representation_optimizer.staging.artifact_validation import (
    ArtifactValidatorRegistry,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_file_digest,
)
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)
from nerve.representation_optimizer.validation.protocols import ProofRequest
from tests.test_representation_optimizer_contracts import (
    hardware_profile_contract,
)


def _write_safetensors(
    package: Path,
    *,
    tensor_name: str,
    filename: str,
    values: tuple[int, ...],
) -> tuple[dict, dict]:
    payload = b"".join(value.to_bytes(2, "little") for value in values)
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "BF16",
                "shape": [len(values)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    relative = f"weights/{filename}"
    path = package / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return (
        {
            "path": relative,
            "safetensors_header_bytes": len(header),
            "metadata": {"format": "nerve", "layout": "row_major"},
        },
        {
            "dtype": "BF16",
            "shape": [len(values)],
            "data_offsets": [0, len(payload)],
            "parameter_count": len(values),
            "byte_count": len(payload),
            "source_file": relative,
            "data_sha256": sha256(payload).hexdigest(),
            "layout": "row_major",
        },
    )


def _provider_problem(
    tmp_path: Path,
    *,
    values_a: tuple[int, ...] = (1, 2, 1, 2),
    values_b: tuple[int, ...] = (2, 3, 2, 3),
    exhaustive: bool = True,
    fused: bool = True,
) -> ProviderProblem:
    package = tmp_path / "package"
    package.mkdir()
    tensor_names = ("arbitrary.first", "unrelated.second")
    source_files = []
    tensors = {}
    for index, (tensor_name, values) in enumerate(
        zip(tensor_names, (values_a, values_b), strict=True)
    ):
        source, tensor = _write_safetensors(
            package,
            tensor_name=tensor_name,
            filename=f"tensor_{index}.safetensors",
            values=values,
        )
        source_files.append(source)
        tensors[tensor_name] = tensor
    (package / "tensors.json").write_text(
        json.dumps(
            {
                "schema": "nerve.tensor_index.v1",
                "source": {"weights_files": source_files},
                "tensors": tensors,
            }
        )
    )

    component_id = "arbitrary_component"
    head_width = len(values_a)
    branch_attrs = (
        {
            "eps": 1e-6,
            "weight_offset": 1.0,
            "head_width": head_width,
            "head_count": 3,
        },
        {
            "eps": 1e-6,
            "weight_offset": 1.0,
            "head_width": head_width,
            "head_count": 1,
        },
    )
    source_nodes = (
        {
            "id": "first_norm",
            "op": "rms_norm_per_head",
            "inputs": ["first_input"],
            "outputs": ["first_normalized"],
            "params": ["first_weight"],
            "attrs": branch_attrs[0],
        },
        {
            "id": "second_norm",
            "op": "rms_norm_per_head",
            "inputs": ["second_input"],
            "outputs": ["second_normalized"],
            "params": ["second_weight"],
            "attrs": branch_attrs[1],
        },
    )
    circuit = {
        "schema": "nerve.stream_circuit.v1",
        "id": "arbitrary_source",
        "source": {"component_id": component_id},
        "nodes": list(source_nodes),
    }
    circuit_path = package / "lowered" / component_id / "circuit.json"
    circuit_path.parent.mkdir(parents=True)
    circuit_path.write_text(json.dumps(circuit))
    execution_ref = "lowered/execution_graph.circuits.json"
    (package / execution_ref).write_text(json.dumps({"component": component_id}))
    physical_nodes = (
        [
            {
                "id": "fused_norm_rope",
                "op": "parallel_head_norm_rope_2way",
                "inputs": ["first_input", "second_input"],
                "outputs": ["first_output", "second_output"],
                "params": ["first_weight", "second_weight"],
                "attrs": {
                    "compiled_from": [
                        "first_norm",
                        "first_rope",
                        "second_norm",
                        "second_rope",
                    ],
                    "branches": [
                        {
                            "norm": branch_attrs[0],
                            "rope": {
                                "head_width": head_width,
                                "rotary_width": head_width,
                                "theta": 10_000.0,
                                "rope_type": "default",
                                "interleaved": False,
                            },
                        },
                        {
                            "norm": branch_attrs[1],
                            "rope": {
                                "head_width": head_width,
                                "rotary_width": head_width,
                                "theta": 10_000.0,
                                "rope_type": "default",
                                "interleaved": False,
                            },
                        },
                    ],
                    "intermediate_rounding": "BF16",
                },
            }
        ]
        if fused
        else list(source_nodes)
    )
    source_shader = "shaders/source_head_norm_rope.spv"
    source_prefill_shader = "shaders/source_head_norm_rope_temporal.spv"
    for shader in (source_shader, source_prefill_shader):
        path = package / shader
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(struct.pack("<5I", 0x07230203, 0x00010600, 0, 1, 0))
    (package / "vulkan_resident_package.json").write_text(
        json.dumps(
            {
                "package_id": "fixture_package",
                "max_context_activations": 131_072,
                "circuit_graph": {
                    "components": [
                        {
                            "component_id": component_id,
                            "operator_type": "arbitrary_attention",
                            "runtime_role": "signal_processor",
                            "implementation": "source_implementation",
                            "behavioral_role": "source_reference_circuit",
                            "circuit": {
                                "schema": "nerve.stream_circuit.v1",
                                "id": "arbitrary_physical_circuit",
                                "source": {
                                    "component_id": component_id,
                                    "source_layer_index": None,
                                    "source_operator_type": "arbitrary_attention",
                                },
                                "runtime_role": "signal_processor",
                                "behavioral_role": "source_reference_circuit",
                                "implementation": "source_implementation",
                                "boundary": {
                                    "inputs": [
                                        {
                                            "id": "first_input",
                                            "signal": "first_input",
                                            "shape": [head_width * 3],
                                            "component_port": "first",
                                        },
                                        {
                                            "id": "second_input",
                                            "signal": "second_input",
                                            "shape": [head_width],
                                            "component_port": "second",
                                        },
                                    ],
                                    "outputs": [
                                        {
                                            "id": "first_output",
                                            "signal": "first_output",
                                            "shape": [head_width * 3],
                                            "source": "first_output",
                                            "component_port": "first",
                                        },
                                        {
                                            "id": "second_output",
                                            "signal": "second_output",
                                            "shape": [head_width],
                                            "source": "second_output",
                                            "component_port": "second",
                                        },
                                    ],
                                    "controls": [],
                                },
                                "state_ports": [],
                                "parameters": {
                                    "layout": "fixture",
                                    "storage": "source_tensor_refs",
                                    "refs": {
                                        "first_weight": {
                                            "tensor": tensor_names[0],
                                            "role": "first_normalization_weight",
                                        },
                                        "second_weight": {
                                            "tensor": tensor_names[1],
                                            "role": "second_normalization_weight",
                                        },
                                    },
                                },
                                "nodes": physical_nodes,
                            },
                            "params": {
                                "schema": "nerve.circuit_params.v1",
                                "circuit": "arbitrary_physical_circuit",
                                "layout": "fixture",
                                "storage": "source_tensor_refs",
                                "refs": {
                                    "first_weight": {
                                        "tensor": tensor_names[0],
                                        "role": "first_normalization_weight",
                                    },
                                    "second_weight": {
                                        "tensor": tensor_names[1],
                                        "role": "second_normalization_weight",
                                    },
                                },
                            },
                            "state": {
                                "schema": "nerve.circuit_state.v1",
                                "circuit": "arbitrary_physical_circuit",
                                "state_ports": [],
                            },
                        }
                    ]
                },
                "component_executions": [
                    {
                        "component_id": component_id,
                        "operator_type": "arbitrary_attention",
                        "implementation": "source_implementation",
                        "kernels": [
                            {
                                "execution_index": 0,
                                "node_id": "fused_norm_rope",
                                "op": "parallel_head_norm_rope_2way",
                                "source_node_ids": [
                                    "first_norm",
                                    "first_rope",
                                    "second_norm",
                                    "second_rope",
                                ],
                                "semantic_module_ids": [
                                    "arbitrary_head_normalization",
                                    "arbitrary_position",
                                ],
                                "execution_domain": "decode",
                                "shader_path": source_shader,
                                "local_size_x": 64,
                                "workgroup_count_x": 4,
                                "batch_mode": "causal_scan",
                                "batch_implementations": [
                                    {
                                        "execution_domain": "prefill",
                                        "lane_tile_width": 64,
                                        "independent_candidate_compatible": False,
                                        "causal_sequence_compatible": True,
                                        "device_requirements": {
                                            "vulkan_device_extensions": [],
                                            "vulkan_features": [],
                                            "subgroup_operations": [
                                                "arithmetic",
                                                "basic",
                                            ],
                                        },
                                        "stages": [
                                            {
                                                "shader_path": source_prefill_shader,
                                                "local_size_x": 64,
                                                "workgroup_count_x": 4,
                                                "control": {
                                                    "kind": "storage_buffer",
                                                    "byte_count": 16,
                                                    "binding": 6,
                                                    "payload": "temporal",
                                                },
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )

    package_id = "fixture_package"
    qualified_nodes = [f"{component_id}/{node['id']}" for node in source_nodes]
    scope_id = stable_contract_id(
        "scope",
        package_id,
        "semantic_module",
        [component_id],
        ["normalization"],
        qualified_nodes,
    )
    parameters = [
        {
            "id": f"parameter:{component_id}/{parameter_id}",
            "component_id": component_id,
            "parameter_ref_id": parameter_id,
            "definition": {
                "tensor": tensor_name,
                "role": "normalization_weight",
            },
        }
        for parameter_id, tensor_name in zip(
            ("first_weight", "second_weight"),
            tensor_names,
            strict=True,
        )
    ]
    contract = {
        "schema": SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
        "scope_id": scope_id,
        "semantic_role": "normalization",
        "interface": {
            "inputs": [],
            "outputs": [],
            "parameters": parameters,
            "states": [],
            "controls": [],
            "randomness": [],
            "dependencies": [],
        },
        "exact_reference": {
            "implementation_id": "exact_reference",
            "artifact_refs": [
                execution_ref,
                f"lowered/{component_id}/circuit.json",
            ],
        },
        "contract_digest": "",
    }
    contract["contract_digest"] = source_behavior_contract_digest(contract)
    scope = {
        "schema": OPTIMIZATION_SCOPE_SCHEMA,
        "scope_id": scope_id,
        "package_id": package_id,
        "kind": "semantic_module",
        "members": {
            "component_ids": [component_id],
            "semantic_module_ids": ["normalization"],
            "source_node_ids": qualified_nodes,
        },
        "boundary": {
            "inputs": [],
            "outputs": [],
            "parameters": parameters,
            "states": [],
            "controls": [],
            "randomness": [],
            "dependencies": [],
        },
        "source_contract_digest": contract["contract_digest"],
    }
    operator_claim = {
        "kind": "operator_structure",
        "status": "supported",
        "exact": True,
        "facts": {
            "node_count": 2,
            "operators": [
                {
                    "node_id": qualified_nodes[index],
                    "component_id": component_id,
                    **node,
                }
                for index, node in enumerate(source_nodes)
            ],
        },
    }
    structure, _ = build_evidence(
        scope_id=scope_id,
        source_contract_digest=contract["contract_digest"],
        analyzer_id="semantic_graph_structure",
        analyzer_version="2",
        claims=(operator_claim,),
        details={},
    )
    codebook_claims = []
    for tensor_name, values in zip(tensor_names, (values_a, values_b), strict=True):
        codebook_claims.append(
            {
                "kind": "low_entropy_codebook",
                "status": "supported",
                "exact": exhaustive,
                "facts": {
                    "tensor": tensor_name,
                    "observation": {
                        "mode": ("exhaustive" if exhaustive else "deterministic_grid"),
                        "storage_dtype": "BF16",
                    },
                },
            }
        )
    codebook, _ = build_evidence(
        scope_id=scope_id,
        source_contract_digest=contract["contract_digest"],
        analyzer_id="elementwise_structure",
        analyzer_version="1",
        claims=tuple(codebook_claims),
        details={},
    )
    return ProviderProblem.from_documents(
        package_id=package_id,
        scopes=(scope,),
        source_contracts=(contract,),
        evidence=(structure, codebook),
        hardware_profile=hardware_profile_contract(),
        source_artifacts=PackageSourceArtifactResolver(package),
    )


def _run(problem: ProviderProblem):
    return ProviderRegistry.from_providers(
        descriptors=load_builtin_representation_descriptors(),
        providers=(ExactHeadNormCodebookProvider(),),
    ).run(problem)


def _construct_codebook_candidate(tmp_path: Path, **problem_options):
    report = _run(_provider_problem(tmp_path, **problem_options))
    plan = report.candidates[0]
    workspace = tmp_path / "workspace"
    root = workspace / "ready" / plan.candidate_id
    root.mkdir(parents=True)
    context = CandidateConstructionContext(
        package_dir=tmp_path / "package",
        staging_dir=root,
        candidate=plan.candidate.to_json(),
        representation_graph=plan.representation_ir.to_json(),
        target_lowering=plan.target_lowering,
        build_plan=plan.construction_requirements,
        started_ns=__import__("time").monotonic_ns(),
        cancel_requested=None,
    )
    toolchain = CodebookToolchainResolver().resolve(plan)
    phases = (
        (
            "semantic_construction",
            toolchain.semantic_constructor.construct_semantic_artifacts,
        ),
        (
            "ordinary_lowering",
            toolchain.ordinary_relowerer.run_ordinary_lowering,
        ),
        (
            "physical_optimization",
            toolchain.physical_optimizer.optimize_physical_artifacts,
        ),
    )
    for phase, service in phases:
        context.begin_phase(phase)
        service(context)
        context.end_phase()
    context.validate_complete()
    contracts = root / "contracts"
    contracts.mkdir()
    (contracts / "target_lowering.json").write_text(json.dumps(plan.target_lowering))
    return plan, workspace, root


def test_exact_codebook_provider_is_structure_generic_and_emits_complete_plan(
    tmp_path: Path,
):
    report = _run(_provider_problem(tmp_path))
    assert report.evaluations[0].status == "completed"
    assert len(report.candidates) == 1
    plan = report.candidates[0]
    candidate = plan.candidate.to_json()
    assert candidate["representation"]["kind"] == "exact_u8_addressed_bf16_codebook"
    assert candidate["representation"]["parameter_format"]["entry_count"] == 3
    assert candidate["representation"]["topology"]["component_id"] == (
        "arbitrary_component"
    )
    assert plan.static_estimate.permanent_bytes == 16
    assert plan.static_estimate.construction_nanoseconds is None
    assert all(
        value is None
        for value in plan.construction_requirements.to_json()[
            "resource_limits"
        ].values()
    )
    assert plan.static_estimate.steady_state_work["dispatch_count_change"] == 0
    assert len(plan.benchmark_workloads) == 2
    validation_checks = plan.validation_requirements.to_json()["checks"]
    assert len(validation_checks) == 9
    component_checks = [
        check
        for check in validation_checks
        if check["regime"]["execution_scope"] == "component"
    ]
    assert len(component_checks) == 4
    assert {check["controls"]["phase"] for check in component_checks} == {
        "decode",
        "prefill",
    }
    assert all(
        check["controls"]["component_id"] == "arbitrary_component"
        and check["controls"]["physical_node_id"] == "fused_norm_rope"
        for check in component_checks
    )
    assert plan.mount_requirements.to_json()["component_replacements"] == [
        {
            "source_component_id": "arbitrary_component",
            "overlay_ref": "overlays/component.json",
        }
    ]


def test_codebook_provider_and_toolchain_are_available_from_builtin_registries(
    tmp_path: Path,
) -> None:
    report = load_builtin_provider_registry().run(_provider_problem(tmp_path))
    assert len(report.candidates) == 1
    toolchain = BuiltinCandidateToolchainResolver().resolve(report.candidates[0])
    assert toolchain.semantic_constructor is not None
    assert toolchain.ordinary_relowerer is not None
    assert toolchain.physical_optimizer is not None


def test_exact_codebook_toolchain_constructs_write_once_complete_candidate(
    tmp_path: Path,
) -> None:
    plan, _workspace, root = _construct_codebook_candidate(tmp_path)
    results = ArtifactValidatorRegistry.with_builtin_validators().validate_artifacts(
        root,
        plan.construction_requirements,
    )

    assert set(results) == {
        output["path"] for output in plan.construction_requirements.outputs
    }
    overlay = json.loads((root / "overlays/component.json").read_text())
    node = overlay["component"]["circuit"]["nodes"][0]
    assert node["op"] == "parallel_head_norm_rope_2way_codebook_u8"
    assert len(node["params"]) == 3
    assert overlay["execution"]["kernels"][0]["op"] == node["op"]
    assert (
        overlay["component"]["params"]["refs"]
        == overlay["component"]["circuit"]["parameters"]["refs"]
    )
    assert (
        overlay["execution"]["kernels"][0]["batch_implementations"][0]["stages"][0][
            "control"
        ]["binding"]
        == 7
    )
    proof = json.loads((root / "proofs/codebook_equivalence.json").read_text())
    assert set(proof["obligations"].values()) == {"proven"}
    assert all(
        source["data_sha256"] == source["reconstructed_sha256"]
        for source in proof["source_tensors"]
    )
    tensor_fragment = json.loads((root / "parameters/tensors.json").read_text())
    codebook = tensor_fragment["tensors"][
        plan.target_lowering["parameters"]["codebook_tensor_name"]
    ]
    assert codebook["byte_count"] == 8
    assert codebook["byte_count"] % 4 == 0


def test_exact_codebook_toolchain_word_aligns_short_address_streams(
    tmp_path: Path,
) -> None:
    plan, _workspace, root = _construct_codebook_candidate(
        tmp_path,
        values_a=(1, 2),
        values_b=(2, 3),
    )
    tensor_fragment = json.loads((root / "parameters/tensors.json").read_text())
    for name, path in zip(
        plan.target_lowering["parameters"]["branch_index_tensor_names"],
        (
            root / "parameters/branch_0_indices.safetensors",
            root / "parameters/branch_1_indices.safetensors",
        ),
        strict=True,
    ):
        metadata = tensor_fragment["tensors"][name]
        assert metadata["shape"] == [4]
        assert metadata["byte_count"] == 4
        payload = path.read_bytes()
        header_bytes = int.from_bytes(payload[:8], "little")
        assert payload[8 + header_bytes :] in (b"\x00\x01\x00\x00", b"\x01\x02\x00\x00")


def test_exact_codebook_proof_verifier_rederives_every_obligation(
    tmp_path: Path,
) -> None:
    plan, workspace, root = _construct_codebook_candidate(tmp_path)
    proof_digest = staged_file_digest(root / "proofs/codebook_equivalence.json")
    verifier = ExactCodebookProofVerifier(
        package_root=tmp_path / "package",
        candidate_workspace_root=workspace,
    )
    common = {
        "plan_id": stable_contract_id(
            "validation_plan",
            plan.candidate_id,
        ),
        "candidate_id": plan.candidate_id,
        "verifier_id": verifier.verifier_id,
        "source_contract_digests": tuple(
            plan.candidate.to_json()["source_contract_digests"]
        ),
        "construction_record_digest": (
            "nerve.optimizer.canonical_json_sha256.v1:" + "0" * 64
        ),
        "reference_implementation": {
            "implementation_id": "source",
            "contract_digest": plan.candidate.to_json()["source_contract_digests"][0],
            "artifact_refs": [],
        },
        "candidate_implementation": {
            "implementation_id": f"staged:{plan.candidate_id}",
            "contract_digest": ("nerve.optimizer.canonical_json_sha256.v1:" + "1" * 64),
            "artifact_refs": [
                {
                    "path": "proofs/codebook_equivalence.json",
                    "digest": proof_digest,
                }
            ],
        },
    }
    for obligation in (
        "codebook_reconstructs_source_bf16_bits",
        "fused_operator_preserves_source_rounding",
    ):
        result = verifier.verify(ProofRequest(obligation=obligation, **common))
        assert result["status"] == "proven", result["diagnostics"]
        assert result["facts"]["exact_bf16_reconstruction"]["tensor_count"] == 2
        artifact = result["artifacts"][0]
        assert (
            b"".join(
                verifier.iter_proof_artifact(
                    artifact["path"],
                    chunk_bytes=7,
                )
            )
            == (root / "proofs/codebook_equivalence.json").read_bytes()
        )

    index_path = root / "parameters/branch_0_indices.safetensors"
    original_index = index_path.read_bytes()
    corrupted = bytearray(original_index)
    corrupted[-1] ^= 0x01
    index_path.write_bytes(corrupted)
    result = verifier.verify(
        ProofRequest(
            obligation="codebook_reconstructs_source_bf16_bits",
            **common,
        )
    )
    assert result["status"] == "inconclusive"
    assert "address storage" in result["diagnostics"][0]

    index_path.write_bytes(original_index)
    shader_path = root / "kernels/head_norm_rope_codebook_u8.spv"
    corrupted = bytearray(shader_path.read_bytes())
    corrupted[-1] ^= 0x01
    shader_path.write_bytes(corrupted)
    result = verifier.verify(
        ProofRequest(
            obligation="fused_operator_preserves_source_rounding",
            **common,
        )
    )
    assert result["status"] == "inconclusive"
    assert "SPIR-V" in result["diagnostics"][0]


def test_exact_codebook_provider_declines_non_exhaustive_evidence(tmp_path: Path):
    report = _run(_provider_problem(tmp_path, exhaustive=False))
    assert report.evaluations[0].status == "declined"
    assert report.candidates == ()
    assert "exhaustive" in report.evaluations[0].structural_match.reasons[0]


def test_exact_codebook_provider_declines_when_physical_fusion_is_absent(
    tmp_path: Path,
):
    report = _run(_provider_problem(tmp_path, fused=False))
    assert report.evaluations[0].status == "declined"
    assert report.candidates == ()
    assert "does not retain" in report.evaluations[0].structural_match.reasons[0]


def test_exact_codebook_provider_rejects_union_larger_than_u8(tmp_path: Path):
    first = tuple(range(256)) * 2
    second = tuple(range(256, 512)) * 2
    report = _run(
        _provider_problem(
            tmp_path,
            values_a=first,
            values_b=second,
        )
    )
    assert report.evaluations[0].status == "declined"
    assert "does not fit" in report.evaluations[0].structural_match.reasons[0]
