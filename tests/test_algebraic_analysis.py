from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.analysis.context import (
    ActivationTrace,
    AnalysisBudget,
    ScopeAnalysisContext,
)
from nerve.representation_optimizer.analysis.elementwise import (
    ElementwiseStructureAnalyzer,
)
from nerve.representation_optimizer.analysis.evidence import (
    build_analysis_run,
    build_evidence,
    validate_analysis_run_directory,
    write_analysis_run,
)
from nerve.representation_optimizer.analysis.graph import GraphStructureAnalyzer
from nerve.representation_optimizer.analysis.joint import JointParameterAnalyzer
from nerve.representation_optimizer.analysis.matrix import MatrixStructureAnalyzer
from nerve.representation_optimizer.analysis.procedural import (
    ProceduralStructureAnalyzer,
)
from nerve.representation_optimizer.analysis.tensor_repository import (
    InMemoryTensorRepository,
)
from nerve.representation_optimizer.analysis.trace import (
    ReachableActivationAnalyzer,
)
from nerve.representation_optimizer.contracts import (
    contract_digest,
    stable_contract_id,
)


def _context(
    tensors: dict[str, np.ndarray],
    *,
    roles: dict[str, str] | None = None,
    nodes: tuple[dict, ...] = (),
    budget: AnalysisBudget | None = None,
    trace: ActivationTrace | None = None,
) -> ScopeAnalysisContext:
    roles = roles or {}
    component_ids = ["component"]
    source_node_ids = [str(node["id"]) for node in nodes] or ["component/node"]
    scope_id = stable_contract_id(
        "scope",
        "fixture_package",
        "semantic_module",
        component_ids,
        ["component/module"],
        source_node_ids,
    )
    parameters = [
        {
            "id": f"parameter:component/{index}",
            "component_id": "component",
            "parameter_ref_id": f"parameter_{index}",
            "definition": {
                "tensor": name,
                "role": roles.get(name, "parameter"),
            },
        }
        for index, name in enumerate(tensors)
    ]
    digest = contract_digest({"scope_id": scope_id})
    scope = {
        "scope_id": scope_id,
        "boundary": {"parameters": parameters},
    }
    return ScopeAnalysisContext(
        package_id="fixture_package",
        scope=scope,
        source_contract={"contract_digest": digest},
        tensors=InMemoryTensorRepository(tensors),
        nodes=nodes,
        budget=budget
        or AnalysisBudget(
            relative_tolerance=0.0,
            decomposition_dimension_limit=128,
        ),
        activation_trace=trace,
    )


def _claim(
    claims: tuple[dict, ...],
    kind: str,
    *,
    tensor: str | None = None,
) -> dict:
    matches = [
        item
        for item in claims
        if item["kind"] == kind
        and (tensor is None or item["facts"].get("tensor") == tensor)
    ]
    assert len(matches) == 1
    return matches[0]


def test_elementwise_analysis_distinguishes_exact_sampled_and_negative_cases():
    tensors = {
        "zero": np.zeros((4, 4), dtype=np.float32),
        "constant": np.full((4, 4), 3.0, dtype=np.float32),
        "sparse": np.diag(np.ones(4, dtype=np.float32)),
        "dense": np.arange(16, dtype=np.float32).reshape(4, 4) + 1,
    }
    result = ElementwiseStructureAnalyzer().analyze(_context(tensors))
    zero = _claim(result.claims, "zero_parameter", tensor="zero")
    assert zero["status"] == "supported"
    assert zero["exact"] is True
    assert (
        _claim(
            result.claims,
            "constant_parameter",
            tensor="constant",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "sparse_parameter",
            tensor="sparse",
        )["status"]
        == "rejected"
    )
    assert (
        _claim(
            result.claims,
            "zero_parameter",
            tensor="dense",
        )["status"]
        == "rejected"
    )

    sampled = ElementwiseStructureAnalyzer().analyze(
        _context(
            {"mostly_zero": np.zeros((32, 32), dtype=np.float32)},
            budget=AnalysisBudget(
                exhaustive_element_limit=8,
                sampled_element_limit=16,
                relative_tolerance=0.0,
            ),
        )
    )
    sampled_zero = _claim(
        sampled.claims,
        "zero_parameter",
        tensor="mostly_zero",
    )
    assert sampled_zero["status"] == "inconclusive"
    assert sampled_zero["exact"] is False
    assert sampled_zero["facts"]["observation"]["mode"] == "deterministic_grid"


def test_numerical_tolerance_never_turns_approximation_into_exact_proof():
    almost_repeated = np.array(
        [[1.0, 2.0], [1.0001, 2.0001]],
        dtype=np.float32,
    )
    result = MatrixStructureAnalyzer().analyze(
        _context(
            {"almost": almost_repeated},
            budget=AnalysisBudget(
                absolute_tolerance=0.001,
                relative_tolerance=0.0,
                decomposition_dimension_limit=16,
            ),
        )
    )
    repeated = _claim(result.claims, "repeated_rows", tensor="almost")
    assert repeated["status"] == "supported"
    assert repeated["exact"] is False


def test_matrix_analyzer_finds_known_structures_and_rejects_controls():
    toeplitz = np.array(
        [
            [1, 2, 3, 4],
            [5, 1, 2, 3],
            [6, 5, 1, 2],
            [7, 6, 5, 1],
        ],
        dtype=np.float32,
    )
    circulant = np.stack(
        [np.roll(np.array([1, 2, 3, 4], dtype=np.float32), row) for row in range(4)]
    )
    block_diagonal = np.zeros((8, 8), dtype=np.float32)
    block_diagonal[:4, :4] = np.eye(4)
    block_diagonal[4:, 4:] = np.eye(4) * 2
    kronecker = np.kron(
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        np.array([[0, 5], [6, 7]], dtype=np.float32),
    )
    low_rank = np.outer(
        np.arange(1, 9, dtype=np.float32),
        np.arange(2, 10, dtype=np.float32),
    )
    butterfly = np.zeros((8, 8), dtype=np.float32)
    for row in range(8):
        butterfly[row, row] = 1
        butterfly[row, row ^ 1] = -1
    random_control = np.random.default_rng(7).normal(size=(7, 5)).astype(np.float32)
    convolution = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    nodes = (
        {
            "id": "component/convolution",
            "op": "convolution_1d",
            "inputs": ["component/input"],
            "outputs": ["component/output"],
            "params": ["convolution"],
        },
    )
    result = MatrixStructureAnalyzer().analyze(
        _context(
            {
                "toeplitz": toeplitz,
                "circulant": circulant,
                "block_diagonal": block_diagonal,
                "kronecker": kronecker,
                "low_rank": low_rank,
                "butterfly": butterfly,
                "random": random_control,
                "convolution": convolution,
            },
            nodes=nodes,
        )
    )
    assert (
        _claim(
            result.claims,
            "toeplitz_structure",
            tensor="toeplitz",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "circulant_structure",
            tensor="circulant",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "block_diagonal_structure",
            tensor="block_diagonal",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "kronecker_tensor_product",
            tensor="kronecker",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "low_rank",
            tensor="low_rank",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "butterfly_structure",
            tensor="butterfly",
        )["status"]
        == "supported"
    )
    assert (
        _claim(
            result.claims,
            "toeplitz_structure",
            tensor="random",
        )["status"]
        == "rejected"
    )
    assert (
        _claim(
            result.claims,
            "circulant_structure",
            tensor="random",
        )["status"]
        == "rejected"
    )
    assert (
        _claim(
            result.claims,
            "convolutional_structure",
            tensor="convolution",
        )["status"]
        == "supported"
    )


def test_joint_analysis_canonicalizes_permutations_and_finds_shared_structure():
    source = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.float32,
    )
    permuted = source[[2, 0, 1]]
    affine = source * 2 + 3
    unrelated = np.array(
        [[3, 1, 8], [2, 9, 4], [7, 6, 5]],
        dtype=np.float32,
    )
    repeated_experts = np.stack((source, source, unrelated))
    result = JointParameterAnalyzer().analyze(
        _context(
            {
                "source": source,
                "permuted": permuted,
                "affine": affine,
                "unrelated": unrelated,
                "experts": repeated_experts,
            },
            roles={
                "source": "projection",
                "permuted": "projection",
                "experts": "expert_bank",
            },
        )
    )
    coordinate = [
        item
        for item in result.claims
        if item["kind"] == "coordinate_equivalence"
        and item["facts"]["left_tensor"] == "source"
        and item["facts"]["right_tensor"] == "permuted"
    ]
    assert len(coordinate) == 1
    assert coordinate[0]["status"] == "supported"
    assert coordinate[0]["exact"] is True
    generators = [
        item
        for item in result.claims
        if item["kind"] == "shared_parameter_generator"
        and item["facts"]["left_tensor"] == "source"
        and item["facts"]["right_tensor"] == "affine"
    ]
    assert generators[0]["status"] == "supported"
    assert generators[0]["facts"]["affine_generator"]["scale"] == pytest.approx(2)
    unrelated_claim = [
        item
        for item in result.claims
        if item["kind"] == "coordinate_equivalence"
        and item["facts"]["left_tensor"] == "source"
        and item["facts"]["right_tensor"] == "unrelated"
    ][0]
    assert unrelated_claim["status"] == "rejected"
    assert (
        _claim(
            result.claims,
            "repeated_experts",
            tensor="experts",
        )["status"]
        == "supported"
    )


def test_joint_analysis_proves_direct_linear_coordinate_symmetry():
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.arange(15, dtype=np.float32).reshape(5, 3)
    nodes = (
        {
            "id": "component/first",
            "component_id": "component",
            "op": "linear",
            "inputs": ["component/input"],
            "outputs": ["component/intermediate"],
            "params": ["parameter_0"],
        },
        {
            "id": "component/second",
            "component_id": "component",
            "op": "linear",
            "inputs": ["component/intermediate"],
            "outputs": ["component/output"],
            "params": ["parameter_1"],
        },
    )
    result = JointParameterAnalyzer().analyze(
        _context({"first": first, "second": second}, nodes=nodes)
    )
    symmetry = _claim(
        result.claims,
        "coupled_coordinate_permutation_symmetry",
    )
    assert symmetry["status"] == "supported"
    assert symmetry["exact"] is True
    assert symmetry["facts"]["symmetry"]["coordinate_width"] == 3


def test_graph_analysis_uses_semantics_and_connectivity_not_model_names():
    nodes = (
        {
            "id": "component/router",
            "op": "topk_select",
            "inputs": ["component/input"],
            "outputs": ["component/routes"],
            "params": [],
        },
        {
            "id": "component/dispatch",
            "op": "expert_dispatch",
            "inputs": ["component/routes"],
            "outputs": ["component/output"],
            "params": [],
        },
        {
            "id": "component/isolated",
            "op": "identity",
            "inputs": ["component/external"],
            "outputs": ["component/other"],
            "params": [],
        },
    )
    result = GraphStructureAnalyzer().analyze(_context({}, nodes=nodes))
    assert _claim(result.claims, "graph_communities")["status"] == "supported"
    routing = _claim(result.claims, "routing_structure")
    assert routing["status"] == "supported"
    assert {record["node_id"] for record in routing["facts"]["routing_nodes"]} == {
        "component/router",
        "component/dispatch",
    }


def test_activation_evidence_always_records_domain_and_remains_non_exhaustive():
    trace = ActivationTrace(
        domain={
            "prompts": ["synthetic prompt"],
            "token_positions": [0, 1, 2],
            "seed": 17,
        },
        signals={"component/output": np.eye(3, dtype=np.float32)},
        trace_digest="trace_fixture_digest",
    )
    result = ReachableActivationAnalyzer().analyze(_context({}, trace=trace))
    evidence = _claim(result.claims, "reachable_activation_refinement")
    assert evidence["status"] == "supported"
    assert evidence["exact"] is False
    assert evidence["facts"]["trace_domain"] == trace.domain
    assert evidence["facts"]["sampled_behavior_is_exhaustive"] is False


def test_procedural_analysis_detects_generators_and_negative_control():
    result = ProceduralStructureAnalyzer().analyze(
        _context(
            {
                "periodic": np.tile(
                    np.array([1, 4, 2], dtype=np.float32),
                    8,
                ),
                "polynomial": np.arange(20, dtype=np.float32) ** 2,
                "random": np.random.default_rng(13).normal(size=31).astype(np.float32),
            }
        )
    )
    periodic = _claim(
        result.claims,
        "periodic_parameter_generator",
        tensor="periodic",
    )
    assert periodic["status"] == "supported"
    assert periodic["exact"] is True
    polynomial = _claim(
        result.claims,
        "polynomial_parameter_generator",
        tensor="polynomial",
    )
    assert polynomial["status"] == "supported"
    assert polynomial["facts"]["degree"] == 2
    assert (
        _claim(
            result.claims,
            "procedural_predictability",
            tensor="random",
        )["status"]
        == "rejected"
    )


def test_evidence_run_is_deterministic_atomic_and_integrity_checked(tmp_path: Path):
    scope_id = stable_contract_id(
        "scope",
        "package",
        "operator",
        ["component"],
        [],
        ["component/node"],
    )
    source_digest = contract_digest({"source": "exact"})
    evidence, details = build_evidence(
        scope_id=scope_id,
        source_contract_digest=source_digest,
        analyzer_id="fixture",
        analyzer_version="1",
        claims=(
            {
                "kind": "known_fact",
                "status": "supported",
                "exact": True,
                "facts": {"value": 3},
            },
        ),
        details={"proof": "synthetic"},
    )
    run = build_analysis_run(
        package_id="package",
        scope_id=scope_id,
        source_contract_digest=source_digest,
        budget=AnalysisBudget().to_json(),
        evidence=(evidence,),
        details=(details,),
    )
    duplicate = build_analysis_run(
        package_id="package",
        scope_id=scope_id,
        source_contract_digest=source_digest,
        budget=AnalysisBudget().to_json(),
        evidence=(deepcopy(evidence),),
        details=(deepcopy(details),),
    )
    assert duplicate.run_id == run.run_id

    output = tmp_path / "analysis"
    write_analysis_run(run, output)
    loaded = validate_analysis_run_directory(output)
    assert loaded.run_id == run.run_id
    with pytest.raises(ModelCompileError, match="refusing to mutate"):
        write_analysis_run(run, output)

    detail_path = output / "details" / "fixture.json"
    detail_path.write_text('{"mutated":true}\n')
    with pytest.raises(ModelCompileError, match="details digest"):
        validate_analysis_run_directory(output)
