from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nerve.compilation import Json, ModelCompileError, read_json
from nerve.representation_optimizer.analysis.claims import StructuralAnalyzer
from nerve.representation_optimizer.analysis.context import (
    ActivationTrace,
    AnalysisBudget,
    ScopeAnalysisContext,
)
from nerve.representation_optimizer.analysis.elementwise import (
    ElementwiseStructureAnalyzer,
)
from nerve.representation_optimizer.analysis.evidence import (
    AnalysisRun,
    build_analysis_run,
    build_evidence,
    write_analysis_run,
)
from nerve.representation_optimizer.analysis.graph import GraphStructureAnalyzer
from nerve.representation_optimizer.analysis.joint import JointParameterAnalyzer
from nerve.representation_optimizer.analysis.matrix import MatrixStructureAnalyzer
from nerve.representation_optimizer.analysis.procedural import (
    ProceduralStructureAnalyzer,
)
from nerve.representation_optimizer.analysis.tensor_repository import (
    PackageTensorRepository,
    TensorRepository,
)
from nerve.representation_optimizer.analysis.trace import (
    ReachableActivationAnalyzer,
)
from nerve.representation_optimizer.scope_enumeration.catalog import (
    load_optimization_scope_catalog,
)
from nerve.representation_optimizer.scope_enumeration.graph import (
    SemanticDependencyGraph,
)


def builtin_analyzers() -> tuple[StructuralAnalyzer, ...]:
    return (
        ElementwiseStructureAnalyzer(),
        MatrixStructureAnalyzer(),
        JointParameterAnalyzer(),
        GraphStructureAnalyzer(),
        ProceduralStructureAnalyzer(),
        ReachableActivationAnalyzer(),
    )


def analyze_scope(
    *,
    package_dir: Path,
    scope_id: str,
    budget: AnalysisBudget | None = None,
    activation_trace: ActivationTrace | None = None,
    analyzers: Iterable[StructuralAnalyzer] | None = None,
    tensors: TensorRepository | None = None,
    output_dir: Path | None = None,
) -> AnalysisRun:
    budget = budget or AnalysisBudget()
    catalog = load_optimization_scope_catalog(
        package_dir / "optimization" / "scopes.json"
    )
    scope = _unique_record(catalog["scopes"], "scope_id", scope_id)
    source_contract = _unique_record(
        catalog["source_contracts"],
        "scope_id",
        scope_id,
    )
    graph = _load_semantic_graph(package_dir)
    context = ScopeAnalysisContext(
        package_id=str(catalog["package_id"]),
        scope=scope,
        source_contract=source_contract,
        tensors=tensors or PackageTensorRepository(package_dir),
        nodes=_scope_nodes(graph, scope),
        budget=budget,
        activation_trace=activation_trace,
    )
    selected = tuple(analyzers or builtin_analyzers())
    identities = [(analyzer.analyzer_id, analyzer.version) for analyzer in selected]
    if len(identities) != len(set(identities)):
        raise ModelCompileError("analysis run contains duplicate analyzer identities")
    evidence = []
    details = []
    for analyzer in selected:
        result = analyzer.analyze(context)
        evidence_document, details_document = build_evidence(
            scope_id=scope_id,
            source_contract_digest=context.source_contract_digest,
            analyzer_id=analyzer.analyzer_id,
            analyzer_version=analyzer.version,
            claims=result.claims,
            details=result.details,
        )
        evidence.append(evidence_document)
        details.append(details_document)
    run = build_analysis_run(
        package_id=context.package_id,
        scope_id=scope_id,
        source_contract_digest=context.source_contract_digest,
        budget=budget.to_json(),
        evidence=tuple(evidence),
        details=tuple(details),
    )
    if output_dir is not None:
        write_analysis_run(run, output_dir)
    return run


def _load_semantic_graph(package_dir: Path) -> SemanticDependencyGraph:
    stage = read_json(package_dir / "optimization" / "stage.json")
    baseline = stage.get("exact_baseline")
    artifact_ref = baseline.get("artifact_ref") if isinstance(baseline, dict) else None
    if not isinstance(artifact_ref, str) or not artifact_ref:
        raise ModelCompileError(
            "compiled package has no optimizer exact-baseline reference"
        )
    relative = Path(artifact_ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCompileError("optimizer exact baseline escapes package")
    lowered = read_json(package_dir / relative)
    return SemanticDependencyGraph.from_lowered_package(
        package_dir=package_dir,
        lowered_index=lowered,
        lowered_index_ref=artifact_ref,
    )


def _scope_nodes(
    graph: SemanticDependencyGraph,
    scope: Json,
) -> tuple[Json, ...]:
    selected = set(str(value) for value in scope["members"]["source_node_ids"])
    result = []
    for component in graph.components:
        for raw_node in component.nodes:
            local_id = str(raw_node["id"])
            qualified_id = f"{component.component_id}/{local_id}"
            if qualified_id not in selected:
                continue
            node = dict(raw_node)
            node["id"] = qualified_id
            node["component_id"] = component.component_id
            node["local_id"] = local_id
            node["inputs"] = [
                f"{component.component_id}/{signal}"
                for signal in raw_node.get("inputs", [])
            ]
            node["outputs"] = [
                f"{component.component_id}/{signal}"
                for signal in raw_node.get("outputs", [])
            ]
            result.append(node)
    found = {str(node["id"]) for node in result}
    missing = sorted(selected - found)
    if missing:
        raise ModelCompileError(
            f"analysis scope references missing source nodes: {missing}"
        )
    return tuple(sorted(result, key=lambda node: str(node["id"])))


def _unique_record(records: object, field: str, value: str) -> Json:
    if not isinstance(records, list):
        raise ModelCompileError("compiled optimizer catalog is malformed")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get(field) == value
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            f"expected one optimizer record with {field}={value!r}, "
            f"found {len(matches)}"
        )
    return matches[0]
