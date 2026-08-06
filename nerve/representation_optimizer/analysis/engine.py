from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import (
    Json,
    ModelCompileError,
    check_compile_cancelled,
    read_json,
)
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
from nerve.representation_optimizer.analysis.memo import AnalysisComputationMemo
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
    cancel_requested: Callable[[], bool] | None = None,
) -> AnalysisRun:
    engine = ScopeAnalysisEngine.from_package(
        package_dir,
        analyzers=analyzers,
        tensors=tensors,
        cancel_requested=cancel_requested,
    )
    return engine.analyze_scope(
        scope_id=scope_id,
        budget=budget,
        activation_trace=activation_trace,
        output_dir=output_dir,
        cancel_requested=cancel_requested,
    )


class ScopeAnalysisEngine:
    """One package-level analysis environment reused across overlapping scopes."""

    def __init__(
        self,
        *,
        package_dir: Path,
        catalog: Json,
        graph: SemanticDependencyGraph,
        scopes: dict[str, Json],
        source_contracts: dict[str, Json],
        nodes: dict[str, Json],
        analyzers: tuple[StructuralAnalyzer, ...],
        tensors: TensorRepository,
        computations: AnalysisComputationMemo | None = None,
    ) -> None:
        self.package_dir = package_dir
        self.catalog = catalog
        self.graph = graph
        self.scopes = scopes
        self.source_contracts = source_contracts
        self.nodes = nodes
        self.analyzers = analyzers
        self.tensors = tensors
        self.computations = computations or AnalysisComputationMemo()

    @classmethod
    def from_package(
        cls,
        package_dir: Path,
        *,
        analyzers: Iterable[StructuralAnalyzer] | None = None,
        tensors: TensorRepository | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ScopeAnalysisEngine:
        check_compile_cancelled(cancel_requested)
        package_dir = package_dir.resolve()
        catalog = load_optimization_scope_catalog(
            package_dir / "optimization" / "scopes.json"
        )
        check_compile_cancelled(cancel_requested)
        graph = _load_semantic_graph(package_dir)
        selected = tuple(analyzers or builtin_analyzers())
        identities = [
            (analyzer.analyzer_id, analyzer.version)
            for analyzer in selected
        ]
        if len(identities) != len(set(identities)):
            raise ModelCompileError(
                "analysis run contains duplicate analyzer identities"
            )
        check_compile_cancelled(cancel_requested)
        scopes = {
            str(scope["scope_id"]): scope
            for scope in catalog["scopes"]
        }
        source_contracts = {
            str(contract["scope_id"]): contract
            for contract in catalog["source_contracts"]
        }
        return cls(
            package_dir=package_dir,
            catalog=catalog,
            graph=graph,
            scopes=scopes,
            source_contracts=source_contracts,
            nodes=_scope_node_index(graph),
            analyzers=selected,
            tensors=tensors or PackageTensorRepository(package_dir),
        )

    def analyze_scope(
        self,
        *,
        scope_id: str,
        budget: AnalysisBudget | None = None,
        activation_trace: ActivationTrace | None = None,
        analyzer_ids: Iterable[str] | None = None,
        output_dir: Path | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> AnalysisRun:
        check_compile_cancelled(cancel_requested)
        budget = budget or AnalysisBudget()
        try:
            scope = self.scopes[scope_id]
            source_contract = self.source_contracts[scope_id]
        except KeyError as error:
            raise ModelCompileError(
                f"expected one optimizer record with scope_id={scope_id!r}"
            ) from error
        context = ScopeAnalysisContext(
            package_id=str(self.catalog["package_id"]),
            scope=scope,
            source_contract=source_contract,
            tensors=self.tensors,
            nodes=_scope_nodes_from_index(self.nodes, scope),
            budget=budget,
            activation_trace=activation_trace,
            computations=self.computations,
        )
        selected_analyzers = self._selected_analyzers(analyzer_ids)
        evidence = []
        details = []
        for analyzer in selected_analyzers:
            check_compile_cancelled(cancel_requested)
            result = analyzer.analyze(context)
            check_compile_cancelled(cancel_requested)
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
            check_compile_cancelled(cancel_requested)
            write_analysis_run(run, output_dir)
        return run

    def _selected_analyzers(
        self,
        analyzer_ids: Iterable[str] | None,
    ) -> tuple[StructuralAnalyzer, ...]:
        if analyzer_ids is None:
            return self.analyzers
        requested = tuple(analyzer_ids)
        if (
            not requested
            or requested != tuple(sorted(set(requested)))
            or any(not isinstance(item, str) or not item for item in requested)
        ):
            raise ModelCompileError(
                "selected analyzer identities must be sorted, unique, and non-empty"
            )
        available = {analyzer.analyzer_id: analyzer for analyzer in self.analyzers}
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ModelCompileError(
                f"selected analyzer identities are unavailable: {unknown}"
            )
        return tuple(available[analyzer_id] for analyzer_id in requested)
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


def _scope_node_index(
    graph: SemanticDependencyGraph,
) -> dict[str, Json]:
    index = {}
    for component in graph.components:
        for raw_node in component.nodes:
            local_id = str(raw_node["id"])
            qualified_id = f"{component.component_id}/{local_id}"
            if qualified_id in index:
                raise ModelCompileError(
                    f"semantic graph contains duplicate node {qualified_id!r}"
                )
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
            index[qualified_id] = node
    return index


def _scope_nodes_from_index(
    nodes: dict[str, Json],
    scope: Json,
) -> tuple[Json, ...]:
    selected = sorted(
        str(value) for value in scope["members"]["source_node_ids"]
    )
    missing = [node_id for node_id in selected if node_id not in nodes]
    if missing:
        raise ModelCompileError(
            f"analysis scope references missing source nodes: {missing}"
        )
    return tuple(deepcopy(nodes[node_id]) for node_id in selected)
