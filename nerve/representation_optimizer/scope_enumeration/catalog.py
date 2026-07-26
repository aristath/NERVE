from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError, read_json, write_json
from nerve.representation_optimizer.contracts import (
    OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    contract_digest,
    validate_contract,
)
from nerve.representation_optimizer.scope_enumeration.enumerator import (
    enumerate_optimization_scope_catalog,
)
from nerve.representation_optimizer.scope_enumeration.graph import (
    SemanticDependencyGraph,
)


OPTIMIZATION_SCOPE_CATALOG_FILE = "scopes.json"


@dataclass(frozen=True)
class OptimizationScopeCatalogArtifact:
    path: Path
    document: Json

    @property
    def digest(self) -> str:
        return contract_digest(self.document)

    def package_reference(self, package_dir: Path) -> str:
        return self.path.relative_to(package_dir).as_posix()


def build_optimization_scope_catalog(
    *,
    package_id: str,
    package_dir: Path,
    lowered_index: Json,
    lowered_index_ref: str,
) -> Json:
    graph = SemanticDependencyGraph.from_lowered_package(
        package_dir=package_dir,
        lowered_index=lowered_index,
        lowered_index_ref=lowered_index_ref,
    )
    return enumerate_optimization_scope_catalog(
        package_id=package_id,
        graph=graph,
    )


def write_optimization_scope_catalog(
    *,
    package_id: str,
    package_dir: Path,
    optimizer_dir: Path,
    lowered_index: Json,
    lowered_index_ref: str,
) -> OptimizationScopeCatalogArtifact:
    try:
        relative_optimizer_dir = optimizer_dir.relative_to(package_dir)
    except ValueError as error:
        raise ModelCompileError(
            "optimization scope catalog must stay inside the compiled package"
        ) from error
    if ".." in relative_optimizer_dir.parts:
        raise ModelCompileError(
            "optimization scope catalog must stay inside the compiled package"
        )
    document = build_optimization_scope_catalog(
        package_id=package_id,
        package_dir=package_dir,
        lowered_index=lowered_index,
        lowered_index_ref=lowered_index_ref,
    )
    path = optimizer_dir / OPTIMIZATION_SCOPE_CATALOG_FILE
    write_json(path, document)
    return OptimizationScopeCatalogArtifact(path=path, document=document)


def load_optimization_scope_catalog(path: Path) -> Json:
    document = read_json(path)
    validate_contract(
        document,
        expected_schema=OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    )
    return document
