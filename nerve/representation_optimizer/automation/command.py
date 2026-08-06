from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled
from nerve.representation_optimizer.analysis.context import AnalysisBudget
from nerve.representation_optimizer.automation.contracts import (
    OptimizationBudget,
)
from nerve.representation_optimizer.automation.orchestrator import (
    run_automated_optimizer,
)
from nerve.representation_optimizer.automation.report import (
    AutomatedOptimizationOutcome,
)
from nerve.representation_optimizer.automation.runtime_target import (
    PreparedOptimizationTargets,
    prepare_runtime_optimization_targets,
)
from nerve.representation_optimizer.providers.builtin import (
    load_builtin_provider_registry,
)


RUNTIME_PACKAGE_MANIFEST = "vulkan_resident_package.json"


@dataclass(frozen=True)
class OptimizePackageOutcome:
    optimization: AutomatedOptimizationOutcome
    targets: PreparedOptimizationTargets

    def to_json(self) -> Json:
        report = self.optimization.report
        return {
            "optimization": {
                "status": report["status"],
                "report_id": report["report_id"],
                "report_path": str(self.optimization.report_path),
                "output_package": str(self.optimization.output_package_dir),
                "summary": report["summary"],
                "publication": report["publication"],
                "event_journal": report["event_journal"],
            },
            "target_preparation": self.targets.to_json(),
        }


def optimize_compiled_package(
    package: Path,
    *,
    output_package_dir: Path | None = None,
    run_root: Path | None = None,
    runtime_bin: Path | None = None,
    component_executor_bin: Path | None = None,
    validation_executor_bin: Path | None = None,
    selected_device_ids: Iterable[str] = (),
    vulkan_driver_files: Iterable[Path] = (),
    speculative_draft_tokens: int = 0,
    residency_policy: str = "demand_retained",
    budget: OptimizationBudget | None = None,
    analysis_budget: AnalysisBudget | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> OptimizePackageOutcome:
    check_compile_cancelled(cancel_requested)
    package_manifest = resolve_package_manifest(package)
    package_dir = package_manifest.parent
    output = (
        output_package_dir.expanduser().resolve()
        if output_package_dir is not None
        else package_dir.with_name(f"{package_dir.name}-optimized")
    )
    workspace = (
        run_root.expanduser().resolve()
        if run_root is not None
        else package_dir.with_name(f".{package_dir.name}-optimizer-run")
    )
    prepared = prepare_runtime_optimization_targets(
        package_manifest=package_manifest,
        run_root=workspace,
        runtime_bin=runtime_bin,
        component_executor_bin=component_executor_bin,
        validation_executor_bin=validation_executor_bin,
        selected_device_ids=selected_device_ids,
        vulkan_driver_files=vulkan_driver_files,
        speculative_draft_tokens=speculative_draft_tokens,
        residency_policy=residency_policy,
        cancel_requested=cancel_requested,
    )
    check_compile_cancelled(cancel_requested)
    outcome = run_automated_optimizer(
        package_dir=package_dir,
        source_artifacts=prepared.source_artifacts,
        output_package_dir=output,
        run_root=workspace,
        providers=load_builtin_provider_registry(),
        targets=prepared.targets,
        budget=budget or OptimizationBudget.explicitly_unbounded(),
        analysis_budget=analysis_budget,
        cancel_requested=cancel_requested,
    )
    return OptimizePackageOutcome(
        optimization=outcome,
        targets=prepared,
    )


def resolve_package_manifest(package: Path) -> Path:
    path = package.expanduser().resolve()
    if path.is_dir():
        manifest = path / RUNTIME_PACKAGE_MANIFEST
        if not manifest.is_file():
            raise ModelCompileError(
                f"{path} does not contain {RUNTIME_PACKAGE_MANIFEST}"
            )
        return manifest
    if path.is_file() and path.name == RUNTIME_PACKAGE_MANIFEST:
        return path
    raise ModelCompileError(f"compiled model package path is invalid: {path}")
