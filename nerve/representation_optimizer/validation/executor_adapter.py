from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    ExecutorArtifactStore,
    LazyExecutorArtifactStore,
    StagedCandidateLoader,
    default_staged_candidate_loader,
)
from nerve.representation_optimizer.benchmarking.executor_client import (
    ResidentExecutorClient,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    ExecutorFactory,
    subprocess_executor,
)
from nerve.representation_optimizer.validation.component_executor import (
    ResidentComponentValidationBackend,
)
from nerve.representation_optimizer.validation.protocols import (
    ValidationRoleMountRequest,
)
from nerve.representation_optimizer.validation.whole_model_executor import (
    ResidentWholeModelValidationBackend,
)


class ResidentBehavioralValidationAdapter:
    """Generic validation over ordinary component and whole-model execution."""

    def __init__(
        self,
        *,
        package_manifest: Path,
        candidate_workspace: Path,
        trace_root: Path,
        component_executor_command: tuple[str, ...],
        whole_model_executor_command: tuple[str, ...],
        vulkan_driver_files: tuple[Path, ...],
        executor_factory: ExecutorFactory = subprocess_executor,
        staged_candidate_loader: StagedCandidateLoader
        | None = None,
    ) -> None:
        self.package_manifest = package_manifest.resolve()
        self.package_dir = self.package_manifest.parent
        self.candidate_workspace = candidate_workspace.resolve()
        self.trace_store = LazyExecutorArtifactStore(
            trace_root,
            label="validation trace",
        )
        self.staged_candidate_loader = (
            staged_candidate_loader
            or default_staged_candidate_loader
        )
        run_nonce = uuid4().hex
        component_client = ResidentExecutorClient(
            package_manifest=self.package_manifest,
            candidate_workspace=self.candidate_workspace,
            executor_command=component_executor_command,
            vulkan_driver_files=vulkan_driver_files,
            executor_factory=executor_factory,
            staged_candidate_loader=self.staged_candidate_loader,
        )
        self.component = ResidentComponentValidationBackend(
            executor_client=component_client,
            trace_store=self.trace_store,
            run_nonce=run_nonce,
        )
        self.whole_model = ResidentWholeModelValidationBackend(
            package_manifest=self.package_manifest,
            candidate_workspace=self.candidate_workspace,
            trace_store=self.trace_store,
            executor_command=whole_model_executor_command,
            vulkan_driver_files=vulkan_driver_files,
            executor_factory=executor_factory,
            staged_candidate_loader=self.staged_candidate_loader,
            run_nonce=run_nonce,
        )

    def iter_fixture_artifact(
        self,
        relative_path: str,
        *,
        candidate_id: str,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        candidate = self.staged_candidate_loader(
            self.candidate_workspace,
            candidate_id,
            self.package_dir,
        )
        return ExecutorArtifactStore(
            candidate.path,
            label="validation fixture",
            create=False,
        ).iter_file(relative_path, chunk_bytes=chunk_bytes)

    def open_session(
        self,
        request: ValidationRoleMountRequest,
    ):
        scope = request.check["regime"]["execution_scope"]
        if scope == "component":
            return self.component.open_session(request)
        if scope == "whole_model":
            return self.whole_model.open_session(request)
        raise ModelCompileError(
            f"unsupported validation execution scope {scope!r}"
        )

    def compare_results(
        self,
        request: Json,
        reference_result: Json,
        candidate_result: Json,
    ) -> Json:
        scope = request["check"]["regime"]["execution_scope"]
        if scope == "component":
            return self.component.compare_results(
                request,
                reference_result,
                candidate_result,
            )
        if scope == "whole_model":
            return self.whole_model.compare_results(
                request,
                reference_result,
                candidate_result,
            )
        raise ModelCompileError(
            f"unsupported validation comparison scope {scope!r}"
        )

    def iter_trace_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        return self.trace_store.iter_file(
            relative_path,
            chunk_bytes=chunk_bytes,
        )
