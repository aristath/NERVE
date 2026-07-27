from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.staging.loading import (
    LoadedStagedCandidate,
    load_staged_candidate,
)


STAGED_IMPLEMENTATION_PREFIX = "staged-representation:"
StagedCandidateLoader = Callable[
    [Path, str, Path],
    LoadedStagedCandidate,
]


class ExecutorArtifactStore:
    """Confined immutable custody for executor inputs and raw traces."""

    def __init__(self, root: Path, *, label: str, create: bool) -> None:
        if root.is_symlink():
            raise ModelCompileError(f"{label} root must not be a symlink")
        root = root.resolve()
        if create:
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            raise ModelCompileError(f"{label} root is unavailable")
        self.root = root
        self.label = label

    def iter_file(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        if (
            isinstance(chunk_bytes, bool)
            or not isinstance(chunk_bytes, int)
            or chunk_bytes <= 0
        ):
            raise ModelCompileError(
                f"{self.label} chunk size must be positive"
            )
        path = self.confined_path(relative_path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                while chunk := stream.read(chunk_bytes):
                    yield chunk
        except OSError as error:
            raise ModelCompileError(
                f"{self.label} is unavailable: {relative_path!r}"
            ) from error

    def publish(self, relative_path: str, payload: bytes) -> dict[str, str]:
        path = self.confined_path(relative_path)
        if not payload:
            raise ModelCompileError(
                f"{self.label} cannot publish an empty artifact"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o644)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise ModelCompileError(
                f"failed to publish {self.label} {relative_path!r}"
            ) from error
        return {
            "path": relative_path,
            "digest": staged_artifact_digest(payload),
        }

    def confined_path(self, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if (
            relative.is_absolute()
            or "." in relative.parts
            or ".." in relative.parts
            or relative.as_posix() != relative_path
        ):
            raise ModelCompileError(
                f"{self.label} path is unsafe: {relative_path!r}"
            )
        path = self.root.joinpath(*relative.parts)
        try:
            path.resolve().relative_to(self.root)
        except (OSError, ValueError) as error:
            raise ModelCompileError(
                f"{self.label} escapes its root: {relative_path!r}"
            ) from error
        return path


def default_staged_candidate_loader(
    workspace_root: Path,
    candidate_id: str,
    package_dir: Path,
) -> LoadedStagedCandidate:
    return load_staged_candidate(
        workspace_root,
        candidate_id,
        package_dir=package_dir,
    )


def resolve_candidate_mount(
    *,
    implementation_id: str,
    workspace_root: Path,
    package_dir: Path,
    loader: StagedCandidateLoader,
) -> tuple[str | None, Path | None]:
    if not implementation_id.startswith(STAGED_IMPLEMENTATION_PREFIX):
        return None, None
    candidate_id = implementation_id.removeprefix(
        STAGED_IMPLEMENTATION_PREFIX
    )
    loaded = loader(workspace_root, candidate_id, package_dir)
    return candidate_id, loaded.path
