from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json
from nerve.representation_optimizer.providers.codebook.member_paths import (
    member_path,
)
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


class MemberConstructionContext:
    """Namespace one member while retaining one sealed construction session."""

    def __init__(
        self,
        parent: CandidateConstructionContext,
        lowering: Json,
    ) -> None:
        self._parent = parent
        self._lowering = deepcopy(lowering)
        self._scope_id = str(lowering["scope_id"])

    @property
    def candidate(self) -> Json:
        return self._parent.candidate

    @property
    def representation_graph(self) -> Json:
        return self._parent.representation_graph

    @property
    def target_lowering(self) -> Json:
        return deepcopy(self._lowering)

    @property
    def build_plan(self):
        return self._parent.build_plan

    @property
    def phase(self) -> str:
        return self._parent.phase

    def write_artifact(self, relative_path: str, payload: bytes) -> None:
        self._parent.write_artifact(
            member_path(self._scope_id, relative_path),
            payload,
        )

    def artifact_reference(self, relative_path: str) -> str:
        reference = member_path(self._scope_id, relative_path)
        return self._parent.artifact_reference(reference)

    def write_artifact_stream(self, relative_path: str, chunks) -> None:
        self._parent.write_artifact_stream(
            member_path(self._scope_id, relative_path),
            chunks,
        )

    def write_json_artifact(self, relative_path: str, document: Json) -> None:
        self._parent.write_json_artifact(
            member_path(self._scope_id, relative_path),
            document,
        )

    def __getattr__(self, name: str):
        return getattr(self._parent, name)
