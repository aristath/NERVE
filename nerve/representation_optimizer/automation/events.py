from __future__ import annotations

import json
import os
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.automation.contracts import (
    OPTIMIZER_EVENT_SCHEMA,
)
from nerve.representation_optimizer.contracts import canonical_json_bytes


class EventJournal:
    def __init__(self, path: Path) -> None:
        if path.exists() or path.is_symlink():
            raise ModelCompileError(
                f"optimizer event journal already exists: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._sequence = 0
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        os.close(descriptor)

    def record(
        self,
        *,
        phase: str,
        status: str,
        scope_id: str | None = None,
        target_id: str | None = None,
        candidate_id: str | None = None,
        evidence_refs: tuple[str, ...] = (),
        details: Json | None = None,
    ) -> Json:
        document = {
            "schema": OPTIMIZER_EVENT_SCHEMA,
            "sequence": self._sequence,
            "phase": phase,
            "status": status,
            "scope_id": scope_id,
            "target_id": target_id,
            "candidate_id": candidate_id,
            "evidence_refs": list(evidence_refs),
            "details": dict(details or {}),
        }
        payload = canonical_json_bytes(document) + b"\n"
        descriptor = os.open(self._path, os.O_WRONLY | os.O_APPEND)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise ModelCompileError(
                    "optimizer event journal write was incomplete"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sequence += 1
        return document

    @property
    def event_count(self) -> int:
        return self._sequence


def read_event_journal(path: Path) -> tuple[Json, ...]:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for expected_sequence, line in enumerate(stream):
            document = json.loads(line)
            if (
                not isinstance(document, dict)
                or document.get("schema") != OPTIMIZER_EVENT_SCHEMA
                or document.get("sequence") != expected_sequence
            ):
                raise ModelCompileError(
                    "optimizer event journal is malformed or non-contiguous"
                )
            events.append(document)
    return tuple(events)
