from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError, read_json
from nerve.representation_optimizer.contracts import canonical_json_bytes


def write_new_json(path: Path, document: Json) -> Path:
    if path.exists() or path.is_symlink():
        raise ModelCompileError(f"optimizer evidence already exists: {path}")
    _write_atomic(path, document)
    return path


def replace_json(path: Path, document: Json) -> Path:
    _write_atomic(path, document)
    return path


def read_object(path: Path) -> Json:
    document = read_json(path)
    if not isinstance(document, dict):
        raise ModelCompileError(f"optimizer document is not an object: {path}")
    return document


def relative_ref(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ModelCompileError("optimizer evidence path escapes run root") from error
    return relative.as_posix()


def _write_atomic(path: Path, document: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ModelCompileError("optimizer evidence directory must not be a symlink")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = canonical_json_bytes(document) + b"\n"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise ModelCompileError("optimizer evidence write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
