from __future__ import annotations

from pathlib import PurePosixPath

from nerve.compilation import ModelCompileError


def member_root(scope_id: str) -> str:
    if (
        not scope_id.startswith("scope_")
        or "/" in scope_id
        or "\\" in scope_id
        or PurePosixPath(scope_id).name != scope_id
    ):
        raise ModelCompileError(
            f"candidate member has unsafe scope identity {scope_id!r}"
        )
    return f"members/{scope_id}"


def member_path(scope_id: str, relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
    ):
        raise ModelCompileError(
            f"candidate member artifact path is unsafe: {relative_path!r}"
        )
    return f"{member_root(scope_id)}/{path.as_posix()}"
