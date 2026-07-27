from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from typing import Protocol

from nerve.compilation import Json, ModelCompileError


class ExecutorTransport(Protocol):
    def request(self, document: Json) -> Json:
        """Send one command and return its complete response."""

    def close(self) -> None:
        """Close a normally released executor process."""

    def abort(self) -> None:
        """Force cleanup after a failed command or protocol exchange."""


ExecutorFactory = Callable[
    [tuple[str, ...], Mapping[str, str]],
    ExecutorTransport,
]


class SubprocessExecutorTransport:
    """Strict line-delimited JSON transport for one resident executor."""

    def __init__(
        self,
        command: tuple[str, ...],
        environment: Mapping[str, str],
    ) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=dict(environment),
            )
        except OSError as error:
            raise ModelCompileError(
                "failed to start resident optimizer executor"
            ) from error
        if (
            self.process.stdin is None
            or self.process.stdout is None
            or self.process.stderr is None
        ):
            self.abort()
            raise ModelCompileError(
                "resident optimizer executor lacks complete pipes"
            )

    def request(self, document: Json) -> Json:
        if self.process.poll() is not None:
            raise self._failure("executor exited before command")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        try:
            self.process.stdin.write(
                json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise self._failure("executor command exchange failed") from error
        if not line:
            raise self._failure("executor returned no response")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise self._failure("executor returned malformed JSON") from error
        if not isinstance(response, dict):
            raise self._failure("executor response is not an object")
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise self._failure(
                f"executor exited with status {return_code}"
            )

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()

    def _failure(self, message: str) -> ModelCompileError:
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read().strip()
        detail = f": {stderr}" if stderr else ""
        return ModelCompileError(message + detail)


def subprocess_executor(
    command: tuple[str, ...],
    environment: Mapping[str, str],
) -> ExecutorTransport:
    return SubprocessExecutorTransport(command, environment)
