from __future__ import annotations

import json
import os
import select
import subprocess
from collections.abc import Callable, Mapping
from typing import Protocol

from nerve.compilation import (
    Json,
    ModelCompileCancelled,
    ModelCompileError,
    check_compile_cancelled,
)


class ExecutorTransport(Protocol):
    def request(
        self,
        document: Json,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Json:
        """Send one command and return its complete response."""

    def close(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
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
                bufsize=0,
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
        self._stdout_buffer = bytearray()

    def request(
        self,
        document: Json,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Json:
        if self.process.poll() is not None:
            raise self._failure("executor exited before command")
        assert self.process.stdin is not None
        try:
            check_compile_cancelled(cancel_requested)
            payload = (
                json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            line = self._readline(cancel_requested)
        except ModelCompileCancelled:
            self.abort()
            raise
        except (BrokenPipeError, OSError) as error:
            raise self._failure("executor command exchange failed") from error
        if not line:
            raise self._failure("executor returned no response")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise self._failure("executor returned malformed JSON") from error
        if not isinstance(response, dict):
            raise self._failure("executor response is not an object")
        return response

    def close(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        while True:
            try:
                return_code = self.process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                try:
                    check_compile_cancelled(cancel_requested)
                except ModelCompileCancelled:
                    self.abort()
                    raise
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
            stderr = self.process.stderr.read().decode(
                "utf-8",
                errors="replace",
            ).strip()
        detail = f": {stderr}" if stderr else ""
        return ModelCompileError(message + detail)

    def _readline(
        self,
        cancel_requested: Callable[[], bool] | None,
    ) -> bytes:
        assert self.process.stdout is not None
        descriptor = self.process.stdout.fileno()
        while True:
            separator = self._stdout_buffer.find(b"\n")
            if separator >= 0:
                line = bytes(self._stdout_buffer[:separator])
                del self._stdout_buffer[: separator + 1]
                return line
            check_compile_cancelled(cancel_requested)
            if self.process.poll() is not None:
                chunk = os.read(descriptor, 64 * 1024)
                if chunk:
                    self._stdout_buffer.extend(chunk)
                    continue
                return bytes(self._stdout_buffer)
            readable, _, _ = select.select(
                (descriptor,),
                (),
                (),
                0.1,
            )
            if not readable:
                continue
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return bytes(self._stdout_buffer)
            self._stdout_buffer.extend(chunk)


def subprocess_executor(
    command: tuple[str, ...],
    environment: Mapping[str, str],
) -> ExecutorTransport:
    return SubprocessExecutorTransport(command, environment)
