from __future__ import annotations

import signal
import sys

import pytest

from nerve.compilation import ModelCompileCancelled
from nerve.representation_optimizer.benchmarking.executor_transport import (
    SubprocessExecutorTransport,
)


def test_subprocess_transport_cancellation_kills_blocked_executor() -> None:
    transport = SubprocessExecutorTransport(
        (
            sys.executable,
            "-u",
            "-c",
            (
                "import signal, sys; "
                "sys.stdin.buffer.readline(); "
                "signal.pause()"
            ),
        ),
        {},
    )
    checks = 0

    def cancel_after_command_is_written() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(ModelCompileCancelled, match="cancelled"):
        transport.request(
            {"command": "never-returns"},
            cancel_requested=cancel_after_command_is_written,
        )

    assert checks >= 2
    assert transport.process.poll() == -signal.SIGKILL


def test_subprocess_transport_preserves_line_framing() -> None:
    transport = SubprocessExecutorTransport(
        (
            sys.executable,
            "-u",
            "-c",
            (
                "import sys; "
                "line = sys.stdin.buffer.readline(); "
                "sys.stdout.buffer.write(line); "
                "sys.stdout.buffer.flush(); "
                "sys.stdin.buffer.read()"
            ),
        ),
        {},
    )

    assert transport.request({"command": "echo"}) == {
        "command": "echo"
    }
    transport.close()
    assert transport.process.returncode == 0
