from __future__ import annotations

import os
import signal
import sys

import pytest

from nerve.compilation import ModelCompileCancelled
from nerve.representation_optimizer.benchmarking.executor_transport import (
    EXECUTOR_PROGRESS_SCHEMA,
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
    assert os.getpgid(transport.process.pid) == transport.process.pid
    assert os.getpgid(transport.process.pid) != os.getpgrp()
    transport.close()
    assert transport.process.returncode == 0


def test_subprocess_transport_delivers_ordered_progress_before_final_response() -> None:
    transport = SubprocessExecutorTransport(
        (
            sys.executable,
            "-u",
            "-c",
            (
                "import json, sys; "
                "request = json.loads(sys.stdin.buffer.readline()); "
                f"schema = {EXECUTOR_PROGRESS_SCHEMA!r}; "
                "events = ["
                "{'schema': schema, 'request_id': request['request_id'], "
                "'sequence': 0, 'payload': {'tokens': 32}},"
                "{'schema': schema, 'request_id': request['request_id'], "
                "'sequence': 1, 'payload': {'tokens': 64}},"
                "{'schema': 'final', 'request_id': request['request_id']}]; "
                "[print(json.dumps(event), flush=True) for event in events]; "
                "sys.stdin.buffer.read()"
            ),
        ),
        {},
    )
    progress = []

    response = transport.request(
        {"request_id": "progress-request"},
        progress_received=progress.append,
    )

    assert [event["payload"]["tokens"] for event in progress] == [
        32,
        64,
    ]
    assert response == {
        "schema": "final",
        "request_id": "progress-request",
    }
    transport.close()
