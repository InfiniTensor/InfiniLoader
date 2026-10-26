from pathlib import Path
import os
import time
from unittest.mock import Mock

import pytest

from infiniloader import cli
from infiniloader.ipc import request
from test_lifecycle import wait_for
from test_runtime_publication import runtime

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux daemon preparation")


@pytest.mark.parametrize("no_wait", [False, True])
def test_slow_local_preparation_does_not_consume_daemon_wait(runtime, monkeypatch, capsys, no_wait):
    model, base = runtime
    real_clock = time.monotonic
    elapsed = [0]
    prepare = cli._prepare_model_source

    def slow_prepare(args):
        result = prepare(args)
        assert result[2] is not None
        elapsed[0] += 20
        return result

    monkeypatch.setattr(cli.time, "monotonic", lambda: real_clock() + elapsed[0])
    monkeypatch.setattr(cli, "_prepare_model_source", slow_prepare)
    args = ["start", str(model), "--name", "unit", "--base-dir", str(base), "--local-cache", "--wait-timeout", "10"]
    if no_wait:
        args.append("--no-wait")
    assert cli.main(args) == 0
    assert capsys.readouterr().out.strip() == str(base / "views/unit")
    wait_for(lambda: (base / "run/unit.sock").exists())
    status = request(base / "run/unit.sock", {"command": "status"})["status"]
    assert status["state"] == "ready"
    assert (Path(status["view"]) / "model.safetensors").read_bytes() == b"0123456789abcdef"


def test_preparation_failure_does_not_spawn_daemon(runtime, monkeypatch, capsys):
    model, base = runtime
    spawn = Mock()

    def fail(args):
        raise OSError("injected preparation failure")

    monkeypatch.setattr(cli, "_prepare_model_source", fail)
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.main(["start", str(model), "--name", "unit", "--base-dir", str(base), "--local-cache"]) == 1
    result = capsys.readouterr()
    assert not result.out and "injected preparation failure" in result.err
    spawn.assert_not_called()
