from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from conftest import make_model_dir
from infiniloader.cli import main
from infiniloader.ipc import request
from infiniloader.runtime import process_token, signal_owned_process

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux integration tests")


def wait_for(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.02)
    raise AssertionError("daemon did not reach expected state")


@contextmanager
def running_daemon(tmp_path, model, *, slow=False, extra=()):
    state = tmp_path / "state"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    code = (
        "import time; from infiniloader.daemon import WeightLoaderDaemon; "
        "from infiniloader.cli import main; "
        "original=WeightLoaderDaemon.load; "
        "WeightLoaderDaemon.load=lambda self: (time.sleep(1), original(self))[1]; "
        "raise SystemExit(main())"
    )
    command = [sys.executable, "-c", code] if slow else [sys.executable, "-m", "infiniloader"]
    command += ["_serve", str(model), "--name", "unit", "--base-dir", str(state), "--replace", "--quiet"]
    command += list(extra)
    with (tmp_path / "daemon.log").open("w+") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            wait_for(lambda: (state / "run/unit.state.json").exists())
            if not slow:
                wait_for(lambda: (state / "run/unit.sock").exists())
            yield proc, state
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)


def test_different_model_and_revision_cannot_reuse_name(tmp_path, capsys):
    a = make_model_dir(tmp_path / "a")
    b = make_model_dir(tmp_path / "b")
    with running_daemon(tmp_path, a) as (_, state):
        assert main(["start", str(b), "--name", "unit", "--base-dir", str(state)]) == 1
        (a / "config.json").write_text('{"changed":true}')
        assert main(["start", str(a), "--name", "unit", "--base-dir", str(state)]) == 1
        assert "different model" in capsys.readouterr().err


def test_start_joins_in_progress_daemon(tmp_path, capsys):
    model = make_model_dir(tmp_path / "model")
    with running_daemon(tmp_path, model, slow=True) as (proc, state):
        assert main(["start", str(model), "--name", "unit", "--base-dir", str(state)]) == 0
        status = request(state / "run/unit.sock", {"command": "status"})["status"]
        assert status["pid"] == proc.pid
        assert (Path(status["view"]) / "model.safetensors").read_bytes() == b"0123456789abcdef"


def test_stop_loading_preserves_unmanaged_directory(tmp_path):
    model = make_model_dir(tmp_path / "model")
    ordinary = tmp_path / "state/views/unit"
    ordinary.mkdir(parents=True)
    (ordinary / "important").write_text("keep")
    with running_daemon(tmp_path, model, slow=True) as (proc, state):
        assert main(["stop", "unit", "--base-dir", str(state), "--remove-view"]) == 1
        proc.wait(timeout=3)
        assert (ordinary / "important").read_text() == "keep"


def test_reused_pid_is_never_signalled():
    with pytest.raises(RuntimeError, match="unverified PID"):
        signal_owned_process(os.getpid(), "wrong-start-time")
    assert process_token(os.getpid()) is not None


@pytest.mark.parametrize("name", ["../model", "/tmp/model", "a/b", "a..b", "", "x" * 97])
def test_invalid_names_are_rejected(name, tmp_path):
    assert main(["start", str(tmp_path), "--name", name, "--base-dir", str(tmp_path / "state")]) == 1
