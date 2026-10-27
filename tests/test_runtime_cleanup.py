import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import Mock

import pytest

from infiniloader.daemon import WeightLoaderDaemon
from infiniloader import cli
from test_runtime_publication import runtime, start, crash_serve

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux runtime cleanup")


def serve_with_cleanup_fault(model, base, fault, target):
    code = f'''from pathlib import Path
import errno, os
from infiniloader.cli import main
from infiniloader.daemon import WeightLoaderDaemon
cleaning = False
publish = WeightLoaderDaemon._publish_socket
cleanup = WeightLoaderDaemon._cleanup_socket
unlink = Path.unlink
def publish_then_stop(self, server):
    publish(self, server)
    self._stop = True
def begin_cleanup(self):
    global cleaning
    cleaning = True
    cleanup(self)
def injected_unlink(path, *args, **kwargs):
    selected = cleaning and path.name == {target!r}
    if selected and {fault!r} == 'unlink_error':
        raise OSError(errno.EIO, 'injected cleanup failure')
    result = unlink(path, *args, **kwargs)
    if selected and {fault!r} == 'exit_after_unlink':
        os._exit(77)
    return result
WeightLoaderDaemon._publish_socket = publish_then_stop
WeightLoaderDaemon._cleanup_socket = begin_cleanup
Path.unlink = injected_unlink
raise SystemExit(main())
'''
    result = subprocess.run([sys.executable, "-c", code, "_serve", str(model), "--name", "unit",
                             "--base-dir", str(base), "--replace", "--quiet"],
                            capture_output=True, text=True, timeout=20)
    expected = 77 if fault == "exit_after_unlink" else 0
    assert result.returncode == expected, result.stderr
    return result


@pytest.mark.parametrize("target", ["unit.pid", "unit.sock", "unit.state.json"])
def test_failed_unlink_keeps_state_and_allows_restart(runtime, target):
    model, base = runtime
    result = serve_with_cleanup_fault(model, base, "unlink_error", target)
    assert (base / "run" / target).exists()
    state = base / "run/unit.state.json"
    assert json.loads(state.read_text())["runtime_format"] == 1
    assert "injected cleanup failure" in result.stderr
    assert target in result.stderr
    assert start(model, base) == 0


@pytest.mark.parametrize("target", ["unit.pid", "unit.sock", "unit.state.json"])
def test_cleanup_interruption_leaves_enough_evidence_to_restart(runtime, target):
    model, base = runtime
    serve_with_cleanup_fault(model, base, "exit_after_unlink", target)
    if target == "unit.state.json":
        assert not (base / "run/unit.pid").exists()
        assert not (base / "run/unit.sock").exists()
    else:
        assert (base / "run/unit.state.json").exists()
    assert start(model, base) == 0


@pytest.mark.parametrize("fault", ["unlink_error", "already_missing", "disappeared_during_unlink", "replaced"])
def test_private_socket_cleanup_preserves_state_on_uncertainty(runtime, monkeypatch, capsys, fault):
    _, base = runtime
    run = base / "run"
    run.mkdir(parents=True)
    daemon = WeightLoaderDaemon.__new__(WeightLoaderDaemon)
    daemon._owns_runtime = True
    daemon.state_path = run / "unit.state.json"
    daemon._claimed_runtime = {}
    daemon.state_path.write_text("owned state")
    daemon._claim_runtime_file(daemon.state_path)
    private = run / ".private"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(private))
    daemon._claim_runtime_file(private)
    real_unlink = Path.unlink
    if fault == "already_missing":
        private.unlink()
    elif fault == "replaced":
        replacement = run / "replacement"
        replacement.write_text("keep this unrelated file")
        replacement.replace(private)
    else:
        def fail(path, *args, **kwargs):
            if path == private:
                if fault == "disappeared_during_unlink":
                    real_unlink(path, *args, **kwargs)
                    raise FileNotFoundError(path)
                raise OSError(errno.EIO, "injected private socket failure")
            return real_unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, "unlink", fail)
    daemon._cleanup_socket()
    if fault in ("already_missing", "disappeared_during_unlink"):
        assert not daemon.state_path.exists()
    else:
        assert daemon.state_path.exists() and private.exists()
        assert "[cleanup] retained" in capsys.readouterr().err
        if fault == "replaced":
            assert private.read_text() == "keep this unrelated file"


def test_private_unlink_error_during_restart_keeps_original_recovery_state(runtime, monkeypatch, capsys):
    model, base = runtime
    crash_serve(model, base, "after_publication")
    state_path = base / "run/unit.state.json"
    original_state = state_path.read_bytes()
    state = json.loads(original_state)
    private = Path(state["runtime_private_socket"])
    daemon = WeightLoaderDaemon.__new__(WeightLoaderDaemon)
    daemon._owns_runtime, daemon.state_path, daemon._claimed_runtime = True, state_path, {}
    for path in (state_path, base / "run/unit.pid", private, base / "run/unit.sock"):
        daemon._claim_runtime_file(path)
    real_unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == private:
            raise OSError(errno.EIO, "persistent private socket failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        daemon._cleanup_socket()
        assert state_path.read_bytes() == original_state
        serve = Mock(return_value=0)
        patch.setattr(WeightLoaderDaemon, "_serve_owned", serve)
        assert cli.main(["_serve", str(model), "--name", "unit", "--base-dir", str(base), "--replace", "--quiet"]) == 1
        serve.assert_not_called()
        assert state_path.read_bytes() == original_state
        assert private.exists()
        assert "persistent private socket failure" in capsys.readouterr().err
    assert start(model, base) == 0
