import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from conftest import make_model_dir
from infiniloader import cli
from infiniloader.daemon import WeightLoaderDaemon
from infiniloader.ipc import request

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux runtime publication")


@pytest.fixture
def runtime(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="ilr-") as directory:
        root = Path(directory)
        model = make_model_dir(root / "model")
        base = root / "state"
        monkeypatch.setenv("PYTHONPATH", str(Path(cli.__file__).resolve().parents[1]))
        yield model, base
        if (base / "run/unit.pid").exists():
            cli.main(["stop", "unit", "--base-dir", str(base), "--remove-view"])


def start(model, base):
    return cli.main(["start", str(model), "--name", "unit", "--base-dir", str(base), "--wait-timeout", "10"])


@pytest.mark.parametrize("through_cli", [False, True])
def test_stop_then_start_can_switch_model_without_retry(runtime, through_cli):
    model, base = runtime
    assert start(model, base) == 0
    sock = base / "run/unit.sock"
    old_pid = request(sock, {"command": "status"})["status"]["pid"]
    if through_cli:
        assert cli.main(["stop", "unit", "--base-dir", str(base), "--remove-view"]) == 0
        assert not (base / "run/unit.pid").exists()
    else:
        assert request(sock, {"command": "stop", "remove_view": True})["stopping"]
    replacement = make_model_dir(model.parent / "replacement", payload=b"new revision")
    assert start(replacement, base) == 0
    status = request(sock, {"command": "status"})["status"]
    assert status["pid"] != old_pid
    assert (Path(status["view"]) / "model.safetensors").read_bytes() == b"new revision"


def crash_serve(model, base, point):
    common = "from infiniloader.cli import main; from infiniloader.daemon import WeightLoaderDaemon; import os\n"
    if point == "before_publication":
        injection = ("original=WeightLoaderDaemon._write_state\n"
                     "def write(self,state):\n"
                     " if self._socket_identity is not None: os._exit(77)\n"
                     " return original(self,state)\n"
                     "WeightLoaderDaemon._write_state=write\n")
    elif point == "after_publication":
        injection = "original=os.link\ndef link(*a,**k):\n original(*a,**k)\n os._exit(77)\nos.link=link\n"
    else:
        injection = ("import infiniloader.daemon as daemon\n"
                     "daemon.atomic_text=lambda *a,**k: os._exit(77)\n")
    result = subprocess.run([sys.executable, "-c", common + injection + "raise SystemExit(main())",
                             "_serve", str(model), "--name", "unit", "--base-dir", str(base), "--replace", "--quiet"],
                            capture_output=True, timeout=20)
    assert result.returncode == 77, result.stderr.decode(errors="replace")


@pytest.mark.parametrize("point", ["before_publication", "after_publication"])
def test_start_recovers_from_socket_publication_crash(runtime, point):
    model, base = runtime
    crash_serve(model, base, point)
    if point == "after_publication":
        state = json.loads((base / "run/unit.state.json").read_text())
        st = (base / "run/unit.sock").stat()
        assert state["runtime_socket"] == {"device": st.st_dev, "inode": st.st_ino}
    else:
        assert not (base / "run/unit.sock").exists()
    assert start(model, base) == 0


def test_start_recovers_between_replacing_state_and_pid(runtime):
    model, base = runtime
    crash_serve(model, base, "after_publication")
    crash_serve(model, base, "before_pid")
    assert not (base / "run/unit.pid").exists()
    assert start(model, base) == 0


def test_publication_does_not_replace_a_concurrent_file(runtime, monkeypatch):
    model, base = runtime
    run = base / "run"
    run.mkdir(parents=True)
    daemon = WeightLoaderDaemon(name="unit", source=model, view=base / "view", socket_path=run / "unit.sock",
                               pid_path=run / "unit.pid", state_path=run / "unit.state.json", upstream_source=None,
                               local_cache_path=None, include_patterns=(), exclude_patterns=(), replace=True,
                               force=False, max_memory_fraction=.9, chunk_size=1024, workers=1)
    original = os.link

    def race(source, destination, **kwargs):
        Path(destination).write_text("preserve this file")
        original(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", race)
    daemon._owns_runtime = True
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            with pytest.raises(FileExistsError):
                daemon._publish_socket(server)
    finally:
        daemon._cleanup_socket()
    assert daemon.socket_path.read_text() == "preserve this file"


def test_private_socket_never_uses_the_public_name(runtime, monkeypatch):
    model, base = runtime
    run = base / "run"
    run.mkdir(parents=True)
    daemon = WeightLoaderDaemon(name="unit", source=model, view=base / "view", socket_path=run / ".abc",
                               pid_path=run / "unit.pid", state_path=run / "unit.state.json", upstream_source=None,
                               local_cache_path=None, include_patterns=(), exclude_patterns=(), replace=True,
                               force=False, max_memory_fraction=.9, chunk_size=1024, workers=1)
    candidates = iter(["fabc", "fdef"])
    monkeypatch.setattr("infiniloader.daemon.uuid.uuid4", lambda: SimpleNamespace(hex=next(candidates)))
    original = daemon._write_state

    def inspect(state):
        assert not daemon.socket_path.exists()
        assert daemon._private_socket != daemon.socket_path
        original(state)

    monkeypatch.setattr(daemon, "_write_state", inspect)
    daemon._owns_runtime = True
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            daemon._publish_socket(server)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(daemon.socket_path))
                accepted, _ = server.accept()
                accepted.close()
    finally:
        daemon._cleanup_socket()
