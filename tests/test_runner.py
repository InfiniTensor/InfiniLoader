import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from conftest import make_model_dir
from infiniloader.local_cache import garbage_collect, materialize_model_cache
from infiniloader.runner import run_program
from infiniloader.runtime import process_matches, process_token, signal_owned_process
from test_lifecycle import wait_for

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux process and lease tests")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_child_signal_exit_uses_shell_status(tmp_path, signum):
    assert run_program([sys.executable, "-c", f"import os; os.kill(os.getpid(), {int(signum)})"], tmp_path) == 128 + signum


@pytest.mark.parametrize("signum,ignore,expected", [
    (signal.SIGTERM, False, 143), (signal.SIGHUP, False, 129),
    (signal.SIGINT, False, 130), (signal.SIGTERM, True, 137),
    (signal.SIGKILL, False, -signal.SIGKILL),
])
def test_wrapper_cancellation_and_surviving_child_lease(tmp_path, signum, ignore, expected):
    cache = materialize_model_cache(make_model_dir(tmp_path / "model"), base_dir=tmp_path / "state")
    marker = tmp_path / "child.pid"
    code = (
        "import os, pathlib, signal, sys, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore else "")
        + "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    log = (tmp_path / "wrapper.log").open("w")
    wrapper = subprocess.Popen([sys.executable, "-m", "infiniloader", "run", "--cache-path", str(cache.cache_path),
                                "--", sys.executable, "-c", code, str(marker)], env=env,
                               stdout=log, stderr=log)
    child, token = None, None
    try:
        def started():
            assert wrapper.poll() is None, (tmp_path / "wrapper.log").read_text()
            return marker.exists() and marker.read_text()
        wait_for(started, timeout=20)
        child = int(marker.read_text())
        token = process_token(child)
        assert token is not None
        assert not garbage_collect(base_dir=tmp_path / "state", max_bytes=0, apply=True)["budget_met"]
        wrapper.send_signal(signum)
        assert wrapper.wait(timeout=8) == expected
        if signum == signal.SIGKILL:
            assert process_matches(child, token)
            assert not garbage_collect(base_dir=tmp_path / "state", max_bytes=0, apply=True)["budget_met"]
        else:
            assert not process_matches(child, token)
            assert garbage_collect(base_dir=tmp_path / "state", max_bytes=0, apply=True)["budget_met"]
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=3)
        if child is not None and process_matches(child, token):
            signal_owned_process(child, token)
            wait_for(lambda: not process_matches(child, token))
        log.close()


def test_signal_handlers_are_restored_when_spawn_fails(tmp_path):
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    with pytest.raises(FileNotFoundError):
        run_program([str(tmp_path / "missing-command")], tmp_path)
    assert {s: signal.getsignal(s) for s in before} == before
