import os
import json
from pathlib import Path
import shutil
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


@pytest.fixture
def default_signals():
    handlers = {signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL,
                signal.SIGINT: signal.default_int_handler}
    previous = {signum: signal.getsignal(signum) for signum in handlers}
    try:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_child_signal_exit_uses_shell_status(tmp_path, signum, default_signals):
    assert run_program([sys.executable, "-c", f"import os; os.kill(os.getpid(), {int(signum)})"], tmp_path) == 128 + signum


@pytest.mark.parametrize("signum,ignore,expected", [
    (signal.SIGTERM, False, 143), (signal.SIGHUP, False, 129),
    (signal.SIGINT, False, 130), (signal.SIGTERM, True, 137),
    (signal.SIGKILL, False, -signal.SIGKILL),
])
def test_wrapper_cancellation_and_surviving_child_lease(tmp_path, signum, ignore, expected, default_signals):
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


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
def test_wrapper_and_child_preserve_inherited_ignored_signals(tmp_path, signum):
    if signum == signal.SIGHUP and not shutil.which("nohup"):
        pytest.skip("requires nohup")
    cache = materialize_model_cache(make_model_dir(tmp_path / "model"), base_dir=tmp_path / "state")
    marker, finish = tmp_path / "consumer.json", tmp_path / "finish"
    code = (
        "import json,os,pathlib,signal,sys,time\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),"
        f"'ignored':signal.getsignal({int(signum)}) == signal.SIG_IGN}}))\n"
        "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.02)\n"
    )
    command = [sys.executable, "-m", "infiniloader", "run", "--cache-path", str(cache.cache_path),
               "--", sys.executable, "-c", code, str(marker), str(finish)]
    if signum == signal.SIGHUP:
        command = ["nohup", *command]
    else:
        command = [sys.executable, "-c",
                   "import os,signal,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); os.execv(sys.argv[1],sys.argv[1:])",
                   *command]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    child, token = None, None
    with (tmp_path / "wrapper.log").open("w") as log:
        wrapper = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            def started():
                assert wrapper.poll() is None, (tmp_path / "wrapper.log").read_text()
                return marker.exists() and marker.read_text()
            wait_for(started, timeout=20)
            data = json.loads(marker.read_text())
            child, token = data["pid"], process_token(data["pid"])
            assert data["ignored"]
            status = Path(f"/proc/{wrapper.pid}/status").read_text()
            mask = int(next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:")), 16)
            assert mask & (1 << (signum - 1))
            wrapper.send_signal(signum)
            os.kill(child, signum)
            assert process_matches(child, token)
            assert not garbage_collect(base_dir=tmp_path / "state", max_bytes=0, apply=True)["budget_met"]
            finish.touch()
            assert wrapper.wait(timeout=8) == 0
            assert garbage_collect(base_dir=tmp_path / "state", max_bytes=0, apply=True)["budget_met"]
        finally:
            finish.touch()
            if wrapper.poll() is None:
                try:
                    wrapper.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    wrapper.kill()
                    wrapper.wait(timeout=3)
            if child is not None:
                wait_for(lambda: not process_matches(child, token))
