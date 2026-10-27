"""Real descriptor ownership across interrupted loads and worker failures."""
import gc
import os
from pathlib import Path
import threading

import pytest

from infiniloader import cache

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux memfd ownership")


def memfds():
    result = set()
    for path in Path("/proc/self/fd").iterdir():
        try:
            target = path.readlink()
        except OSError:
            continue
        if "memfd:infiniloader-" in str(target):
            result.add((path.name, str(target)))
    return result


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    for name in ("first.bin", "second.bin", "third.bin"):
        (root / name).write_bytes(b"weights" * 1024)
    return root


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_callback_interruption_releases_loaded_files(model, workers, error):
    before = memfds()

    def interrupt(done, *_):
        if done:
            raise error("interrupt after copying weights")

    with pytest.raises(error):
        cache.load_model_into_memory(model, workers=workers, progress=interrupt)
    gc.collect()
    assert memfds() == before


def test_copy_interruption_releases_raw_descriptor(model, monkeypatch):
    before = memfds()

    def interrupt(*_):
        raise KeyboardInterrupt("interrupt before ownership transfer")

    monkeypatch.setattr(cache, "_copy_with_copy_file_range", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cache.copy_file_to_memfd(cache.collect_source_files(model)[0])
    gc.collect()
    assert memfds() == before


@pytest.mark.parametrize("enqueue", [False, True])
def test_partial_submit_failure_releases_worker_results(model, monkeypatch, enqueue):
    before = memfds()
    original = cache.ThreadPoolExecutor.submit
    submitted = 0

    def fail_second(pool, *args, **kwargs):
        nonlocal submitted
        submitted += 1
        if submitted != 2 or enqueue:
            result = original(pool, *args, **kwargs)
        if submitted == 2:
            raise RuntimeError("cannot start another worker")
        return result

    monkeypatch.setattr(cache.ThreadPoolExecutor, "submit", fail_second)
    with pytest.raises(RuntimeError, match="cannot start"):
        cache.load_model_into_memory(model, workers=2)
    gc.collect()
    assert memfds() == before


def test_worker_finishing_after_interrupted_shutdown_closes_its_file(model, monkeypatch):
    before = memfds()
    release = threading.Event()
    waiting = threading.Event()
    original_copy = cache.copy_file_to_memfd
    original_shutdown = cache.ThreadPoolExecutor.shutdown
    pools = []

    def delayed_copy(source, **kwargs):
        if source.rel.name == "second.bin":
            waiting.set()
            assert release.wait(10)
        return original_copy(source, **kwargs)

    def interrupted_shutdown(pool, wait=True, **kwargs):
        pools.append(pool)
        original_shutdown(pool, wait=False, **kwargs)
        raise KeyboardInterrupt("second interrupt while waiting for workers")

    def interrupt(done, *_):
        if done:
            assert waiting.wait(10)
            raise KeyboardInterrupt("first interrupt")

    monkeypatch.setattr(cache, "copy_file_to_memfd", delayed_copy)
    monkeypatch.setattr(cache.ThreadPoolExecutor, "shutdown", interrupted_shutdown)
    try:
        with pytest.raises(KeyboardInterrupt, match="second interrupt"):
            cache.load_model_into_memory(model, workers=2, progress=interrupt)
    finally:
        release.set()
        for pool in pools:
            original_shutdown(pool, wait=True)
    gc.collect()
    assert memfds() == before
