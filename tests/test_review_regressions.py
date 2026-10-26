"""Cache integrity and allocation regressions from independent expert review."""
import json
import os
import tracemalloc

import pytest

from conftest import make_model_dir
from infiniloader import cache
from infiniloader.runtime import atomic_json

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux review regressions")








@pytest.mark.parametrize("backend", ["memfd"])
def test_copy_workers_hold_only_one_chunk_at_a_time(tmp_path, monkeypatch, backend):
    chunk = 1024 * 1024
    model = make_model_dir(tmp_path / "model", payload=b"x" * (3 * chunk))
    source = next(item for item in cache.collect_source_files(model) if item.rel.name == "model.safetensors")
    def fallback(*_):
        raise OSError("force read/write fallback")
    monkeypatch.setattr(cache, "_copy_with_copy_file_range", fallback)
    tracemalloc.start()
    try:
        if backend == "memfd":
            cached = cache.copy_file_to_memfd(source, chunk_size=chunk)
            cached.close()
        else:
            from infiniloader.local_cache import _copy_one
            _copy_one(source, tmp_path / "copy", chunk)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < chunk * 1.5, peak
    finally:
        tracemalloc.stop()
