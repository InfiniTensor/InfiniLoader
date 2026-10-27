import os
from pathlib import Path
import sys
import tempfile

import pytest

from conftest import make_model_dir
from infiniloader import cli, paths
from infiniloader.local_cache import materialize_model_cache
from infiniloader.runtime import validate_socket_path


@pytest.mark.parametrize("option", ["base", "view"])
@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux daemon integration")
def test_relative_runtime_paths_survive_consumer_directory_change(monkeypatch, option):
    with tempfile.TemporaryDirectory(prefix="ilp-") as directory:
        root = Path(directory)
        first, second = root / "first", root / "second"
        first.mkdir()
        second.mkdir()
        model = make_model_dir(root / "model")
        base = first / "state"
        monkeypatch.setenv("PYTHONPATH", str(Path(cli.__file__).resolve().parents[1]))
        monkeypatch.chdir(first)
        args = ["start", str(model), "--name", "unit", "--base-dir", "state" if option == "base" else str(base)]
        if option == "view":
            args += ["--view", "relative-view"]
        started = False
        try:
            assert cli.main(args) == 0
            started = True
            monkeypatch.chdir(second)
            script = ("import os; from pathlib import Path; p=Path(os.environ['ILOADER_MODEL_PATH']); "
                      "assert p.is_absolute(); assert (p/'model.safetensors').read_bytes()==b'0123456789abcdef'")
            assert cli.main(["run", "--base-dir", str(base), "unit", "--", sys.executable, "-c", script]) == 0
        finally:
            if started:
                assert cli.main(["stop", "unit", "--base-dir", str(base), "--remove-view"]) == 0
                from test_lifecycle import wait_for
                wait_for(lambda: not (base / "run/unit.pid").exists())


def test_configured_relative_base_is_anchored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ILOADER_BASE_DIR", "state")
    assert paths.default_base_dir() == tmp_path / "state"


def test_unix_socket_limit_counts_encoded_bytes():
    validate_socket_path(Path("/" + "a" * 106))
    with pytest.raises(ValueError, match="107 bytes"):
        validate_socket_path(Path("/" + "a" * 107))
    with pytest.raises(ValueError, match="107 bytes"):
        validate_socket_path(Path("/" + "模" * 36))


def test_long_socket_is_rejected_before_cache_preparation(tmp_path, monkeypatch, capsys):
    def forbidden(*_):
        pytest.fail("must validate socket before preparing model data")

    monkeypatch.setattr(cli, "_prepare_model_source", forbidden)
    model = make_model_dir(tmp_path / "model")
    assert cli.main(["start", str(model), "--name", "x" * 96, "--base-dir", str(tmp_path / "state")]) == 1
    assert "107 bytes" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["explicit", "default", "symlink"])
def test_nested_cache_root_is_rejected_before_copy(tmp_path, monkeypatch, mode):
    model = make_model_dir(tmp_path / "model")
    nested = model / "local-cache"
    kwargs = {"cache_root": nested}
    if mode == "default":
        kwargs = {"base_dir": model / "state"}
    elif mode == "symlink":
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(model, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation unavailable")
        kwargs = {"cache_root": alias / "local-cache"}

    def forbidden(*_):
        pytest.fail("must reject overlapping roots before copying")

    monkeypatch.setattr("infiniloader.local_cache._copy_one", forbidden)
    with pytest.raises(ValueError, match="outside the source"):
        materialize_model_cache(model, **kwargs)
    assert not nested.exists()
    assert not (model / "state").exists()
