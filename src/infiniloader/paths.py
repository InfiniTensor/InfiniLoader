from __future__ import annotations

import os
from pathlib import Path


APP_DIR_NAME = "infiniloader"


def _writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def default_base_dir() -> Path:
    configured = os.environ.get("ILOADER_BASE_DIR")
    if configured:
        candidate = Path(configured).expanduser()
        if _writable_dir(candidate):
            return candidate
        raise RuntimeError(f"ILOADER_BASE_DIR is not writable: {candidate}")

    candidates = []
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home).expanduser() / APP_DIR_NAME)
    candidates.append(Path.home() / ".cache" / APP_DIR_NAME)
    for candidate in candidates:
        if _writable_dir(candidate):
            return candidate
    fallback = Path(os.environ.get("TMPDIR", "/tmp")) / APP_DIR_NAME
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def runtime_dir(base_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(base_dir).expanduser() if base_dir else default_base_dir()
    path = base / "run"
    path.mkdir(parents=True, exist_ok=True)
    return path


def view_root(base_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(base_dir).expanduser() if base_dir else default_base_dir()
    path = base / "views"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dist_view_root(base_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(base_dir).expanduser() if base_dir else default_base_dir()
    path = base / "dist-views"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir(base_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(base_dir).expanduser() if base_dir else default_base_dir()
    path = base / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_cache_root(base_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(base_dir).expanduser() if base_dir else default_base_dir()
    path = base / "model-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_socket_path(name: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return runtime_dir(base_dir) / f"{name}.sock"


def default_pid_path(name: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return runtime_dir(base_dir) / f"{name}.pid"


def default_state_path(name: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return runtime_dir(base_dir) / f"{name}.state.json"


def default_log_path(name: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return logs_dir(base_dir) / f"{name}.log"


def default_view_path(name: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return view_root(base_dir) / name
