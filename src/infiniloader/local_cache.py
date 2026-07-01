from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cache import DEFAULT_EXCLUDES, SourceFile, collect_source_files, human_bytes
from .paths import model_cache_root


@dataclass
class LocalCacheResult:
    source_root: Path
    cache_path: Path
    total_bytes: int
    file_count: int
    reused: bool
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "source_root": str(self.source_root),
            "cache_path": str(self.cache_path),
            "total_bytes": self.total_bytes,
            "total_human": human_bytes(self.total_bytes),
            "file_count": self.file_count,
            "reused": self.reused,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def cache_key(source_root: Path, name: str | None = None) -> str:
    source_root = source_root.expanduser().resolve()
    digest = hashlib.sha1(str(source_root).encode("utf-8")).hexdigest()[:12]
    prefix = _safe_name(name or source_root.name)
    return f"{prefix}-{digest}"


def cache_path_for(
    source_root: Path,
    *,
    name: str | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    root = Path(cache_root).expanduser() if cache_root else model_cache_root(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / cache_key(source_root, name)


def _manifest_path(path: Path) -> Path:
    return path / ".iloader-cache" / "manifest.json"


def _files_manifest(files: list[SourceFile]) -> list[dict]:
    return [
        {
            "relative_path": item.rel_posix,
            "size": item.size,
            "mtime_ns": item.mtime_ns,
        }
        for item in files
    ]


def _build_manifest(
    source_root: Path,
    files: list[SourceFile],
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> dict:
    return {
        "format_version": 1,
        "created_at": time.time(),
        "source_root": str(source_root),
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
        "include_patterns": list(include_patterns),
        "exclude_patterns": list(exclude_patterns),
        "files": _files_manifest(files),
    }


def _load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cache_is_valid(
    source_root: Path,
    cache_path: Path,
    files: list[SourceFile],
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> bool:
    manifest = _load_manifest(cache_path)
    if not manifest:
        return False
    if manifest.get("source_root") != str(source_root):
        return False
    if tuple(manifest.get("include_patterns", ())) != include_patterns:
        return False
    if tuple(manifest.get("exclude_patterns", ())) != exclude_patterns:
        return False
    if manifest.get("files") != _files_manifest(files):
        return False
    for item in files:
        dest = cache_path / item.rel
        try:
            st = dest.stat()
        except OSError:
            return False
        if st.st_size != item.size:
            return False
    return True


def _safe_remove_cache(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not _manifest_path(path).exists() and ".tmp." not in path.name:
        raise RuntimeError(f"refusing to remove non-InfiniLoader cache directory {path}")
    shutil.rmtree(path)


def _copy_one(item: SourceFile, source_root: Path, cache_path: Path, chunk_size: int) -> int:
    src = source_root / item.rel
    dst = cache_path / item.rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_file_range = getattr(os, "copy_file_range", None)

    src_fd = os.open(src, os.O_RDONLY)
    try:
        dst_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, item.mode & 0o777)
        try:
            copied = 0
            if copy_file_range is not None:
                try:
                    while copied < item.size:
                        sent = copy_file_range(src_fd, dst_fd, min(chunk_size, item.size - copied))
                        if sent == 0:
                            break
                        copied += sent
                except OSError:
                    os.lseek(src_fd, 0, os.SEEK_SET)
                    os.ftruncate(dst_fd, 0)
                    os.lseek(dst_fd, 0, os.SEEK_SET)
                    copied = 0

            if copied != item.size:
                os.ftruncate(dst_fd, 0)
                os.lseek(src_fd, 0, os.SEEK_SET)
                os.lseek(dst_fd, 0, os.SEEK_SET)
                copied = 0
                while copied < item.size:
                    data = os.read(src_fd, min(chunk_size, max(item.size - copied, 1)))
                    if not data:
                        break
                    view = memoryview(data)
                    while view:
                        written = os.write(dst_fd, view)
                        view = view[written:]
                    copied += len(data)

            if copied != item.size:
                raise IOError(f"short copy for {src}: {copied} != {item.size}")
            os.fsync(dst_fd)
        finally:
            os.close(dst_fd)
    finally:
        os.close(src_fd)

    os.utime(dst, ns=(item.mtime_ns, item.mtime_ns))
    return item.size


def materialize_model_cache(
    source_root: Path,
    *,
    name: str | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
    refresh: bool = False,
    workers: int = 4,
    chunk_size: int = 64 * 1024 * 1024,
    progress: Callable[[int, int, Path, bool], None] | None = None,
) -> LocalCacheResult:
    start = time.perf_counter()
    source_root = source_root.expanduser().resolve()
    files = collect_source_files(source_root, include_patterns, exclude_patterns)
    if not files:
        raise RuntimeError(f"no regular files selected under {source_root}")
    total = sum(item.size for item in files)
    cache_path = cache_path_for(source_root, name=name, cache_root=cache_root, base_dir=base_dir).resolve(strict=False)

    if not refresh and cache_is_valid(source_root, cache_path, files, include_patterns, exclude_patterns):
        return LocalCacheResult(source_root, cache_path, total, len(files), True, time.perf_counter() - start)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp.{os.getpid()}"
    _safe_remove_cache(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=False)

    copied = 0
    try:
        if workers <= 1:
            for item in files:
                copied += _copy_one(item, source_root, tmp_path, chunk_size)
                if progress:
                    progress(copied, total, item.rel, False)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_copy_one, item, source_root, tmp_path, chunk_size): item for item in files}
                for future in as_completed(futures):
                    item = futures[future]
                    copied += future.result()
                    if progress:
                        progress(copied, total, item.rel, False)

        manifest = _build_manifest(source_root, files, include_patterns, exclude_patterns)
        manifest_dir = tmp_path / ".iloader-cache"
        manifest_dir.mkdir(exist_ok=True)
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        if cache_path.exists():
            _safe_remove_cache(cache_path)
        os.replace(tmp_path, cache_path)
    except Exception:
        _safe_remove_cache(tmp_path)
        raise

    return LocalCacheResult(source_root, cache_path, total, len(files), False, time.perf_counter() - start)


def cache_status(
    source_root: Path,
    *,
    name: str | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> dict:
    source_root = source_root.expanduser().resolve()
    files = collect_source_files(source_root, include_patterns, exclude_patterns)
    cache_path = cache_path_for(source_root, name=name, cache_root=cache_root, base_dir=base_dir).resolve(strict=False)
    valid = cache_is_valid(source_root, cache_path, files, include_patterns, exclude_patterns)
    return {
        "source_root": str(source_root),
        "cache_path": str(cache_path),
        "exists": cache_path.exists(),
        "valid": valid,
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
        "total_human": human_bytes(sum(item.size for item in files)),
    }


def list_model_caches(
    *,
    cache_root: str | os.PathLike[str] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> list[dict]:
    root = Path(cache_root).expanduser() if cache_root else model_cache_root(base_dir)
    rows: list[dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        manifest = _load_manifest(path)
        if not manifest:
            continue
        rows.append(
            {
                "cache_path": str(path),
                "source_root": manifest.get("source_root"),
                "file_count": manifest.get("file_count"),
                "total_bytes": manifest.get("total_bytes"),
                "total_human": human_bytes(int(manifest.get("total_bytes") or 0)),
                "created_at": manifest.get("created_at"),
            }
        )
    return rows
