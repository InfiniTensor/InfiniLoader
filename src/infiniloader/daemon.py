from __future__ import annotations

import json
import os
import signal
import socket
import time
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import struct
import threading
from pathlib import Path
from typing import Any

from .cache import CachedModel, build_view, human_bytes, load_model_into_memory, source_identity, _safe_remove_view
from .runtime import FileLock, atomic_json, process_token


class WeightLoaderDaemon:
    def __init__(
        self,
        *,
        name: str,
        source: Path,
        view: Path,
        socket_path: Path,
        pid_path: Path,
        state_path: Path,
        upstream_source: Path | None,
        local_cache_path: Path | None,
        include_patterns: tuple[str, ...],
        exclude_patterns: tuple[str, ...],
        replace: bool,
        force: bool,
        max_memory_fraction: float,
        chunk_size: int,
        workers: int,
        quiet: bool = False,
        identity: dict | None = None,
    ) -> None:
        self.name = name
        self.source = source
        self.view = view
        self.socket_path = socket_path
        self.pid_path = pid_path
        self.state_path = state_path
        self.upstream_source = upstream_source
        self.local_cache_path = local_cache_path
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.replace = replace
        self.force = force
        self.max_memory_fraction = max_memory_fraction
        self.chunk_size = chunk_size
        self.workers = workers
        self.quiet = quiet
        self.identity = identity or {}
        self.process_token = process_token(os.getpid())
        self._owns_runtime = False
        self.cached: CachedModel | None = None
        self.started_at = time.time()
        self.load_started_at: float | None = None
        self.load_finished_at: float | None = None
        self._stop = False
        self._remove_view_on_exit = False
        self._connections = threading.BoundedSemaphore(16)

    def log(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    def _write_state(self, state: dict[str, Any]) -> None:
        state.setdefault("name", self.name)
        state.setdefault("pid", os.getpid())
        state.setdefault("source", str(self.source))
        state.setdefault("view", str(self.view))
        state.setdefault("socket", str(self.socket_path))
        state.setdefault("started_at", self.started_at)
        state.setdefault("identity", self.identity)
        state.setdefault("process_token", self.process_token)
        atomic_json(self.state_path, state)

    def _progress(self, copied: int, total: int, rel: Path) -> None:
        pct = copied / total * 100 if total else 100.0
        elapsed = time.time() - self.load_started_at if self.load_started_at else 0.0
        rate = copied / elapsed if elapsed > 0 and copied > 0 else None
        remaining = total - copied
        eta = remaining / rate if rate and remaining > 0 else None
        self._write_state(
            {
                "state": "loading",
                "load_started_at": self.load_started_at,
                "copied_bytes": copied,
                "copied_human": human_bytes(copied),
                "total_bytes": total,
                "total_human": human_bytes(total),
                "progress": copied / total if total else 1.0,
                "current_file": None if rel.as_posix() == "." else rel.as_posix(),
                "elapsed_seconds": elapsed,
                "rate_bytes_per_second": rate,
                "rate_human_per_second": f"{human_bytes(int(rate))}/s" if rate else None,
                "eta_seconds": eta,
            }
        )
        self.log(f"[load] {pct:6.2f}% {human_bytes(copied)} / {human_bytes(total)} cached; last={rel.as_posix()}")

    def load(self) -> None:
        self.log(f"[load] source={self.source}")
        self.load_started_at = time.time()
        self._write_state({"state": "loading", "load_started_at": self.load_started_at, "copied_bytes": 0})
        self.cached = load_model_into_memory(
            self.source,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
            max_memory_fraction=self.max_memory_fraction,
            force=self.force,
            progress=self._progress,
            chunk_size=self.chunk_size,
            workers=self.workers,
        )
        self.load_finished_at = time.time()
        build_view(self.cached, self.view, replace=self.replace)
        load_seconds = self.load_finished_at - self.load_started_at
        self._write_state(
            {
                "state": "ready",
                "load_started_at": self.load_started_at,
                "load_finished_at": self.load_finished_at,
                "load_seconds": load_seconds,
                "copied_bytes": self.cached.total_bytes,
                "copied_human": human_bytes(self.cached.total_bytes),
                "total_bytes": self.cached.total_bytes,
                "total_human": human_bytes(self.cached.total_bytes),
                "progress": 1.0,
                "file_count": len(self.cached.files),
                "workers": self.workers,
                "rate_bytes_per_second": self.cached.total_bytes / load_seconds if load_seconds > 0 else None,
            }
        )
        self.log(f"[ready] name={self.name} view={self.view} files={len(self.cached.files)} bytes={human_bytes(self.cached.total_bytes)}")

    def _status(self) -> dict[str, Any]:
        assert self.cached is not None
        return {
            "name": self.name,
            "identity": self.identity,
            "process_token": self.process_token,
            "pid": os.getpid(),
            "source": str(self.source),
            "upstream_source": str(self.upstream_source) if self.upstream_source else None,
            "local_cache_path": str(self.local_cache_path) if self.local_cache_path else None,
            "view": str(self.view),
            "socket": str(self.socket_path),
            "started_at": self.started_at,
            "uptime_seconds": time.time() - self.started_at,
            "load_seconds": (self.load_finished_at - self.load_started_at) if self.load_started_at and self.load_finished_at else None,
            "file_count": len(self.cached.files),
            "total_bytes": self.cached.total_bytes,
            "total_human": human_bytes(self.cached.total_bytes),
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "workers": self.workers,
        }

    def _handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = payload.get("command")
        if command == "ping":
            return {"ok": True, "pong": True}
        if command == "status":
            return {"ok": True, "status": self._status()}
        if command == "manifest":
            assert self.cached is not None
            return {"ok": True, "manifest": self.cached.to_manifest(self.view)}
        if command == "stop":
            self._remove_view_on_exit = bool(payload.get("remove_view", False))
            self._stop = True
            return {"ok": True, "stopping": True}
        return {"ok": False, "error": f"unknown command: {command!r}"}

    def _serve_connection(self, conn: socket.socket) -> None:
        try:
            with conn:
                conn.settimeout(1.0)
                if hasattr(socket, "SO_PEERCRED"):
                    _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        return
                data = bytearray()
                deadline = time.monotonic() + 1.0
                while b"\n" not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    conn.settimeout(remaining)
                    chunk = conn.recv(min(65536 - len(data), 4096))
                    if not chunk:
                        return
                    data.extend(chunk)
                    if len(data) >= 65536:
                        return
                try:
                    payload = json.loads(data.split(b"\n", 1)[0])
                    if not isinstance(payload, dict):
                        raise ValueError("request must be a JSON object")
                    response = self._handle(payload)
                except (ValueError, TypeError, RuntimeError) as exc:
                    response = {"ok": False, "error": str(exc)}
                conn.settimeout(1.0)
                conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
        except OSError:
            pass  # A stalled or disconnected client must not terminate the daemon.
        finally:
            self._connections.release()

    def _cleanup_socket(self) -> None:
        if not self._owns_runtime:
            return
        try:
            if self.socket_path.exists() or self.socket_path.is_socket():
                self.socket_path.unlink()
        except OSError:
            pass
        try:
            if self.pid_path.exists():
                self.pid_path.unlink()
        except OSError:
            pass
        try:
            if self.state_path.exists():
                self.state_path.unlink()
        except OSError:
            pass

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self.log(f"[signal] received {signum}, shutting down")
        if self.cached is None:
            self._cleanup_socket()
            os._exit(0)
        self._stop = True

    def serve_forever(self) -> int:
        with ExitStack() as locks:
            locks.enter_context(FileLock(self.socket_path.with_suffix(".owner.lock"), timeout=0))
            locks.enter_context(FileLock(self.view.parent / f".{self.view.name}.owner.lock", timeout=0))
            self._owns_runtime = True
            try:
                return self._serve_owned()
            finally:
                self._owns_runtime = False

    def _serve_owned(self) -> int:
        if not self.identity:
            self.identity = source_identity(self.upstream_source or self.source, self.include_patterns, self.exclude_patterns)
        self._install_signal_handlers()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self._write_state({"state": "starting"})

        server: socket.socket | None = None
        clients = ThreadPoolExecutor(max_workers=4, thread_name_prefix="iloader-ipc")

        try:
            self.load()

            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.settimeout(1.0)
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(16)
            self.log(f"[listen] socket={self.socket_path}")

            while not self._stop:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                if self._connections.acquire(blocking=False):
                    clients.submit(self._serve_connection, conn)
                else:
                    conn.close()
        finally:
            clients.shutdown(wait=True)
            if server is not None:
                server.close()
            self._cleanup_socket()
            if self.cached is not None:
                self.cached.close()
            if self._remove_view_on_exit:
                try:
                    _safe_remove_view(self.view, expected_pid=os.getpid())
                except (OSError, RuntimeError) as exc:
                    self.log(f"[cleanup] {exc}")
            self.log("[stop] daemon exited")
        return 0
