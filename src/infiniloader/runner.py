"""Run one foreground consumer while its caller holds the cache lease."""
from __future__ import annotations

import os
import signal
import subprocess
import time

from .ipc import request
from .runtime import process_token


def run_program(program, path, *, pass_fds=(), socket_path=None, lease=None) -> int:
    cancelled = None
    proc = None
    handlers = {}

    def cancel(signum, _frame):
        nonlocal cancelled
        cancelled = signum

    try:
        # Install before spawn so cancellation cannot strand a just-created child.
        # SIGINT retains Python's KeyboardInterrupt handling: terminal interrupts
        # already reach both processes in the foreground process group.
        for signum in (signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, cancel)
        env = dict(os.environ, ILOADER_MODEL_PATH=str(path))
        proc = subprocess.Popen([part.replace("{model}", str(path)) for part in program],
                                env=env, pass_fds=pass_fds)
        if lease:
            token = process_token(proc.pid)
            if proc.poll() is None:
                try:
                    request(socket_path, {"command": "lease-update", "lease": lease, "pid": proc.pid,
                                          "process_token": token})
                except RuntimeError:
                    if proc.poll() is None:
                        raise
        deadline = None
        while proc.poll() is None:
            if cancelled is not None and deadline is None:
                proc.send_signal(cancelled)
                deadline = time.monotonic() + 3
            if deadline is not None and time.monotonic() >= deadline:
                proc.kill()
            try:
                proc.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                continue
        return proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            for signum, previous in handlers.items():
                signal.signal(signum, previous)
