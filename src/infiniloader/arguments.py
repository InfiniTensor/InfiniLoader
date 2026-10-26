"""Argument definitions for the public CLI and internal daemon entry point."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
import sys

from . import __version__
from .memory import parse_bytes

CommandHandler = Callable[[argparse.Namespace], int]


def build_parser(handlers: Mapping[str, CommandHandler]) -> argparse.ArgumentParser:
    prog = Path(sys.argv[0]).name
    if prog in {"__main__.py", "python", "python3"}:
        prog = "iloader"
    parser = argparse.ArgumentParser(prog=prog, description="InfiniLoader: LLM weight loading and residency manager")
    parser.add_argument("--version", action="version", version=f"InfiniLoader {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    inspect_p = sub.add_parser("inspect", help="inspect selected files and memory budget")
    inspect_p.add_argument("model")
    inspect_p.add_argument("--pattern", action="append", help="include glob; can be repeated")
    inspect_p.add_argument("--exclude", action="append", help="exclude glob; replaces defaults when provided")
    inspect_p.add_argument("--json", action="store_true")
    inspect_p.set_defaults(func=handlers["cmd_inspect"])
    from .shards import configure_parser as configure_shards
    configure_shards(sub)

    serve_p = sub.add_parser("_serve", help=argparse.SUPPRESS)
    serve_p.add_argument("model")
    serve_p.add_argument("--name", required=True)
    serve_p.add_argument("--view")
    serve_p.add_argument("--socket")
    serve_p.add_argument("--pid-file")
    serve_p.add_argument("--state-file", help=argparse.SUPPRESS)
    serve_p.add_argument("--base-dir")
    serve_p.add_argument("--upstream-source", help=argparse.SUPPRESS)
    serve_p.add_argument("--local-cache-path", help=argparse.SUPPRESS)
    serve_p.add_argument("--identity", help=argparse.SUPPRESS)
    serve_p.add_argument("--pattern", action="append")
    serve_p.add_argument("--exclude", action="append")
    serve_p.add_argument("--replace", action="store_true")
    serve_p.add_argument("--force", action="store_true")
    serve_p.add_argument("--quiet", action="store_true")
    serve_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    serve_p.add_argument("--workers", type=int, default=1, help="parallel memfd preload workers")
    serve_p.add_argument("--max-memory-fraction", type=float, default=0.90)
    serve_p.add_argument("--local-cache", action="store_true", help="copy model to local cache before memfd preload")
    serve_p.add_argument("--cache-root")
    serve_p.add_argument("--cache-name")
    serve_p.add_argument("--refresh-cache", action="store_true")
    serve_p.add_argument("--cache-workers", type=int, default=4)
    serve_p.set_defaults(func=handlers["cmd_serve"])
    sub._choices_actions = [action for action in sub._choices_actions if action.dest != "_serve"]

    start_p = sub.add_parser("start", help="start background daemon")
    start_p.add_argument("model")
    start_p.add_argument("--name", required=True)
    start_p.add_argument("--view")
    start_p.add_argument("--socket")
    start_p.add_argument("--pid-file")
    start_p.add_argument("--base-dir")
    start_p.add_argument("--log-file")
    start_p.add_argument("--pattern", action="append")
    start_p.add_argument("--exclude", action="append")
    start_p.add_argument("--force", action="store_true")
    start_p.add_argument("--no-wait", action="store_true")
    start_p.add_argument("--wait-timeout", type=float, default=600.0)
    start_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    start_p.add_argument("--workers", type=int, default=1, help="parallel memfd preload workers")
    start_p.add_argument("--max-memory-fraction", type=float, default=0.90)
    start_p.add_argument("--local-cache", action="store_true", help="copy model to local cache before memfd preload")
    start_p.add_argument("--cache-root")
    start_p.add_argument("--cache-name")
    start_p.add_argument("--refresh-cache", action="store_true")
    start_p.add_argument("--cache-workers", type=int, default=4)
    start_p.set_defaults(func=handlers["cmd_start"])

    for memory_parser in (serve_p, start_p):
        memory_parser.add_argument("--max-memory-bytes", type=parse_bytes, help="per-cache preload cap including worker buffers")
        memory_parser.add_argument("--reserve-bytes", type=parse_bytes, default=256 * 1024 * 1024, help="headroom left for the engine; default 256 MiB")
        memory_parser.add_argument("--max-resident-bytes", type=parse_bytes, help="set persistent total resident quota for this base directory")
        memory_parser.add_argument("--idle-ttl", type=float, help="evict after this many idle seconds; use run to protect readers")

    cache_p = sub.add_parser("cache", help="materialize a local model cache")
    cache_p.add_argument("model")
    cache_p.add_argument("--name")
    cache_p.add_argument("--cache-name")
    cache_p.add_argument("--cache-root")
    cache_p.add_argument("--base-dir")
    cache_p.add_argument("--pattern", action="append")
    cache_p.add_argument("--exclude", action="append")
    cache_p.add_argument("--refresh", action="store_true")
    cache_p.add_argument("--cache-workers", type=int, default=4)
    cache_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    cache_p.add_argument("--quiet", action="store_true")
    cache_p.add_argument("--json", action="store_true")
    cache_p.set_defaults(func=handlers["cmd_cache"])

    cache_status_p = sub.add_parser("cache-status", help="check whether a local model cache is valid")
    cache_status_p.add_argument("model")
    cache_status_p.add_argument("--name")
    cache_status_p.add_argument("--cache-name")
    cache_status_p.add_argument("--cache-root")
    cache_status_p.add_argument("--base-dir")
    cache_status_p.add_argument("--pattern", action="append")
    cache_status_p.add_argument("--exclude", action="append")
    cache_status_p.add_argument("--json", action="store_true")
    cache_status_p.set_defaults(func=handlers["cmd_cache_status"])

    cache_list_p = sub.add_parser("cache-list", help="list local model caches")
    cache_list_p.add_argument("--cache-root")
    cache_list_p.add_argument("--base-dir")
    cache_list_p.add_argument("--json", action="store_true")
    cache_list_p.set_defaults(func=handlers["cmd_cache_list"])

    verify_p = sub.add_parser("verify", help="verify a versioned local cache using SHA-256")
    verify_p.add_argument("path")
    verify_p.add_argument("--quick", action="store_true", help="check file metadata without reading all bytes")
    verify_p.add_argument("--json", action="store_true")
    verify_p.set_defaults(func=handlers["cmd_verify"])

    gc_p = sub.add_parser("gc", help="preview or apply LRU/TTL collection of unleased disk caches")
    gc_p.add_argument("--base-dir")
    gc_p.add_argument("--cache-root")
    gc_p.add_argument("--max-bytes", type=parse_bytes, help="target total cache bytes; accepts GiB/MiB")
    gc_p.add_argument("--resident", action="store_true", help="collect resident daemons instead of disk caches")
    gc_p.add_argument("--ttl", type=float, help="collect caches unused for this many seconds")
    gc_p.add_argument("--apply", action="store_true", help="delete the selected caches; default is dry run")
    gc_p.add_argument("--json", action="store_true")
    gc_p.set_defaults(func=handlers["cmd_gc"])

    run_p = sub.add_parser("run", help="run a foreground command while protecting its cache from eviction")
    run_p.add_argument("--base-dir")
    run_p.add_argument("--cache-path")
    run_p.add_argument("name_arg", nargs="?")
    run_p.add_argument("program", nargs=argparse.REMAINDER)
    run_p.set_defaults(func=handlers["cmd_run"])

    doctor_p = sub.add_parser("doctor", help="check Linux, memfd, proc access, locks and resource limits")
    doctor_p.add_argument("--base-dir")
    doctor_p.add_argument("--json", action="store_true")
    doctor_p.set_defaults(func=handlers["cmd_doctor"])

    list_p = sub.add_parser("list", help="list active resident caches")
    list_p.add_argument("--base-dir")
    list_p.add_argument("--timeout", type=float, default=1.0)
    list_p.add_argument("--json", action="store_true")
    list_p.add_argument("--names-only", action="store_true", help=argparse.SUPPRESS)
    list_p.set_defaults(func=handlers["cmd_list"])

    dist_views_p = sub.add_parser("dist-views", help="create full HF-compatible model views for simulated ranks")
    dist_views_p.add_argument("model")
    dist_views_p.add_argument("--name", required=True)
    dist_views_p.add_argument("--ranks", type=int, required=True)
    dist_views_p.add_argument("--base-dir")
    dist_views_p.add_argument("--replace", action="store_true")
    dist_views_p.add_argument("--json", action="store_true")
    dist_views_p.set_defaults(func=handlers["cmd_dist_views"])

    dist_sim_p = sub.add_parser("dist-simulate", help="simulate concurrent multi-rank model reads on this host")
    dist_sim_p.add_argument("model")
    dist_sim_p.add_argument("--ranks", type=int, required=True)
    dist_sim_p.add_argument("--rank-view-root", help="root containing rank0/rank1/... directories")
    dist_sim_p.add_argument("--pattern", action="append")
    dist_sim_p.add_argument("--exclude", action="append")
    dist_sim_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    dist_sim_p.add_argument("--timeout", type=float, default=60.0)
    dist_sim_p.add_argument("--json", action="store_true")
    dist_sim_p.set_defaults(func=handlers["cmd_dist_simulate"])

    dist_plan_p = sub.add_parser("dist-plan", help="analyze safetensors metadata for TP split candidates")
    dist_plan_p.add_argument("model")
    dist_plan_p.add_argument("--tp", type=int, required=True)
    dist_plan_p.add_argument("--limit", type=int, default=40)
    dist_plan_p.add_argument("--json", action="store_true")
    dist_plan_p.set_defaults(func=handlers["cmd_dist_plan"])

    for name, help_text, func in [
        ("status", "show daemon status", handlers["cmd_status"]),
        ("view", "print daemon view path", handlers["cmd_view"]),
        ("manifest", "print daemon manifest", handlers["cmd_manifest"]),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name_arg", nargs="?")
        p.add_argument("--name")
        p.add_argument("--socket")
        p.add_argument("--base-dir")
        p.add_argument("--timeout", type=float, default=5.0)
        if name in {"status"}:
            p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    stop_p = sub.add_parser("stop", help="stop daemon")
    stop_p.add_argument("name_arg", nargs="?")
    stop_p.add_argument("--name")
    stop_p.add_argument("--all", action="store_true", help="stop every active resident cache in the run directory")
    stop_p.add_argument("--socket")
    stop_p.add_argument("--base-dir")
    stop_p.add_argument("--timeout", type=float, default=5.0)
    stop_p.add_argument("--remove-view", action="store_true")
    stop_p.add_argument("--force", action="store_true", help="stop even when commands hold active leases")
    stop_p.set_defaults(func=handlers["cmd_stop"])

    bench_p = sub.add_parser("bench", help="sequential read benchmark")
    bench_p.add_argument("path")
    bench_p.add_argument("--pattern", action="append")
    bench_p.add_argument("--exclude", action="append")
    bench_p.add_argument("--repeat", type=int, default=3)
    bench_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    bench_p.add_argument("--json", action="store_true")
    bench_p.set_defaults(func=handlers["cmd_bench"])

    selftest_p = sub.add_parser("selftest", help="verify memfd view with open/stat/mmap")
    selftest_p.set_defaults(func=handlers["cmd_selftest"])

    completion_p = sub.add_parser("completion", help="print shell completion script")
    completion_p.add_argument("shell", choices=["bash"])
    completion_p.set_defaults(func=handlers["cmd_completion"])
    return parser
