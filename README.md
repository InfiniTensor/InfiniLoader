# InfiniLoader

[中文说明](README.zh-CN.md)

InfiniLoader keeps model files in a long-lived memory cache so inference processes
can restart without fetching the same weights from their original storage again.
It is a Linux userspace file cache; model execution and request scheduling remain
the inference framework's responsibility.

The core uses Python's standard library. It copies selected files into sealed
`memfd` objects and publishes a directory of links to `/proc/<daemon-pid>/fd/<fd>`.
Frameworks use their existing file loaders with this directory as the model path.
The cache outlives individual inference processes and can be reused across restarts.

## 1. Install

Linux and Python 3.10+ are required. From the InfiniLoader checkout, install into
each inference environment where you want to use `iloader`:

```bash
python3 -m pip install .
iloader doctor
iloader selftest
```

For development, `source env.sh` from this checkout exposes the source tree and
enables Bash completion. Install inference frameworks separately. The daemon and
consumers must run as the same user in the same PID namespace, with access to the
daemon's `/proc` entries. The runtime directory must be node-local and support
locks, Unix sockets and hard links; `doctor` checks these capabilities.
See [container setup and operations](docs/operations.md).

## 2. Understand names and paths

| Term | Meaning | Example |
| --- | --- | --- |
| `MODEL` / `MODEL_DIR` | Existing **local model directory**, including weights, configuration and tokenizer files. Download the model first; this is not an online model ID. | `/mnt/models/Qwen3-8B` |
| `--name NAME` | Local identifier for a resident cache, used by `run`, `status` and `stop`. Independent of the model directory name and the API's served model name. | `--name dev` |
| `VIEW` | Published directory whose files point to the cached bytes in memory. `start` and `view` print its absolute path. | `~/.cache/infiniloader/views/dev` |
| `'{model}'` | **Literal placeholder recognized by `iloader run`**. Keep the braces; `run` substitutes the cache directory into command arguments. It is not a shell variable or a model ID. | `--model '{model}'` |

Names start with an ASCII letter or digit, contain only ASCII letters, digits,
`.`, `_`, `-`, are at most 96 characters, and cannot contain `..`. Names are scoped
to the registry selected by `--base-dir` / `ILOADER_BASE_DIR`.

**Pass the cache view as the model path.** InfiniLoader does not currently intercept
reads from the original disk directory. Passing `/mnt/models/Qwen3-8B` directly to
a framework still reads that source. Use `'{model}'` through `run` or an explicit
`VIEW` path. Compatible loaders need no changes to their framework code.

## 3. Use iloader independently

These Bash commands prepare and inspect a cache before involving any engine:

```bash
MODEL=/absolute/path/to/model
iloader inspect "$MODEL"
iloader start "$MODEL" --name dev
iloader status dev
iloader list
iloader view dev

# A small consumer: print the substituted path and read a cached config file.
iloader run dev -- python3 -c \
  'from pathlib import Path; import sys; p = Path(sys.argv[1]); print(p); print((p / "config.json").read_text()[:200])' \
  '{model}'
```

Choose a model containing `config.json` for this consumer. By default, `start`
waits until the view is ready and writes only its path to stdout; progress goes
to stderr. You can capture it with `VIEW=$(iloader start "$MODEL" --name dev)`.

The daemon remains alive after a consumer exits. Repeat `iloader run dev -- ...`
to reuse it. Starting the same unchanged source and selection under the same name
reuses the existing cache. Use immutable model revisions: a different or changed
source requires a new name or an explicit stop and rebuild. Stop and restart the
daemon to change its loading settings.

`run` holds a lease while its foreground child runs, protecting the cache from
normal eviction and stopping. Keep that child alive while its workers need files.
It sets `ILOADER_MODEL_PATH` in the child's environment, executes arguments without
a shell, and returns the child's exit status. Outer shell variables expand before
`run` starts; use the literal `'{model}'` shown above.

### Commands and argument placement

```text
iloader start MODEL_DIR --name NAME [START_OPTIONS]
iloader run [--base-dir PATH] [--socket PATH] NAME -- COMMAND [ARGUMENTS...]
iloader run --cache-path LOCAL_GENERATION -- COMMAND [ARGUMENTS...]
iloader status|view|manifest|stop NAME [OPTIONS]
```

`start` requires `--name`; `run` takes a **positional name**, with InfiniLoader
options **before the name**. The actual command and engine options follow `--`.
For example, `--workers` belongs to `start`; `--tensor-parallel-size` after
`vllm serve` belongs to vLLM.

| Command | Purpose and useful options |
| --- | --- |
| `inspect MODEL_DIR` | Preview selected file sizes and available memory; `--pattern`, `--exclude`, `--json`. |
| `start MODEL_DIR --name NAME` | Preload files and launch the resident daemon; see the options below. |
| `run NAME -- COMMAND ...` | Launch a foreground consumer; optional `--base-dir`, `--socket` before `NAME`. The disk-only form uses `--cache-path` instead of a name. |
| `status NAME` | Show state, cached bytes and leases; `--json` for structured output. |
| `view NAME` / `manifest NAME` | Print the view path / JSON file manifest. |
| `list` | List resident caches; `--base-dir`, `--timeout` (default 1 second), `--json`. |
| `stop NAME` | Stop a daemon; `--remove-view` also removes its published directory. `--all` selects resident caches using the default registry paths; `--force` overrides active leases. |
| `cache MODEL_DIR` / `cache-status MODEL_DIR` / `cache-list` | Build, check or list local **disk** generations; see section 5. |
| `verify LOCAL_GENERATION` | Verify a disk generation using SHA-256; `--quick` checks metadata only; `--json`. |
| `gc` | Preview disk cleanup; `--resident` selects memory caches, `--max-bytes SIZE` sets a target, `--ttl SECONDS` selects idle caches, `--apply` performs cleanup. |
| `doctor` / `selftest` | Check the environment / exercise the memfd interface. `doctor` accepts `--base-dir` and `--json`. |
| `bench PATH` | Sequential file-read benchmark; `--repeat` (default 3), `--pattern`, `--exclude`, `--chunk-size`, `--json`. Does not clear OS page caches or measure engine startup. |
| `completion bash` | Print the Bash completion script. |

`status`, `view`, `manifest` and `stop` also accept `--name NAME`, `--socket PATH`,
`--base-dir PATH` and `--timeout SECONDS` (default 5). `gc` accepts `--base-dir`,
`--cache-root` for disk caches, and `--json`; cleanup is a dry run unless `--apply`
is present, and leased caches are skipped. `--resident` and `--cache-root` cannot
be combined.

### start options

| Option | Default | Meaning |
| --- | --- | --- |
| `--name NAME` | Required | Resident cache identifier. |
| `--view PATH` | `BASE/views/NAME` | Published directory; must be separate from the source. |
| `--base-dir PATH` | See below | Root for runtime metadata, views, logs and the default disk cache. |
| `--socket PATH` | `BASE/run/NAME.sock` | Override the Unix socket; use the same override in later commands, including `run`. |
| `--pid-file PATH`, `--log-file PATH` | `BASE/run/NAME.pid`, `BASE/logs/NAME.log` | Override daemon PID and log paths. |
| `--pattern GLOB` | All files, subject to exclusions | Include relative paths; repeat for multiple patterns. Quote globs to prevent shell expansion. |
| `--exclude GLOB` | Built-in exclusions | Exclude relative paths; repeat as needed. Supplying exclusions **replaces** the default list. |
| `--workers N` | `1` | Parallel memory-preload workers, not GPU count or tensor parallelism. |
| `--chunk-size BYTES` | `67108864` (64 MiB) | Per-worker copy buffer; integer bytes only, not `64MiB`. |
| `--max-memory-fraction FRACTION` | `0.90` | Fraction used in the available-memory admission check. |
| `--max-memory-bytes SIZE` | Unset | Per-cache preload cap, including worker buffers. |
| `--reserve-bytes SIZE` | `256MiB` | Headroom in the admission check; adjust for your engine. Does not reserve physical RAM. |
| `--max-resident-bytes SIZE` | Unset | Set a persistent shared resident quota for the socket's parent directory (normally `BASE/run`). |
| `--idle-ttl SECONDS` | Disabled | Evict after this idle interval; use `run` to lease active readers. |
| `--force` | Off | Bypass the available-memory check; explicit per-cache and shared caps still apply. |
| `--wait-timeout SECONDS` | `600` | Daemon readiness/registration wait; local disk-cache preparation is excluded. |
| `--no-wait` | Off | Return after daemon registration; the view may not yet be ready. |
| `--local-cache` | Off | Build/reuse a local disk generation before memory preload. |
| `--cache-root PATH` | `BASE/model-cache` | Disk-cache root when `--local-cache` is enabled; keep outside the source. |
| `--cache-name NAME` | Resident `NAME` | Disk-cache label for `--local-cache`. |
| `--refresh-cache` | Off | Rebuild the disk generation during `--local-cache` preparation. |
| `--cache-workers N` | `4` | Parallel disk-copy workers for `--local-cache`. |

`SIZE` accepts byte counts or binary suffixes such as `512MiB` and `32GiB`.
`BASE` uses `ILOADER_BASE_DIR` if set, otherwise `$XDG_CACHE_HOME/infiniloader`
or `~/.cache/infiniloader`, with a temporary-directory fallback if necessary.
To use one registry across commands and environments, export
`ILOADER_BASE_DIR=/node-local/path/infiniloader` or pass the same `--base-dir`
to each command that supports it. Unix socket paths must fit in 107 encoded bytes.

Prefer default socket and PID paths. A custom socket parent creates a separate
resident quota group; sharing `--base-dir` alone does not combine those quotas.
Registry-wide discovery and cleanup assume default paths: a custom PID file can
hide a daemon from listing, and `stop --all` does not automatically reuse custom
sockets. Stop such a daemon explicitly with `iloader stop --socket PATH NAME`.

Keep config and tokenizer files in the selected set. Using only
`--pattern '*.safetensors'` can leave an engine without required metadata.
Run `iloader COMMAND --help` for complete command options.

## 4. Load the cache with InfiniLM, vLLM or SGLang

First prepare `dev` using section 3. Install the engine in its environment and
choose a checkpoint it supports. Run these examples **one at a time**; compatible
loaders can reuse the same file cache. These are integration examples, not a
versioned compatibility matrix: current CI covers the cache service and reference
examples, not all engine versions.

### InfiniLM

Follow [InfiniLM's installation and server instructions](https://github.com/InfiniTensor/InfiniLM).
From the **InfiniLM checkout**, with `iloader` available in that environment:

```bash
cd /absolute/path/to/InfiniLM
iloader run dev -- python3 python/infinilm/server/inference_server.py \
  --model '{model}' --device nvidia --tp 1
```

`--model` receives the memory-cache directory. `--device nvidia` selects InfiniLM's
NVIDIA platform; select the platform supported by your installation. `--tp 1` sets
the engine's tensor-parallel degree. Add other InfiniLM server options as usual.

### vLLM

With [vLLM installed](https://docs.vllm.ai/en/latest/serving/online_serving/):

```bash
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 1
```

The positional argument after `serve` is the model directory.
`--tensor-parallel-size 1` is vLLM's tensor-parallel degree. Append other vLLM
options after the engine command.

### SGLang

With [SGLang installed](https://docs.sglang.io/docs/basic_usage/send_request):

```bash
iloader run dev -- python3 -m sglang.launch_server \
  --model-path '{model}' --tp 1
```

`--model-path` receives the cache directory; `--tp 1` is SGLang's tensor-parallel
degree. See its [server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
for other settings.

### Use an explicit path in an existing script

```bash
VIEW=$(iloader view dev)
# Example: replace only the model path in an existing vLLM command.
vllm serve "$VIEW" --tensor-parallel-size 1
```

For InfiniLM use `--model "$VIEW"`; for SGLang use `--model-path "$VIEW"`.
Direct launches have **no automatic lease**: keep idle eviction disabled and do
not stop or collect the cache while readers are active. Prefer `iloader run` for
managed lifetimes. The view needs its daemon alive and is local to its host/PID
namespace. Tensor parallelism does not distribute cached files to other machines;
each node needs its own accessible cache.

When all consumers have exited, release memory and remove the view:

```bash
iloader stop dev --remove-view
```

This leaves the original model directory and any local disk generations intact.

## 5. Optional local disk cache and memory budgets

A local disk generation avoids repeatedly copying a model from network or other
slow storage. It is a separate object from a resident memory cache:

```bash
MODEL=/absolute/path/to/model
LOCAL=$(iloader cache "$MODEL" --name local-dev --cache-root /node-local/model-cache)
iloader cache-status "$MODEL" --name local-dev --cache-root /node-local/model-cache
iloader cache-list --cache-root /node-local/model-cache
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- python3 -c \
  'from pathlib import Path; import sys; print((Path(sys.argv[1]) / "config.json").read_text()[:200])' \
  '{model}'
```

Here `cache --name local-dev` labels a **disk** generation; it does not launch a
resident daemon named `local-dev`. Use `run --cache-path "$LOCAL"` for this form;
you can replace the Python consumer with any of the engine commands above.

`cache` and `cache-status` accept `--name` (or `--cache-name`), `--cache-root`,
`--base-dir`, `--pattern`, `--exclude` and `--json`. `cache` also accepts `--refresh`
to rebuild, `--cache-workers` (default 4), integer `--chunk-size` (default 67108864)
and `--quiet` to suppress progress. Keep the disk-cache root outside the source.
Disk-only caching does not preload another model-sized allocation into memfd;
the OS controls page-cache residency.

To load a local disk generation into memory, use `start --local-cache`:

```bash
iloader start "$MODEL" --name dev --local-cache --cache-root /node-local/model-cache \
  --cache-name local-dev --max-memory-bytes 32GiB --reserve-bytes 4GiB \
  --max-resident-bytes 64GiB
iloader gc --resident --max-bytes 32GiB          # preview only
iloader gc --resident --max-bytes 32GiB --apply  # run after consumers exit
```

These limits are examples; size them for your model and engine. The memory cache
holds selected file bytes plus loading buffers. Engine allocations and client-held
mappings/COW copies are additional. Sealed memfd pages are immutable but are not
mlocked, so memory pressure and swap can affect latency.

## Further documentation

- [Operations, verification, cleanup and containers](docs/operations.md)
- [Contributing and running tests](CONTRIBUTING.md)
- [Optional tensor sharding, TensorStore and expert examples](docs/distributed.md)
- [Cache-focused development scope](docs/roadmap.zh-CN.md)
- [Historical measurements and maintenance archive](docs/archive/README.md)

TensorStore, explicit TP/PP layouts and the GPU expert pool are experimental APIs
with optional PyTorch dependencies. The reference MLP/MoE checks numerical behavior;
it does not provide a production model adapter, quantized kernel or serving engine.
They are not needed for the core `start/run` cache path.

[MIT license](LICENSE).
