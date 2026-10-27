# InfiniLoader usage guide

[Back to README](../README.md)

This walkthrough uses a custom view directory, while the README uses the default.
If `qwen` is already running from the quick start, finish its readers and run
`iloader stop qwen --remove-view` before following this walkthrough.

[中文说明](usage.zh-CN.md)

InfiniLoader keeps model files in a background memory cache so inference processes
can restart without fetching the same weights from their original storage again.
It publishes a normal-looking model directory that compatible framework file
loaders can read. Model execution and request scheduling remain with the framework.

Start the cache once, then choose either of these ways to use it:

- Put `iloader run CACHE_NAME --` before your engine command and use `'{model}'`
  for its model path. InfiniLoader fills in that path and protects the running reader.
- Pass the published cache directory directly to your engine. The inference command
  then contains no `iloader` invocation and no `'{model}'` placeholder.

The examples below use the source `/models/Qwen3-8B`, cache name `qwen`, and published
directory `$HOME/model-views/qwen`. Replace the source with your downloaded local
checkpoint. If you change the cache name or published directory, update later
commands to match; keep `'{model}'` literally as written. Run engine examples as
alternatives, not all at once.

## Install

Linux and Python 3.10+ are required. From this repository, install InfiniLoader into
each environment where you need the `iloader` command:

```bash
python3 -m pip install .
iloader doctor
iloader selftest
```

Install the inference framework and its dependencies separately. For development,
`source env.sh` from this repository exposes the source tree and Bash completion.
The daemon and readers must use the same user and PID namespace, with access to
the daemon's `/proc` entries. The runtime directory must be node-local and support
locks, Unix sockets and hard links. See [operations and containers](operations.md).

## 1. Prepare the memory cache

The examples use Bash. These commands do not launch a GPU or an inference server:

```bash
iloader inspect /models/Qwen3-8B
iloader start /models/Qwen3-8B --name qwen --view "$HOME/model-views/qwen"
iloader status qwen
iloader view qwen
```

`inspect` shows selected file sizes and available memory. `start` copies the files
into memory, starts a background daemon and waits for the cache directory to be
ready. `status` checks the daemon; `view` prints its published absolute path.
If you omit `--view`, InfiniLoader chooses `BASE/views/qwen` instead. `BASE` here
stands for the runtime root, normally `~/.cache/infiniloader`; it is explanatory
notation, not a shell variable you need to define. Overrides are listed in section 5.

| Value in the example | Who chooses it? | What it means |
| --- | --- | --- |
| `/models/Qwen3-8B` | You | Original local directory containing weights, config and tokenizer files. Download it first; this is not an online model ID. |
| `qwen` | You, via `--name qwen` | Local cache identifier for `run`, `status`, `view` and `stop`. It is independent of the checkpoint directory and the engine's API model name. |
| `$HOME/model-views/qwen` | You, via `--view` | Directory the engine should read to use cached files. `$HOME` is an ordinary shell variable for your home directory. |

The daemon remains alive after an inference process exits. Start another reader to
reuse it. By default there is no idle eviction. Starting the same unchanged source
and selection under the same name reuses the cache. Use immutable model revisions;
changed sources or daemon settings require a new name or an explicit stop/rebuild.

### What exactly is `'{model}'`?

It is a fixed placeholder meaning **"insert this cache's directory here"**.
Copy it literally, including the braces. Only `iloader run` knows how to replace it:

```bash
iloader run qwen -- cat '{model}/config.json'
```

If your home directory is `/home/alice`, the cache created above makes this launch:

```text
cat /home/alice/model-views/qwen/config.json
```

So `qwen` selects **which cache** to use, and `'{model}'` marks **where its path goes**
in the child command. It is not the original model name or a shell variable. Outside
`iloader run`, pass a real path instead; a bare engine does not expand `'{model}'`.

## 2. Choose how to launch the engine

| | Through `iloader run` | Direct engine command |
| --- | --- | --- |
| Prepare the cache first? | Yes, with `start`. | Yes, with `start`, possibly by an operator/startup script. |
| Model argument | Literal `'{model}'`. | Actual published directory, e.g. `"$HOME/model-views/qwen"`. |
| Other engine arguments | Keep them after the engine command, unchanged. | Keep them as usual. |
| Reader protection | Holds a lease while the foreground child runs. | No automatic lease; keep idle eviction disabled and do not stop/collect active readers' caches. |
| Does inference invoke `iloader`? | Yes. | No; only cache preparation uses it. |

For the wrapper form, keep the command in this order:

```text
iloader run [ILOADER_OPTIONS] CACHE_NAME -- ENGINE_COMMAND [ENGINE_ARGUMENTS]
```

InfiniLoader's `run` options, such as `--base-dir` or `--socket`, go **before the
cache name**. Everything after `--` is the child command: engine flags, positional
arguments and quoted values stay in their original positions. `run` substitutes
`{model}` inside arguments and otherwise passes them to the child. It does not
reinterpret engine flags or execute the command through a shell.

The child also receives `ILOADER_MODEL_PATH` as an environment variable. This is
provided by `run`; setting it yourself does not make an engine switch paths.

**Can I keep the original disk path and just enable caching with an engine flag?**
Currently no. This repository supplies neither native enable flags for the engines
below nor an environment switch that redirects their original disk paths. Reading
`/models/Qwen3-8B` directly still reads the source. Use `run` or explicitly point
the engine at the published directory. Compatible loaders need no source-code changes.

## 3. Complete framework examples

Prepare `qwen` with section 1 first. Use a checkpoint and hardware supported by the
chosen engine. The following show the **original command**, the **wrapper form**,
and the **direct cache-path form**, with other server arguments preserved.
They are integration examples; current CI covers the cache service and reference
examples, not a versioned compatibility matrix for all three engines.

### InfiniLM

Follow [InfiniLM's installation instructions](https://github.com/InfiniTensor/InfiniLM).
Run from its checkout, replacing the `cd` path with your own InfiniLM repository
path. The original disk-loading command is:

```bash
cd /absolute/path/to/InfiniLM
python3 python/infinilm/server/inference_server.py \
  --model /models/Qwen3-8B --device nvidia --tp 1 \
  --host 127.0.0.1 --port 8000
```

To use `iloader run`, prepend the wrapper and replace only the model path:

```bash
iloader run qwen -- python3 python/infinilm/server/inference_server.py \
  --model '{model}' --device nvidia --tp 1 \
  --host 127.0.0.1 --port 8000
```

To launch InfiniLM directly against the cache, use the real published path:

```bash
python3 python/infinilm/server/inference_server.py \
  --model "$HOME/model-views/qwen" --device nvidia --tp 1 \
  --host 127.0.0.1 --port 8000
```

`--model` is the model directory. `--device nvidia` selects InfiniLM's NVIDIA
platform; adjust it for your installation. `--tp` is tensor parallelism; `--host`
and `--port` configure its server. These are all InfiniLM arguments, not InfiniLoader
options. Other InfiniLM arguments can remain on the engine command in the same way.

### vLLM

With [vLLM installed](https://docs.vllm.ai/en/latest/serving/online_serving/), your
original command might be:

```bash
vllm serve /models/Qwen3-8B \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

Through the wrapper, keeping all other arguments:

```bash
iloader run qwen -- vllm serve '{model}' \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

Or directly, without `iloader` in the inference command:

```bash
vllm serve "$HOME/model-views/qwen" \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

The positional argument after `serve` is the model directory. The tensor-parallel,
host, port and any additional [vLLM arguments](https://docs.vllm.ai/en/latest/cli/serve/)
remain vLLM's responsibility. Preserve an explicit served model name if your clients
depend on it; the cache identifier does not configure the engine's API model name.

### SGLang

With [SGLang installed](https://docs.sglang.io/docs/basic_usage/send_request), the
original command is:

```bash
python3 -m sglang.launch_server --model-path /models/Qwen3-8B \
  --tp 1 --host 127.0.0.1 --port 30000
```

Through the wrapper:

```bash
iloader run qwen -- python3 -m sglang.launch_server --model-path '{model}' \
  --tp 1 --host 127.0.0.1 --port 30000
```

Or directly against the published cache directory:

```bash
python3 -m sglang.launch_server --model-path "$HOME/model-views/qwen" \
  --tp 1 --host 127.0.0.1 --port 30000
```

`--model-path` is the model directory; `--tp`, `--host`, `--port` and other
[server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
are passed to SGLang unchanged.

### Reuse, existing scripts and stopping

If you used the default view location, retrieve it rather than guessing the path:

```bash
VIEW=$(iloader view qwen)
vllm serve "$VIEW" --tensor-parallel-size 1 --port 8000
```

`VIEW` here is a shell variable containing a real path. Existing scripts can use
it for InfiniLM's `--model`, vLLM's positional model argument or SGLang's
`--model-path`; there is no placeholder substitution in this direct form.

Stop the inference process first. The cache remains available for the next run;
when all readers have finished and you want to free its memory, run:

```bash
iloader stop qwen --remove-view
```

This stops the cache daemon and removes the published view, leaving the source and
any local disk generations intact. The view needs its daemon alive and is local
to its host/PID namespace. Tensor parallelism does not distribute cached files
to other machines; each node needs its own accessible cache.

## 4. Local disk caching and memory budgets


A local disk generation avoids repeatedly copying a model from network or other
slow storage. It is a separate object from a resident memory cache:

```bash
LOCAL=$(iloader cache /models/Qwen3-8B --name disk-qwen --cache-root /node-local/model-cache)
iloader cache-status /models/Qwen3-8B --name disk-qwen --cache-root /node-local/model-cache
iloader cache-list --cache-root /node-local/model-cache
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- cat '{model}/config.json'
```

Here `cache --name disk-qwen` labels a **disk** generation; it does not launch a
resident daemon named `disk-qwen`. Use `run --cache-path "$LOCAL"` for this form;
you can replace the `cat` consumer with any of the engine commands above.

`cache` and `cache-status` accept `--name` (or `--cache-name`), `--cache-root`,
`--base-dir`, `--pattern`, `--exclude` and `--json`. `cache` also accepts `--refresh`
to rebuild, `--cache-workers` (default 4), integer `--chunk-size` (default 67108864)
and `--quiet` to suppress progress. Keep the disk-cache root outside the source.
Disk-only caching does not preload another model-sized allocation into memfd;
the OS controls page-cache residency.

To load a local disk generation into memory, use `start --local-cache`. This is an
alternative way to prepare `qwen`. If section 1's daemon is still running, finish
all readers and run `iloader stop qwen --remove-view` first. Loading settings and
memory caps cannot be changed by starting over an existing daemon.

```bash
iloader start /models/Qwen3-8B --name qwen --view "$HOME/model-views/qwen" \
  --local-cache --cache-root /node-local/model-cache \
  --cache-name disk-qwen --max-memory-bytes 32GiB --reserve-bytes 4GiB \
  --max-resident-bytes 64GiB
iloader gc --resident --max-bytes 32GiB          # preview only
iloader gc --resident --max-bytes 32GiB --apply  # run after consumers exit
```

These limits are examples; size them for your model and engine. The memory cache
holds selected file bytes plus loading buffers. Engine allocations and client-held
mappings/COW copies are additional. Sealed memfd pages are immutable but are not
mlocked, so memory pressure and swap can affect latency.

## 5. CLI reference

`MODEL_DIR` below means your local source directory; `NAME` means a cache identifier.
Names start with an ASCII letter or digit, contain only ASCII letters, digits,
`.`, `_`, `-`, are at most 96 characters and cannot contain `..`. They are scoped
to the registry selected by `--base-dir` / `ILOADER_BASE_DIR`.

### Command forms and options

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
| `cache MODEL_DIR` / `cache-status MODEL_DIR` / `cache-list` | Build, check or list local **disk** generations; see section 4. |
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

## How the file cache works

The core uses Python's standard library. Selected files are copied into sealed
`memfd` objects and exposed through links to `/proc/<daemon-pid>/fd/<fd>`.
Engine allocations and transfers to the GPU are still the engine's responsibility;
this cache does not promise that every stage of a restart becomes faster.

## Further documentation

- [Operations, verification, cleanup and containers](operations.md)
- [Contributing and running tests](../CONTRIBUTING.md)
- [Optional tensor sharding, TensorStore and expert examples](distributed.md)
- [Cache-focused development scope](roadmap.zh-CN.md)
- [Historical measurements and maintenance archive](archive/README.md)

TensorStore, explicit TP/PP layouts and the GPU expert pool are experimental APIs
with optional PyTorch dependencies. The reference MLP/MoE checks numerical behavior;
it does not provide a production model adapter, quantized kernel or serving engine.
They are not needed for the core `start/run` cache path.

[MIT license](../LICENSE).
