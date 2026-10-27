# InfiniLoader

[中文说明](README.zh-CN.md)

InfiniLoader keeps model files in a long-lived cache so inference-engine processes
can restart without fetching the same weights from their original storage again.
It is a Linux userspace cache: it does not implement model execution or request scheduling.

The core uses Python's standard library. It copies selected files into sealed
`memfd` objects and exposes a directory of links to `/proc/<daemon-pid>/fd/<fd>`.
Consumers keep their existing file loaders and use the published directory as the model path.

## Install and check the environment

Linux and Python 3.10+ are required. Install inside the engine's environment, using
the same user and PID namespace as the consumer:

```bash
python3 -m pip install .
iloader doctor
iloader selftest
```

For development, `source env.sh` exposes the checkout without installing it and
enables Bash completion. The runtime directory must be node-local and support file
locks, Unix sockets and hard links; `doctor` checks these capabilities.

## Cache once and reuse

```bash
MODEL=/absolute/path/to/model
iloader start "$MODEL" --name dev
iloader status dev

# Example when vLLM is already installed; validate your engine version.
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 1

# After the engine exits, the cache remains available for another run.
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 1
iloader stop dev --remove-view
```

`run` replaces the literal `{model}` argument and sets `ILOADER_MODEL_PATH`.
It holds a lease while the foreground consumer runs. Put InfiniLoader options
before the cache name, and keep the consumer alive while its workers need the view.
`stop` waits for the original daemon to exit, or returns an error on timeout.

The file interface is the integration boundary. Engine-specific formats, individual
file-path resolution, containers and loader behavior can affect compatibility;
test your actual engine. Current CI exercises the cache service and reference
examples, not a versioned vLLM/SGLang/InfiniLM compatibility matrix.

## Disk caching and memory limits

For a reusable local disk generation without another model-sized RAM allocation:

```bash
LOCAL=$(iloader cache "$MODEL" --name dev)
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- vllm serve '{model}'
```

To preload into RAM from the local generation, use `start --local-cache`.
The disk-cache root must be outside the source model directory.

```bash
iloader start "$MODEL" --name dev --local-cache \
  --max-memory-bytes 32GiB --reserve-bytes 4GiB --max-resident-bytes 64GiB
iloader gc --resident --max-bytes 32GiB          # preview only
iloader gc --resident --max-bytes 32GiB --apply
```

The limits above are examples; size them for the model and engine. The cache holds
selected file bytes plus loading buffers. Engine allocations and client-held
mappings/COW copies are additional. Sealed memfd pages are immutable but are not
mlocked, so memory pressure and swap can affect latency.

Use immutable model revisions. A changed source requires a new cache name or an
explicit stop/rebuild. Readers launched directly against `iloader view dev` have
no automatic lease; keep idle eviction disabled for those readers.

## Documentation and optional APIs

- [Operations, verification, cleanup and containers](docs/operations.md)
- [Contributing and running tests](CONTRIBUTING.md)
- [Optional tensor sharding, TensorStore and expert examples](docs/distributed.md)
- [Cache-focused development scope](docs/roadmap.zh-CN.md)
- [Historical measurements and maintenance archive](docs/archive/README.md)

TensorStore, explicit TP/PP layouts and the GPU expert pool are experimental APIs
with optional PyTorch dependencies. The reference MLP/MoE checks numerical behavior;
it does not provide a production model adapter, quantized kernel or serving engine.
They are not needed for the core `start/run` cache path.

Run `iloader --help` and `iloader COMMAND --help` for the complete CLI. Typical core
commands are `start`, `run`, `status`, `list`, `stop`, `cache`, `verify`, `gc`, `doctor`
and `bench`. `bench` measures file reads without clearing the OS page cache;
it does not measure end-to-end engine startup.

[MIT license](LICENSE).
