# InfiniLoader

[中文](README.zh-CN.md)

A reusable model-file cache for Linux. Load a checkpoint into a background memory
cache, then reuse it across inference-process restarts through ordinary file reads.
Compatible loaders need no framework source-code changes.

## Installation

Requires Linux and Python 3.10+. From this repository, install into your inference
environment. Install the inference framework separately.

```bash
python3 -m pip install .
iloader doctor
```

## Quick start

Prepare a cache from an existing local model directory:

```bash
iloader start /models/Qwen3-8B --name qwen
```

Replace `/models/Qwen3-8B` with your model directory. `qwen` is a cache name you
choose. The command waits until the cache is ready; it stays available after an
inference process exits.

### Start an inference engine

Choose an installed engine that supports your checkpoint. Add `iloader run qwen --`
before its command and replace the model path with **`'{model}'`**, literally as
written. InfiniLoader fills in the cache path; keep all other engine arguments.

**InfiniLM** — run from your [InfiniLM checkout](https://github.com/InfiniTensor/InfiniLM):

```bash
cd /absolute/path/to/InfiniLM
iloader run qwen -- python3 python/infinilm/server/inference_server.py \
  --model '{model}' --device nvidia --tp 1 --host 127.0.0.1 --port 8000
```

Replace the `cd` path with your checkout; adjust `--device` for your hardware.

**vLLM:**

```bash
iloader run qwen -- vllm serve '{model}' \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

**SGLang:**

```bash
iloader run qwen -- python3 -m sglang.launch_server --model-path '{model}' \
  --tp 1 --host 127.0.0.1 --port 30000
```

Repeat your chosen command to reuse the cache. `run` holds a lease while its
foreground child runs, protecting the cache from normal eviction and stopping.
Keep that child alive while its workers need the files. Engine-version support
must be checked in your environment; these examples are not a compatibility matrix.

### Use an existing launch command directly

You can also pass the published cache path without using `iloader run`:

```bash
VIEW=$(iloader view qwen)
vllm serve "$VIEW" --tensor-parallel-size 1 --port 8000
```

For InfiniLM use `--model "$VIEW"`; for SGLang use `--model-path "$VIEW"`.
Direct launches have no automatic lease: leave idle eviction disabled and do not
stop or collect the cache while readers are active. The original disk path is not
redirected automatically, and there is no native engine flag to enable InfiniLoader.

### Manage the cache

```bash
iloader status qwen
iloader list
# After all readers have exited, release memory and remove the cache view.
iloader stop qwen --remove-view
```

Stopping the cache leaves the source model files intact. The daemon and readers
must use the same user and PID namespace; see [container setup](docs/operations.md).

## Distributed use (experimental)

Run a cache on each node for multi-node workloads. Experimental `iloader shard`
commands export per-rank weights from an **explicit TP/PP layout**; users specify
the tensor partition rules, distribute the exported files and start local caches.
This is not automatic cross-node deployment. The exported format needs a model
adapter; it is not a drop-in checkpoint for the engine commands above.
See [distributed usage](docs/distributed.md#deployment-boundary).

## Performance

Caching avoids repeated reads from the source storage, but costs preload time and
memory. When the OS page cache is already warm, an additional cache may offer little
speed benefit. Framework initialization and CPU-to-GPU transfers still take time;
measure the complete workload, including preparation, for your environment.

## Documentation

- [Usage guide and CLI reference](docs/usage.md) — command comparisons, every `start` option, memory budgets and disk caching.
- [Operations](docs/operations.md) — verification, cleanup and containers.
- [Contributing](CONTRIBUTING.md) — development setup and tests.
- [Experimental APIs](docs/distributed.md) — optional tensor sharding, TensorStore and expert examples.

For command help, use `iloader --help` or `iloader COMMAND --help`.

[MIT license](LICENSE).
