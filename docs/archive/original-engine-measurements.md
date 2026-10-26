# Original engine measurements

Source: the README at original commit `5014a6a`. These observations were not
remeasured by the cache reliability fixes. The original table does not provide a
complete engine-version matrix, model revisions, raw samples or reproducible commands.
Retain the numbers as historical observations, not current release guarantees.
Weight-stage, engine-load and file-read measurements are different scopes and must
not be combined into an end-to-end speedup claim.


Measurements below were collected on a single NVIDIA L20 server using remote
model storage plus local cache storage. They are meant as order-of-magnitude
guidance; exact results depend on model size, storage, framework version, JIT
state, and OS page cache state.

| Scenario | Baseline | InfiniLoader path | Result |
| --- | ---: | ---: | --- |
| Bloom-560M safetensors read from NFS vs local cache | 0.91 GiB/s | 1.98 GiB/s | 2.18x read throughput |
| Multi-shard memfd preload with 1 vs 4 workers | 0.856 s | 0.240 s | 3.57x preload speedup |
| vLLM SmolLM2 cold-ish NFS weight stage vs memfd view | 33.82 s | 0.15 s | large cold-start improvement |
| vLLM SmolLM2 hot weight stage | 0.20 s | 0.15 s | small hot-cache improvement |
| SGLang 0.5.2 SmolLM2 hot engine load | 12.18 s | 10.37 s | faster hot reload |
| InfiniLM SmolLM2 internal load-only log | 902.7 ms | 219.5 ms | 4.1x weight-stage improvement |
| 4-rank simulated read from InfiniLoader-derived rank views | - | 2.70 GiB in 0.45 s | about 6.0 GiB/s aggregate |

The most stable production-like gain comes from avoiding repeated NFS reads via
local cache plus memfd views.
