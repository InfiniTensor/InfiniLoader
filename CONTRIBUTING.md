# Contributing

Keep changes focused on the cache's file interface, correctness and usability.
Optional tensor/device integrations should remain separate from the standard-library
core. Include a regression that reproduces a behavior failure rather than testing
only the implementation's current structure.

## Core checks

Use Linux with Python 3.10+:

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
iloader selftest
iloader doctor
```

The file-cache tests intentionally use arbitrary file bytes; they test filesystem
and lifecycle behavior, not model semantics. Runtime tests use real subprocesses,
memfds and Unix sockets. Fault-injection tests use only their own temporary paths.

## Optional PyTorch checks

Install a PyTorch build appropriate for the target device, then:

```bash
python3 -m pip install -e '.[test]' safetensors numpy
python3 -m pytest -q
python3 examples/hybrid_moe.py --device cpu --output hybrid-cpu.json
```

The optional tests cover tensor views, real local Gloo TP/PP execution and a small
reference MoE. Device tests skip when no compatible GPU is available; a CPU pass
does not validate CUDA/ROCm execution. Remote/device measurements must state their
commit, hardware, software versions and actual commands separately.

## Review and documentation

- Run the checks relevant to changed behavior and include their actual output.
- Preserve ordinary files and unknown runtime resources when testing cleanup.
- Keep CLI help, completion and both READMEs consistent. Completion is generated
  from the argument parser and has executable Bash tests.
- Document memory ownership and lifetime when changing descriptors, mappings,
  leases or device copies.
- Performance claims need a named baseline, reproducible commands and repeated
  measurements; file-read bandwidth is not engine-startup latency.
- Historical maintenance records belong in [the archive](docs/archive/README.md),
  while current operational contracts belong in [operations](docs/operations.md).
