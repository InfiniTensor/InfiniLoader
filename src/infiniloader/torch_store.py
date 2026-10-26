"""Optional PyTorch access to canonical CPU weights and explicit TP/PP slices.

Private COW mappings protect the shared checkpoint against in-place tensor writes.
No pinning, dtype conversion, GPU transfer or framework weight packing is implicit.
"""
from __future__ import annotations

import math
import mmap
import hashlib
from pathlib import Path

from .shards import catalog, verify_rank, read_json
from .layouts import validate_layout


class TensorStore:
    def __init__(self, path: str | Path):
        import torch
        self.torch = torch
        self.path = Path(path).expanduser().resolve()
        marker = self.path / "shard.json"
        has_receipt = marker.exists()
        self.receipt = verify_rank(self.path) if has_receipt else None
        self.catalog = catalog(self.path)
        if self.receipt and (set(self.catalog["files"]) != {"weights.safetensors"}
                             or self.catalog["files"]["weights.safetensors"]["sha256"] != self.receipt["weights_sha256"]):
            raise ValueError("mapped catalog does not match shard receipt")
        self.content_id = self.receipt["plan"]["content_id"] if self.receipt else self.catalog["content_id"]
        self._maps = {}
        self._closed = False
        import fcntl
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
        try:
            for file, record in self.catalog["files"].items():
                with (self.path / file).open("rb") as stream:
                    try:
                        sealed = fcntl.fcntl(stream.fileno(), fcntl.F_GET_SEALS) & seals == seals
                    except OSError:
                        sealed = False
                    if not sealed:
                        raise ValueError("TensorStore requires sealed InfiniLoader weights; preload and use the resident view")
                    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
                self._maps[file] = mapping
                if hashlib.sha256(mapping).hexdigest() != record["sha256"]:
                    raise ValueError("weights changed between catalog and mapping")
            if marker.exists() != has_receipt or has_receipt and read_json(marker) != self.receipt:
                raise ValueError("shard receipt changed during store initialization")
        except BaseException:
            self.close()
            raise

    def tensor(self, name: str):
        if self._closed:
            raise RuntimeError("tensor store is closed")
        spec = self.catalog["tensors"][name]
        types = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
                 "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
                 "BOOL": "bool", "U16": "uint16", "U32": "uint32", "U64": "uint64",
                 "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
        dtype = getattr(self.torch, types[spec["dtype"]], None)
        if dtype is None:
            raise ValueError(f"installed torch does not support {spec['dtype']}")
        count = math.prod(spec["shape"])
        if not count:
            return self.torch.empty(spec["shape"], dtype=dtype)
        file = spec["file"]
        offset = self.catalog["files"][file]["data_start"] + spec["data_offsets"][0]
        return self.torch.frombuffer(self._maps[file], dtype=dtype, count=count, offset=offset).reshape(spec["shape"])

    def rank_tensors(self, layout: dict, rank: int) -> dict:
        """Canonical replicas can change layout without rereading/rebuilding weights.

        Pre-sharded stores are bound to their exported layout and global rank.
        The caller implements the actual model's collective/bias semantics.
        """
        if self._closed:
            raise RuntimeError("tensor store is closed")
        if type(rank) is not int:
            raise ValueError("invalid tensor layout/rank")
        if self.receipt and self.receipt["plan"]["mode"] == "sharded":
            if layout != self.receipt["plan"]["layout"] or rank != self.receipt["rank"]:
                raise ValueError("sharded store requires its exported layout and rank")
            return {n: self.tensor(n) for n in self.catalog["tensors"]}
        tp, pp, rules = validate_layout(self.catalog["tensors"], layout)
        if not 0 <= rank < tp * pp:
            raise ValueError("invalid tensor layout/rank")
        result = {}
        for name, spec in self.catalog["tensors"].items():
            rule = rules[name]
            stage, dim = rule["stage"], rule["split_dim"]
            if stage != rank // tp:
                continue
            tensor = self.tensor(name)
            if dim is not None:
                width = tensor.shape[dim] // tp
                tensor = tensor.narrow(dim, rank % tp * width, width)
            result[name] = tensor
        return result

    def close(self):
        self._closed = True
        # torch.frombuffer keeps its buffer alive through storage lifetime, including
        # tensor views. Never explicitly close mappings still referenced by tensors.
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
