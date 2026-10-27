"""Shared safetensors metadata validation; payload hashing is a caller concern."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}

MAX_HEADER = 100_000_000


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _signature(path: Path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _int(value, *, minimum=0):
    return type(value) is int and value >= minimum


def tensor_header(path: Path, *, include_metadata: bool = False) -> tuple[dict, int]:
    """Reject holes, overlaps, truncated payloads and unsupported packed dtypes."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors: {path}")
        length, = struct.unpack("<Q", raw)
        if not 2 <= length <= min(MAX_HEADER, size - 8):
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length).decode("utf-8"), object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    has_metadata = "__metadata__" in header
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("safetensors metadata must contain strings")
    intervals = []
    for name, spec in header.items():
        if not name or not isinstance(spec, dict):
            raise ValueError("invalid tensor record")
        dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES or not isinstance(shape, list) or not all(_int(d) for d in shape):
            raise ValueError(f"unsupported dtype or shape: {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(_int(v) for v in offsets) or offsets[1] < offsets[0]
                or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]):
            raise ValueError(f"invalid tensor offsets/size: {name}")
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("overlapping tensor data or hole in payload")
        end = stop
    if end != size - 8 - length:
        raise ValueError("truncated or trailing safetensors payload")
    if include_metadata and has_metadata:
        header = {"__metadata__": metadata, **header}
    return header, 8 + length


def tensor_inventory(source: Path) -> tuple[dict, dict]:
    """Validate the complete tensor/index inventory without reading tensor data."""
    source = source.expanduser().resolve()
    paths = sorted(source.rglob("*.safetensors"))
    signatures = {path: _signature(path) for path in paths}
    index_path = source / "model.safetensors.index.json"
    index_signature = _signature(index_path) if index_path.exists() else None
    files, tensors = {}, {}
    for path in paths:
        header, data_start = tensor_header(path)
        relative = path.relative_to(source).as_posix()
        files[relative] = {"size": signatures[path][2], "data_start": data_start}
        for name, spec in header.items():
            if name in tensors:
                raise ValueError(f"duplicate tensor across files: {name}")
            tensors[name] = dict(spec, file=relative)
    if not files or not tensors:
        raise ValueError("no safetensors tensors found")
    if index_signature is not None:
        index = read_json(index_path)
        if not isinstance(index, dict) or index.get("weight_map") != {name: spec["file"] for name, spec in tensors.items()}:
            raise ValueError("safetensors index does not match tensor inventory")
    if (paths != sorted(source.rglob("*.safetensors"))
            or any(_signature(path) != signature for path, signature in signatures.items())
            or (_signature(index_path) if index_path.exists() else None) != index_signature):
        raise RuntimeError("source inventory changed while reading metadata")
    return files, tensors
