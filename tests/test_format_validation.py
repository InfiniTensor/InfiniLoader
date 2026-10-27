import json
import struct

import pytest

from infiniloader import distributed, shards
from infiniloader.formats import canonical, tensor_inventory


def write(path, header, payload=b"abcd"):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(header) if isinstance(header, dict) else header
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def valid():
    return {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}


@pytest.mark.parametrize("case", ["missing_payload", "wrong_shape", "unknown_dtype", "overlap", "duplicate_json", "metadata"])
def test_both_planners_reject_the_same_malformed_checkpoint(tmp_path, case):
    source = tmp_path / "model"
    header, payload = valid(), b"abcd"
    if case == "missing_payload":
        header["weight"]["shape"] = [4, 2]
        header["weight"]["data_offsets"] = [0, 32000]
        payload = b""
    elif case == "wrong_shape":
        header["weight"]["shape"] = [2]
    elif case == "unknown_dtype":
        header["weight"]["dtype"] = "UNSUPPORTED"
    elif case == "overlap":
        header["another"] = dict(header["weight"])
    elif case == "duplicate_json":
        raw = canonical(header["weight"])
        header = b'{"weight":' + raw + b',"weight":' + raw + b'}'
    else:
        header["__metadata__"] = {"invalid": 1}
    write(source / "weights.safetensors", header, payload)
    layout = {"format": 1, "tp": 1, "pp": 1, "tensors": {"weight": {"stage": 0, "split_dim": None}}}
    for planner in (lambda: distributed.build_tp_plan(source, 1), lambda: shards.build_plan(source, layout)):
        with pytest.raises(ValueError):
            planner()


def test_metadata_analysis_is_recursive_and_does_not_hash_payload(tmp_path, monkeypatch):
    source = tmp_path / "model"
    write(source / "nested/weights.safetensors", dict(valid(), __metadata__={"format": "pt"}))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "nested/weights.safetensors"}}))
    monkeypatch.setattr(shards, "file_hash", lambda *_: pytest.fail("header analysis must not hash weight data"))
    result = distributed.build_tp_plan(source, 1)
    assert result["total_bytes"] == 4
    assert result["tensors"][0]["file"] == "nested/weights.safetensors"
    assert distributed.parse_safetensors_header(source / "nested/weights.safetensors")["__metadata__"] == {"format": "pt"}


@pytest.mark.parametrize("case", ["duplicate_tensor", "bad_index", "empty"])
def test_analysis_rejects_inconsistent_inventory(tmp_path, case):
    source = tmp_path / "model"
    source.mkdir()
    if case != "empty":
        write(source / "weights.safetensors", valid())
        if case == "duplicate_tensor":
            write(source / "nested/weights.safetensors", valid())
        else:
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing"}}))
    with pytest.raises(ValueError):
        distributed.build_tp_plan(source, 1)


@pytest.mark.parametrize("change", ["replace", "add", "remove"])
def test_catalog_detects_index_changes_during_payload_hashing(tmp_path, monkeypatch, change):
    source = tmp_path / "model"
    write(source / "weights.safetensors", valid())
    index = source / "model.safetensors.index.json"
    if change != "add":
        index.write_text(json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
    original = shards.file_hash

    def mutate(path):
        result = original(path)
        if change == "remove":
            index.unlink()
        else:
            index.write_text(json.dumps({"weight_map": {"missing": "missing.safetensors"}}))
        return result

    monkeypatch.setattr(shards, "file_hash", mutate)
    with pytest.raises(RuntimeError, match="inventory changed"):
        shards.catalog(source)
