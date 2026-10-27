"""CPU model of event bookkeeping; actual CUDA lifetime tests remain separate."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from infiniloader.experts import ExpertPool, _Entry


class Stream:
    def __init__(self, device):
        self.device = device


class Event:
    def __init__(self):
        self.synchronized = False

    def record(self, stream):
        self.stream = stream

    def query(self):
        raise AssertionError("hot release must not rescan historical events")

    def synchronize(self):
        self.synchronized = True


class Cuda:
    def __init__(self, stream):
        self.current = stream
        self.events = []
        self.synchronizations = 0

    def Event(self):
        event = Event()
        self.events.append(event)
        return event

    def current_stream(self, device):
        return self.current

    def device(self, device):
        return nullcontext()

    def stream(self, stream):
        return nullcontext()

    def synchronize(self, device):
        self.synchronizations += 1


def pool_with_stream():
    device = object()
    stream = Stream(device)
    cuda = Cuda(stream)
    recorded = []
    tensor = SimpleNamespace(record_stream=recorded.append)
    store = SimpleNamespace(torch=SimpleNamespace(cuda=cuda),
                            catalog={"tensors": {"w": {"shape": [1], "dtype": "F32"}}})
    pool = ExpertPool(store, {"0": {"up": "w"}}, max_gpu_bytes=4)
    pool.device = device
    pool.entries["0"] = _Entry({"up": tensor}, 4)
    return pool, cuda, stream, recorded


def test_repeated_hot_leases_keep_only_latest_event_per_stream():
    pool, cuda, stream, recorded = pool_with_stream()
    for _ in range(1000):
        with pool.acquire("0"):
            pass
    assert len(pool.entries["0"].events) == 1
    assert recorded == [stream] * 1000
    pool.evict("0")
    assert cuda.events[-1].synchronized
    assert not any(event.synchronized for event in cuda.events[:-1])


def test_eviction_waits_for_latest_event_on_every_consumer_stream():
    pool, cuda, first, recorded = pool_with_stream()
    second = Stream(pool.device)
    with pool.acquire("0") as lease:
        lease.record_stream(second)
    initial = list(cuda.events)
    with pool.acquire("0"):
        pass
    assert len(pool.entries["0"].events) == 2
    pool.close()
    assert initial[1].synchronized and cuda.events[-1].synchronized
    assert not initial[0].synchronized
    assert recorded == [first, second, first]


def test_event_record_failure_synchronizes_device_and_releases_users(monkeypatch):
    pool, cuda, stream, recorded = pool_with_stream()

    def fail(self, stream):
        raise RuntimeError("injected event failure")

    monkeypatch.setattr(Event, "record", fail)
    with pytest.raises(RuntimeError, match="injected"):
        with pool.acquire("0") as lease:
            pass
    assert cuda.synchronizations == 1
    assert pool.status()["active_leases"] == 0
    assert pool.entries["0"].users == 0
    assert not lease.weights
