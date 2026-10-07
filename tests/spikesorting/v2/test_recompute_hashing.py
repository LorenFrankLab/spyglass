"""Analyzer-content hashing contracts independent of database state."""

from types import SimpleNamespace

import numpy as np
import pytest

from spyglass.spikesorting.v2 import _recompute as hashes


class _Analyzer:
    def __init__(self, data):
        self.data = data

    def has_extension(self, name):
        return name == "waveforms"

    def get_extension(self, name):
        return SimpleNamespace(get_data=lambda: self.data)


def _hash(data):
    return hashes.hash_extension_data(_Analyzer(data))["waveforms"]


@pytest.mark.parametrize(
    "other",
    [
        lambda data: data.reshape(3, 2),
        lambda data: data.view("int32"),
        lambda data: [data[:1], data[1:]],
    ],
    ids=["shape", "dtype", "array-boundaries"],
)
def test_hash_distinguishes_array_interpretation(other):
    data = np.arange(6, dtype="int64").reshape(2, 3)
    assert _hash(data) != _hash(other(data))


def test_hash_canonicalizes_endian_and_layout(monkeypatch):
    data = np.arange(60, dtype="float32").reshape(3, 4, 5) / 7
    expected = _hash(data)
    assert _hash(data.astype(">f4")) == expected
    assert _hash(np.asfortranarray(data)) == expected
    # Changing the buffering budget must leave the digest unchanged.
    monkeypatch.setattr(hashes, "_HASH_CHUNK_BYTES", 13)
    assert _hash(data.astype(">f4")) == expected
    assert _hash(np.asfortranarray(data)) == expected
    assert _hash(data[:, :, ::2]) == _hash(data[:, :, ::2].copy())


def test_hash_handles_empty_structured_and_rejects_object_arrays():
    assert _hash(np.empty((0, 2))) != _hash(np.empty((0, 3)))
    spikes = np.array(
        [(7, 2), (10, 3)], dtype=[("sample", "i8"), ("unit", "i4")]
    )
    assert _hash(spikes) == _hash(spikes.astype(spikes.dtype.newbyteorder(">")))
    with pytest.raises(TypeError, match="non-object"):
        _hash(np.array([object()], dtype=object))


@pytest.mark.parametrize("nested", [False, True])
def test_structured_hash_ignores_alignment_padding(monkeypatch, nested):
    inner = np.dtype([("a", "u1"), ("b", "f8")], align=True)
    dtype = (
        np.dtype([("tag", "u1"), ("records", inner, (2,))], align=True)
        if nested
        else inner
    )
    data = np.zeros(3, dtype=dtype)
    other = np.empty_like(data)
    # Change every storage byte, then restore only named values. All outer
    # and nested padding remains different from the zero-initialized source.
    other.view("u1")[:] = 13
    if nested:
        data["tag"] = 7
        data["records"]["a"] = 2
        data["records"]["b"] = [0.25, 0.5]
        other["tag"] = data["tag"]
        other["records"]["a"] = data["records"]["a"]
        other["records"]["b"] = data["records"]["b"]
    else:
        data["a"] = 2
        data["b"] = [0.25, 0.5, 0.75]
        other["a"] = data["a"]
        other["b"] = data["b"]
    assert np.array_equal(data, other)
    assert not np.array_equal(data.view("u1"), other.view("u1"))
    expected = _hash(data)
    assert _hash(other) == expected
    assert _hash(other.astype(dtype.newbyteorder(">"))) == expected
    monkeypatch.setattr(hashes, "_HASH_CHUNK_BYTES", dtype.itemsize)
    assert _hash(other[::-1][::-1]) == expected


def test_legacy_hash_inventory_is_readable_but_requires_refresh():
    manifest = {
        "extension_content_hashes": {"noise_levels": "a" * 32},
        "base_extension_seed_modes": {"noise_levels": 0},
        "storage_fingerprint": "unchanged",
    }
    assert "shapes and canonical dtypes" in (
        hashes.analyzer_recompute_unverifiable_reason(manifest)
    )
    assert hashes.analyzer_inventory_refresh_needed(
        manifest, "unchanged", reclaimed=False
    )
    # Retain an audit for deliberately reclaimed storage, including legacy
    # inventories; an algorithm upgrade must not erase that audit.
    assert not hashes.analyzer_inventory_refresh_needed(
        manifest, None, reclaimed=True
    )
    assert not hashes.analyzer_content_hash_is_current("a" * 64)
    assert hashes.analyzer_content_hash_is_current(_hash(np.arange(3)))


@pytest.mark.parametrize(
    "dtype",
    [np.dtype("float32"), np.dtype([("a", "u1"), ("b", "i8")], align=True)],
    ids=["float", "structured-padding"],
)
def test_production_role_hash_keeps_memmap_allocation_bounded(tmp_path, dtype):
    import tracemalloc

    path = tmp_path / "waveforms.npy"
    volume = 32 * 1024**2
    data = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=(volume // dtype.itemsize,)
    )
    if dtype.names:
        data["a"] = 2
        data["b"] = 17
    else:
        data[:] = 0.012345
    data.flush()
    tracemalloc.start()
    try:
        result = hashes.analyzer_role_hashes(_Analyzer(data))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert hashes.analyzer_content_hash_is_current(result["display"])
    # The old rounding + tobytes path allocated twice the complete volume.
    # Allow multiple chunk buffers and fixed overhead, never a volume copy.
    assert peak < 8 * hashes._HASH_CHUNK_BYTES, peak
    assert peak < volume / 4, peak
