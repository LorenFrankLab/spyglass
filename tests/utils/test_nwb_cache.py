"""Unit tests for NWBFileCache LRU memory management."""

import time
from unittest.mock import MagicMock, patch

import pytest

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def NWBFileCache():
    from spyglass.utils.nwb_helper_fn import NWBFileCache

    return NWBFileCache


@pytest.fixture(scope="module")
def configure_nwb_cache():
    from spyglass.utils.nwb_helper_fn import configure_nwb_cache

    return configure_nwb_cache


def _fake_vm(available_gb, total_gb=32.0):
    """Return a psutil.virtual_memory()-like object."""
    vm = MagicMock()
    vm.available = int(available_gb * 1e9)
    vm.total = int(total_gb * 1e9)
    return vm


def _make_io():
    io = MagicMock()
    io.close = MagicMock()
    return io


# ── basic dict interface ───────────────────────────────────────────────────────


def test_setitem_getitem(NWBFileCache):
    cache = NWBFileCache()
    io, nwb = _make_io(), MagicMock()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io, nwb)
    assert "/a.nwb" in cache
    assert cache["/a.nwb"] == (io, nwb)


def test_get_returns_default_for_missing(NWBFileCache):
    cache = NWBFileCache()
    assert cache.get("/missing.nwb") == (None, None)
    assert cache.get("/missing.nwb", ("x", "y")) == ("x", "y")


def test_len(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (_make_io(), MagicMock())
        cache["/b.nwb"] = (_make_io(), MagicMock())
    assert len(cache) == 2


def test_close_all(NWBFileCache):
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io_a, MagicMock())
        cache["/b.nwb"] = (io_b, MagicMock())
    cache.close_all()
    io_a.close.assert_called_once()
    io_b.close.assert_called_once()
    assert len(cache) == 0


# ── last-used tracking ────────────────────────────────────────────────────────


def test_get_updates_last_used(NWBFileCache):
    """Accessing via .get() should bump the last-used timestamp."""
    cache = NWBFileCache()
    io, nwb = _make_io(), MagicMock()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io, nwb)
    t_before = cache._cache["/a.nwb"][2]
    time.sleep(0.01)
    cache.get("/a.nwb")
    t_after = cache._cache["/a.nwb"][2]
    assert t_after > t_before


def test_getitem_updates_last_used(NWBFileCache):
    cache = NWBFileCache()
    io, nwb = _make_io(), MagicMock()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io, nwb)
    t_before = cache._cache["/a.nwb"][2]
    time.sleep(0.01)
    _ = cache["/a.nwb"]
    t_after = cache._cache["/a.nwb"][2]
    assert t_after > t_before


# ── eviction ──────────────────────────────────────────────────────────────────


def test_no_eviction_when_memory_ok(NWBFileCache):
    """No files evicted when there is plenty of free RAM."""
    cache = NWBFileCache()
    io_a = _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io_a, MagicMock())
        cache["/b.nwb"] = (_make_io(), MagicMock())
    assert len(cache) == 2
    io_a.close.assert_not_called()


def test_eviction_on_low_memory(NWBFileCache):
    """When free RAM is below threshold, LRU file is evicted before adding."""
    cache = NWBFileCache()
    io_a = _make_io()
    # Add first file with plenty of RAM
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io_a, MagicMock())

    # Simulate low available RAM for the second insert
    with patch("psutil.virtual_memory", return_value=_fake_vm(0.5)):
        cache["/b.nwb"] = (_make_io(), MagicMock())

    # /a.nwb was LRU and should have been evicted
    io_a.close.assert_called_once()
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_lru_eviction_order(NWBFileCache):
    """The least-recently-used entry is the one evicted."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io_a, MagicMock())
        time.sleep(0.01)
        cache["/b.nwb"] = (io_b, MagicMock())
        # Touch /a so /b is now the LRU
        time.sleep(0.01)
        cache.get("/a.nwb")

    # First memory check reports low; after /b is evicted, report OK so the
    # loop stops before also evicting /a.
    mem_responses = [_fake_vm(0.5), _fake_vm(16)]
    with patch("psutil.virtual_memory", side_effect=mem_responses):
        cache["/c.nwb"] = (_make_io(), MagicMock())

    io_b.close.assert_called_once()
    io_a.close.assert_not_called()
    assert "/b.nwb" not in cache
    assert "/a.nwb" in cache
    assert "/c.nwb" in cache


# ── configure_nwb_cache ───────────────────────────────────────────────────────


def test_configure_updates_thresholds(configure_nwb_cache):
    import spyglass.utils.nwb_helper_fn as mod

    original_gb = mod._NWB_CACHE_MIN_FREE_GB
    original_pct = mod._NWB_CACHE_MIN_FREE_PCT
    try:
        configure_nwb_cache(min_free_gb=8.0, min_free_pct=0.2)
        assert mod._NWB_CACHE_MIN_FREE_GB == 8.0
        assert mod._NWB_CACHE_MIN_FREE_PCT == 0.2
    finally:
        mod._NWB_CACHE_MIN_FREE_GB = original_gb
        mod._NWB_CACHE_MIN_FREE_PCT = original_pct


def test_configure_partial_update(configure_nwb_cache):
    import spyglass.utils.nwb_helper_fn as mod

    original_gb = mod._NWB_CACHE_MIN_FREE_GB
    try:
        configure_nwb_cache(min_free_gb=4.0)
        assert mod._NWB_CACHE_MIN_FREE_GB == 4.0
        # pct should be unchanged
        assert mod._NWB_CACHE_MIN_FREE_PCT == 0.1
    finally:
        mod._NWB_CACHE_MIN_FREE_GB = original_gb
