"""Unit tests for NWBFileCache LRU memory management."""

import resource
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


# ── file descriptor limit eviction ───────────────────────────────────────────


def test_eviction_on_fd_limit(NWBFileCache):
    """LRU file is evicted when cache size reaches the OS fd soft limit."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    # Soft limit of 10 fds; fraction 0.99 → threshold is 9.9, so 10 entries
    # would exceed it. Add two files: first with plenty of memory and a limit
    # that allows it, then tighten the limit so the second insert evicts the first.
    fake_limits = (10, 1024)  # (soft, hard)

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=fake_limits),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())

    # Drop soft limit to 1 so one open file already exceeds the fraction
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1, 1024)),
    ):
        cache["/b.nwb"] = (io_b, MagicMock())

    io_a.close.assert_called_once()
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_no_eviction_when_fd_ok(NWBFileCache):
    """No eviction when cache is well within the OS fd limit."""
    cache = NWBFileCache()
    io_a = _make_io()
    fake_limits = (1024, 4096)

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=fake_limits),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())
        cache["/b.nwb"] = (_make_io(), MagicMock())

    assert len(cache) == 2
    io_a.close.assert_not_called()


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


def test_configure_max_file_fraction(configure_nwb_cache):
    import spyglass.utils.nwb_helper_fn as mod

    original = mod._NWB_CACHE_MAX_FILE_FRACTION
    try:
        configure_nwb_cache(max_file_fraction=0.5)
        assert mod._NWB_CACHE_MAX_FILE_FRACTION == 0.5
    finally:
        mod._NWB_CACHE_MAX_FILE_FRACTION = original


def test_configure_max_file_fraction_invalid(configure_nwb_cache):
    import pytest

    with pytest.raises(ValueError):
        configure_nwb_cache(max_file_fraction=0.0)
    with pytest.raises(ValueError):
        configure_nwb_cache(max_file_fraction=1.5)


# ── hybrid eviction (idle time + ref count) ───────────────────────────────────


def test_acquire_increments_refcount(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (_make_io(), MagicMock())
    assert cache._cache["/a.nwb"][3] == 0
    cache.acquire("/a.nwb")
    assert cache._cache["/a.nwb"][3] == 1
    cache.release("/a.nwb")
    assert cache._cache["/a.nwb"][3] == 0


def test_release_floors_at_zero(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (_make_io(), MagicMock())
    cache.release("/a.nwb")  # release without prior acquire
    assert cache._cache["/a.nwb"][3] == 0


def test_released_evicted_before_active(NWBFileCache):
    """A released (refcount=0) file is evicted before an active (refcount>0) one."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 4096)),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())
        cache["/b.nwb"] = (io_b, MagicMock())

    cache.acquire("/b.nwb")  # protect /b; /a stays at refcount=0

    mem_responses = [_fake_vm(0.5), _fake_vm(16)]
    with (
        patch("psutil.virtual_memory", side_effect=mem_responses),
        patch("resource.getrlimit", return_value=(1024, 4096)),
    ):
        cache["/c.nwb"] = (_make_io(), MagicMock())

    io_a.close.assert_called_once()  # /a evicted (refcount=0)
    io_b.close.assert_not_called()  # /b protected (refcount=1)
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_tier3_eviction_warns(NWBFileCache):
    """A warning is emitted when the only eviction candidate is an active file."""
    cache = NWBFileCache()
    io_a = _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 4096)),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())

    cache.acquire("/a.nwb")  # all files are now active

    mem_responses = [_fake_vm(0.5), _fake_vm(16)]
    with (
        patch("psutil.virtual_memory", side_effect=mem_responses),
        patch("resource.getrlimit", return_value=(1024, 4096)),
        patch("spyglass.utils.nwb_helper_fn.logger") as mock_logger,
    ):
        cache["/b.nwb"] = (_make_io(), MagicMock())
        warning_calls = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert any("active" in w for w in warning_calls)


def test_close_all_warns_on_active(NWBFileCache):
    """close_all emits a warning when files have outstanding holds."""
    cache = NWBFileCache()
    io_a = _make_io()

    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io_a, MagicMock())

    cache.acquire("/a.nwb")

    with patch("spyglass.utils.nwb_helper_fn.logger") as mock_logger:
        cache.close_all()
        warning_calls = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert any(
            "active" in w.lower() or "hold" in w.lower() for w in warning_calls
        )

    io_a.close.assert_called_once()
    assert len(cache) == 0
