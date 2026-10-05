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
def sg_config():
    from spyglass.settings import sg_config

    return sg_config


@pytest.fixture(scope="module")
def SpyglassConfig():
    from spyglass.settings import SpyglassConfig

    return SpyglassConfig


@pytest.fixture(scope="module")
def nwb_mod():
    import spyglass.utils.nwb_helper_fn as mod

    return mod


def _fake_vm(available_gb, total_gb=32.0):
    """Return a psutil.virtual_memory()-like object."""
    vm = MagicMock()
    vm.available = int(available_gb * 1e9)
    vm.total = int(total_gb * 1e9)
    return vm


def _fake_proc(num_fds):
    """Return a psutil.Process()-like object reporting *num_fds* descriptors.

    *num_fds* may be an int, or a list consumed one value per call to mimic
    the count dropping as files are closed.
    """
    proc = MagicMock()
    if isinstance(num_fds, list):
        proc.num_fds = MagicMock(side_effect=num_fds)
    else:
        proc.num_fds = MagicMock(return_value=num_fds)
    return proc


def _make_io():
    io = MagicMock()
    io.close = MagicMock()
    return io


def _warn_texts(mock_warn):
    """Return the messages passed to a patched ``_warn_msg``.

    Warnings are asserted through ``_warn_msg`` rather than log capture
    because it routes to debug in test mode. See BaseMixin._warn_msg.
    """
    return [call.args[0] for call in mock_warn.call_args_list]


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
    """LRU file is evicted when process descriptors reach the fd budget."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())

    # 1020 descriptors against a 0.8 × 1024 = 819 budget
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc([1020, 800])),
    ):
        cache["/b.nwb"] = (io_b, MagicMock())

    io_a.close.assert_called_once()
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_no_eviction_when_fd_ok(NWBFileCache):
    """No eviction when the process is well within the OS fd limit."""
    cache = NWBFileCache()
    io_a = _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 4096)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())
        cache["/b.nwb"] = (_make_io(), MagicMock())

    assert len(cache) == 2
    io_a.close.assert_not_called()


def test_fd_eviction_counts_non_cache_fds(NWBFileCache, sg_config):
    """Descriptors held outside the cache consume the same budget."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())
        time.sleep(0.01)
        cache["/b.nwb"] = (io_b, MagicMock())

    # Only two cache entries, but 900 process descriptors against a budget of
    # 0.8 × 1024 = 819. Counting cache entries alone would not evict here.
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc([900, 700])),
        patch.dict(sg_config._nwb_cache, {"max_file_fraction": 0.8}),
    ):
        cache["/c.nwb"] = (_make_io(), MagicMock())

    io_a.close.assert_called_once()  # LRU of the two
    io_b.close.assert_not_called()
    assert "/a.nwb" not in cache
    assert len(cache) == 2


def test_fd_count_falls_back_to_cache_size(NWBFileCache):
    """Where num_fds is unavailable, the cache's own size is the count."""
    cache = NWBFileCache()
    io_a = _make_io()
    no_num_fds = MagicMock()
    no_num_fds.num_fds = MagicMock(side_effect=AttributeError)

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=no_num_fds),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())

    # Soft limit of 1 → budget 0.8 → the one cached entry exceeds it
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1, 1024)),
        patch("psutil.Process", return_value=no_num_fds),
    ):
        cache["/b.nwb"] = (_make_io(), MagicMock())

    io_a.close.assert_called_once()
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_fd_check_skipped_without_resource(NWBFileCache, nwb_mod):
    """Without `resource`, only the RAM check governs eviction."""
    cache = NWBFileCache()
    io_a = _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("psutil.Process", return_value=_fake_proc(10**6)),
        patch.object(nwb_mod, "resource", None),
    ):
        cache["/a.nwb"] = (io_a, MagicMock())
        cache["/b.nwb"] = (_make_io(), MagicMock())

    assert len(cache) == 2
    io_a.close.assert_not_called()


def test_warns_once_when_fd_pressure_outlives_cache(NWBFileCache):
    """An empty cache under fd pressure warns once instead of looping."""
    cache = NWBFileCache()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(1020)),
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        cache["/a.nwb"] = (_make_io(), MagicMock())
        cache["/b.nwb"] = (_make_io(), MagicMock())
        warnings = _warn_texts(mock_warn)

    assert len(warnings) == 1  # one warning despite two inserts
    assert "ulimit" in warnings[0]
    assert len(cache) == 1  # /a evicted on /b's insert, /b still added


# ── threshold settings ────────────────────────────────────────────────────────


def test_default_max_file_fraction(sg_config):
    """The default leaves descriptor headroom for non-cache handles."""
    assert sg_config.nwb_max_file_fraction == 0.8


def test_config_overrides_defaults(SpyglassConfig):
    """Supplied keys win; unsupplied keys keep their default."""
    ret = SpyglassConfig._validated_nwb_cache(
        {"min_free_gb": 8.0, "min_free_pct": 0.2}
    )
    assert ret["min_free_gb"] == 8.0
    assert ret["min_free_pct"] == 0.2
    assert ret["max_file_fraction"] == 0.8  # untouched


def test_config_coerces_to_float(SpyglassConfig):
    """Integers from JSON become floats."""
    ret = SpyglassConfig._validated_nwb_cache({"min_free_gb": 4})
    assert isinstance(ret["min_free_gb"], float)


def test_cache_reads_live_config(NWBFileCache, sg_config):
    """The cache honors thresholds changed after it was built."""
    cache = NWBFileCache()
    with patch.dict(sg_config._nwb_cache, {"min_free_gb": 1e6}):
        assert not cache._free_ram_ok()  # 1M GB free RAM is never available
    assert cache._free_ram_ok()


@pytest.mark.parametrize(
    "bad",
    [
        {"max_file_fraction": 0.0},
        {"max_file_fraction": 1.5},
        {"min_free_gb": -1.0},
        {"min_free_pct": 1.5},
    ],
)
def test_config_rejects_out_of_range(SpyglassConfig, bad):
    with pytest.raises(ValueError):
        SpyglassConfig._validated_nwb_cache(bad)


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


def test_release_clears_repeated_holds(NWBFileCache):
    """One release clears every hold, so repeat fetches need one close."""
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (_make_io(), MagicMock())

    cache.acquire("/a.nwb")  # as two fetch_nwb() calls would
    cache.acquire("/a.nwb")
    assert cache._cache["/a.nwb"][3] == 2

    cache.release("/a.nwb")  # one close_nwb() releases both
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
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        cache["/b.nwb"] = (_make_io(), MagicMock())
        assert any("active" in w for w in _warn_texts(mock_warn))


def test_close_all_warns_on_active(NWBFileCache):
    """close_all emits a warning when files have outstanding holds."""
    cache = NWBFileCache()
    io_a = _make_io()

    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (io_a, MagicMock())

    cache.acquire("/a.nwb")

    with patch.object(NWBFileCache, "_warn_msg") as mock_warn:
        cache.close_all()
        assert any(
            "active" in w.lower() or "hold" in w.lower()
            for w in _warn_texts(mock_warn)
        )

    io_a.close.assert_called_once()
    assert len(cache) == 0


# ── held-eviction warning volume ──────────────────────────────────────────────


def test_held_eviction_warns_once(NWBFileCache):
    """Repeat evictions of held files are counted, not re-warned."""
    cache = NWBFileCache()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        for path in ("/a.nwb", "/b.nwb"):
            cache[path] = (_make_io(), MagicMock())
            cache.acquire(path)

    # Each insert finds only held files to evict. RAM reads low, then OK, so
    # one file is evicted per insert and the cache never empties.
    low_then_ok = [_fake_vm(0.5), _fake_vm(16)] * 3
    with (
        patch("psutil.virtual_memory", side_effect=low_then_ok),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        for path in ("/c.nwb", "/d.nwb", "/e.nwb"):
            cache[path] = (_make_io(), MagicMock())
            cache.acquire(path)
        warnings = _warn_texts(mock_warn)

    assert len(warnings) == 1  # one warning for three evictions
    assert cache._held_evictions == 3


def test_close_all_reports_held_eviction_count(NWBFileCache):
    """close_all logs the running total of held evictions, then resets."""
    cache = NWBFileCache()
    cache._held_evictions = 2  # stands in for earlier memory pressure
    cache._warned.add("held_eviction")

    with patch.object(NWBFileCache, "_warn_msg") as mock_warn:
        cache.close_all()
        warnings = _warn_texts(mock_warn)

    assert any("2 NWB file(s)" in w for w in warnings)
    assert cache._held_evictions == 0
    assert "held_eviction" not in cache._warned  # next batch warns again


def test_no_held_summary_when_clean(NWBFileCache):
    """No summary when every evicted file had been released."""
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = (_make_io(), MagicMock())

    with patch.object(NWBFileCache, "_warn_msg") as mock_warn:
        cache.close_all()
        mock_warn.assert_not_called()
