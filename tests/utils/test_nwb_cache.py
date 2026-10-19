"""Unit tests for NWBFileCache LRU memory management."""

import errno
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


def _opened(io=None, streamed=False):
    """Wrap a mock IO handle in the ``Opened`` record the cache stores."""
    from spyglass.utils.file_backends import Opened

    return Opened(io=io or _make_io(), nwbfile=MagicMock(), streamed=streamed)


def _add(cache, path, opened=None):
    """Insert in the order ``_open_nwb_file`` uses: make room, then add.

    ``__setitem__`` makes no room of its own, so a test that exercises either
    limit -- to show a file closes or to show none does -- has to call
    ``make_room()`` where the opener does. Plain setup inserts need not.
    """
    opened = opened if opened is not None else _opened()
    cache.make_room()
    cache[path] = opened
    return opened


def _write_nwb(path):
    """Write a small NWB file with a 10-sample TimeSeries. Return its path."""
    from datetime import datetime, timezone

    import numpy as np
    import pynwb

    nwbf = pynwb.NWBFile(
        session_description="nwb cache test",
        identifier=path.stem,
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    nwbf.add_acquisition(
        pynwb.TimeSeries(name="ts", data=np.arange(10.0), unit="m", rate=1.0)
    )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbf)
    return str(path)


def _warn_texts(mock_warn):
    """Return the messages passed to a patched ``_warn_msg``.

    Warnings are asserted through ``_warn_msg`` rather than log capture
    because it routes to debug in test mode. See BaseMixin._warn_msg.
    """
    return [call.args[0] for call in mock_warn.call_args_list]


# ── basic dict interface ───────────────────────────────────────────────────────


def test_setitem_getitem(NWBFileCache):
    cache = NWBFileCache()
    opened = _opened()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = opened
    assert "/a.nwb" in cache
    assert cache["/a.nwb"] is opened


def test_get_returns_default_for_missing(NWBFileCache):
    cache = NWBFileCache()
    assert cache.get("/missing.nwb") is None
    assert cache.get("/missing.nwb", ("x", "y")) == ("x", "y")


def test_len(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened()
        cache["/b.nwb"] = _opened()
    assert len(cache) == 2


def test_iter_and_delete(NWBFileCache):
    """Iterating yields keys and ``del`` drops an entry, as on a dict."""
    cache = NWBFileCache()
    io_a = _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)
        cache["/b.nwb"] = _opened()

    assert set(cache) == {"/a.nwb", "/b.nwb"}

    del cache["/a.nwb"]
    assert set(cache) == {"/b.nwb"}
    io_a.close.assert_not_called()  # del drops, it does not close


def test_close_all(NWBFileCache):
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)
        cache["/b.nwb"] = _opened(io_b)
    cache.close_all()
    io_a.close.assert_called_once()
    io_b.close.assert_called_once()
    assert len(cache) == 0


# ── last-used tracking ────────────────────────────────────────────────────────


def test_get_updates_last_used(NWBFileCache):
    """Accessing via .get() should bump the last-used timestamp."""
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened()
    t_before = cache._cache["/a.nwb"][1]
    time.sleep(0.01)
    cache.get("/a.nwb")
    t_after = cache._cache["/a.nwb"][1]
    assert t_after > t_before


def test_getitem_updates_last_used(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened()
    t_before = cache._cache["/a.nwb"][1]
    time.sleep(0.01)
    _ = cache["/a.nwb"]
    t_after = cache._cache["/a.nwb"][1]
    assert t_after > t_before


# ── eviction ──────────────────────────────────────────────────────────────────


def test_no_eviction_when_memory_ok(NWBFileCache):
    """No files evicted when there is plenty of free RAM."""
    cache = NWBFileCache()
    io_a = _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        _add(cache, "/a.nwb", _opened(io_a))
        _add(cache, "/b.nwb")
    assert len(cache) == 2
    io_a.close.assert_not_called()


def test_eviction_on_low_memory(NWBFileCache):
    """When free RAM is below threshold, LRU file is evicted before adding."""
    cache = NWBFileCache()
    io_a = _make_io()
    # Add first file with plenty of RAM
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)

    # Simulate low available RAM for the second open
    with patch("psutil.virtual_memory", return_value=_fake_vm(0.5)):
        _add(cache, "/b.nwb")

    # /a.nwb was LRU and should have been evicted
    io_a.close.assert_called_once()
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_low_ram_closes_one_unheld_per_check(NWBFileCache):
    """Low RAM closes a single unheld file, not the whole cache.

    Closing an NWB file returns almost nothing to the OS, so a loop that ran
    until free RAM recovered would drain the cache on every open.
    """
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)
        time.sleep(0.01)
        cache["/b.nwb"] = _opened(io_b)

    # RAM stays low for the whole open; only the LRU unheld file closes
    with patch(
        "psutil.virtual_memory", return_value=_fake_vm(40, total_gb=512)
    ):
        _add(cache, "/c.nwb")

    io_a.close.assert_called_once()  # LRU of the unheld
    io_b.close.assert_not_called()  # cache not drained
    assert set(cache) == {"/b.nwb", "/c.nwb"}


def test_low_ram_keeps_held_file_readable(NWBFileCache, nwb_mod, tmp_path):
    """Low RAM never closes a held file, so its lazy reads still work.

    Mirrors the reviewer's repro: 40 GB free of 512 GB total is under the 10%
    default, and the first file is held when the second opens.
    """
    cache = NWBFileCache()
    path_a = _write_nwb(tmp_path / "a_.nwb")
    path_b = _write_nwb(tmp_path / "b_.nwb")

    try:
        with patch.dict(vars(nwb_mod), {"__open_nwb_files": cache}):
            lazy = nwb_mod.get_nwb_file(path_a).acquisition["ts"].data
            nwb_mod._acquire_nwb_file(path_a)
            with patch(
                "psutil.virtual_memory",
                return_value=_fake_vm(40, total_gb=512),
            ):
                nwb_mod.get_nwb_file(path_b)

        assert path_a in cache  # held, so low RAM left it open
        assert lazy[:3].tolist() == [0.0, 1.0, 2.0]
    finally:
        cache.close_all()


def test_lru_eviction_order(NWBFileCache):
    """The least-recently-used entry is the one evicted."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)
        time.sleep(0.01)
        cache["/b.nwb"] = _opened(io_b)
        # Touch /a so /b is now the LRU
        time.sleep(0.01)
        cache.get("/a.nwb")

    # First memory check reports low; after /b is evicted, report OK so the
    # loop stops before also evicting /a.
    mem_responses = [_fake_vm(0.5), _fake_vm(16)]
    with patch("psutil.virtual_memory", side_effect=mem_responses):
        _add(cache, "/c.nwb")

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
        cache["/a.nwb"] = _opened(io_a)

    # 1020 descriptors against a 0.8 × 1024 = 819 budget
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc([1020, 800])),
    ):
        _add(cache, "/b.nwb", _opened(io_b))

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
        _add(cache, "/a.nwb", _opened(io_a))
        _add(cache, "/b.nwb")

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
        cache["/a.nwb"] = _opened(io_a)
        time.sleep(0.01)
        cache["/b.nwb"] = _opened(io_b)

    # Only two cache entries, but 900 process descriptors against a budget of
    # 0.8 × 1024 = 819. Counting cache entries alone would not evict here.
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc([900, 700])),
        patch.dict(sg_config._nwb_cache, {"max_file_fraction": 0.8}),
    ):
        _add(cache, "/c.nwb")

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
        cache["/a.nwb"] = _opened(io_a)

    # Soft limit of 1 → budget 0.8 → the one cached entry exceeds it
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1, 1024)),
        patch("psutil.Process", return_value=no_num_fds),
    ):
        _add(cache, "/b.nwb")

    io_a.close.assert_called_once()
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_emfile_from_num_fds_counts_as_over_budget(NWBFileCache):
    """A descriptor-exhaustion error from num_fds means "no room", not "unknown".

    psutil reads /proc, which needs a descriptor. Falling back to the cache's
    own size here would report headroom at the one moment there is none.
    """
    cache = NWBFileCache()
    exhausted = MagicMock()
    exhausted.num_fds = MagicMock(
        side_effect=OSError(errno.EMFILE, "Too many open files")
    )

    with (
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=exhausted),
    ):
        assert not cache._num_open_ok()


def test_fd_check_skipped_without_resource(NWBFileCache, nwb_mod):
    """Without `resource`, only the RAM check governs eviction."""
    cache = NWBFileCache()
    io_a = _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("psutil.Process", return_value=_fake_proc(10**6)),
        patch.object(nwb_mod, "resource", None),
    ):
        _add(cache, "/a.nwb", _opened(io_a))
        _add(cache, "/b.nwb")

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
        _add(cache, "/a.nwb")
        _add(cache, "/b.nwb")
        warnings = _warn_texts(mock_warn)

    assert len(warnings) == 1  # one warning despite two opens
    assert "ulimit" in warnings[0]
    assert len(cache) == 1  # /a evicted on /b's open, /b still added


# ── room made before the open ─────────────────────────────────────────────────


def test_room_made_before_open(NWBFileCache, nwb_mod):
    """At the descriptor budget, a cached file closes before the next open.

    Closing after the open is too late: the open is what raises ``EMFILE``.
    """
    cache = NWBFileCache()
    io_a = _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)

    def open_asserting_room_exists(path):
        io_a.close.assert_called_once()  # /a closed before this open ran
        return _opened()

    backend = MagicMock()
    backend.open.side_effect = open_asserting_room_exists

    with (
        patch.dict(vars(nwb_mod), {"__open_nwb_files": cache}),
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc([1020, 800, 800])),
    ):
        nwb_mod._open_nwb_file("/b.nwb", source=backend)

    backend.open.assert_called_once()
    assert set(cache) == {"/b.nwb"}


def test_one_open_closes_one_unheld_file(NWBFileCache, nwb_mod):
    """Sustained low RAM closes one unheld file per open, not two.

    The limits used to be checked both before the open and again on insert, so
    a single open closed two files where the policy allows one.
    """
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)
        time.sleep(0.01)
        cache["/b.nwb"] = _opened(io_b)

    backend = MagicMock()
    backend.open.side_effect = lambda path: _opened()

    # RAM stays low throughout, so a second check would close /b as well
    with (
        patch.dict(vars(nwb_mod), {"__open_nwb_files": cache}),
        patch("psutil.virtual_memory", return_value=_fake_vm(40, total_gb=512)),
        patch("resource.getrlimit", return_value=(1024, 4096)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        nwb_mod._open_nwb_file("/c.nwb", source=backend)

    io_a.close.assert_called_once()  # LRU of the unheld
    io_b.close.assert_not_called()
    assert set(cache) == {"/b.nwb", "/c.nwb"}


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
    # Fake the RAM for both asserts: the second one reads the default
    # thresholds, which a busy host can genuinely fail.
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
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


def test_acquire_and_release_set_held(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened()
    assert cache._cache["/a.nwb"][2] is False
    cache.acquire("/a.nwb")
    assert cache._cache["/a.nwb"][2] is True
    cache.release("/a.nwb")
    assert cache._cache["/a.nwb"][2] is False


def test_release_without_acquire(NWBFileCache):
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened()
    cache.release("/a.nwb")  # release without prior acquire
    assert cache._cache["/a.nwb"][2] is False


def test_release_clears_repeated_holds(NWBFileCache):
    """Repeat fetches leave one hold, which one close releases."""
    cache = NWBFileCache()
    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened()

    cache.acquire("/a.nwb")  # as two fetch_nwb() calls would
    cache.acquire("/a.nwb")

    cache.release("/a.nwb")  # one close_nwb() is enough
    assert cache._cache["/a.nwb"][2] is False


def test_unheld_closed_before_held(NWBFileCache):
    """An unheld file is closed before a held one."""
    cache = NWBFileCache()
    io_a, io_b = _make_io(), _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 4096)),
    ):
        cache["/a.nwb"] = _opened(io_a)
        cache["/b.nwb"] = _opened(io_b)

    cache.acquire("/b.nwb")  # protect /b; /a stays unheld

    mem_responses = [_fake_vm(0.5), _fake_vm(16)]
    with (
        patch("psutil.virtual_memory", side_effect=mem_responses),
        patch("resource.getrlimit", return_value=(1024, 4096)),
    ):
        _add(cache, "/c.nwb")

    io_a.close.assert_called_once()  # /a closed, it was unheld
    io_b.close.assert_not_called()  # /b protected by its hold
    assert "/a.nwb" not in cache
    assert "/b.nwb" in cache


def test_held_close_warns_under_fd_pressure(NWBFileCache):
    """Descriptor pressure closes a held file as a last resort, and warns.

    Only this path may close a held file; low RAM never does.
    """
    cache = NWBFileCache()
    io_a = _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        cache["/a.nwb"] = _opened(io_a)

    cache.acquire("/a.nwb")  # every file is now held

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        # over budget, still over the held-close fraction, then under
        patch("psutil.Process", return_value=_fake_proc([1020, 1020, 800])),
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        _add(cache, "/b.nwb")
        held = [w for w in _warn_texts(mock_warn) if "still held" in w]
        assert held
        # A populate() job holds files from internal code, so name the two
        # actions its caller does have: release between calls, or raise the
        # limit.
        assert "close_nwb_files() between populate()" in held[0]
        assert "ulimit -n" in held[0]

    io_a.close.assert_called_once()
    assert cache._held_evictions == 1


def test_held_file_kept_while_fd_headroom_remains(NWBFileCache):
    """Between the two fractions, a held file survives descriptor pressure.

    900 of a 1024 limit is over the 0.8 budget but under the 0.95 held-close
    fraction, so the next open still succeeds and closing the held file would
    break its reads for nothing.
    """
    cache = NWBFileCache()
    io_a = _make_io()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        cache["/a.nwb"] = _opened(io_a)

    cache.acquire("/a.nwb")  # every file is now held

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(900)),
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        _add(cache, "/b.nwb")

    io_a.close.assert_not_called()
    mock_warn.assert_not_called()
    assert "/a.nwb" in cache
    assert cache._held_evictions == 0


def test_held_file_readable_under_outside_fd_pressure(
    NWBFileCache, nwb_mod, tmp_path
):
    """A held file's lazy read survives fd pressure from outside the cache.

    The reviewer's repro: descriptors consumed elsewhere push the process over
    the budget while headroom to the real limit remains.
    """
    cache = NWBFileCache()
    path_a = _write_nwb(tmp_path / "a_.nwb")
    path_b = _write_nwb(tmp_path / "b_.nwb")

    try:
        with patch.dict(vars(nwb_mod), {"__open_nwb_files": cache}):
            lazy = nwb_mod.get_nwb_file(path_a).acquisition["ts"].data
            nwb_mod._acquire_nwb_file(path_a)
            with (
                patch("psutil.virtual_memory", return_value=_fake_vm(16)),
                patch("resource.getrlimit", return_value=(1024, 1024)),
                patch("psutil.Process", return_value=_fake_proc(900)),
            ):
                nwb_mod.get_nwb_file(path_b)

        assert path_a in cache  # held, and headroom was left
        assert lazy[:3].tolist() == [0.0, 1.0, 2.0]
    finally:
        cache.close_all()


def test_low_ram_never_closes_held_file(NWBFileCache):
    """With only held files, low RAM closes nothing and does not warn."""
    cache = NWBFileCache()
    io_a = _make_io()

    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)

    cache.acquire("/a.nwb")

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(40, total_gb=512)),
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        _add(cache, "/b.nwb")

    io_a.close.assert_not_called()
    mock_warn.assert_not_called()
    assert cache._held_evictions == 0
    assert set(cache) == {"/a.nwb", "/b.nwb"}


def test_close_all_does_not_warn_on_held(NWBFileCache):
    """close_all closes held files silently.

    Reaching it means the caller said they were finished, so warning that
    their reads will fail is wrong -- and a per-session loop would warn on
    every pass.
    """
    cache = NWBFileCache()
    io_a = _make_io()

    with patch("psutil.virtual_memory", return_value=_fake_vm(16)):
        cache["/a.nwb"] = _opened(io_a)

    cache.acquire("/a.nwb")

    with patch.object(NWBFileCache, "_warn_msg") as mock_warn:
        cache.close_all()
        mock_warn.assert_not_called()

    io_a.close.assert_called_once()
    assert len(cache) == 0


# ── held-eviction warning volume ──────────────────────────────────────────────


def test_held_eviction_warns_once(NWBFileCache):
    """Repeat closes of held files are counted, not re-warned."""
    cache = NWBFileCache()

    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(10)),
    ):
        for path in ("/a.nwb", "/b.nwb"):
            cache[path] = _opened()
            cache.acquire(path)

    # Each open finds only held files to close. Descriptors read over the
    # budget then under, so one file closes per open and the cache persists.
    # Per open: over budget, still over the held-close fraction, then under
    over_then_under = [1020, 1020, 800] * 3
    with (
        patch("psutil.virtual_memory", return_value=_fake_vm(16)),
        patch("resource.getrlimit", return_value=(1024, 1024)),
        patch("psutil.Process", return_value=_fake_proc(over_then_under)),
        patch.object(NWBFileCache, "_warn_msg") as mock_warn,
    ):
        for path in ("/c.nwb", "/d.nwb", "/e.nwb"):
            _add(cache, path)
            cache.acquire(path)
        warnings = _warn_texts(mock_warn)

    assert len(warnings) == 1  # one warning for three closes
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
        cache["/a.nwb"] = _opened()

    with patch.object(NWBFileCache, "_warn_msg") as mock_warn:
        cache.close_all()
        mock_warn.assert_not_called()
