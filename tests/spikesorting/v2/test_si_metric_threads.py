"""DB-free checks that concurrent SpikeInterface metric computes stay separate.

SpikeInterface keeps each metric's defaults in class-level dicts that a
compute updates in place, and ``warnings.catch_warnings`` swaps
process-wide state. Each test runs two threads, ``A`` and ``B``, on their
own tiny in-memory analyzers. Their computes are forced to overlap if the
code allows it: A blocks inside a SpikeInterface metric until B has either
reached its own metric or asked for Spyglass's SpikeInterface metric lock
(recorded by the instrumented lock below). Every wait has a timeout, so a
regression fails instead of hanging. No database is used.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

_TIMEOUT_S = 30.0

# SI 0.104.3 ``PresenceRatio.metric_params``
# (spikeinterface/metrics/quality/misc_metrics.py:137).
_PRISTINE_PRESENCE = {"bin_duration_s": 60, "mean_fr_ratio_thresh": 0.0}


class _SignallingLock:
    """Reentrant lock that records which threads have asked for it."""

    def __init__(self):
        self._lock = threading.RLock()
        self._guard = threading.Lock()
        self._requested: dict[str, threading.Event] = {}

    def requested(self, thread_name: str) -> threading.Event:
        """Event set once ``thread_name`` has asked for the lock."""
        with self._guard:
            return self._requested.setdefault(thread_name, threading.Event())

    def acquire(self, blocking=True, timeout=-1):
        self.requested(threading.current_thread().name).set()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc_info):
        self.release()


@pytest.fixture
def lock_requests(monkeypatch):
    """Swap Spyglass's SI metric lock for one that records requests."""
    from spyglass.spikesorting.v2._core import (
        si_metric_patches as _si_metric_patches,
    )

    lock = _SignallingLock()
    monkeypatch.setattr(_si_metric_patches, "SI_METRIC_STATE_LOCK", lock)
    return lock


@pytest.fixture(scope="module")
def analyzers():
    """Two independent in-memory analyzers, one per thread."""
    import spikeinterface.full as sf
    from spikeinterface.core import generate_ground_truth_recording

    built = {}
    for seed, name in enumerate(("A", "B")):
        recording, sorting = generate_ground_truth_recording(
            durations=[10.0],
            sampling_frequency=30_000.0,
            num_channels=2,
            num_units=2,
            seed=seed,
        )
        built[name] = sf.create_sorting_analyzer(
            sorting, recording, format="memory", sparse=False
        )
    return built


def _wait(event: threading.Event, what: str) -> None:
    """Wait for ``event``; fail the calling thread on timeout."""
    if not event.wait(_TIMEOUT_S):
        raise TimeoutError(f"timed out waiting for {what}")


def _wait_any(*events: threading.Event) -> None:
    """Wait until any of ``events`` is set, or give up after the timeout.

    Giving up is not a failure: it only means no overlap was possible.
    """
    deadline = time.monotonic() + _TIMEOUT_S
    while not any(event.is_set() for event in events):
        if time.monotonic() > deadline:
            return
        time.sleep(0.001)


def _run_threads(targets: dict) -> dict:
    """Run ``{name: callable}`` on named threads; ``{name: result or error}``."""
    outcome = {}

    def _call(name, target):
        try:
            outcome[name] = target()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            outcome[name] = exc

    threads = [
        threading.Thread(target=_call, args=(name, target), name=name)
        for name, target in targets.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_TIMEOUT_S * 3)
    assert not any(thread.is_alive() for thread in threads), "thread hung"
    return outcome


def test_concurrent_metric_computes_keep_their_own_defaults(
    analyzers, lock_requests, monkeypatch
):
    """Two overlapping isolated computes each apply only their own kwargs.

    A computes ``presence_ratio`` with ``bin_duration_s=0.5`` and B with
    SI's defaults; B starts while A is inside its compute (A's kwargs are
    then merged into SI's class defaults). Each extension must record its
    own params, and SI's defaults must be the untouched originals after.
    """
    import spikeinterface.metrics.quality.misc_metrics as mm
    from spikeinterface.metrics.quality import compute_quality_metrics

    from spyglass.spikesorting.v2._core.si_metric_patches import (
        isolated_si_metric_defaults,
    )

    defaults_before = mm.PresenceRatio.metric_params
    assert defaults_before == _PRISTINE_PRESENCE

    a_inside = threading.Event()
    a_done = threading.Event()
    b_inside = threading.Event()
    b_waiting = lock_requests.requested("B")
    compute_presence_ratios = mm.PresenceRatio.metric_function

    def presence_ratio(*args, **kwargs):
        if threading.current_thread().name == "A":
            a_inside.set()
            _wait_any(b_inside, b_waiting)
        else:
            b_inside.set()
            # B finishes last, as in the reported leak.
            _wait(a_done, "A to finish its compute")
        return compute_presence_ratios(*args, **kwargs)

    monkeypatch.setattr(mm.PresenceRatio, "metric_function", presence_ratio)

    def compute(name, metric_params):
        with isolated_si_metric_defaults():
            compute_quality_metrics(
                analyzers[name],
                metric_names=["presence_ratio"],
                metric_params=metric_params,
                skip_pc_metrics=True,
                delete_existing_metrics=True,
            )

    def run_a():
        compute("A", {"presence_ratio": {"bin_duration_s": 0.5}})
        a_done.set()

    def run_b():
        _wait(a_inside, "A to enter its compute")
        compute("B", None)

    outcome = _run_threads({"A": run_a, "B": run_b})

    assert outcome == {"A": None, "B": None}
    applied = {
        name: analyzers[name]
        .get_extension("quality_metrics")
        .params["metric_params"]["presence_ratio"]
        for name in ("A", "B")
    }
    assert applied["A"] == {"bin_duration_s": 0.5, "mean_fr_ratio_thresh": 0.0}
    assert applied["B"] == _PRISTINE_PRESENCE
    assert mm.PresenceRatio.metric_params is defaults_before
    assert mm.PresenceRatio.metric_params == _PRISTINE_PRESENCE


@pytest.mark.parametrize(
    "rules, a_raises",
    [
        # A's firing_rate failure is unreferenced by A's rules: A logs it.
        # B's rules reference firing_rate, which B computes fine.
        ({"A": {"num_spikes"}, "B": {"firing_rate"}}, False),
        # A's rules reference its failed firing_rate: A raises. B's rules
        # do not reference firing_rate.
        ({"A": {"firing_rate"}, "B": {"num_spikes"}}, True),
    ],
    ids=["unreferenced-failure", "referenced-failure"],
)
def test_concurrent_si_warning_capture_is_attributed_to_its_own_evaluation(
    analyzers, lock_requests, monkeypatch, caplog, rules, a_raises
):
    """A metric error is escalated or logged only by the thread it hit.

    SI's ``firing_rate`` raises in thread A (SI turns that into an "Error
    computing metric" warning and a NaN column) and succeeds in thread B.
    B enters its warning capture while A is inside its compute, and A's
    error is emitted before B's capture ends.
    """
    import spikeinterface.metrics.quality.misc_metrics as mm
    from spikeinterface.metrics.quality import compute_quality_metrics

    from spyglass.spikesorting.v2._curation.metrics import (
        escalate_si_metric_errors,
    )

    a_inside = threading.Event()
    a_computed = threading.Event()
    b_inside = threading.Event()
    b_done = threading.Event()
    b_waiting = lock_requests.requested("B")
    compute_firing_rates = mm.FiringRate.metric_function

    def firing_rate(*args, **kwargs):
        if threading.current_thread().name == "A":
            a_inside.set()
            _wait_any(b_inside, b_waiting)
            raise RuntimeError("thread A firing-rate failure")
        b_inside.set()
        # A's error is emitted before B's capture ends.
        _wait(a_computed, "A to finish its compute")
        return compute_firing_rates(*args, **kwargs)

    monkeypatch.setattr(mm.FiringRate, "metric_function", firing_rate)

    def compute(name):
        return compute_quality_metrics(
            analyzers[name],
            metric_names=["firing_rate", "num_spikes"],
            skip_pc_metrics=True,
            delete_existing_metrics=True,
        )

    def run_a():
        with escalate_si_metric_errors(rules["A"]):
            metrics = compute("A")
            a_computed.set()
            # Leave A's capture after B's when both can be open at once.
            _wait_any(b_done, b_waiting)
        return metrics

    def run_b():
        _wait(a_inside, "A to enter its compute")
        try:
            with escalate_si_metric_errors(rules["B"]):
                return compute("B")
        finally:
            b_done.set()

    with caplog.at_level("WARNING", logger="spyglass"):
        outcome = _run_threads({"A": run_a, "B": run_b})

    failure_logs = [
        record.threadName
        for record in caplog.records
        if "failed to compute metric 'firing_rate'" in record.getMessage()
    ]
    assert not isinstance(outcome["B"], BaseException), outcome["B"]
    assert np.isfinite(outcome["B"]["firing_rate"].astype(float)).all()
    if a_raises:
        assert isinstance(outcome["A"], ValueError), outcome["A"]
        assert "'firing_rate'" in str(outcome["A"])
        assert "thread A firing-rate failure" in str(outcome["A"])
        assert failure_logs == []
    else:
        assert not isinstance(outcome["A"], BaseException), outcome["A"]
        assert outcome["A"]["firing_rate"].isna().all()
        assert failure_logs == ["A"]
