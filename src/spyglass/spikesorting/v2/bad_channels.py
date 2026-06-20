"""Bad-channel detection (SpikeInterface ``coherence+psd``).

Two helpers:

- ``detect_bad_channels`` -- a thin, pure wrapper over
  ``spikeinterface.preprocessing.detect_bad_channels`` that pins only the
  detection method and returns the flagged channel ids plus the per-channel
  label map (``good`` / ``dead`` / ``noise`` / ``out``).
- ``suggest_bad_channels`` -- a suggest-then-confirm helper that loads a
  session's raw recording, band-pass filters it, runs detection **per full
  shank** (the coherence method is spatially local), and reports -- or, with
  ``write=True``, persists -- the quality-bad (``dead`` / ``noise``) channels
  onto ``Electrode.bad_channel``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: SpikeInterface / spyglass (``Electrode``,
``read_raw_nwb_recording``, ``spikeinterface_channel_ids``, ``logger``)
dependencies are imported lazily inside the functions. ``suggest_bad_channels``
does touch the DB at CALL time (an ``Electrode`` fetch, plus ``Electrode.update1``
when ``write=True``) via those lazy imports -- mirroring
``_recording_materialization.py``'s contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._enums import BadChannelLabel

__all__ = ["detect_bad_channels", "suggest_bad_channels"]

# We pin ONLY the method. Every threshold is left to SpikeInterface's own
# signature default (e.g. SI 0.104.3: dead -0.5, noisy 1.0, outside -0.75,
# psd_hf 0.02, seed None) so "ships SI defaults" is literally true and a future
# SI default change flows through. Do NOT hardcode threshold values here -- they
# would silently diverge from SI and masquerade as defaults.
BAD_CHANNEL_DETECT_DEFAULTS: dict[str, str] = {"method": "coherence+psd"}

# Labels that ``write=True`` persists to ``Electrode.bad_channel``. These are
# the *quality-bad* classes (a dead/flat contact, a noisy contact) -- safe to
# interpolate from neighbours or to remove. ``"out"`` (outside-brain) is
# deliberately excluded: the boolean flag cannot carry the label, and a
# persisted ``out`` would later be wrongly *interpolated* by downstream handling.
_WRITABLE_LABELS: tuple[BadChannelLabel, ...] = ("dead", "noise")


def detect_bad_channels(recording, **overrides) -> dict:
    """Detect bad channels, returning ids and per-channel labels.

    Thin wrapper over ``spikeinterface.preprocessing.detect_bad_channels``.
    Only ``method="coherence+psd"`` (the IBL coherence/PSD method) is pinned;
    every other knob falls through to SpikeInterface's signature default unless
    the caller overrides it. The method is force-pinned *after* the override
    merge, so a caller-supplied ``method=`` cannot silently replace it. ``None``
    overrides are **dropped** so a "leave at SI default" sentinel never reaches
    SI, where e.g. ``psd_hf_threshold=None`` would be invalid (SI expects
    ``0.02``).

    The recording must already be band-pass / high-pass filtered (the method
    assumes spike-band data) and carry µV-scaled traces (SI asserts
    ``recording.has_scaleable_traces()``; set channel gains/offsets if working
    with synthetic data). Thresholds are Neuropixels-derived -- recalibrate for
    other probe geometries (e.g. polymer) if over/under-flagging.

    Parameters
    ----------
    recording : si.BaseRecording
        A filtered, µV-scaled recording. For the spatially-local coherence
        method this should be a single physical shank.
    **overrides
        Any keyword argument of ``detect_bad_channels`` (e.g.
        ``dead_channel_threshold``, ``psd_hf_threshold``). ``None`` values are
        dropped; ``method`` is ignored (always ``"coherence+psd"``).

    Returns
    -------
    dict
        ``{"bad_channel_ids": [...], "labels": {channel_id: label}}`` where
        ``label`` is one of ``"good"`` / ``"dead"`` / ``"noise"`` / ``"out"``.
        The caller decides per label what to do (``out`` is not interpolatable).
    """
    import spikeinterface.preprocessing as sip

    params = {
        **BAD_CHANNEL_DETECT_DEFAULTS,
        **{k: v for k, v in overrides.items() if v is not None},
    }
    params["method"] = "coherence+psd"  # hard-pin; override cannot replace
    bad_ids, labels = sip.detect_bad_channels(recording, **params)
    return {
        "bad_channel_ids": [c for c in bad_ids],
        "labels": {
            c: str(label)
            for c, label in zip(recording.get_channel_ids(), labels)
        },
    }


def _shank_groups(
    electrodes: list[dict], nwb_file_name: str
) -> dict[tuple[str, int], list[int]]:
    """Group fetched ``Electrode`` rows by physical shank.

    Returns ``{(electrode_group_name, probe_shank): [electrode_id, ...]}``,
    ordered by ``(electrode_group_name, probe_shank)`` with each id list
    sorted, so iteration is deterministic.

    ``probe_shank`` rides into ``Electrode`` through the *nullable*
    ``Electrode -> Probe.Electrode -> Probe.Shank`` FK chain (``probe_shank``
    is the ``Probe.Shank`` key), so it is unset when an NWB lacks novela probe
    metadata. Per-shank detection is undefined without it (and ``int(None)``
    would otherwise crash), so this raises a clear ``ValueError`` naming the
    fix. Pure / DB-free -- it operates only on the already-fetched rows.
    """
    no_shank = sorted(
        int(r["electrode_id"]) for r in electrodes if r["probe_shank"] is None
    )
    if no_shank:
        raise ValueError(
            f"suggest_bad_channels: {len(no_shank)} electrode(s) in "
            f"{nwb_file_name!r} have no probe_shank (the nullable "
            "Probe.Electrode metadata is unset), so per-shank detection "
            f"cannot run. First few electrode_ids: {no_shank[:5]}. Populate "
            "the probe metadata, or restrict electrode_group_names to groups "
            "that have it."
        )
    groups: dict[tuple[str, int], list[int]] = {}
    for row in electrodes:
        key = (str(row["electrode_group_name"]), int(row["probe_shank"]))
        groups.setdefault(key, []).append(int(row["electrode_id"]))
    return {k: sorted(v) for k, v in sorted(groups.items())}


def _persist_quality_bad(nwb_file_name: str, report: list[dict]) -> int:
    """Additively persist a report's ``dead``/``noise`` electrodes.

    Sets ``Electrode.bad_channel='True'`` for every ``dead``/``noise`` entry in
    ``report``; ``out`` is never written (see :func:`suggest_bad_channels`).
    Additive and idempotent: an electrode already flagged ``'True'`` is skipped,
    so a curated flag is never cleared and a re-run is a no-op. All writes share
    one transaction, so a mid-loop DB error is all-or-nothing rather than
    leaving a half-written flag set. Returns the count of newly-written rows.
    """
    from spyglass.common.common_ephys import Electrode

    pks = [
        {
            "nwb_file_name": nwb_file_name,
            "electrode_group_name": e["electrode_group_name"],
            "electrode_id": int(e["electrode_id"]),
        }
        for e in report
        if e["label"] in _WRITABLE_LABELS
    ]
    n_written = 0
    if pks:
        with Electrode.connection.transaction:
            for pk in pks:
                if (Electrode & pk).fetch1("bad_channel") != "True":
                    Electrode.update1({**pk, "bad_channel": "True"})
                    n_written += 1
    return n_written


def suggest_bad_channels(
    nwb_file_name: str,
    *,
    electrode_group_names=None,
    bandpass: tuple[float, float] = (300.0, 6000.0),
    detection_params: dict | None = None,
    write: bool = False,
    report: list[dict] | None = None,
) -> list[dict]:
    """Suggest -- and optionally persist -- a session's bad channels.

    Suggest-then-confirm helper. By default (``write=False``) it changes
    nothing: it loads the session's raw recording, band-pass filters it, runs
    :func:`detect_bad_channels` **per full shank** (the coherence method is
    spatially local; mixing shanks corrupts it), maps SpikeInterface channel
    ids back to spyglass ``electrode_id`` s, and returns one report dict per
    flagged electrode carrying its label so you can review what kind of bad it
    is before persisting anything.

    To make the confirm step persist **exactly** what you reviewed, pass the
    reviewed report back: ``suggest_bad_channels(nwb, write=True,
    report=reviewed)`` skips detection entirely and writes from ``reviewed``.
    This matters because the ``coherence+psd`` method samples random chunks with
    SpikeInterface's ``seed=None`` default, so a *fresh* ``write=True`` call (no
    ``report=``) re-detects and may flag a slightly different set than the
    ``write=False`` review returned. (Passing a fixed
    ``detection_params={"seed": ...}`` to both calls makes the fresh path
    reproducible too.)

    With ``write=True`` it sets ``Electrode.bad_channel='True'`` for **only**
    the ``dead`` / ``noise`` electrodes. The write is **additive** -- it never
    clears an existing curated ``'True'`` and never writes ``'False'``. ``out``
    (outside-brain) channels are **report-only**: they are surfaced for
    awareness but **never** written, because ``Electrode.bad_channel`` is a
    boolean that cannot carry the ``out`` label and a persisted ``out`` would
    later be wrongly *interpolated* by downstream handling.

    Invariant: ``Electrode.bad_channel='True'`` means a *quality-bad*
    (``dead`` / ``noise`` class) channel that is safe to interpolate or remove;
    it must NOT mark an outside-brain channel. That is why this helper refuses
    to write ``out``. To keep an ``out`` channel out of a sort, omit it from the
    sort group's membership (it is not auto-handled anywhere), e.g. build the
    group with ``SortGroupV2.set_group_by_electrode_table_column(nwb_file_name,
    column="electrode_id", groups=[[...in-brain electrode_ids...]])``.

    Ordering: finalize ``bad_channel`` flags **before** creating sort groups.
    ``SortGroupV2.set_group_by_*`` excludes flagged channels at creation and a
    flag added after a group exists does not retroactively drop its members --
    recreate the group to apply later flags.

    Two caveats on the detection itself. The ``coherence+psd`` method estimates
    from random data chunks with SpikeInterface's ``seed=None`` default, so the
    flagged set can vary run-to-run; pass ``detection_params={"seed": ...}`` for
    a reproducible result (relevant when ``write=True`` persists the outcome).
    The method is also Neuropixels-density-tuned (default ``n_neighbors=11``): on
    a shank with only a few contacts (e.g. a tetrode) the spatial-coherence
    statistic is degenerate and the scan tends to report "no bad channels" --
    treat small-shank results with skepticism rather than as a clean bill.

    Parameters
    ----------
    nwb_file_name : str
        Session whose electrodes should be scanned.
    electrode_group_names : list[str], optional
        Restrict the scan to these electrode groups. Default ``None`` scans
        every group in the session.
    bandpass : tuple[float, float], optional
        ``(freq_min, freq_max)`` Hz for the pre-detection band-pass filter.
        Default ``(300.0, 6000.0)`` (spike band). ``freq_max`` must be below
        the recording's Nyquist frequency.
    detection_params : dict, optional
        Overrides forwarded to :func:`detect_bad_channels` (e.g.
        ``{"dead_channel_threshold": -0.4}``). ``None`` values within it are
        dropped. Default ``None`` uses SI's defaults.
    write : bool, optional
        ``False`` (default) suggests only -- mutates nothing. ``True`` persists
        ``dead`` / ``noise`` flags (additive).
    report : list[dict], optional
        A report previously returned by this function. When given (with
        ``write=True``), detection is **skipped** and these exact entries are
        persisted -- so the confirm step writes precisely what was reviewed.
        Default ``None`` detects fresh.

    Returns
    -------
    list[dict]
        One dict per flagged electrode (all labels, including ``out``):
        ``{"electrode_group_name", "electrode_id", "probe_shank", "label"}``.
        When ``report`` is given, it is returned unchanged.

    Raises
    ------
    ValueError
        If no electrodes are found for the session / requested groups, or if
        ``report`` is given without ``write=True``.
    """
    from spyglass.utils import logger

    # Apply mode: persist a previously-reviewed report verbatim, no re-detection
    # (so the confirm step writes exactly what was reviewed -- detection with
    # ``seed=None`` is otherwise non-deterministic run-to-run).
    if report is not None:
        if not write:
            raise ValueError(
                "suggest_bad_channels: `report` is only for persisting a "
                "previously reviewed report -- pass write=True, or omit "
                "`report` to detect fresh."
            )
        n_written = _persist_quality_bad(nwb_file_name, report)
        logger.info(
            f"suggest_bad_channels: {nwb_file_name!r} -- persisted reviewed "
            f"report ({len(report)} flagged); wrote {n_written} dead/noise to "
            "Electrode.bad_channel (out channels never written)."
        )
        return report

    # Detect mode.
    import spikeinterface.preprocessing as sip
    from spikeinterface.core.channelslice import ChannelSliceRecording

    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.utils import read_raw_nwb_recording
    from spyglass.spikesorting.v2._recording_materialization import (
        spikeinterface_channel_ids,
    )

    # 1. Electrode metadata, grouped by physical shank (fail loud on a missing
    #    shank). ``probe_shank`` is read exactly as ``set_group_by_shank`` reads
    #    it (off a plain ``Electrode`` fetch, via the Probe FK chain).
    query = Electrode() & {"nwb_file_name": nwb_file_name}
    if electrode_group_names is not None:
        query = query & [
            {"electrode_group_name": g} for g in electrode_group_names
        ]
    electrodes = query.fetch(
        "electrode_group_name", "electrode_id", "probe_shank", as_dict=True
    )
    if len(electrodes) == 0:
        scope = (
            f" in electrode groups {list(electrode_group_names)!r}"
            if electrode_group_names is not None
            else ""
        )
        raise ValueError(
            f"suggest_bad_channels: no electrodes found for "
            f"{nwb_file_name!r}{scope}. Check Electrode population."
        )
    shanks = _shank_groups(electrodes, nwb_file_name)

    # 2. Load the raw recording and band-pass filter ONCE so detection runs on
    #    spike-band data. The NWB conversion makes the recording µV-scaled, so
    #    ``coherence+psd`` (which asserts ``has_scaleable_traces``) is satisfied.
    #    Resolve every electrode_id -> SI channel id in a single NWB read (not
    #    once per shank).
    rec = read_raw_nwb_recording(Nwbfile.get_abs_path(nwb_file_name))
    # freq_max must be below the recording's Nyquist (fs/2); a value at/above
    # it fails opaquely inside scipy's filter design (shared check with
    # apply_pre_motion_preprocessing).
    from spyglass.spikesorting.v2._signal_math import (
        assert_freq_max_below_nyquist,
    )

    assert_freq_max_below_nyquist(
        bandpass[1],
        rec.get_sampling_frequency(),
        context=f"suggest_bad_channels ({nwb_file_name!r}): ",
    )
    rec_f = sip.bandpass_filter(rec, freq_min=bandpass[0], freq_max=bandpass[1])
    all_ids = [eid for ids in shanks.values() for eid in ids]
    si_by_eid = dict(
        zip(all_ids, spikeinterface_channel_ids(nwb_file_name, all_ids))
    )

    # 3. Detect within each shank; build the report. Detection is read-only --
    #    nothing is persisted until the (optional) atomic write below.
    report = []
    n_out = 0
    for (e_group, shank), electrode_ids in shanks.items():
        shank_rec = ChannelSliceRecording(
            rec_f,
            channel_ids=[si_by_eid[eid] for eid in electrode_ids],
            renamed_channel_ids=electrode_ids,
        )
        result = detect_bad_channels(shank_rec, **(detection_params or {}))
        labels = result["labels"]

        tally: dict[str, int] = {}
        for eid in result["bad_channel_ids"]:
            label = labels[eid]
            tally[label] = tally.get(label, 0) + 1
            report.append(
                {
                    "electrode_group_name": e_group,
                    "electrode_id": int(eid),
                    "probe_shank": int(shank),
                    "label": label,
                }
            )
            if label == "out":
                n_out += 1

        summary = (
            ", ".join(f"{n} {lbl}" for lbl, n in sorted(tally.items()))
            or "no bad channels"
        )
        logger.info(
            f"suggest_bad_channels: {nwb_file_name!r} group {e_group} "
            f"shank {shank}: {summary}."
        )

    # 4. Persist dead/noise (additive, atomic) from the freshly-built report.
    n_written = _persist_quality_bad(nwb_file_name, report) if write else 0
    write_msg = (
        f"wrote {n_written} dead/noise to Electrode.bad_channel"
        if write
        else "write=False (no changes made)"
    )
    logger.info(
        f"suggest_bad_channels: {nwb_file_name!r} -- {len(report)} flagged "
        f"across {len(shanks)} shank(s); {write_msg}; {n_out} out channel(s) "
        "report-only (never written). Thresholds are SpikeInterface defaults "
        "(Neuropixels-derived; recalibrate for polymer if needed)."
    )
    return report
