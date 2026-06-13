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

# We pin ONLY the method. Every threshold is left to SpikeInterface's own
# signature default (e.g. SI 0.104.3: dead -0.5, noisy 1.0, outside -0.75,
# psd_hf 0.02, seed None) so "ships SI defaults" is literally true and a future
# SI default change flows through. Do NOT hardcode threshold values here -- they
# would silently diverge from SI and masquerade as defaults.
BAD_CHANNEL_DETECT_DEFAULTS: dict = {"method": "coherence+psd"}

# Labels that ``write=True`` persists to ``Electrode.bad_channel``. These are
# the *quality-bad* classes (a dead/flat contact, a noisy contact) -- safe to
# interpolate from neighbours or to remove. ``"out"`` (outside-brain) is
# deliberately excluded: the boolean flag cannot carry the label, and a
# persisted ``out`` would later be wrongly *interpolated* by downstream handling.
_WRITABLE_LABELS = ("dead", "noise")


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


def suggest_bad_channels(
    nwb_file_name: str,
    *,
    electrode_group_names=None,
    bandpass: tuple[float, float] = (300.0, 6000.0),
    detection_params: dict | None = None,
    write: bool = False,
) -> list[dict]:
    """Suggest -- and optionally persist -- a session's bad channels.

    Suggest-then-confirm helper. By default (``write=False``) it changes
    nothing: it loads the session's raw recording, band-pass filters it, runs
    :func:`detect_bad_channels` **per full shank** (the coherence method is
    spatially local; mixing shanks corrupts it), maps SpikeInterface channel
    ids back to spyglass ``electrode_id`` s, and returns one report dict per
    flagged electrode carrying its label so you can review what kind of bad it
    is before persisting anything.

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
    group with ``SortGroupV2.set_group_by_electrode_table_column(
    column="electrode_id", groups=[[...in-brain electrode_ids...]])``.

    Ordering: finalize ``bad_channel`` flags **before** creating sort groups.
    ``SortGroupV2.set_group_by_*`` excludes flagged channels at creation and a
    flag added after a group exists does not retroactively drop its members --
    recreate the group to apply later flags.

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

    Returns
    -------
    list[dict]
        One dict per flagged electrode (all labels, including ``out``):
        ``{"electrode_group_name", "electrode_id", "probe_shank", "label"}``.

    Raises
    ------
    ValueError
        If no electrodes are found for the session / requested groups.
    """
    import spikeinterface.preprocessing as sip
    from spikeinterface.core.channelslice import ChannelSliceRecording

    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.utils import read_raw_nwb_recording
    from spyglass.spikesorting.v2._recording_materialization import (
        spikeinterface_channel_ids,
    )
    from spyglass.utils import logger

    # 1. Electrode metadata for shank grouping. ``probe_shank`` rides in via the
    #    ``Electrode -> Probe.Electrode`` FK, exactly as ``set_group_by_shank``
    #    reads it.
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

    # 2. Load the raw recording and band-pass filter so detection runs on
    #    spike-band data. The NWB conversion makes the recording µV-scaled, so
    #    ``coherence+psd`` (which asserts ``has_scaleable_traces``) is satisfied.
    rec = read_raw_nwb_recording(Nwbfile.get_abs_path(nwb_file_name))
    rec_f = sip.bandpass_filter(rec, freq_min=bandpass[0], freq_max=bandpass[1])

    # 3. Group electrodes by (electrode group, shank) and detect within each.
    shanks: dict[tuple[str, int], list[int]] = {}
    for row in electrodes:
        key = (str(row["electrode_group_name"]), int(row["probe_shank"]))
        shanks.setdefault(key, []).append(int(row["electrode_id"]))

    report: list[dict] = []
    n_written = 0
    n_out = 0
    for (e_group, shank), electrode_ids in sorted(shanks.items()):
        electrode_ids = sorted(electrode_ids)
        si_ids = spikeinterface_channel_ids(nwb_file_name, electrode_ids)
        shank_rec = ChannelSliceRecording(
            rec_f,
            channel_ids=si_ids,
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
            elif write and label in _WRITABLE_LABELS:
                pk = {
                    "nwb_file_name": nwb_file_name,
                    "electrode_group_name": e_group,
                    "electrode_id": int(eid),
                }
                if (Electrode & pk).fetch1("bad_channel") != "True":
                    Electrode.update1({**pk, "bad_channel": "True"})
                    n_written += 1

        summary = (
            ", ".join(f"{n} {lbl}" for lbl, n in sorted(tally.items()))
            or "no bad channels"
        )
        logger.info(
            f"suggest_bad_channels: {nwb_file_name!r} group {e_group} "
            f"shank {shank}: {summary}."
        )

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
