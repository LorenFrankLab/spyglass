"""Bad-channel detection (SpikeInterface ``coherence+psd``).

- ``detect_bad_channels``: wrapper over
  ``spikeinterface.preprocessing.detect_bad_channels`` that pins the method
  and returns flagged ids plus per-channel labels.
- ``suggest_bad_channels``: detects per shank on a session's band-passed raw
  recording and reports, or with ``persist=True`` writes, the ``dead`` /
  ``noise`` channels to ``Electrode.bad_channel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._core.signal_math import (
    assert_freq_max_below_nyquist,
)
from spyglass.spikesorting.v2._recording.geometry import (
    spikeinterface_channel_ids,
)
from spyglass.utils import logger
from spyglass.utils.nwb_helper_fn import get_raw_eseries_path

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._core.enums import BadChannelLabel

__all__ = ["detect_bad_channels", "suggest_bad_channels"]

# Pin only the method; thresholds stay at SpikeInterface's defaults.
BAD_CHANNEL_DETECT_DEFAULTS: dict[str, str] = {"method": "coherence+psd"}

# Labels persisted to Electrode.bad_channel. "out" is excluded: the boolean
# flag cannot carry it, and downstream would wrongly interpolate it.
_WRITABLE_LABELS: tuple[BadChannelLabel, ...] = ("dead", "noise")


def detect_bad_channels(recording, **overrides) -> dict:
    """Detect bad channels, returning ids and per-channel labels.

    Wraps ``spikeinterface.preprocessing.detect_bad_channels`` with
    ``method="coherence+psd"`` pinned; other arguments use SpikeInterface's
    defaults unless overridden.

    The recording must be high-pass filtered and µV-scaled. Thresholds are
    Neuropixels-derived; recalibrate for other probes (e.g. polymer).

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
        "bad_channel_ids": list(bad_ids),
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
    sorted by key with each id list sorted.

    Raises ``ValueError`` if any row has no ``probe_shank`` (the nullable
    ``Probe.Electrode`` FK is unset when the NWB lacks probe metadata).
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

    Sets ``Electrode.bad_channel='True'`` for each ``dead``/``noise`` entry,
    skipping rows already flagged, in one transaction. Never clears a flag.
    Returns the count of newly written rows.
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
    persist: bool = False,
    reviewed_report: list[dict] | None = None,
) -> list[dict]:
    """Suggest, and optionally persist, a session's bad channels.

    By default nothing is written: the raw recording is band-pass filtered,
    :func:`detect_bad_channels` runs per shank (the coherence method is
    spatially local), and one report dict is returned per flagged electrode.

    - Persist exactly what you reviewed with ``persist=True,
      reviewed_report=reviewed``. A fresh ``persist=True`` call re-detects,
      and detection is random unless ``detection_params={"seed": ...}``.
    - ``persist=True`` writes ``Electrode.bad_channel='True'`` only for
      ``dead`` / ``noise``, additively. ``out`` (outside-brain) is
      report-only, because downstream would interpolate a flagged channel.
      Exclude ``out`` channels from the sort group instead, e.g. with
      ``SortGroupV2.set_group_by_electrode_table_column``.
    - Set flags before creating sort groups; existing groups keep their
      members until recreated.
    - The method is tuned for Neuropixels density (``n_neighbors=11``). On a
      few-contact shank such as a tetrode it tends to report no bad channels.

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
    persist : bool, optional
        ``False`` (default) writes nothing. ``True`` persists ``dead`` /
        ``noise`` flags additively.
    reviewed_report : list[dict], optional
        A report previously returned by this function. With ``persist=True``,
        skips detection and persists these entries. Default ``None`` detects
        fresh.

    Returns
    -------
    list[dict]
        One dict per flagged electrode (all labels, including ``out``):
        ``{"electrode_group_name", "electrode_id", "probe_shank", "label"}``.
        When ``reviewed_report`` is given, it is returned unchanged.

    Raises
    ------
    ValueError
        If no electrodes are found for the session / requested groups, or if
        ``reviewed_report`` is given without ``persist=True``.
    """
    # Apply mode: persist the reviewed report without re-detecting.
    if reviewed_report is not None:
        if not persist:
            raise ValueError(
                "suggest_bad_channels: `reviewed_report` is only for "
                "persisting a previously reviewed report -- pass "
                "persist=True, or omit `reviewed_report` to detect fresh."
            )
        n_written = _persist_quality_bad(nwb_file_name, reviewed_report)
        logger.info(
            f"suggest_bad_channels: {nwb_file_name!r} -- persisted reviewed "
            f"report ({len(reviewed_report)} flagged); wrote {n_written} "
            "dead/noise to Electrode.bad_channel (out channels never written)."
        )
        return reviewed_report

    # Detect mode.
    import spikeinterface.preprocessing as sip
    from spikeinterface.core.channelslice import ChannelSliceRecording

    from spyglass.common.common_ephys import Electrode, Raw
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb

    # Read probe_shank the same way set_group_by_shank does.
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

    # Read the series Recording reads; fall back to name-based selection
    # when the session has no Raw row.
    raw_path = Nwbfile.get_abs_path(nwb_file_name)
    raw_object_ids = (Raw & {"nwb_file_name": nwb_file_name}).fetch(
        "raw_object_id"
    )
    rec = read_recording_nwb(
        raw_path,
        electrical_series_path=get_raw_eseries_path(
            raw_path,
            object_id=raw_object_ids[0] if len(raw_object_ids) else None,
        ),
        load_time_vector=False,
    )
    # scipy's filter design fails opaquely at or above Nyquist.
    assert_freq_max_below_nyquist(
        bandpass[1],
        rec.get_sampling_frequency(),
        context=f"suggest_bad_channels ({nwb_file_name!r}): ",
    )
    rec_f = sip.bandpass_filter(rec, freq_min=bandpass[0], freq_max=bandpass[1])
    all_ids = [eid for ids in shanks.values() for eid in ids]
    si_by_eid = dict(
        zip(all_ids, spikeinterface_channel_ids(raw_path, all_ids))
    )

    detected_report = []
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
            detected_report.append(
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

    n_written = (
        _persist_quality_bad(nwb_file_name, detected_report) if persist else 0
    )
    persist_msg = (
        f"wrote {n_written} dead/noise to Electrode.bad_channel"
        if persist
        else "persist=False (no changes made)"
    )
    logger.info(
        f"suggest_bad_channels: {nwb_file_name!r} -- {len(detected_report)} flagged "
        f"across {len(shanks)} shank(s); {persist_msg}; {n_out} out channel(s) "
        "report-only (never written). Thresholds are SpikeInterface defaults "
        "(Neuropixels-derived; recalibrate for polymer if needed)."
    )
    return detected_report
