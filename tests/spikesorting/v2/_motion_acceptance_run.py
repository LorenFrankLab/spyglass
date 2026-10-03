"""Run one motion acceptance benchmark case through the v2 stage functions.

One case = one ``(scenario, seed, recipe)`` of a manifest
(``_motion_acceptance.AcceptanceManifest``), run in its own process so its
peak RSS is its own::

    python -m tests.spikesorting.v2._motion_acceptance_run case \\
        --manifest M.json --scenario rigid --seed 0 --recipe dredge --out DIR

writes ``DIR/<scenario>__s<seed>__<recipe>.json``. A case generates the
scenario's recording and its noise-free twin (SpikeInterface
``generate_drifting_recording``), bandpasses them, derives continuity with the
stage's own helpers (``continuity_from_timestamps`` / ``concat_continuity``),
masks planted artifacts (``statistics_spans``, ``silence_frame_ranges``),
estimates with ``_motion.estimate_motion_in_spans`` on the estimation clock
(``build_estimation_clock``), applies with
``_motion.apply_motion_on_estimation_clock``, sorts with the sorting stage's
``run_si_sorter`` + ``remove_excess_spikes`` and compares with the ground
truth. The metrics (motion error, corrected-signal fidelity, sorting accuracy,
border channels, cost) are defined in :func:`plain_motion_error`,
:func:`source_clock_motion_error`, :func:`fidelity_sums`,
:func:`sorting_metrics`, :func:`predicted_removed_channels` and
:func:`run_case`.

The ``representative`` command runs a paired off / dredge / dredge_fast
estimate-and-apply (and optionally sorting) on one shank of the MEArec polymer
drift fixture and records quality, border effects, runtime and memory; it
applies no gate.

DB-FREE: no DataJoint connection is opened. The sorter's scratch lives under
``spyglass.settings.temp_dir``, which a test session's teardown clears: do not
run another pytest session on the same checkout while cases run.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np


def peak_rss_bytes() -> int:
    """This process's peak resident set size (bytes), for the cost metric.

    ``getrusage(RUSAGE_SELF).ru_maxrss`` counts this process only. It equals
    the maximum resident set size ``/usr/bin/time -l`` reports for the case
    process because everything runs in it: SpikeInterface jobs use the
    manifest's job kwargs (``n_jobs=1``, no worker processes) and
    MountainSort5 runs in-process through ``run_sorter`` (local backend).
    On the 108 cases of the development manifest the two agreed to the byte.
    With ``n_jobs > 1`` or a container sorter, worker or container memory would
    be missing from it.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ``ru_maxrss`` is bytes on macOS and KiB on Linux.
    return int(rss if sys.platform == "darwin" else rss * 1024)


class ScenarioRecording(NamedTuple):
    """One scenario's recording as the stages see it.

    Attributes
    ----------
    recording : si.BaseRecording
        Bandpassed (float64) drifting recording: windows joined back to back
        for a windowed scenario, artifacts planted for a masked one.
    static : si.BaseRecording
        The static twin, bandpassed and joined the same way.
    gt_sorting : si.BaseSorting
        Ground-truth spikes in ``recording``'s frames (masked frames removed).
    displacement : numpy.ndarray
        ``(n_t,)`` ground-truth displacement of the deepest unit (um) on the
        generated recording's clock, SpikeInterface ``Motion`` sign.
    unit_depths_um : numpy.ndarray
        ``(num_units,)`` ground-truth unit depths (for the nonrigid factor).
    continuity : _sorting_artifact_mask.Continuity
        Continuity spans with their first and last source timestamps.
    excluded : list[tuple[int, int]]
        Masked frame ranges (empty without planted artifacts).
    """

    recording: object
    static: object
    gt_sorting: object
    displacement: np.ndarray
    unit_depths_um: np.ndarray
    continuity: object
    excluded: list


def _zigzag_kwargs(manifest, amplitude_um: float, gradient) -> dict:
    duration = manifest.generator.duration_s
    return dict(
        displacement_sampling_frequency=(
            manifest.generator.displacement_sampling_frequency
        ),
        drift_start_um=[0, amplitude_um],
        drift_stop_um=[0, -amplitude_um],
        drift_step_um=1,
        motion_list=[
            dict(
                drift_mode="zigzag",
                non_rigid_gradient=gradient,
                t_start_drift=0.0,
                t_end_drift=None,
                period_s=duration,
            )
        ],
    )


def _generate(manifest, scenario_name: str, seed: int, *, noise_free: bool):
    """``generate_drifting_recording`` for one scenario (unfiltered)."""
    from spikeinterface.generation import generate_drifting_recording

    from tests.spikesorting.v2._motion_fixtures import (
        _step_displacement_data,
        polymer_shank_probe,
    )

    spec = manifest.scenarios[scenario_name]
    gen = manifest.generator
    probe = manifest.probe
    duration = gen.duration_s
    if getattr(spec, "windows_s", None) is not None:
        duration = float(spec.windows_s[-1][1])
    kwargs = dict(
        num_units=gen.num_units,
        sampling_frequency=gen.sampling_frequency,
        probe=polymer_shank_probe(
            probe.n_contacts,
            pitch_um=probe.pitch_um,
            contact_radius_um=probe.contact_radius_um,
        ),
        extra_outputs=True,
        seed=seed,
    )
    if spec.kind == "step":
        kwargs["displacement_data"] = _step_displacement_data(
            duration,
            spec.change_times_s,
            spec.levels_um,
            gen.num_units,
            displacement_sampling_frequency=gen.displacement_sampling_frequency,
        )
    else:
        gradient = getattr(spec, "non_rigid_gradient", None)
        kwargs["generate_displacement_vector_kwargs"] = _zigzag_kwargs(
            manifest, spec.amplitude_um, gradient
        )
    if noise_free:
        kwargs["generate_noise_kwargs"] = dict(
            noise_levels=0.0, spatial_decay=None
        )
    static, drifting, gt_sorting, extra = generate_drifting_recording(
        duration=duration, **kwargs
    )
    if spec.kind == "static":
        drifting = static
        displacement = np.zeros(extra["displacement_vectors"].shape[0])
    else:
        displacement = extra["displacement_vectors"][:, 1, 0].astype(float)
    return static, drifting, gt_sorting, displacement, extra


def _cut_sorting(gt_sorting, windows_frames, excluded):
    """Ground-truth spikes inside the kept windows, in joined frames, minus
    spikes inside ``excluded`` (joined frames)."""
    from spikeinterface.core import NumpySorting

    vector = gt_sorting.to_spike_vector()
    frames, labels = [], []
    offset = 0
    for start, end in windows_frames:
        inside = (vector["sample_index"] >= start) & (
            vector["sample_index"] < end
        )
        frames.append(vector["sample_index"][inside] - start + offset)
        labels.append(vector["unit_index"][inside])
        offset += end - start
    frames = np.concatenate(frames)
    labels = np.concatenate(labels)
    keep = np.ones(frames.size, dtype=bool)
    for lo, hi in excluded:
        keep &= ~((frames >= lo) & (frames < hi))
    unit_ids = np.asarray(gt_sorting.unit_ids)
    return NumpySorting.from_samples_and_labels(
        [frames[keep]],
        [unit_ids[labels[keep]]],
        gt_sorting.get_sampling_frequency(),
        unit_ids=unit_ids,
    )


def build_scenario(
    manifest, scenario_name: str, seed: int, *, noise_free: bool
) -> ScenarioRecording:
    """Generate, filter, cut, join and mask one scenario's recording."""
    from spikeinterface.core import concatenate_recordings

    from spyglass.spikesorting.v2._concat_recording import concat_continuity
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        continuity_from_timestamps,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        bandpass,
        plant_artifact_bursts,
    )

    spec = manifest.scenarios[scenario_name]
    fs = manifest.generator.sampling_frequency
    static, drifting, gt_sorting, displacement, extra = _generate(
        manifest, scenario_name, seed, noise_free=noise_free
    )

    def _filter(recording):
        return bandpass(
            recording,
            freq_min=manifest.bandpass.freq_min,
            freq_max=manifest.bandpass.freq_max,
        )

    windows = getattr(spec, "windows_s", None)
    if windows is None:
        n = drifting.get_num_samples()
        windows_frames = [(0, n)]
        joined, joined_static = _filter(drifting), _filter(static)
        continuity = continuity_from_timestamps(joined)
    else:
        windows_frames = [
            (int(round(a * fs)), int(round(b * fs))) for a, b in windows
        ]
        spec_members = getattr(spec, "members", None)
        members = spec_members or [list(range(len(windows)))]

        def _join(recording):
            filtered = _filter(recording)
            member_recordings = []
            for member in members:
                pieces = []
                for i in member:
                    start, end = windows_frames[i]
                    piece = filtered.frame_slice(start, end)
                    piece.set_times(
                        np.arange(start, end) / fs, with_warning=False
                    )
                    pieces.append(piece)
                if len(pieces) == 1:
                    member_recordings.append(pieces[0])
                    continue
                member_recording = concatenate_recordings(
                    pieces, ignore_times=True
                )
                member_recording.set_times(
                    np.concatenate([p.get_times() for p in pieces]),
                    with_warning=False,
                )
                member_recordings.append(member_recording)
            return member_recordings

        member_recordings = _join(drifting)
        static_members = _join(static)
        if spec_members is None:
            joined, joined_static = member_recordings[0], static_members[0]
            continuity = continuity_from_timestamps(joined)
        else:
            counts = [r.get_num_samples() for r in member_recordings]
            continuity = concat_continuity(member_recordings, counts)
            joined = concatenate_recordings(
                member_recordings, ignore_times=True
            )
            joined_static = concatenate_recordings(
                static_members, ignore_times=True
            )
    excluded: list = []
    artifacts = getattr(spec, "masked_artifacts", None)
    if artifacts is not None:
        joined, excluded = plant_artifact_bursts(
            joined,
            artifacts.windows_s,
            rate_hz=artifacts.rate_hz,
            amplitude_uv=artifacts.amplitude_uv,
        )
    return ScenarioRecording(
        recording=joined,
        static=joined_static,
        gt_sorting=_cut_sorting(gt_sorting, windows_frames, excluded),
        displacement=displacement,
        unit_depths_um=np.asarray(extra["unit_locations"][:, 1], dtype=float),
        continuity=continuity,
        excluded=[(int(a), int(b)) for a, b in excluded],
    )


def depth_factor(scenario_spec, unit_depths_um, depths_um) -> np.ndarray:
    """Ground-truth displacement factor at each depth, ``(n_depths,)``.

    1 for rigid drift. For nonrigid zigzag drift SpikeInterface scales each
    unit's displacement linearly in its depth from ``non_rigid_gradient``
    (most superficial unit) to 1 (deepest unit); the same line is evaluated
    at ``depths_um`` and clipped to that range.
    """
    gradient = getattr(scenario_spec, "non_rigid_gradient", None)
    depths_um = np.asarray(depths_um, dtype=float)
    if gradient is None:
        return np.ones_like(depths_um)
    top, bottom = unit_depths_um.max(), unit_depths_um.min()
    fraction = np.clip((top - depths_um) / (top - bottom), 0.0, 1.0)
    return gradient + (1.0 - gradient) * fraction


def _error_summary(estimate, truth) -> dict:
    """RMS / p95 / max of ``estimate - truth`` after ONE global offset."""
    diff = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    offset = float(diff.mean())
    err = np.abs(diff - offset)
    return dict(
        global_offset_um=offset,
        rms_um=float(np.sqrt(np.mean(err**2))),
        p95_um=float(np.percentile(err, 95)),
        max_abs_um=float(err.max()),
    )


def _correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def plain_motion_error(motion, displacement, factor_fn, channel_depths, dfs):
    """Motion error on an unwindowed, unmasked recording (estimation clock =
    source).

    The estimate is evaluated at its own temporal-bin centers at every
    channel depth. The truth of bin ``i`` is the mean of the ground-truth
    vector (samples at ``(k + 0.5) / dfs``) over ``[edge_i, edge_{i+1})`` of
    ``Motion.temporal_bin_edges_s``, times the depth factor. One scalar
    offset over all bins and depths is removed; RMS, 95th percentile and max
    of ``|error|`` are reported. Sign: the Pearson correlation of the
    estimate's native displacement averaged over its spatial bins with the
    truth averaged over the same bins.
    """
    edges = motion.temporal_bin_edges_s[0]
    centers = motion.temporal_bins_s[0]
    t = (np.arange(displacement.size) + 0.5) / dfs
    vbin = np.array(
        [
            (
                displacement[(t >= lo) & (t < hi)].mean()
                if ((t >= lo) & (t < hi)).any()
                else np.nan
            )
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    estimate = np.stack(
        [
            motion.get_displacement_at_time_and_depth(
                np.full(channel_depths.size, c), channel_depths, segment_index=0
            )
            for c in centers
        ]
    )
    truth = vbin[:, None] * factor_fn(channel_depths)[None, :]
    ok = np.isfinite(truth).all(axis=1)
    result = _error_summary(estimate[ok], truth[ok])
    native = np.asarray(motion.displacement[0])
    native_truth = vbin[:, None] * factor_fn(motion.spatial_bins_um)[None, :]
    result["sign_corr"] = _correlation(
        native[ok].mean(axis=1), native_truth[ok].mean(axis=1)
    )
    result["variant"] = "bin_edges"
    result["n_bins"] = int(ok.sum())
    return result


def source_clock_motion_error(motion, clock, displacement, channel_depths, dfs):
    """Motion error on the source clock (windowed, joined or masked scenarios).

    Temporal bins are mapped to source time with the stage's accessor
    (``_motion.displacement_on_source_clock``); bins in a capped gap are
    skipped. The truth of a bin is the mean rigid ground truth over the bin's
    width on the source clock, clipped to its continuity span
    (``_motion_fixtures.source_clock_estimate_and_truth``). One scalar offset
    over all kept bins and channel depths is removed; RMS, 95th percentile
    and max of ``|error|`` are reported. Sign: correlation of the estimate
    averaged over channel depths with the truth, per kept bin.
    """
    from spyglass.spikesorting.v2._motion import displacement_on_source_clock
    from tests.spikesorting.v2._motion_fixtures import (
        source_clock_estimate_and_truth,
    )

    estimate, truth = source_clock_estimate_and_truth(
        motion,
        clock,
        displacement,
        channel_depths,
        displacement_sampling_frequency=dfs,
    )
    result = _error_summary(
        estimate, np.broadcast_to(truth[:, None], estimate.shape)
    )
    result["sign_corr"] = _correlation(estimate.mean(axis=1), truth)
    result["variant"] = "source_clock"
    result["n_bins"] = int(truth.size)
    result["n_gap_bins"] = int(
        displacement_on_source_clock(motion, clock).in_gap.sum()
    )
    return result


def oracle_motion(displacement, factor_fn, channel_depths, clock, dfs):
    """The ground-truth displacement as a ``Motion`` on the estimation clock.

    Ground-truth samples (centers ``(k + 0.5) / dfs`` on the source clock)
    inside continuity span ``i`` are placed at ``e_i + (t - t_i)``; spatial
    bins are the sorted channel depths, each scaled by the depth factor. On
    a single span starting at 0 this is the truth on its own clock.
    """
    from spikeinterface.core.motion import Motion

    t = (np.arange(displacement.size) + 0.5) / dfs
    fs = clock.sampling_frequency
    times, values = [], []
    for i in range(len(clock.spans)):
        lo = clock.source_start_s[i]
        hi = clock.source_end_s[i] + 1.0 / fs
        inside = (t >= lo) & (t < hi)
        times.append(clock.estimation_start_s[i] + (t[inside] - lo))
        values.append(displacement[inside])
    depths = np.sort(np.asarray(channel_depths, dtype=float))
    values = np.concatenate(values)
    field = values[:, None] * factor_fn(depths)[None, :]
    return Motion(
        [field.astype("float64")],
        [np.concatenate(times)],
        depths,
        direction="y",
    )


def predicted_removed_channels(recording, motion) -> list:
    """Channels ``border_mode="remove_channels"`` must drop for ``motion``.

    SpikeInterface keeps a contact only when its depth plus the displacement
    at that depth stays within the contacts' depth range in every temporal
    bin (``sortingcomponents/motion/motion_interpolation.py:364-384``).
    """
    locs = np.asarray(recording.get_channel_locations())[:, motion.dim]
    lo, hi = locs.min(), locs.max()
    moved = locs[:, None] + motion.get_displacement_at_time_and_depth(
        times_s=motion.temporal_bins_s[0], locations_um=locs, grid=True
    )
    inside = (np.clip(moved, lo, hi) == moved).all(axis=1)
    return [str(c) for c in np.asarray(recording.channel_ids)[~inside]]


def fidelity_sums(test, reference, channel_ids, starts, window) -> dict:
    """Per-channel ``sum (test - ref)^2`` and ``sum ref^2`` over windows.

    Returns
    -------
    dict
        ``channel_ids`` (str), ``num`` and ``den`` lists, and ``pooled`` =
        ``sqrt(sum num / sum den)``.
    """
    num = np.zeros(len(channel_ids))
    den = np.zeros(len(channel_ids))
    for start in starts:
        a = test.get_traces(
            start_frame=start, end_frame=start + window, channel_ids=channel_ids
        ).astype("float64")
        b = reference.get_traces(
            start_frame=start, end_frame=start + window, channel_ids=channel_ids
        ).astype("float64")
        num += np.sum((a - b) ** 2, axis=0)
        den += np.sum(b**2, axis=0)
    return dict(
        channel_ids=[str(c) for c in channel_ids],
        num=num.tolist(),
        den=den.tolist(),
        pooled=float(np.sqrt(num.sum() / den.sum())),
    )


def paired_fidelity(corrected, uncorrected, static, starts, window) -> dict:
    """Fidelity of one twin: the corrected and the uncorrected recording against
    the static twin, both over the corrected recording's channels.

    A ``remove_channels`` correction drops the end contacts, which carry the
    largest residual; measuring the uncorrected baseline over every channel
    would credit the dropped channels to the correction.

    Returns
    -------
    dict
        ``corrected`` and ``uncorrected`` :func:`fidelity_sums`.
    """
    kept = list(corrected.channel_ids)
    return dict(
        corrected=fidelity_sums(corrected, static, kept, starts, window),
        uncorrected=fidelity_sums(uncorrected, static, kept, starts, window),
    )


def sorting_metrics(gt_sorting, sorting, comparison) -> dict:
    """Sorting accuracy: ground-truth comparison of one sort (manifest
    scores)."""
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    cmp = compare_sorter_to_ground_truth(
        gt_sorting,
        sorting,
        exhaustive_gt=True,
        delta_time=comparison.delta_time_ms,
        match_score=comparison.match_score,
        well_detected_score=comparison.well_detected_score,
        redundant_score=comparison.redundant_score,
        overmerged_score=comparison.overmerged_score,
    )
    perf = cmp.get_performance(method="by_unit")
    agreement = cmp.agreement_scores
    oversplit = int(
        ((agreement > comparison.oversplit_agreement).sum(axis=1) > 1).sum()
    )
    return dict(
        n_sorted_units=int(sorting.get_num_units()),
        n_gt_units=int(gt_sorting.get_num_units()),
        n_gt_spikes=int(gt_sorting.to_spike_vector().size),
        mean_accuracy=float(perf["accuracy"].mean()),
        mean_precision=float(perf["precision"].mean()),
        mean_recall=float(perf["recall"].mean()),
        n_well_detected=int(
            cmp.count_well_detected_units(comparison.well_detected_score)
        ),
        n_redundant=int(cmp.count_redundant_units()),
        n_overmerged=int(cmp.count_overmerged_units()),
        n_false_positive=int(cmp.count_false_positive_units()),
        n_bad=int(cmp.count_bad_units()),
        n_gt_oversplit=oversplit,
        per_unit_accuracy=[float(v) for v in perf["accuracy"]],
    )


def _sort(manifest, recording, spans, tag: str):
    """Sort through the v2 sorting stage's DB-free runner."""
    from spyglass.spikesorting.v2._params.sorter import MountainSort5Schema
    from spyglass.spikesorting.v2._sorting_dispatch import (
        remove_excess_spikes,
        run_si_sorter,
    )

    params = MountainSort5Schema(**manifest.sorter.params).model_dump()
    sorting = run_si_sorter(
        manifest.sorter.name,
        params,
        recording,
        tag,
        {"random_seed": manifest.sorter.random_seed},
        statistics_spans=spans,
    )
    return remove_excess_spikes(sorting, recording)


def run_case(
    manifest, manifest_path, scenario: str, seed: int, recipe: str, out_dir
) -> dict:
    """Run one case and write its result JSON; return the result.

    Raises
    ------
    ValueError
        If the case is not in the manifest's grid, the seed is reserved
        for held-out runs while the manifest is a development one, or the
        harness files differ from the manifest's harness pin.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._motion import (
        apply_motion_on_estimation_clock,
        build_estimation_clock,
        estimate_motion_in_spans,
        resolve_estimation_params,
        resolve_interpolation_params,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        silence_frame_ranges,
        statistics_spans,
    )
    from tests.spikesorting.v2._motion_acceptance import (
        HELD_OUT_SEED_MIN,
        case_tag,
        check_harness_pin,
        harness_fingerprint,
        manifest_sha256,
    )

    if (scenario, seed, recipe) not in set(manifest.iter_cases()):
        raise ValueError(f"case {scenario} seed {seed} {recipe} not in grid.")
    if manifest.purpose != "held_out" and seed >= HELD_OUT_SEED_MIN:
        raise ValueError(f"seed {seed} is reserved for held-out runs.")
    fingerprint = harness_fingerprint()
    check_harness_pin(manifest, fingerprint)

    t_start = time.perf_counter()
    timings: dict = {}
    job_kwargs = dict(manifest.evaluation.job_kwargs)
    si.set_global_job_kwargs(**job_kwargs)
    spec = manifest.scenarios[scenario]
    recipe_spec = manifest.recipes[recipe]
    dfs = manifest.generator.displacement_sampling_frequency
    fs = manifest.generator.sampling_frequency

    t = time.perf_counter()
    noisy = build_scenario(manifest, scenario, seed, noise_free=False)
    clean = build_scenario(manifest, scenario, seed, noise_free=True)
    timings["generate"] = time.perf_counter() - t
    if not np.array_equal(
        noisy.gt_sorting.to_spike_vector(), clean.gt_sorting.to_spike_vector()
    ):
        raise AssertionError("noise-free twin spike trains differ")

    recording = noisy.recording
    n_samples = recording.get_num_samples()
    channel_ids = list(recording.channel_ids)
    channel_depths = np.asarray(recording.get_channel_locations())[:, 1]
    continuity = noisy.continuity
    spans = statistics_spans(n_samples, noisy.excluded, continuity.spans)

    def factor_fn(depths):
        return depth_factor(spec, noisy.unit_depths_um, depths)

    result: dict = dict(
        scenario=scenario,
        seed=seed,
        recipe=recipe,
        manifest=manifest.name,
        manifest_sha256=manifest_sha256(manifest_path),
        harness=fingerprint,
        spikeinterface_version=si.__version__,
        n_samples=int(n_samples),
        continuity_spans=[list(s) for s in continuity.spans],
        continuity_start_s=list(continuity.start_s),
        excluded_frame_ranges=[list(r) for r in noisy.excluded],
        statistics_spans=[list(s) for s in spans],
        gt=dict(
            n_spikes=int(noisy.gt_sorting.to_spike_vector().size),
            displacement_ptp_um=float(np.ptp(noisy.displacement)),
        ),
    )

    motion = None
    border_mode = None
    corrected = silence_frame_ranges(recording, noisy.excluded)
    corrected_clean = silence_frame_ranges(clean.recording, clean.excluded)
    removed: list = []
    if recipe_spec.kind != "off":
        interpolation = resolve_interpolation_params(recipe_spec.interpolation)
        border_mode = interpolation["border_mode"]
        if recipe_spec.kind == "oracle":
            max_gap_s = recipe_spec.max_gap_s
        else:
            noise_seed = (
                seed
                if manifest.noise_levels_seed == "case_seed"
                else int(manifest.noise_levels_seed)
            )
            resolved = resolve_estimation_params(
                {**recipe_spec.estimation, "noise_levels_seed": noise_seed}
            )
            max_gap_s = resolved["max_gap_s"]
        clock = build_estimation_clock(
            continuity.spans,
            continuity.start_s,
            continuity.end_s,
            fs,
            max_gap_s=max_gap_s,
        )
        if recipe_spec.kind == "oracle":
            motion = oracle_motion(
                noisy.displacement, factor_fn, channel_depths, clock, dfs
            )
        else:
            masked = silence_frame_ranges(recording, noisy.excluded)
            t = time.perf_counter()
            motion, diagnostics = estimate_motion_in_spans(
                masked,
                statistics_spans=spans,
                clock=clock,
                resolved_params=resolved,
                job_kwargs=job_kwargs,
            )
            timings["estimate"] = time.perf_counter() - t
            result["estimation"] = dict(
                n_peaks_detected=diagnostics.n_peaks_detected,
                n_peaks_kept=diagnostics.n_peaks_kept,
                peaks_per_continuity_span=(
                    diagnostics.peaks_per_continuity_span.tolist()
                ),
                n_temporal_bins=int(motion.displacement[0].shape[0]),
                spatial_bins_um=np.asarray(motion.spatial_bins_um).tolist(),
            )
            np.savez(
                Path(out_dir)
                / f"{case_tag(scenario, seed, recipe)}_motion.npz",
                displacement=motion.displacement[0],
                temporal_bins_s=motion.temporal_bins_s[0],
                spatial_bins_um=motion.spatial_bins_um,
            )
        windowed = (
            getattr(spec, "windows_s", None) is not None
            or getattr(spec, "masked_artifacts", None) is not None
        )
        if windowed:
            result["motion"] = source_clock_motion_error(
                motion, clock, noisy.displacement, channel_depths, dfs
            )
        else:
            result["motion"] = plain_motion_error(
                motion, noisy.displacement, factor_fn, channel_depths, dfs
            )
        t = time.perf_counter()
        applied = apply_motion_on_estimation_clock(
            recording,
            motion,
            clock=clock,
            statistics_spans=spans,
            resolved_interpolation=interpolation,
        )
        applied_clean = apply_motion_on_estimation_clock(
            clean.recording,
            motion,
            clock=clock,
            statistics_spans=spans,
            resolved_interpolation=interpolation,
        )
        timings["apply_setup"] = time.perf_counter() - t
        corrected, corrected_clean = applied.recording, applied_clean.recording
        removed = [str(c) for c in applied.removed_channel_ids]
        result["interpolation"] = interpolation

    kept = list(corrected.channel_ids)
    result["border"] = dict(
        border_mode=border_mode,
        n_in_channels=len(channel_ids),
        n_out_channels=len(kept),
        removed_channel_ids=removed,
        predicted_removed_channel_ids=(
            predicted_removed_channels(recording, motion)
            if border_mode == "remove_channels"
            else None
        ),
    )

    t = time.perf_counter()
    window = int(round(manifest.evaluation.fidelity_window_s * fs))
    starts = np.linspace(
        window, n_samples - 2 * window, manifest.evaluation.fidelity_n_windows
    ).astype(int)
    static_clean = silence_frame_ranges(clean.static, clean.excluded)
    uncorrected_clean = silence_frame_ranges(clean.recording, clean.excluded)
    static_noisy = silence_frame_ranges(noisy.static, noisy.excluded)
    uncorrected_noisy = silence_frame_ranges(recording, noisy.excluded)
    result["fidelity_signal"] = paired_fidelity(
        corrected_clean, uncorrected_clean, static_clean, starts, window
    )
    result["fidelity_noisy_pooled"] = {
        name: sums["pooled"]
        for name, sums in paired_fidelity(
            corrected, uncorrected_noisy, static_noisy, starts, window
        ).items()
    }
    timings["fidelity"] = time.perf_counter() - t
    # ``ru_maxrss`` only grows, so this is the peak of everything before the
    # sort: generation, estimation and the interpolation the fidelity windows
    # read (the sort interpolates the whole recording again).
    result["peak_rss_before_sort_bytes"] = peak_rss_bytes()

    t = time.perf_counter()
    sorting = _sort(
        manifest, corrected, spans, case_tag(scenario, seed, recipe)
    )
    timings["sort"] = time.perf_counter() - t
    result["sorting"] = sorting_metrics(
        noisy.gt_sorting, sorting, manifest.comparison
    )
    timings["total"] = time.perf_counter() - t_start
    result["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    result["peak_rss_bytes"] = peak_rss_bytes()
    out = Path(out_dir) / f"{case_tag(scenario, seed, recipe)}.json"
    out.write_text(json.dumps(result, indent=1, default=str))
    return result


# ---- representative polymer drift fixture -----------------------------------


def run_representative(
    nwb_path, out_dir, *, shank: int, sort: bool, gt_h5=None
):
    """Paired off / dredge / dredge_fast on one shank of a MEArec fixture.

    Reads the fixture with SpikeInterface's NWB reader, keeps one 32-contact
    shank, bandpasses it like the v2 hippocampus recipe, and runs the shipped
    ``dredge_v1`` / ``dredge_fast_v1`` estimation rows and
    ``kriging_force_extrapolate_v1`` interpolation through the stage
    functions. Records the estimate, its correlation with the MEArec drift
    vector when ``gt_h5`` is given (both signs: the fixture's convention is
    unconfirmed), per-channel RMS of the corrected over the uncorrected
    traces (border effects), runtimes and peak RSS; with ``sort`` also sorts
    off and corrected with MountainSort5 and reports unit counts and, with
    ``gt_h5``, agreement with the shank's ground-truth units
    (:func:`_representative_sorting_summary`).
    """
    import spikeinterface as si
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2._motion import (
        apply_motion_on_estimation_clock,
        build_estimation_clock,
        estimate_motion_in_spans,
        resolve_estimation_params,
        resolve_interpolation_params,
    )
    from spyglass.spikesorting.v2._recipe_catalog import (
        KRIGING_FORCE_EXTRAPOLATE,
        motion_estimation_default_contents,
        motion_interpolation_default_contents,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        continuity_from_timestamps,
    )
    from tests.spikesorting.v2._motion_fixtures import JOB_KWARGS, bandpass

    si.set_global_job_kwargs(**JOB_KWARGS)
    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    raw = se.read_nwb_recording(str(nwb_path))
    shanks = np.asarray(raw.get_property("probe_shank"))
    ids = raw.channel_ids[shanks == shank]
    recording = bandpass(raw.select_channels(ids))
    n = recording.get_num_samples()
    fs = recording.get_sampling_frequency()
    continuity = continuity_from_timestamps(recording)
    estimation_rows = {
        row[0]: row[1] for row in motion_estimation_default_contents()
    }
    interpolation = resolve_interpolation_params(
        {r[0]: r[1] for r in motion_interpolation_default_contents()}[
            KRIGING_FORCE_EXTRAPOLATE
        ]
    )
    result: dict = dict(
        nwb=str(nwb_path),
        shank=shank,
        n_channels=len(ids),
        n_samples=int(n),
        duration_s=n / fs,
        continuity_spans=[list(s) for s in continuity.spans],
        interpolation=interpolation,
        load_s=round(time.perf_counter() - t0, 3),
        recipes={},
    )
    gt = None
    if gt_h5 is not None:
        import h5py

        with h5py.File(gt_h5, "r") as f:
            gt = (
                f["drift_list/0/drift_vector_um"][:],
                f["drift_list/0/drift_times"][:],
            )
    window = int(fs)
    starts = np.linspace(window, n - 2 * window, 24).astype(int)
    depths = np.asarray(recording.get_channel_locations())[:, 1]
    order = np.argsort(depths)
    sortings = {}
    if sort:
        t = time.perf_counter()
        spans = continuity.spans
        sortings["off"] = _sort_representative(recording, spans)
        result["off_sort_s"] = round(time.perf_counter() - t, 3)
    for name in ("dredge_v1", "dredge_fast_v1"):
        resolved = resolve_estimation_params(estimation_rows[name])
        clock = build_estimation_clock(
            continuity.spans,
            continuity.start_s,
            continuity.end_s,
            fs,
            max_gap_s=resolved["max_gap_s"],
        )
        t = time.perf_counter()
        motion, diagnostics = estimate_motion_in_spans(
            recording,
            statistics_spans=continuity.spans,
            clock=clock,
            resolved_params=resolved,
            job_kwargs=JOB_KWARGS,
        )
        estimate_s = time.perf_counter() - t
        applied = apply_motion_on_estimation_clock(
            recording,
            motion,
            clock=clock,
            statistics_spans=continuity.spans,
            resolved_interpolation=interpolation,
        )
        corrected = applied.recording
        num = np.zeros(len(ids))
        den = np.zeros(len(ids))
        finite = True
        for start in starts:
            a = corrected.get_traces(
                start_frame=start, end_frame=start + window
            )
            b = recording.get_traces(
                start_frame=start, end_frame=start + window
            )
            finite &= bool(np.isfinite(a).all())
            num += np.sum(a.astype(float) ** 2, axis=0)
            den += np.sum(b.astype(float) ** 2, axis=0)
        ratio = np.sqrt(num / den)[order]
        displacement = np.asarray(motion.displacement[0])
        entry = dict(
            estimate_s=round(estimate_s, 3),
            n_peaks_kept=diagnostics.n_peaks_kept,
            n_temporal_bins=int(displacement.shape[0]),
            spatial_bins_um=np.asarray(motion.spatial_bins_um).tolist(),
            displacement_finite=bool(np.isfinite(displacement).all()),
            displacement_ptp_um=float(np.ptp(displacement)),
            displacement_max_abs_um=float(np.max(np.abs(displacement))),
            n_out_channels=int(corrected.get_num_channels()),
            removed_channel_ids=[str(c) for c in applied.removed_channel_ids],
            corrected_traces_finite=finite,
            rms_ratio_corrected_over_uncorrected_by_depth=ratio.tolist(),
            rms_ratio_edge_contacts=[
                float(ratio[0]),
                float(ratio[1]),
                float(ratio[-2]),
                float(ratio[-1]),
            ],
            rms_ratio_interior_median=float(np.median(ratio[2:-2])),
        )
        if gt is not None:
            vector, times = gt
            edges = motion.temporal_bin_edges_s[0]
            truth = np.array(
                [
                    vector[(times >= lo) & (times < hi)].mean()
                    for lo, hi in zip(edges[:-1], edges[1:])
                ]
            )
            est = displacement.mean(axis=1)
            entry["corr_with_plus_gt"] = _correlation(est, truth)
            for sign, key in ((1, "plus_gt"), (-1, "minus_gt")):
                summary = _error_summary(
                    displacement,
                    sign * truth[:, None] * np.ones_like(displacement),
                )
                entry[f"error_vs_{key}"] = summary
            entry["zero_estimate_rms_um"] = float(
                np.sqrt(np.mean((truth - truth.mean()) ** 2))
            )
        if sort:
            t = time.perf_counter()
            sortings[name] = _sort_representative(corrected, continuity.spans)
            entry["sort_s"] = round(time.perf_counter() - t, 3)
        result["recipes"][name] = entry
    if sort:
        result["sorting"] = _representative_sorting_summary(
            nwb_path, sortings, shank, raw, gt_h5
        )
    result["total_s"] = round(time.perf_counter() - t0, 3)
    result["peak_rss_bytes"] = peak_rss_bytes()
    (out_dir / f"representative_shank{shank}.json").write_text(
        json.dumps(result, indent=1, default=str)
    )
    return result


def _sort_representative(recording, spans):
    from spyglass.spikesorting.v2._params.sorter import MountainSort5Schema
    from spyglass.spikesorting.v2._sorting_dispatch import (
        remove_excess_spikes,
        run_si_sorter,
    )

    sorting = run_si_sorter(
        "mountainsort5",
        MountainSort5Schema().model_dump(),
        recording,
        "representative",
        {"random_seed": 0},
        statistics_spans=spans,
    )
    return remove_excess_spikes(sorting, recording)


def _representative_sorting_summary(
    nwb_path, sortings, shank: int, raw, gt_h5
) -> dict:
    """Sorted unit counts and, with the MEArec work file, agreement with the
    fixture's ground-truth units whose largest template peak is on this shank.

    The fixture's ground-truth spike times (NWB ``processing/ground_truth``)
    and the MEArec file's ``voltage_peaks`` (units x channels) are matched by
    unit index; the spike counts and channel positions are checked to agree
    first. Without ``gt_h5`` only unit counts are reported.
    """
    import h5py
    from spikeinterface.comparison import compare_sorter_to_ground_truth
    from spikeinterface.core import NumpySorting

    summary = {
        name: dict(n_sorted_units=int(sorting.get_num_units()))
        for name, sorting in sortings.items()
    }
    if gt_h5 is None:
        return summary
    with h5py.File(nwb_path, "r") as f:
        units = f["processing/ground_truth/units"]
        times = units["spike_times"][:]
        bounds = np.r_[0, units["spike_times_index"][:]]
    with h5py.File(gt_h5, "r") as h:
        peaks = h["voltage_peaks"][:]
        positions = h["channel_positions"][:]
        counts = [
            h["spiketrains"][str(u)]["times"].shape[0]
            for u in range(peaks.shape[0])
        ]
    if counts != np.diff(bounds).tolist() or not np.allclose(
        positions[:, [2, 1]], raw.get_channel_locations()
    ):
        raise ValueError(
            "the MEArec work file does not match the fixture's units or "
            "channel positions."
        )
    shanks = np.asarray(raw.get_property("probe_shank"))
    on_shank = shanks[np.argmax(np.abs(peaks), axis=1)] == shank
    fs = raw.get_sampling_frequency()
    t0 = raw.get_start_time()
    frames, labels = [], []
    for u in np.flatnonzero(on_shank):
        spikes = times[bounds[u] : bounds[u + 1]]
        frames.append(np.round((spikes - t0) * fs).astype(np.int64))
        labels.append(np.full(spikes.size, u))
    summary["n_gt_units_on_shank"] = int(on_shank.sum())
    if not frames:
        return summary
    frames = np.concatenate(frames)
    labels = np.concatenate(labels)
    order = np.argsort(frames, kind="stable")
    gt = NumpySorting.from_samples_and_labels(
        [frames[order]], [labels[order]], fs
    )
    for name, sorting in sortings.items():
        cmp = compare_sorter_to_ground_truth(gt, sorting, exhaustive_gt=False)
        perf = cmp.get_performance(method="by_unit")
        summary[name].update(
            mean_accuracy_on_shank_gt=float(perf["accuracy"].mean()),
            n_well_detected=int(cmp.count_well_detected_units(0.8)),
        )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--manifest", required=True)
    case.add_argument("--scenario", required=True)
    case.add_argument("--seed", type=int, required=True)
    case.add_argument("--recipe", required=True)
    case.add_argument("--out", required=True)
    rep = sub.add_parser("representative")
    rep.add_argument("--nwb", required=True)
    rep.add_argument("--out", required=True)
    rep.add_argument("--shank", type=int, default=2)
    rep.add_argument("--sort", action="store_true")
    rep.add_argument("--gt-h5", default=None)
    args = parser.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.command == "case":
        from tests.spikesorting.v2._motion_acceptance import load_manifest

        run_case(
            load_manifest(args.manifest),
            args.manifest,
            args.scenario,
            args.seed,
            args.recipe,
            args.out,
        )
    else:
        run_representative(
            args.nwb,
            args.out,
            shank=args.shank,
            sort=args.sort,
            gt_h5=args.gt_h5,
        )


if __name__ == "__main__":
    main()
