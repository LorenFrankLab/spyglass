# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Cross-Session Unit Tracking with UnitMatch
#
# This notebook walks the [UnitMatchPy](https://github.com/EnnyvanBeest/UnitMatch)
# API end-to-end against a modern (SortingAnalyzer-based) spike-sorting analyzer,
# to pin the matcher's real API surface, input/output layout, and failure modes
# before a DataJoint wrapper is written.
#
# **What it does**
# 1. Builds two pseudo-sessions from the 128-channel Frank-lab polymer-probe
#    MEArec fixture (`mearec_polymer_128ch_60s.nwb`) using its ground-truth
#    spike trains, with a known overlapping unit set so we can score matching.
# 2. Builds a modern `SortingAnalyzer` per session (`templates` / `waveforms` /
#    `unit_locations` extensions) — the analyzer the production pipeline produces.
# 3. Extracts a **self-contained UnitMatch bundle** per session (per-unit
#    split-half average waveforms + channel positions). The matcher consumes the
#    bundle only — never the analyzer object, the recording, or any database key.
# 4. Runs UnitMatch end-to-end and scores the cross-session match probabilities
#    against the ground-truth correspondence.
#
# **Environment.** Isolated `uv` virtualenv, Python 3.11, `UnitMatchPy==3.2.7`,
# `spikeinterface==0.104.3`, `numpy==2.4.6`. UnitMatchPy and `mat73` are an
# optional dependency group; they are not installed into the base environment.

# +
import time
import resource
import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless; figures are closed explicitly
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core import NumpySorting
from spikeinterface.extractors import read_nwb_recording
from pynwb import NWBHDF5IO

from importlib.metadata import version

import UnitMatchPy  # noqa: F401  (top-level import runs GUI; see caveat below)

print("UnitMatchPy", version("UnitMatchPy"))
print("spikeinterface", si.__version__, "| numpy", np.__version__)
# -


# ## numpy-2 compatibility shim for UnitMatchPy 3.2.7
#
# UnitMatchPy 3.2.7 declares `numpy>=2` support, but its metric path does **not**
# run under numpy 2 unmodified. `param_functions.get_avg_waveform_per_tp` builds a
# per-unit good-time window with
# `np.arange(wave_duration_tmp[0], wave_duration_tmp[-1] + 1)`, where
# `wave_duration_tmp` is a `np.argwhere` result (shape `(k, 1)`) — so the
# endpoints are 1-element arrays. numpy < 2 accepted 1-element arrays as scalars;
# numpy 2 raises `TypeError: only 0-dimensional arrays can be converted to Python
# scalars`. A bare `except` swallows this for every unit, mislabels them
# "very likely bad", and leaves the per-time-point trajectory written at relative
# indices instead of absolute samples → every centroid distance becomes NaN → the
# candidate-pair set is empty → the auto-threshold step crashes.
#
# The fix is a wrapper-owned numpy proxy, scoped to `param_functions`, whose
# `arange` coerces 1-element-array endpoints to scalars (a no-op for scalar args).
# It does not edit the installed package and keeps the numpy>=2 baseline.

# +
import numpy as _np
import UnitMatchPy.param_functions as _pf


class _ArangeProxy:
    """numpy proxy that coerces 1-element-array ``arange`` endpoints to scalars."""

    def arange(self, start, stop=None, *args, **kwargs):
        start = start.item() if getattr(start, "size", None) == 1 else start
        if stop is not None:
            stop = stop.item() if getattr(stop, "size", None) == 1 else stop
            return _np.arange(start, stop, *args, **kwargs)
        return _np.arange(start, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(_np, name)


_pf.np = _ArangeProxy()

import UnitMatchPy.extract_raw_data as erd
import UnitMatchPy.utils as uutil
import UnitMatchPy.overlord as ov
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.save_utils as su
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

# -

# ## Read the polymer fixture (recording + ground-truth spike trains)
#
# The fixture stores its recording as an `ElectricalSeries` named `"e-series"`
# and its planted ground-truth units in a sidecar `ground_truth/units` table
# (the top-level `Units` slot is intentionally left free). We read both with
# SpikeInterface + pynwb — no Spyglass/DataJoint import is required for the spike.


# +
def find_fixture(name="mearec_polymer_128ch_60s.nwb"):
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "tests" / "spikesorting" / "v2" / "fixtures" / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"could not locate {name} from {here}")


FIXTURE = find_fixture()
recording = read_nwb_recording(
    FIXTURE, electrical_series_path="acquisition/e-series"
)
fs = recording.get_sampling_frequency()
recording = spre.bandpass_filter(recording, freq_min=300.0, freq_max=6000.0)
# The fixture stores contacts in the XY plane with rel_z = 0, so SpikeInterface
# builds a 3-D probe. Flatten it to 2-D (the z column is all zeros) or the
# `unit_locations` monopolar/center-of-mass solvers break on the extra axis.
recording = recording.set_probe(recording.get_probe().to_2d())

with NWBHDF5IO(str(FIXTURE), "r") as io:
    gt = io.read().processing["ground_truth"].data_interfaces["units"]
    # NumpySorting.from_times_and_labels takes spike times in SECONDS (the
    # docstring's "spike samples" wording is wrong); pass GT seconds verbatim.
    gt_trains = {
        int(u): np.asarray(gt["spike_times"][i]) for i, u in enumerate(gt.id[:])
    }

print(
    f"{recording.get_num_channels()} channels, {fs/1000:.0f} kHz, "
    f"{recording.get_num_samples()/fs:.0f} s, {len(gt_trains)} ground-truth units"
)
# -

# ## Two pseudo-sessions with a known overlapping unit set
#
# We have one short recording, so we synthesize two "sessions" from disjoint time
# windows of the same neurons and keep an **overlapping** subset of ground-truth
# units in each. The intersection are true cross-session correspondences; the rest
# are session-unique. (A dedicated validation fixture — a true two-session
# polymer recording with planted correspondences and inter-session drift — is
# future work; this walkthrough only needs a realistic input to exercise the API.)

# +
S1_IDS = list(range(0, 16))  # session 1 keeps units 0..15 over the first half
S2_IDS = list(range(8, 24))  # session 2 keeps units 8..23 over the second half
SHARED = sorted(
    set(S1_IDS) & set(S2_IDS)
)  # 8..15 -> 8 true cross-session pairs
print("shared (true match) units:", SHARED)


def session_sorting(unit_ids):
    labels = np.concatenate([np.full(len(gt_trains[u]), u) for u in unit_ids])
    times = np.concatenate([gt_trains[u] for u in unit_ids])  # seconds
    order = np.argsort(times)
    return NumpySorting.from_times_and_labels(
        times[order], labels[order], sampling_frequency=fs
    )


n = recording.get_num_samples()
mid = n // 2
sessions = {
    "Session1": (
        recording.frame_slice(0, mid),
        session_sorting(S1_IDS).frame_slice(0, mid),
    ),
    "Session2": (
        recording.frame_slice(mid, n),
        session_sorting(S2_IDS).frame_slice(mid, n),
    ),
}
# -

# ## A modern SortingAnalyzer (the production analyzer)
#
# The production pipeline builds a sparse `SortingAnalyzer` per sort, carrying the
# `templates`, `waveforms`, and `unit_locations` extensions. We build one here for
# Session 1 to show what the matcher's wrapper starts from. Note it is **sparse**
# and computed over the **whole** recording — it does not, on its own, provide the
# two cross-validation half-templates UnitMatch needs (see the next section).

srec1, ssort1 = sessions["Session1"]
v2_analyzer = si.create_sorting_analyzer(ssort1, srec1, sparse=True)
v2_analyzer.compute(["random_spikes", "noise_levels", "templates", "waveforms"])
v2_analyzer.compute("unit_locations")
print(
    "v2 analyzer extensions:", sorted(v2_analyzer.get_loaded_extension_names())
)
print(
    "templates extension shape:",
    v2_analyzer.get_extension("templates").get_data().shape,
    "(sparse)",
)

# ## Wrapper-owned UnitMatch bundle
#
# UnitMatch needs, per unit, an average waveform of shape
# `(spike_width, n_channels, 2)` — the last axis is the two cross-validation
# halves — **dense across all channels**. The sparse single-recording analyzer
# above cannot supply this, so the wrapper re-extracts dense templates on the two
# time-halves of the (curated) recording. In the DataJoint wrapper this recording
# comes from `CurationV2.get_recording(key)`; here it is the SpikeInterface
# recording in hand. The matcher only ever sees the written bundle.
#
# **Bundle layout** (one directory per session — the `paths_from_KS` convention):
# - `RawWaveforms/Unit{id}_RawSpikes.npy` — one file per unit, `(spike_width, n_channels, 2)`
# - `channel_positions.npy` — `(n_channels, 2)`
# - `cluster_group.tsv` — `cluster_id`/`group` columns; `good` rows are matched


# +
def write_bundle(session_dir, srec, ssort):
    session_dir.mkdir(parents=True, exist_ok=True)
    half = srec.get_num_samples() // 2
    t_halves = []
    for a0, a1 in [(0, half), (half, srec.get_num_samples())]:
        analyzer = si.create_sorting_analyzer(
            ssort.frame_slice(a0, a1), srec.frame_slice(a0, a1), sparse=False
        )
        analyzer.compute(
            "random_spikes", method="uniform", max_spikes_per_unit=100, seed=0
        )
        # Symmetric window so the trough sits at the centre sample; UnitMatch's
        # load_good_waveforms forces peak_loc = spike_width // 2, so an asymmetric
        # window would misplace the peak and flag every unit "bad".
        analyzer.compute("waveforms", ms_before=1.5, ms_after=1.5)
        analyzer.compute("templates")
        t_halves.append(analyzer.get_extension("templates").get_data())
    avg_waves = np.stack(
        t_halves, axis=-1
    )  # (n_units, spike_width, n_channels, 2)
    unit_ids = np.asarray(ssort.get_unit_ids(), dtype=int)
    np.save(session_dir / "channel_positions.npy", srec.get_channel_locations())
    erd.save_avg_waveforms(
        avg_waves,
        str(session_dir),
        unit_ids,
        unit_ids,
        extract_good_units_only=False,
    )
    rows = [np.array(("cluster_id", "group"))] + [
        np.array((str(i), "good")) for i in unit_ids
    ]
    np.savetxt(
        session_dir / "cluster_group.tsv",
        np.vstack(rows),
        fmt=["%s", "%s"],
        delimiter="\t",
    )


bundle_root = Path(tempfile.mkdtemp(prefix="unitmatch_bundle_"))
session_dirs = []
for name, (srec, ssort) in sessions.items():
    sdir = bundle_root / name
    write_bundle(sdir, srec, ssort)
    session_dirs.append(str(sdir))

sample = next(
    (Path(session_dirs[0]) / "RawWaveforms").glob("Unit*_RawSpikes.npy")
)
print("bundle files:", sorted(p.name for p in Path(session_dirs[0]).iterdir()))
print(
    "per-unit RawWaveforms shape:",
    np.load(sample).shape,
    "(spike_width, n_channels, 2)",
)
# -

# ## Run UnitMatch
#
# The pipeline is: load default params → estimate probe geometry → load the
# per-session bundles into one stacked `waveform` array → zero-centre (SI
# templates carry a DC offset) → extract waveform parameters → extract metric
# scores → naive-Bayes probability. `get_probe_geometry` must see the **raw 2-D**
# `(x, y)` positions: `paths_from_KS` prepends a column of ones (its 3-D
# convention), which would otherwise collapse the four shanks to one.

# +
param = default_params.get_default_param()
param["KS_dirs"] = session_dirs
wave_paths, label_paths, channel_pos = uutil.paths_from_KS(session_dirs)
raw_positions = np.load(Path(session_dirs[0]) / "channel_positions.npy")
param = uutil.get_probe_geometry(raw_positions, param)
print(f"probe geometry: {param['no_shanks']} shanks")

waveform, session_id, session_switch, within_session, good_units, param = (
    uutil.load_good_waveforms(
        wave_paths, label_paths, param, good_units_only=True
    )
)
# Zero-centre: subtract the mean of the first 15 samples (per-unit, per-channel).
waveform = waveform - np.broadcast_to(
    waveform[:, :15, :, :].mean(axis=1)[:, np.newaxis, :, :], waveform.shape
)
clus_info = {
    "good_units": good_units,
    "session_switch": session_switch,
    "session_id": session_id,
    "original_ids": np.concatenate(good_units),
}
print(
    f"stacked waveform {waveform.shape}: n_units={param['n_units']} "
    f"(per session {param['n_units_per_session']}), spike_width={param['spike_width']}, "
    f"peak_loc={param['peak_loc']}, n_channels={param['n_channels']}"
)

# +
t_start = time.monotonic()
extracted = ov.extract_parameters(waveform, channel_pos, clus_info, param)
total_score, candidate_pairs, scores_to_include, predictors = (
    ov.extract_metric_scores(
        extracted, session_switch, within_session, param, niter=2
    )
)
# n_expected_matches is populated by extract_metric_scores (this call order matters)
prior_match = 1 - (param["n_expected_matches"] / param["n_units"] ** 2)
priors = np.array((prior_match, 1 - prior_match))
labels = candidate_pairs.astype(int)
cond = np.unique(labels)
kernels = bf.get_parameter_kernels(
    scores_to_include, labels, cond, param, add_one=1
)
probability = bf.apply_naive_bayes(kernels, priors, predictors, param, cond)
prob_matrix = probability[:, 1].reshape(param["n_units"], param["n_units"])
runtime_s = time.monotonic() - t_start

print(
    f"probability matrix {prob_matrix.shape}, range [{prob_matrix.min():.3f}, {prob_matrix.max():.3f}]"
)
print(f"metric scores: {list(scores_to_include.keys())}")
# -

# ## Output: match table and cross-session unique IDs
#
# `save_utils.make_match_table` returns a pandas `DataFrame` with one row per unit
# pair. `assign_unique_id` assigns each unit a cross-session biological-identity
# label at three stringency tiers — **Conservative** (a unit joins a group only if
# it matches *every* member: a maximal clique), **Intermediate** (matches every
# member in the same/adjacent session), **Liberal** (matches *any* member: a
# connected component). The Conservative tier is the direct analog of a strict
# maximal-clique tracked-unit derivation.
#
# Note what UnitMatch does **not** emit as per-pair columns: there is no per-pair
# drift or FDR value. Drift is estimated and applied internally per session-pair;
# the false-positive rate is a session-level diagnostic printed by
# `utils.evaluate_output`, not a column.

threshold = param["match_threshold"]
above_threshold = prob_matrix > threshold
output_threshold = above_threshold.astype(int)
matches = np.argwhere(above_threshold & (within_session == 0))
unique_ids = aid.assign_unique_id(prob_matrix, param, clus_info)
# Return order is [Liberal, Intermediate, Conservative, default] (verified
# against assign_unique_id source). Conservative (index 2) is the maximal-clique
# tier — the strict tracked-unit analog; index it explicitly, not by guesswork.
conservative_uids = unique_ids[2]
match_table = su.make_match_table(
    scores_to_include,
    matches,
    prob_matrix,
    total_score,
    output_threshold,
    clus_info,
    param,
    UIDs=unique_ids,
)
print(
    "assign_unique_id -> list of",
    np.shape(unique_ids),
    "(rows: Liberal / Intermediate / Conservative / default);",
    "n distinct Conservative groups:",
    len(np.unique(conservative_uids)),
)
print("make_match_table ->", type(match_table).__name__, match_table.shape)
print("columns:", list(match_table.columns))
match_table.head()

# ## Match quality vs. ground truth
#
# The cross-session block is `prob_matrix[:n1, n1:]` (Session 1 rows × Session 2
# columns). True correspondences are pairs sharing a ground-truth unit id.

# +
# Session boundary in the stacked matrix. Use session_switch (the matched-unit
# boundary), NOT param["n_units_per_session"], which holds TOTAL tsv rows rather
# than good-unit counts — they coincide here only because every unit is `good`.
n1 = int(session_switch[1])
cross = prob_matrix[:n1, n1:]
gt_match = np.array(S1_IDS)[:, None] == np.array(S2_IDS)[None, :]
true_pairs = cross[gt_match]
false_pairs = cross[~gt_match]

scores = np.concatenate([true_pairs, false_pairs])
order = np.argsort(scores)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(scores) + 1)
auc = (
    ranks[: true_pairs.size].sum() - true_pairs.size * (true_pairs.size + 1) / 2
) / (true_pairs.size * false_pairs.size)

peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print(f"true pairs:  mean={true_pairs.mean():.3f}  min={true_pairs.min():.3f}")
print(
    f"false pairs: mean={false_pairs.mean():.3f}  max={false_pairs.max():.3f}"
)
print(
    f"true > {threshold}: {(true_pairs > threshold).sum()}/{true_pairs.size}  |  "
    f"AUC = {auc:.3f}"
)
print(
    f"compute cost: {runtime_s:.1f} s inference, {peak_rss_gb:.2f} GB peak RSS"
)
assert (
    auc > 0.85
), "cross-session match probability failed to separate true pairs"
# -

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cross, cmap="magma", vmin=0, vmax=1, aspect="auto")
ax.set_xlabel("Session 2 unit index")
ax.set_ylabel("Session 1 unit index")
ax.set_title("Cross-session match probability")
fig.colorbar(im, ax=ax, label="UnitMatch probability")
fig.tight_layout()
plt.close(fig)
print("cross-session probability heatmap rendered (Agg backend; figure closed)")

# ## API summary (for the DataJoint wrapper)
#
# - **Entry points** (not `run.py`, which is a non-importable script):
#   `default_params.get_default_param`, `utils.paths_from_KS`,
#   `utils.get_probe_geometry`, `utils.load_good_waveforms`,
#   `overlord.extract_parameters`, `overlord.extract_metric_scores`,
#   `bayes_functions.get_parameter_kernels`, `bayes_functions.apply_naive_bayes`,
#   `save_utils.make_match_table`, `assign_unique_id.assign_unique_id`.
# - **Matcher input** = a per-session directory bundle
#   (`RawWaveforms/Unit{id}_RawSpikes.npy` `(spike_width, n_channels, 2)`,
#   `channel_positions.npy` `(n_channels, 2)`, `cluster_group.tsv`). The wrapper
#   builds it from the curated sorting + recording; the matcher never receives the
#   analyzer object, the recording, or any database key.
# - **Matcher output** = an `(n_total_units, n_total_units)` probability matrix
#   plus a `make_match_table` DataFrame (per-pair probability + sub-scores) and
#   `assign_unique_id` cross-session identities. No per-pair drift/FDR columns.
# - **Required adaptations**: the numpy-2 `arange` shim; a symmetric waveform
#   window; pass raw 2-D positions to `get_probe_geometry`; spike times in seconds
#   for `NumpySorting.from_times_and_labels`.
