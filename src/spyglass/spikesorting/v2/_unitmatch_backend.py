"""UnitMatch cross-session matcher backend.

Concrete :class:`~spyglass.spikesorting.v2.matcher_protocol.MatcherProtocol`
implementation wrapping `UnitMatchPy <https://github.com/EnnyvanBeest/UnitMatch>`_.

Two roles live here:

- :func:`extract_unitmatch_bundle` -- the *wrapper* helper that turns a curated
  SpikeInterface sorting + recording into a UnitMatch directory bundle (dense
  split-half templates + channel positions + good-unit labels). ``UnitMatch.make``
  calls this per session; it reads the recording (e.g. via
  ``CurationV2.get_recording``) but writes only the self-contained bundle.
- :class:`UnitMatchBackend.match` -- the *matcher*: it reads the prepared bundle
  directories and runs the UnitMatch inference, returning cross-session pairs.
  It never touches a recording, a ``SortingAnalyzer``, or a Spyglass key.

UnitMatchPy is an optional dependency (``pip install -e
".[spikesorting-v2-matching]"``); the import is guarded so a missing install --
or a Tk-less top-level ``import UnitMatchPy`` -- raises a clear, actionable error
rather than a cryptic one. UnitMatchPy 3.2.7's metric path is numpy-2 broken
(``param_functions.get_avg_waveform_per_tp`` calls ``np.arange`` with 1-element
``np.argwhere`` endpoints, which numpy>=2 rejects and UnitMatch's bare ``except``
silently swallows -- corrupting the waveform trajectory), so the guard installs
a numpy-2 ``arange`` shim scoped to its ``param_functions`` module.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
from spyglass.spikesorting.v2.matcher_protocol import (
    MatchPair,
    SessionMatcherInput,
    register_matcher,
)

_INSTALL_HINT = (
    'UnitMatchPy is required for cross-session matching. Install it with '
    '`pip install -e ".[spikesorting-v2-matching]"` (UnitMatchPy>=3.2.6,<3.2.8 '
    "+ mat73). If the import fails on `_tkinter`, the top-level "
    "`import UnitMatchPy` is loading its Tk GUI; run in a Tk-enabled "
    "environment."
)


class _ArangeProxy:
    """numpy proxy that coerces 1-element-array ``arange`` endpoints to scalars.

    UnitMatchPy 3.2.7's ``param_functions.get_avg_waveform_per_tp`` calls
    ``np.arange`` with ``np.argwhere`` endpoints (1-element arrays). numpy<2
    accepted those as scalars; numpy>=2 raises ``TypeError``, which UnitMatch's
    bare ``except`` swallows and silently corrupts the waveform trajectory.
    Coercing the endpoints (a no-op for true scalars) restores numpy<2 behavior
    without editing the installed package.
    """

    def arange(self, start, stop=None, *args, **kwargs):
        start = start.item() if getattr(start, "size", None) == 1 else start
        if stop is not None:
            stop = stop.item() if getattr(stop, "size", None) == 1 else stop
            return np.arange(start, stop, *args, **kwargs)
        return np.arange(start, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(np, name)


@functools.lru_cache(maxsize=1)
def _require_unitmatch() -> SimpleNamespace:
    """Import the UnitMatchPy submodules + install the numpy-2 shim (cached).

    Returns a namespace of the submodules used by the backend. Raises
    ``ImportError`` with an actionable install hint if UnitMatchPy is absent or
    its top-level (GUI) import fails.
    """
    try:
        import UnitMatchPy.assign_unique_id as assign_unique_id
        import UnitMatchPy.bayes_functions as bayes_functions
        import UnitMatchPy.default_params as default_params
        import UnitMatchPy.extract_raw_data as extract_raw_data
        import UnitMatchPy.overlord as overlord
        import UnitMatchPy.param_functions as param_functions
        import UnitMatchPy.utils as utils
    except ImportError as exc:  # missing package or Tk-less GUI import
        raise ImportError(_INSTALL_HINT) from exc

    # numpy-2 compatibility shim (idempotent; only swaps once).
    if not isinstance(param_functions.np, _ArangeProxy):
        param_functions.np = _ArangeProxy()

    return SimpleNamespace(
        assign_unique_id=assign_unique_id,
        bayes_functions=bayes_functions,
        default_params=default_params,
        extract_raw_data=extract_raw_data,
        overlord=overlord,
        param_functions=param_functions,
        utils=utils,
    )


def _bundle_compute_kwargs(
    seed: int, job_kwargs: dict | None
) -> tuple[int, dict]:
    """Resolve the bundle random seed + the SI ``compute`` job kwargs.

    The bundle ``seed`` is authoritative: a stray ``random_seed`` in
    ``job_kwargs`` (an ambient ``dj.config`` seed, or a value leaked from a
    params blob) is stripped and IGNORED -- it never overrides ``seed``. Letting
    it win would make the stored, identity-bearing ``seed`` disagree with the
    seed actually used. ``random_seed`` is also not a valid
    ``SortingAnalyzer.compute`` kwarg (SI raises "please remove
    {'random_seed'}"), so stripping it is required regardless.
    """
    compute_job_kwargs = dict(job_kwargs or {})
    compute_job_kwargs.pop("random_seed", None)
    return seed, compute_job_kwargs


def extract_unitmatch_bundle(
    session_dir,
    recording,
    sorting,
    *,
    ms_before: float = 1.5,
    ms_after: float = 1.5,
    max_spikes_per_unit: int = 100,
    seed: int = 0,
    job_kwargs: dict | None = None,
):
    """Write a UnitMatch directory bundle for one curated session.

    UnitMatch needs a dense per-unit average waveform of shape
    ``(spike_width, n_channels, 2)`` -- two cross-validation halves across ALL
    channels. The v2 canonical analyzer is sparse and single-recording, so this
    re-extracts: split the recording into two halves, build a dense
    (``sparse=False``) analyzer per half, compute templates, and stack. A
    *symmetric* waveform window keeps the trough at the centre sample, matching
    UnitMatch's ``peak_loc = spike_width // 2`` assumption.

    Parameters
    ----------
    session_dir : path-like
        Output directory; created if absent. Receives ``RawWaveforms/`` plus
        ``channel_positions.npy`` and ``cluster_group.tsv``.
    recording, sorting : spikeinterface objects
        The curated recording + sorting for this session.
    ms_before, ms_after : float
        Symmetric waveform window (default 1.5/1.5 ms).
    max_spikes_per_unit : int
        Random-spike cap per unit per half (default 100).
    seed : int
        Random-spikes seed for determinism (default 0).
    job_kwargs : dict or None
        SpikeInterface job kwargs (``n_jobs`` / ``chunk_duration`` / ...) splatted
        into the ``waveforms`` / ``templates`` compute calls. ``UnitMatch.make``
        resolves these from ``MatcherParameters.job_kwargs``; ``None`` uses the
        SpikeInterface defaults.
    """
    import spikeinterface as si

    um = _require_unitmatch()
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    random_seed, compute_job_kwargs = _bundle_compute_kwargs(seed, job_kwargs)

    half = recording.get_num_samples() // 2
    t_halves = []
    for a0, a1 in [(0, half), (half, recording.get_num_samples())]:
        analyzer = si.create_sorting_analyzer(
            sorting.frame_slice(a0, a1),
            recording.frame_slice(a0, a1),
            sparse=False,
        )
        analyzer.compute(
            "random_spikes",
            method="uniform",
            max_spikes_per_unit=max_spikes_per_unit,
            seed=random_seed,
        )
        analyzer.compute(
            "waveforms",
            ms_before=ms_before,
            ms_after=ms_after,
            **compute_job_kwargs,
        )
        analyzer.compute("templates", **compute_job_kwargs)
        t_halves.append(analyzer.get_extension("templates").get_data())

    avg_waves = np.stack(t_halves, axis=-1)  # (n_units, spike_width, n_chan, 2)
    unit_ids = np.asarray(sorting.get_unit_ids(), dtype=int)
    np.save(session_dir / "channel_positions.npy", recording.get_channel_locations())
    um.extract_raw_data.save_avg_waveforms(
        avg_waves, str(session_dir), unit_ids, unit_ids, extract_good_units_only=False
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


def _zero_center(waveform: np.ndarray) -> np.ndarray:
    """Subtract the mean of the first 15 samples (SI templates carry a DC offset).

    The 15-sample window assumes the pre-spike baseline occupies at least the
    first 15 samples, i.e. the peak sits well after sample 15. This holds for the
    bundles written here -- the default symmetric ``ms_before`` (~1.5 ms) at the
    recording's sampling rate puts the trough tens of samples in. ``ms_before``
    is not user-configurable today; if it ever is, derive this window from the
    peak location / ``ms_before`` instead of the literal 15.
    """
    return waveform - np.broadcast_to(
        waveform[:, :15, :, :].mean(axis=1)[:, np.newaxis, :, :], waveform.shape
    )


def assert_consistent_channel_geometry(named_positions) -> None:
    """Reject sessions that do not share one probe geometry.

    UnitMatch derives geometry from the FIRST session's channel positions and
    runs per-channel loops that require every session to share that geometry;
    cross-probe / cross-day geometry matching is out of scope. This pure check
    raises a clear ``ValueError`` on the first session whose positions differ
    (by shape or value) from the first session's. Shared by the backend
    ``match()`` (after bundle extraction) and ``UnitMatchSelection.insert_selection``
    (a preflight BEFORE the expensive dense bundle extraction), so both reject a
    geometry mismatch the same way.

    Parameters
    ----------
    named_positions : sequence of (label, ndarray)
        Ordered ``(session_label, channel_positions)`` pairs. Fewer than two
        sessions trivially passes (nothing to compare against).

    Raises
    ------
    ValueError
        On the first session whose channel-position array differs from the
        first session's (shape mismatch or not ``np.allclose``).
    """
    named_positions = list(named_positions)
    if len(named_positions) < 2:
        return
    ref_label, ref_positions = named_positions[0]
    ref_positions = np.asarray(ref_positions)
    for label, positions in named_positions[1:]:
        positions = np.asarray(positions)
        if positions.shape != ref_positions.shape or not np.allclose(
            positions, ref_positions
        ):
            raise ValueError(
                "UnitMatch requires all sessions to share one probe geometry "
                "(cross-probe / cross-day geometry matching is out of scope), "
                f"but session {label} has channel positions {positions.shape} "
                f"that differ from session {ref_label}'s {ref_positions.shape}. "
                "Group only sessions recorded on the same probe."
            )


class UnitMatchBackend:
    """The ``unitmatch`` cross-session matcher backend."""

    name = "unitmatch"

    @staticmethod
    def backend_version() -> str | None:
        """Installed ``unitmatchpy`` distribution version, or ``None`` if absent.

        The producing-library provenance the ``UnitMatch`` table records. Kept on
        the backend (single source of truth) so the table does not hardcode the
        package name; ``None`` rather than a guess when the package is missing.
        """
        import importlib.metadata

        try:
            return importlib.metadata.version("unitmatchpy")
        except importlib.metadata.PackageNotFoundError:
            return None

    def match(
        self,
        session_inputs: list[SessionMatcherInput],
        params: dict,
    ) -> list[MatchPair]:
        """Run UnitMatch over the prepared per-session bundles.

        Returns ``[]`` for the degenerate single-session case without importing
        or calling UnitMatch.
        """
        if len(session_inputs) < 2:
            return []

        um = _require_unitmatch()
        param = um.default_params.get_default_param()
        match_threshold = float(
            params.get("match_threshold", param["match_threshold"])
        )
        param["match_threshold"] = match_threshold

        # UnitMatch assumes ONE probe across the group: it derives geometry from
        # the first session's channel positions and runs per-channel loops that
        # require every session to share that geometry. Cross-probe matching is
        # out of scope, so reject mismatched geometry up front with a clear error
        # rather than letting UnitMatch fail deep in a shape mismatch. The same
        # check runs as a preflight in UnitMatchSelection.insert_selection (before
        # bundle extraction); this is the post-extraction backstop.
        assert_consistent_channel_geometry(
            [
                (s.session_key, np.load(s.channel_positions_path))
                for s in session_inputs
            ]
        )
        raw_positions = np.load(session_inputs[0].channel_positions_path)

        session_dirs = [str(s.waveform_dir) for s in session_inputs]
        param["KS_dirs"] = session_dirs
        wave_paths, label_paths, channel_pos = um.utils.paths_from_KS(session_dirs)
        # get_probe_geometry needs the raw 2-D positions (paths_from_KS prepends
        # a ones column that would collapse the shanks to one).
        param = um.utils.get_probe_geometry(raw_positions, param)

        waveform, session_id, session_switch, within_session, good_units, param = (
            um.utils.load_good_waveforms(
                wave_paths, label_paths, param, good_units_only=True
            )
        )
        # load_good_waveforms silently DROPS a session whose bundle fails to
        # load (it does not raise). If that happened, the compact session
        # indexes below would attribute one session's units to another
        # session's Spyglass key -- fail loudly instead.
        if len(good_units) != len(session_inputs):
            raise RuntimeError(
                f"UnitMatch loaded {len(good_units)} session(s) but "
                f"{len(session_inputs)} were provided; a session bundle failed "
                "to load (UnitMatchPy drops it silently), so units cannot be "
                "attributed to the correct session. Check each session bundle's "
                "RawWaveforms/ and cluster_group.tsv."
            )
        # No good units across the loaded sessions -> no pairs. Return early
        # before the prior-probability computation below, which divides by
        # ``n_units ** 2``. The table layer never reaches this (make_fetch
        # rejects an empty matchable set), but this backend is a public
        # MatcherProtocol implementation and must not assume that precondition.
        if param["n_units"] == 0:
            return []
        waveform = _zero_center(waveform)
        clus_info = {
            "good_units": good_units,
            "session_switch": session_switch,
            "session_id": session_id,
            "original_ids": np.concatenate(good_units),
        }
        extracted = um.overlord.extract_parameters(
            waveform, channel_pos, clus_info, param
        )
        total_score, candidate_pairs, scores_to_include, predictors = (
            um.overlord.extract_metric_scores(
                extracted, session_switch, within_session, param, niter=2
            )
        )
        prior_match = 1 - (param["n_expected_matches"] / param["n_units"] ** 2)
        priors = np.array((prior_match, 1 - prior_match))
        labels = candidate_pairs.astype(int)
        cond = np.unique(labels)
        kernels = um.bayes_functions.get_parameter_kernels(
            scores_to_include, labels, cond, param, add_one=1
        )
        probability = um.bayes_functions.apply_naive_bayes(
            kernels, priors, predictors, param, cond
        )
        prob_matrix = probability[:, 1].reshape(
            param["n_units"], param["n_units"]
        )

        return self._pairs_from_matrix(
            prob_matrix,
            session_switch,
            clus_info["original_ids"],
            session_inputs,
            match_threshold,
        )

    @staticmethod
    def _pairs_from_matrix(
        prob_matrix,
        session_switch,
        original_ids,
        session_inputs,
        match_threshold,
    ) -> list[MatchPair]:
        """Unflatten the probability matrix into cross-session MatchPairs.

        UnitMatch's probability matrix is asymmetric (one entry per directed
        cross-validation comparison). Following UnitMatch's own grouping
        (``assign_unique_id``), a pair is emitted only when BOTH directions
        ``prob_matrix[i, j]`` and ``prob_matrix[j, i]`` clear the threshold, and
        the reported ``match_probability`` is their mean -- so a one-sided /
        borderline match that UnitMatch would reject is dropped, and the
        probability is orientation-independent.

        Side A is the session that appears first in ``session_inputs`` (the
        caller owns that ordering); the ``i < j`` loop with the within-session
        skip emits each unordered cross-session pair exactly once, so reversed
        duplicates cannot both appear.
        """
        boundaries = np.asarray(session_switch).ravel()
        n = prob_matrix.shape[0]
        if n == 0:
            return []
        prob = np.asarray(prob_matrix, dtype=float)

        # Build the survivor mask in NumPy rather than looping over every
        # (i, j): a pair survives iff i < j (each unordered pair once), the two
        # units are in different sessions, and BOTH directed probabilities clear
        # the threshold (``min(p_ij, p_ji) > threshold``). Then loop only over
        # the surviving matches to construct MatchPairs.
        session_ids = (
            np.searchsorted(boundaries, np.arange(n), side="right") - 1
        )
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        cross_session = session_ids[:, None] != session_ids[None, :]
        both_pass = (prob > match_threshold) & (prob.T > match_threshold)
        mask = upper & cross_session & both_pass
        mean_prob = (prob + prob.T) / 2.0  # symmetric, orientation-independent

        def unit_of(stacked_index) -> int:
            return int(np.asarray(original_ids[stacked_index]).item())

        pairs: list[MatchPair] = []
        # np.argwhere yields (i, j) in row-major (i then j) order, so each
        # unordered cross-session pair appears once, side A = the earlier
        # session block (i < j).
        for i, j in np.argwhere(mask):
            key_a = session_inputs[int(session_ids[i])].session_key
            key_b = session_inputs[int(session_ids[j])].session_key
            pairs.append(
                MatchPair(
                    session_a_sorting_id=str(key_a["sorting_id"]),
                    session_a_curation_id=int(key_a["curation_id"]),
                    unit_a_id=unit_of(i),
                    session_b_sorting_id=str(key_b["sorting_id"]),
                    session_b_curation_id=int(key_b["curation_id"]),
                    unit_b_id=unit_of(j),
                    match_probability=float(mean_prob[i, j]),
                )
            )
        return pairs


def register() -> None:
    """Register the UnitMatch backend + schema (idempotent).

    Called at import for the usual side-effect path, and re-callable by
    ``matcher_protocol.register_default_matchers`` so the registry self-heals
    even if it was cleared (e.g. by a test fixture). No ``replace`` flag: this
    re-registers the SAME ``UnitMatchBackend`` class, which ``register_matcher``
    treats as idempotent. ``replace`` is reserved for a deliberate maintenance
    swap to genuinely different code, so the built-in path never masks a real
    name collision.
    """
    register_matcher(UnitMatchBackend(), UnitMatchParamsSchema)


register()
