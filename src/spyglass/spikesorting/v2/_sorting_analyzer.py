"""``SortingAnalyzer`` cache build/rebuild behind ``Sorting``.

``build_analyzer`` builds the binary-folder ``SortingAnalyzer`` and its base
extensions (``random_spikes`` / ``noise_levels`` / ``templates`` /
``waveforms``) for ``Sorting.make_compute`` and the rebuild path -- with the
deterministic-seed pins, the 3D->2D probe projection, the zero-unit
short-circuit, and partial-folder cleanup on failure. The table threads the
fetched ``SorterParameters`` row and resolved job kwargs in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute).

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The analyzer build
needs none of that at import, so ``Sorting`` becomes a thin orchestrator. Same
"thin DataJoint shell over pure/IO services" direction as ``_artifact_compute``
/ ``_selection_identity`` / ``_analyzer_cache`` / ``_curation_transforms`` /
``_units_nwb`` / ``_sorting_units`` / ``_sorting_artifact_mask`` /
``_sorting_dispatch``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / spyglass dependencies are imported
lazily inside the function. ``build_analyzer``'s rebuild fallback (only when
``sorter_row`` is not threaded in) does a single ``SorterParameters`` /
``SortingSelection`` fetch via a CALL-TIME lazy import of those table classes
from ``sorting`` -- by then ``sorting`` is fully imported, so there is no import
cycle.
"""

from __future__ import annotations


def build_analyzer(
    sorting,
    recording,
    key,
    *,
    sorter_row=None,
    job_kwargs=None,
    analyzer_folder=None,
):
    """Build the binary-folder SortingAnalyzer + base extensions.

    ``sorter_row`` is the already-fetched ``SorterParameters`` row
    from ``make_fetch``; passing it through avoids three redundant
    DB round-trips during ``make_compute`` (we cannot read inside
    ``make_compute`` per the tri-part contract). The rebuild path
    does not have the row pre-fetched, so it leaves
    ``sorter_row=None`` and pays the DB cost once -- that path is
    rare (missing analyzer folder) and not on the populate hot
    path, so the lookup is acceptable there.

    ``job_kwargs`` is the already-resolved dict from
    ``_resolved_job_kwargs`` -- when ``make_compute`` calls
    ``run_sorter`` and ``build_analyzer`` from the same
    invocation it resolves once and threads through; the rebuild
    path falls back to resolving locally (same DB-read tradeoff
    as ``sorter_row``).

    ``analyzer_folder`` is the resolved cache folder to write. When
    ``None`` (the make_compute path) it is resolved from ``sorting_id``
    and RETURNED for the caller to thread on. Callers that already
    resolved it (``_rebuild_analyzer_folder``, which needs the path for
    its own cleanup) pass it in so the folder built equals the folder
    they clean up -- one resolution, no config-drift edge.

    Parameters
    ----------
    sorting : spikeinterface.BaseSorting
        The (excess-trimmed) sorting to build the analyzer from.
    recording : si.BaseRecording
        The recording the sorting was computed on.
    key : dict
        Restriction dict carrying ``sorting_id``; used to resolve the
        analyzer folder and the fallback ``SorterParameters`` fetch.
    sorter_row : dict, optional
        Keyword-only. Pre-fetched ``SorterParameters`` row. ``None``
        (the rebuild path) triggers a one-time DB fetch. Default
        ``None``.
    job_kwargs : dict, optional
        Keyword-only. Pre-resolved job kwargs. ``None`` resolves them
        locally from ``sorter_row``. Default ``None``.
    analyzer_folder : pathlib.Path, optional
        Keyword-only. Resolved cache folder to write. ``None`` resolves
        it from ``key["sorting_id"]``. Default ``None``.

    Returns
    -------
    pathlib.Path
        The analyzer cache folder path. For a zero-unit sort the folder
        is returned without being built.
    """
    import shutil as _shutil

    import spikeinterface as si

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs
    from spyglass.utils import logger

    folder = (
        analyzer_folder
        if analyzer_folder is not None
        else analyzer_path(key["sorting_id"])
    )

    # Zero-unit short-circuit BEFORE any I/O or DB fetch:
    # ``create_sorting_analyzer(sparse=True)`` -> ``estimate_sparsity``
    # -> ``random_spikes_selection`` crashes on ``np.concatenate([])``
    # for empty sortings. ``_populate_unit_part`` iterates an empty
    # ``sorting.unit_ids`` and writes zero Unit rows; the Sorting
    # master row commits with ``n_units=0``. Return the (not yet
    # created) folder path for the row; ``Sorting.get_analyzer``
    # raises ``ZeroUnitAnalyzerError`` for a zero-unit sort rather
    # than trying to load this never-built folder.
    if sorting.get_num_units() == 0:
        logger.warning(
            "Sorting._build_analyzer: sorting_id="
            f"{key.get('sorting_id')!r} has zero units; skipping "
            "analyzer build. Check ``detect_threshold`` / "
            "artifact masking if you expected non-zero output."
        )
        return folder

    folder.parent.mkdir(parents=True, exist_ok=True)

    if sorter_row is None:
        from spyglass.spikesorting.v2.sorting import (
            SorterParameters,
            SortingSelection,
        )

        sorter_row = (
            SorterParameters
            & ((SortingSelection & key).proj("sorter", "sorter_params_name"))
        ).fetch1()
    if job_kwargs is None:
        job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])

    # Project the probe to 2D before building the analyzer. Spyglass electrode
    # geometry is stored in 3D (the z coordinate is typically 0), but several
    # SortingAnalyzer extensions and consumers assume 2D contact positions:
    # ``unit_locations`` (monopolar_triangulation / center_of_mass) and the
    # spikeinterface-gui probe view both raise
    # ``could not broadcast input array from shape (3,) into shape (2,)`` on a
    # 3D probe. The in-plane (x, y) coordinates are unchanged by the projection,
    # so sparsity, templates, and waveforms are unaffected ONLY when the z axis
    # is degenerate (all-zero, as Spyglass stores it). Done here (rather than at
    # recording materialization) so the sort itself still sees the recording
    # untouched.
    probe = recording.get_probe()
    if probe.ndim == 3:
        import numpy as _np

        from spyglass.utils import logger

        # Dropping a non-degenerate z axis would shift channel distances (and
        # therefore sparsity), so the "x/y unchanged" guarantee above no longer
        # holds. Frank-lab probes are planar (z=0); warn on a genuinely
        # non-planar probe, which may need a different projection axis.
        z = _np.asarray(probe.contact_positions)[:, 2]
        if z.size and not _np.allclose(z, z[0]):
            logger.warning(
                "build_analyzer: projecting a 3D probe to 2D (axes='xy'), but "
                "the contact z coordinates are not constant (range "
                f"{float(z.max() - z.min()):.3g} um). Depth geometry is "
                "discarded, which can shift channel distances and sparsity "
                "relative to the 3D probe."
            )
        recording = recording.set_probe(probe.to_2d())

    try:
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            sparse=True,
            format="binary_folder",
            folder=folder,
            return_in_uV=True,
            overwrite=True,
        )
        # ``random_seed`` is a Spyglass-side knob (consumed by the sorter
        # and the whitening pin in ``run_si_sorter``), not a valid
        # ``SortingAnalyzer.compute`` keyword -- SI raises
        # "please remove {'random_seed'}". Strip it here, mirroring the
        # detect_peaks path. It stays in ``job_kwargs`` upstream so the
        # sorter/whitening still read the seed.
        analyzer_job_kwargs = {
            k: v for k, v in job_kwargs.items() if k != "random_seed"
        }
        analyzer.compute(
            ["random_spikes", "noise_levels", "templates", "waveforms"],
            extension_params={
                "random_spikes": {
                    "max_spikes_per_unit": 500,
                    "method": "uniform",
                    # Pin the stochastic spike subsampling. When a
                    # unit has more than ``max_spikes_per_unit`` (500)
                    # spikes, ``random_spikes`` draws a uniform random
                    # subset; the SI 0.104 extension defaults
                    # ``seed=None`` (verified against
                    # ``ComputeRandomSpikes._set_params``), so an
                    # unseeded build selects a different subset each
                    # time -- and the persisted ``peak_amplitude_uv``
                    # / peak channel (computed from the subset's
                    # templates) drifts across rebuilds of the same
                    # sort. Defaults to 0 but honors the per-row
                    # ``job_kwargs={"random_seed": N}`` override, the
                    # same knob the whitening / noise_levels pins read
                    # in ``run_si_sorter`` / ``run_clusterless_
                    # thresholder`` (lines using
                    # ``(job_kwargs or {}).get("random_seed", 0)``) --
                    # so a user changing the seed gets a consistent
                    # seed across the sort AND the analyzer subsample.
                    "seed": (job_kwargs or {}).get("random_seed", 0),
                },
                "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
            },
            **analyzer_job_kwargs,
        )
    except Exception:
        try:
            if folder.exists():
                _shutil.rmtree(folder, ignore_errors=False)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Sorting._build_analyzer: failed to remove partial "
                f"analyzer folder {folder!r}: {cleanup_exc!r}"
            )
        raise
    return folder
