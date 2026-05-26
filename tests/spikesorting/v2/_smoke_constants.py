"""Shared constants for the MEArec smoke-fixture sort path.

Several test sites need the same low-amplitude ``clusterless_thresholder``
row to find any peaks on the synthetic MEArec smoke fixture (its
templates max out around a few microvolts; the Frank-lab production
default at 100 uV rejects every peak). Hoisting the row's name and
parameter payload here keeps all five call sites
(``baseline_capture.py``, ``test_phase1_baseline_regen.py``,
``test_run_v2_pipeline_clusterless_preset``,
``test_clusterless_thresholder_end_to_end``,
``test_v2_real_data_v1_parity``) in lockstep -- a future param tweak
lands in one place and the v1↔v2 parity gate stays meaningful.

The ``V1_TO_V2_*_NAMES`` dicts encode the v1-vs-v2 row-name asymmetry
that the v1↔v2 parity test crosses: v1 ships ``preproc_param_name=
"default"`` / ``sorter_param_name="default_clusterless"`` while v2
ships ``preproc_params_name="default_franklab"`` /
``sorter_params_name="default"``. Captures done with v1's shipping
default names get mapped to v2's equivalents at parity-test time.
"""

from __future__ import annotations

#: Name of the smoke-fixture-friendly ``clusterless_thresholder`` row.
#: Used by both v1 ``SpikeSorterParameters`` (via ``baseline_capture``)
#: and v2 ``SorterParameters`` (via the parity test); the schemas
#: differ but the name is the same.
SMOKE_CLUSTERLESS_PARAM_NAME = "smoke_clusterless_5uv"

#: 5 uV threshold + ``locally_exclusive`` peak detection tuned for the
#: MEArec smoke fixture. ``noise_levels`` is INTENTIONALLY OMITTED so
#: SpikeInterface computes per-channel MAD and the threshold is
#: interpreted as a MAD multiplier (matching the v1 baseline-capture
#: row); forwarding ``[1.0]`` would silently flip the semantic to raw
#: microvolts and explode the detection count by ~1,400x on this
#: fixture (the regression that drove v2 spike-sorting followup #11).
SMOKE_CLUSTERLESS_PARAMS: dict = {
    "detect_threshold": 5.0,
    "method": "locally_exclusive",
    "peak_sign": "neg",
    "exclude_sweep_ms": 0.1,
    "local_radius_um": 100.0,
}

#: v1 -> v2 ``preproc_param_name`` translation for shipped default
#: rows. v1's ``"default"`` is the bandpass + common_reference
#: configuration; v2 ships the same semantics under the explicit name
#: ``"default_franklab"``.
V1_TO_V2_PREPROC_PARAM_NAMES: dict = {
    "default": "default_franklab",
}

#: v1 -> v2 ``sorter_params_name`` translation for shipped default
#: rows. v1's ``"default_clusterless"`` is the 100 uV +
#: ``noise_levels=[1.0]`` row; v2 ships the same configuration under
#: the shorter name ``"default"``. The smoke row name is unchanged
#: across pipelines (both sides use ``smoke_clusterless_5uv``).
V1_TO_V2_SORTER_PARAM_NAMES: dict = {
    "default_clusterless": "default",
}


def v2_preproc_name_for_v1(v1_name: str) -> str:
    """Map a v1 ``preproc_param_name`` to the v2 equivalent.

    Returns the v1 name unchanged when no entry exists in
    :data:`V1_TO_V2_PREPROC_PARAM_NAMES`; this lets non-default
    baselines (captured under a custom v1 row) be honored verbatim
    on the v2 side as long as the v2 row exists under the same
    name.
    """
    return V1_TO_V2_PREPROC_PARAM_NAMES.get(v1_name, v1_name)


def v2_sorter_name_for_v1(v1_name: str) -> str:
    """Map a v1 ``sorter_param_name`` to the v2 equivalent.

    Same fallback semantics as :func:`v2_preproc_name_for_v1`.
    """
    return V1_TO_V2_SORTER_PARAM_NAMES.get(v1_name, v1_name)
