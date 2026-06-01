"""Audit-derived correctness & v1-parity regression slice.

Tests here pin behavioral fixes surfaced by the v1↔v2 parity audit that
the original 6-agent review missed. Each test either reproduces a
confirmed bug (raise / no-leak / no-silent-mask) or pins an intentional
v2 design choice against future drift. Where a test locks a v1↔v2
divergence the docstring cites the v1 source so a reviewer can confirm
the choice was deliberate.

Pure-Pydantic tests import only ``_params`` (no DB). Tests that touch a
``dj.schema``-activated runtime module request ``dj_conn`` and import the
table inside the test body (mirroring ``test_v1_parity``).
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._params.sorter import MountainSort4Schema


# ---------- A3: MS4 adjacency_radius -1 sentinel ---------------------------


def test_ms4_schema_accepts_adjacency_radius_minus_one():
    """A3: ``adjacency_radius=-1`` is SI's "use all channels" sentinel.

    SpikeInterface's MountainSort4 wrapper documents ``adjacency_radius=-1``
    as "use all channels in the adjacency graph". The earlier
    ``Field(ge=0.0)`` floor rejected the sentinel, so a v1 user porting
    ``{"adjacency_radius": -1}`` was blocked at insert time. ``ge=-1.0``
    plus a validator that rejects the open interval ``(-1, 0)`` accepts
    the sentinel and any non-negative radius while still catching
    nonsensical negatives.
    """
    assert MountainSort4Schema(adjacency_radius=-1).adjacency_radius == -1.0
    assert MountainSort4Schema(adjacency_radius=0).adjacency_radius == 0.0
    assert MountainSort4Schema(adjacency_radius=75.0).adjacency_radius == 75.0

    # The open interval (-1, 0) is meaningless and must be rejected.
    for bad in (-0.5, -0.999, -0.0001):
        with pytest.raises(ValueError, match="adjacency_radius"):
            MountainSort4Schema(adjacency_radius=bad)

    # Below the sentinel is also invalid.
    with pytest.raises(ValueError):
        MountainSort4Schema(adjacency_radius=-2.0)


# ---------- A2: Franklab MS4 v1-name back-compat aliases -------------------


@pytest.mark.usefixtures("dj_conn")
def test_franklab_ms4_v1_alias_rows_present():
    """A2: v1's bare ``30KHz`` preset names still resolve on v2.

    v2 renamed the Franklab MS4 presets to lowercase-k + ``_ms4``
    (``v1/sorting.py:158,163`` -> ``franklab_*_30kHz_ms4``), silently
    breaking v1 code that looks the row up by its old name. The catalog
    ships back-compat alias rows carrying the IDENTICAL validated params
    blob, so both the new and old names resolve to the same parameters.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    catalog = {
        (row[0], row[1]): row for row in SorterParameters._DEFAULT_CONTENTS
    }
    pairs = [
        (
            "franklab_tetrode_hippocampus_30kHz_ms4",
            "franklab_tetrode_hippocampus_30KHz",
        ),
        ("franklab_probe_ctx_30kHz_ms4", "franklab_probe_ctx_30KHz"),
    ]
    for ms4_name, alias_name in pairs:
        ms4_row = catalog[("mountainsort4", ms4_name)]
        alias_row = catalog[("mountainsort4", alias_name)]
        # params blob (index 2) and schema_version (index 3) identical.
        assert alias_row[2] == ms4_row[2], (
            f"alias {alias_name!r} params blob diverged from {ms4_name!r}"
        )
        assert alias_row[3] == ms4_row[3]

    # When MS4 is installed the aliases actually land in the table, so a
    # v1-style lookup by the old name resolves. (On the CI SI-0.104 image
    # MS4 is not installed and insert_default skips every MS4 row -- the
    # catalog assertion above is the install-independent invariant.)
    if "mountainsort4" in sis.installed_sorters():
        SorterParameters().insert_default()
        for _, alias_name in pairs:
            assert (
                SorterParameters
                & {
                    "sorter": "mountainsort4",
                    "sorter_params_name": alias_name,
                }
            ), f"alias row {alias_name!r} did not insert with MS4 present"


# ---------- A5: params_schema_version must be supplied ---------------------


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_rejects_missing_schema_version():
    """A5: a custom row omitting ``params_schema_version`` is rejected.

    The column default is the sentinel 0 ("unspecified"). A clusterless
    row whose validated ``params`` carries ``schema_version=4`` but whose
    outer column silently defaulted to 0 (or the stale 1) would mis-tag
    the blob. ``insert1`` requires the caller to pass the version
    explicitly and the error names both the sorter and the expected
    version.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    base = {
        "sorter": "clusterless_thresholder",
        "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        "job_kwargs": None,
    }

    # Omitted entirely -> raise naming the field, the sorter, and the
    # expected inner version (4 for clusterless).
    omitted = dict(base, sorter_params_name="audit_a5_missing")
    with pytest.raises(ValueError) as excinfo:
        SorterParameters().insert1(omitted, skip_duplicates=True)
    msg = str(excinfo.value)
    assert "params_schema_version" in msg
    assert "clusterless_thresholder" in msg
    assert "4" in msg

    # Explicit sentinel 0 is rejected the same way.
    sentinel = dict(
        base, sorter_params_name="audit_a5_zero", params_schema_version=0
    )
    with pytest.raises(ValueError, match="params_schema_version"):
        SorterParameters().insert1(sentinel, skip_duplicates=True)

    # The correct explicit version inserts cleanly.
    ok = dict(
        base, sorter_params_name="audit_a5_ok", params_schema_version=4
    )
    SorterParameters().insert1(ok, skip_duplicates=True)
    assert (
        SorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "audit_a5_ok",
        }
    )

    # A wrong (non-sentinel) version that disagrees with the blob still
    # trips the existing drift check.
    drift = dict(
        base, sorter_params_name="audit_a5_drift", params_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        SorterParameters().insert1(drift, skip_duplicates=True)
