"""Review-profile normalization, hashing, and persistence contracts."""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._review_profile import (
    normalize_review_profile,
    profile_display_property_vocabulary,
    review_profile_label_policy,
)

_AVAILABLE = ("snr", "isi_violation", "nn_isolation", "nn_noise_overlap")
_BASE = {
    "review_profile_name": "review_a_2026_06",
    "metric_params_name": "franklab_default",
    "auto_curation_rules_name": "franklab_rules_2026_06",
    "displayed_unit_properties": ["snr", "isi_violation"],
    "label_options": ["accept", "mua", "noise", "reject"],
    "label_import_mode": "replace",
}


def _normalized(**changes):
    return normalize_review_profile(
        {**_BASE, **changes}, available_displayed_properties=_AVAILABLE
    )


def test_review_profile_hash_covers_semantics():
    """Every recipe/display/import semantic, including list order, is hashed."""
    base_hash = _normalized()["profile_hash"]
    changes = (
        {"metric_params_name": "other_metrics"},
        {"auto_curation_rules_name": "other_rules"},
        {"displayed_unit_properties": ["isi_violation", "snr"]},
        {"label_options": ["mua", "accept", "noise", "reject"]},
        {"label_import_mode": "overlay"},
    )
    assert all(
        _normalized(**change)["profile_hash"] != base_hash for change in changes
    )


@pytest.mark.parametrize(
    "runtime_field",
    [
        "upload",
        "ephemeral",
        "credentials",
        "local_destination",
        "annotation_sets",
    ],
)
def test_review_profile_separates_delivery_options(runtime_field):
    """Per-review delivery/annotation choices cannot enter profile identity."""
    with pytest.raises(ValueError, match="runtime/review fields"):
        _normalized(**{runtime_field: True})


def test_review_profile_normalizes_and_validates_ordered_lists():
    """Profiles preserve valid order and reject unknown/duplicate entries."""
    row = _normalized()
    assert row["displayed_unit_properties"] == ["snr", "isi_violation"]
    assert row["label_options"] == ["accept", "mua", "noise", "reject"]
    with pytest.raises(ValueError, match="not produced"):
        _normalized(displayed_unit_properties=["not_a_metric"])
    assert _normalized(label_options=["good"])["label_options"] == ["good"]
    with pytest.raises(ValueError, match="non-empty"):
        _normalized(label_options=[""])
    with pytest.raises(ValueError, match="duplicate"):
        _normalized(label_options=["accept", "accept"])


def test_review_profile_property_vocabulary_expands_builtin_outputs():
    """The NN recipe name expands to both concrete display columns."""
    metric_row = {
        "metric_names": ["snr", "nn_advanced"],
        "template_metric_columns": ["trough_half_width"],
    }
    assert profile_display_property_vocabulary(metric_row) == (
        "observed_duration_s",
        "observed_firing_rate_hz",
        "observed_presence_ratio",
        "snr",
        "nn_isolation",
        "nn_noise_overlap",
        "trough_half_width",
    )


def test_review_profile_property_vocabulary_matches_pinned_si():
    """The DB-free column vocabulary tracks every pinned SI quality metric."""
    from spikeinterface.metrics import ComputeQualityMetrics

    for metric in ComputeQualityMetrics.metric_list:
        actual = profile_display_property_vocabulary(
            {
                "metric_names": [metric.metric_name],
                "template_metric_columns": [],
            }
        )
        expected = tuple(metric.metric_columns)
        if metric.metric_name == "isi_violation":
            expected = ("isi_violation", *expected)
        assert actual[:3] == (
            "observed_duration_s",
            "observed_firing_rate_hz",
            "observed_presence_ratio",
        )
        assert actual[3:] == expected, metric.metric_name


def test_review_profile_import_mode_reuses_curation_label_policy():
    """Public overlay maps to the existing expert-layer inherit policy."""
    assert review_profile_label_policy("replace") == "replace"
    assert review_profile_label_policy("overlay") == "inherit"
    with pytest.raises(ValueError, match="replace.*overlay"):
        review_profile_label_policy("merge")


@pytest.mark.database
def test_review_profile_persisted_and_immutable(dj_conn):
    """DB rows are normalized, idempotent, immutable, and name-stable."""
    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile

    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    names = ["test_review_profile_a", "test_review_profile_alias"]
    (
        CurationReviewProfile
        & [{"review_profile_name": name} for name in names]
    ).delete(safemode=False)
    row = {
        **_BASE,
        "review_profile_name": names[0],
        "auto_curation_rules_name": ("franklab_default_auto_curation_2026_06"),
    }
    try:
        CurationReviewProfile.insert1(row)
        stored = (
            CurationReviewProfile & {"review_profile_name": names[0]}
        ).fetch1()
        assert stored["metric_params_name"] == "franklab_default"
        assert stored["displayed_unit_properties"] == ["snr", "isi_violation"]
        assert stored["label_options"] == ["accept", "mua", "noise", "reject"]
        assert stored["label_import_mode"] == "replace"
        assert len(stored["profile_hash"]) == 64

        CurationReviewProfile.insert1(row, skip_duplicates=True)
        assert (
            len(CurationReviewProfile & {"review_profile_name": names[0]}) == 1
        )
        with pytest.raises(
            ValueError, match="already exists.*different content"
        ):
            CurationReviewProfile.insert1(
                {**row, "label_import_mode": "overlay"}
            )
        with pytest.raises(DuplicateParameterContentError, match="duplicates"):
            CurationReviewProfile.insert1(
                {**row, "review_profile_name": names[1]}
            )
        with pytest.raises(dj.errors.DataJointError, match="In-place update1"):
            CurationReviewProfile.update1(
                {**stored, "label_import_mode": "overlay"}
            )
        assert (
            CurationReviewProfile.label_policy(
                {"review_profile_name": names[0]}
            )
            == "replace"
        )
    finally:
        (
            CurationReviewProfile
            & [{"review_profile_name": name} for name in names]
        ).delete(safemode=False)


@pytest.mark.database
def test_franklab_review_profile_is_shipped(dj_conn):
    """The one-call initializer persists the approved dated Frank-lab profile."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.review_profile import (
        FRANKLAB_REVIEW_PROFILE,
        CurationReviewProfile,
    )

    initialize_v2_defaults()
    row = (
        CurationReviewProfile & {"review_profile_name": FRANKLAB_REVIEW_PROFILE}
    ).fetch1()
    assert row["metric_params_name"] == "franklab_default"
    assert row["auto_curation_rules_name"] == (
        "franklab_default_auto_curation_2026_06"
    )
    assert row["label_import_mode"] == "replace"


def test_review_display_options_are_bounded_and_serializable():
    """Display budget validates, round-trips, and stays display-only."""
    from spyglass.spikesorting.v2._review_profile import (
        REVIEW_DISPLAY_OPTIONS_VERSION,
        ReviewDisplayOptions,
    )

    default = ReviewDisplayOptions()
    assert default.max_amplitudes_per_unit is None
    assert default.point_limit("amplitudes", 3600) == 180000
    assert default.point_limit("raster", 60) == 3000
    old = ReviewDisplayOptions.from_mapping(
        {"version": 1, "max_amplitudes_per_unit": 2000}
    )
    assert old.point_limit("amplitudes", 3600) == 2000
    assert old.point_limit("raster", 3600) == 2000
    assert default.as_dict()["version"] == REVIEW_DISPLAY_OPTIONS_VERSION
    assert ReviewDisplayOptions.from_mapping(default.as_dict()) == default
    assert ReviewDisplayOptions.from_mapping(None) == default
    custom = ReviewDisplayOptions.from_mapping(
        {
            "max_amplitudes_per_unit": None,
            "min_similarity_for_correlograms": 0.5,
        }
    )
    assert custom.max_amplitudes_per_unit is None
    assert "floor(duration × 50)" in custom.describe()
    assert "all spikes" not in custom.describe()
    for bad in (
        {"max_amplitudes_per_unit": 0},
        {"amplitude_sampling_seed": -1},
        {"min_similarity_for_correlograms": 1.5},
        {"version": 99},
        {"max_spikes_per_unit": 10},  # not a field: display != science
    ):
        with pytest.raises((ValueError, TypeError)):
            ReviewDisplayOptions.from_mapping(bad)
