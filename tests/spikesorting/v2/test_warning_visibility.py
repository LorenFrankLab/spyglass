"""Unexpected pipeline warnings survive the shared pytest configuration."""

import warnings

import pytest
from hdmf.build.warnings import MissingRequiredBuildWarning


@pytest.mark.parametrize(
    "category",
    [
        UserWarning,
        FutureWarning,
        DeprecationWarning,
        ResourceWarning,
        MissingRequiredBuildWarning,
        pytest.PytestUnhandledThreadExceptionWarning,
    ],
)
def test_unexpected_pipeline_warnings_remain_visible(category):
    # Preserve the configured filters: pytest.warns() would force "always"
    # itself and hide a regression back to blanket category suppression.
    with warnings.catch_warnings(record=True) as seen:
        warnings.warn("Unexpected v2 pipeline validation problem", category)
    assert len(seen) == 1
    assert seen[0].category is category


def test_nwb_warning_proxy_only_suppresses_known_legacy_field():
    import hdmf.build.objectmapper as objectmapper

    with warnings.catch_warnings(record=True) as seen:
        objectmapper.warnings.warn(
            "Missing required attribute: source_script_file_name",
            MissingRequiredBuildWarning,
        )
        objectmapper.warnings.warn(
            "Missing required attribute: unrelated_v2_field",
            MissingRequiredBuildWarning,
        )
    assert len(seen) == 1
    assert "unrelated_v2_field" in str(seen[0].message)
