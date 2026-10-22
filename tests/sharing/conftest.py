"""Fixtures shared by the sharing tests."""

import pytest


@pytest.fixture(autouse=True)
def analysis_store_spec():
    """Point `dj.config["stores"]` back at the resolved Spyglass dirs.

    `SpyglassConfig._set_dj_config_stores` writes its instance's dirs into the
    global `dj.config`, so a test elsewhere that builds a config under
    `tmp_path` leaves the analysis store staged there. This suite uploads
    analysis filepaths, which DataJoint refuses when the file sits outside the
    staged directory.
    """
    try:
        from spyglass.settings import sg_config

        sg_config._set_dj_config_stores()
    except Exception:  # no config loaded, nothing to repair
        pass

    yield
