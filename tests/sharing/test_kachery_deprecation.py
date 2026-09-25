"""Tests that kachery entry points announce their own retirement.

The point of logging rather than only warning is that removal can be timed
against real use, so what matters is that a call reaches `ActivityLog` under
the name someone would search for — and that an instance without kachery
installed records nothing.
"""

import pytest


@pytest.fixture
def kachery(common):
    """The kachery module, imported after the schema is available."""
    from spyglass.sharing import sharing_kachery

    _ = common  # connection must be live before the schema is declared

    return sharing_kachery


@pytest.fixture
def logged(kachery, monkeypatch):
    """Record deprecation logs instead of writing them to the database."""
    calls = []

    def fake_log(cls, name, alt=None, warning=True, doc=None):
        calls.append(dict(name=name, alt=alt, doc=doc))

    # A classmethod, so the real call form `ActivityLog().deprecate_log(...)`
    # binds the same way and `name` is not shadowed by the instance.
    monkeypatch.setattr(
        "spyglass.common.common_usage.ActivityLog.deprecate_log",
        classmethod(fake_log),
    )

    return calls


def test_share_data_to_kachery_is_logged(kachery, logged):
    """The headline entry point names itself and its replacement."""
    if not kachery._kachery_available:
        pytest.skip("kachery_cloud is not installed")

    with pytest.raises(ValueError, match="restriction"):
        kachery.share_data_to_kachery(zone_name="test.zone")

    assert [c["name"] for c in logged] == ["share_data_to_kachery"]
    assert logged[0]["alt"] == "spyglass.sharing.share_file"
    assert logged[0]["doc"].startswith("https://")


def test_download_file_is_logged(kachery, logged):
    """The read path logs too; it is what the backend chain reaches."""
    if not kachery._kachery_available:
        pytest.skip("kachery_cloud is not installed")

    assert (
        kachery.AnalysisNwbfileKachery.download_file(
            "no_such_file.nwb", permit_fail=True
        )
        is False
    )

    assert [c["name"] for c in logged] == [
        "AnalysisNwbfileKachery.download_file"
    ]


def test_nothing_is_logged_without_kachery(kachery, logged, monkeypatch):
    """An instance with no kachery installed is not a kachery user.

    `download_file` is called for every chain miss, so logging before the
    availability check would count installs that never used the feature.
    """
    monkeypatch.setattr(kachery, "_kachery_available", False)

    assert (
        kachery.AnalysisNwbfileKachery.download_file(
            "no_such_file.nwb", permit_fail=True
        )
        is False
    )

    assert logged == []
