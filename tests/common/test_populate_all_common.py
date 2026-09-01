"""#1497: populate_all_common reported errors left over from earlier attempts."""

from importlib import import_module

import pytest


@pytest.fixture
def populate_module(common):
    """The populate_all_common module, not the same-named function on common."""
    return import_module("spyglass.common.populate_all_common")


@pytest.fixture
def stale_error(populate_module):
    """Seed an InsertError row that looks like it came from an earlier run."""
    import datajoint as dj

    from spyglass.common.common_usage import InsertError

    nwb_file_name = "stale_error_1497_.nwb"
    constants = dict(
        dj_user=dj.config["database.user"],
        connection_id=dj.conn().connection_id,
        nwb_file_name=nwb_file_name,
    )
    InsertError.insert1(
        dict(
            **constants,
            table="Session",
            error_type="ValueError",
            error_message="stale failure from a previous attempt",
            error_raw="stale failure from a previous attempt",
        )
    )

    yield nwb_file_name, constants

    (InsertError & constants).delete_quick()


def test_clears_errors_from_previous_attempt(
    populate_module, stale_error, monkeypatch
):
    """#1497: a clean run must not report an earlier attempt's errors."""
    from spyglass.common.common_usage import InsertError

    nwb_file_name, constants = stale_error
    assert len(InsertError & constants) == 1, "Stale error was not seeded"

    # Stand in for ingestion so nothing new is logged and the only candidate
    # error is the stale row seeded above.
    monkeypatch.setattr(
        populate_module, "single_transaction_make", lambda **kwargs: None
    )

    assert populate_module.populate_all_common(nwb_file_name) is None
    assert len(InsertError & constants) == 0


def test_still_reports_errors_from_this_attempt(
    populate_module, stale_error, monkeypatch
):
    """Clearing the log must not hide failures from the current run."""
    from spyglass.common.common_session import Session
    from spyglass.common.common_usage import InsertError

    nwb_file_name, constants = stale_error

    def _fail(**kwargs):
        populate_module.log_insert_error(
            table=Session,
            err=ValueError("failure from this attempt"),
            error_constants=kwargs["error_constants"],
        )

    monkeypatch.setattr(populate_module, "single_transaction_make", _fail)

    result = populate_module.populate_all_common(nwb_file_name)
    messages = (InsertError & constants).fetch("error_message")

    assert result is not None, "Errors from this attempt were not reported"
    assert len(result) == len(messages)
    assert all("this attempt" in message for message in messages)
    assert not any("previous attempt" in message for message in messages)
