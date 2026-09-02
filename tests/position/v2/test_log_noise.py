"""Tests for inference log/warning noise reduction (issue #1676).

A real populate emitted, for one NWB read, two hdmf warnings about the
same schema-migration fact -- differing only in an interpolated device
name -- plus three info lines all announcing the same "inference is
starting" event. ``dedupe_warnings`` addresses the first; the log lines
were demoted to debug, leaving one info line in the tool runner.
"""

import warnings

import pytest


@pytest.fixture
def dedupe():
    """Import under test (deferred: keeps DB imports out of collection)."""
    from spyglass.position.utils import dedupe_warnings

    return dedupe_warnings


# The real pair, verbatim apart from truncation -- Python's own duplicate
# filter does not collapse these, because the message text differs.
HDMF_MSG = (
    "Device.model was detected as a string, but NWB 2.9 specifies "
    'Device.model as a link to a DeviceModel. Remapping "{name}" to a '
    "new DeviceModel."
)


class TestDedupeWarnings:
    def test_collapses_messages_differing_only_in_quoted_value(self, dedupe):
        """The reported case: two warnings, one underlying fact."""
        with warnings.catch_warnings(record=True) as out:
            warnings.simplefilter("always")
            with dedupe():
                warnings.warn(HDMF_MSG.format(name="unknown"), UserWarning)
                warnings.warn(HDMF_MSG.format(name="unknown2"), UserWarning)
        assert len(out) == 1

    def test_genuinely_distinct_warnings_all_survive(self, dedupe):
        """Deduping must not become silencing."""
        with warnings.catch_warnings(record=True) as out:
            warnings.simplefilter("always")
            with dedupe():
                warnings.warn("ndx-pose version mismatch", UserWarning)
                warnings.warn(HDMF_MSG.format(name="unknown"), UserWarning)
        assert len(out) == 2

    def test_same_text_different_category_both_survive(self, dedupe):
        """Category is part of identity, not just the message text."""
        with warnings.catch_warnings(record=True) as out:
            warnings.simplefilter("always")
            with dedupe():
                warnings.warn("same text", UserWarning)
                warnings.warn("same text", DeprecationWarning)
        assert len(out) == 2

    def test_original_source_location_is_preserved(self, dedupe):
        """Re-emission must not reattribute blame to the helper.

        Uses warn_explicit under the hood precisely so a user chasing a
        warning still lands on the code that raised it.
        """
        with warnings.catch_warnings(record=True) as out:
            warnings.simplefilter("always")
            with dedupe():
                warnings.warn("attributed here", UserWarning)
        assert out[0].filename == __file__

    def test_exceptions_propagate(self, dedupe):
        """A failure inside the block is not swallowed by the capture."""
        with pytest.raises(ValueError, match="boom"):
            with dedupe():
                raise ValueError("boom")

    def test_yields_captured_records(self, dedupe):
        """The yielded list is populated for callers that want to inspect."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with dedupe() as caught:
                warnings.warn("a", UserWarning)
                warnings.warn("b", UserWarning)
        assert len(caught) == 2, "captures every raw warning, pre-dedupe"
