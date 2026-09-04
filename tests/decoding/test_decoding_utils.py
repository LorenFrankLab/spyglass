"""Unit tests for ``spyglass.decoding.utils.resolve_orientation_col``.

``resolve_orientation_col`` is the single source of truth for orientation-column
resolution, used by ``DecodingOutput.create_decoding_view`` and by the
``get_orientation_col`` methods on both v1 decoding tables. These tests pin its
contract directly: the merge-path tests exercise it only indirectly (and always
with a non-None requested name), while the ``orientation_name=None`` default
that both v1 methods rely on is otherwise only reachable through a heavy DB
fixture. See issue #1616.
"""

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "columns,orientation_name,expected",
    [
        # A present requested name wins, even over the otherwise-preferred
        # "orientation" when both known columns are present.
        (
            ["orientation", "head_orientation"],
            "head_orientation",
            "head_orientation",
        ),
        # Auto-detect (orientation_name=None, as both v1 methods call it)
        # prefers "orientation".
        (["orientation", "head_orientation"], None, "orientation"),
        (["orientation"], None, "orientation"),
        # Falls back to "head_orientation" when "orientation" is absent.
        (["head_orientation"], None, "head_orientation"),
        # Neither known column present -> None. This None return is what
        # drives the v1 `resolve_orientation_col(df) or "head_orientation"`
        # fallback.
        (["position_x", "position_y"], None, None),
        # A requested-but-absent name is ignored; auto-detect takes over.
        (["orientation"], "__absent__", "orientation"),
        # A present but non-orientation requested name is honored verbatim
        # (the function trusts an explicit, present request).
        (["position_x"], "position_x", "position_x"),
    ],
)
def test_resolve_orientation_col(columns, orientation_name, expected):
    from spyglass.decoding.utils import resolve_orientation_col

    df = pd.DataFrame(columns=columns)
    assert (
        resolve_orientation_col(df, orientation_name=orientation_name)
        == expected
    )
