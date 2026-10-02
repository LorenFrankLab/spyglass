"""The README's v2 quick example uses only receipt keys and accessors that exist.

DB-free: reads README.md and the ``run_v2_pipeline`` receipt types; no
DataJoint connection is opened.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# A receipt variable in the README example: ``run_summary``,
# ``analysis_summary``, ...
_SUMMARY_SUBSCRIPT = re.compile(r"\b(\w*summary)\[\"(\w+)\"\]")
_SUMMARY_ATTRIBUTE = re.compile(r"\b(\w*summary)\.(\w+)\b")


@pytest.fixture(scope="module")
def readme_python() -> str:
    readme = Path(__file__).resolve().parents[3] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)
    return "\n".join(blocks)


def test_readme_receipt_keys_exist(readme_python):
    """Every ``*summary["key"]`` in the README is a run_v2_pipeline key."""
    from spyglass.spikesorting.v2 import _pipeline_types

    documented = set()
    for summary_type in (
        _pipeline_types.RunV2SingleSessionSummary,
        _pipeline_types.RunV2ConcatSummary,
    ):
        documented |= summary_type.__required_keys__
        documented |= summary_type.__optional_keys__

    used = {key for _, key in _SUMMARY_SUBSCRIPT.findall(readme_python)}
    assert used, "found no receipt keys in the README; the pattern is stale"
    assert used <= documented, f"unknown receipt keys: {used - documented}"


def test_readme_receipt_attributes_exist(readme_python):
    """Every ``*summary.attr`` in the README is a RunResult attribute."""
    from spyglass.spikesorting.v2.curation_api import RunResult

    used = {attr for _, attr in _SUMMARY_ATTRIBUTE.findall(readme_python)}
    missing = {attr for attr in used if not hasattr(RunResult, attr)}
    assert not missing, f"unknown RunResult attributes: {missing}"
