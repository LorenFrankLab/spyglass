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
_SUMMARY_SUBSCRIPT = re.compile(r"\b\w*summary\[\s*[\"'](\w+)[\"']\s*\]")
_SUMMARY_ATTRIBUTE = re.compile(r"\b\w*summary\.(\w+)\b")


@pytest.fixture(scope="module")
def readme_python() -> str:
    readme = Path(__file__).resolve().parents[3] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)
    return "\n".join(blocks)


def _check_receipt_accessors(python):
    """Check both supported receipt access styles without accepting an empty example."""
    from spyglass.spikesorting.v2._orchestration import types as _pipeline_types
    from spyglass.spikesorting.v2.curation_api import RunResult

    documented = set()
    for summary_type in (
        _pipeline_types.RunV2SingleSessionSummary,
        _pipeline_types.RunV2ConcatSummary,
    ):
        documented |= summary_type.__required_keys__
        documented |= summary_type.__optional_keys__

    keys = set(_SUMMARY_SUBSCRIPT.findall(python))
    attributes = set(_SUMMARY_ATTRIBUTE.findall(python))
    assert (
        keys or attributes
    ), "found no receipt accessors; the pattern is stale"
    assert keys <= documented, f"unknown receipt keys: {keys - documented}"
    missing = {attr for attr in attributes if not hasattr(RunResult, attr)}
    assert not missing, f"unknown RunResult attributes: {missing}"


def test_readme_receipt_accessors_exist(readme_python):
    """The README uses valid dictionary keys or RunResult attributes."""
    _check_receipt_accessors(readme_python)


@pytest.mark.parametrize(
    "python",
    [
        'run_summary["root_curation_id"]',
        "run_summary['root_curation_id']",
        "analysis_summary.auto_labeled_curation",
    ],
)
def test_receipt_check_accepts_supported_access_styles(python):
    _check_receipt_accessors(python)


@pytest.mark.parametrize(
    "python, message",
    [
        ("print('no receipt example')", "found no receipt accessors"),
        ('run_summary["unknown_key"]', "unknown receipt keys"),
        ("analysis_summary.unknown_attribute", "unknown RunResult attributes"),
    ],
)
def test_receipt_check_rejects_missing_or_unknown_accessors(python, message):
    with pytest.raises(AssertionError, match=message):
        _check_receipt_accessors(python)
