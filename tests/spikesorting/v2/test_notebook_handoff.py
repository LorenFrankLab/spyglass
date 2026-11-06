"""Execute the published analysis handoff without a database or browser."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit

_NOTEBOOK = (
    Path(__file__).resolve().parents[3]
    / "notebooks"
    / "10_Spike_SortingV2.ipynb"
)


def _execute_handoff(monkeypatch, namespace, select):
    pipeline = ModuleType("spyglass.spikesorting.v2.pipeline")
    pipeline.select_units_for_analysis = select
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)
    cells = json.loads(_NOTEBOOK.read_text())["cells"]
    source = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
        and "if use_browser_result:" in "".join(cell["source"])
    )
    exec(  # noqa: S102 - exercise the repository-owned notebook cell
        compile(source, f"{_NOTEBOOK.name}:analysis-handoff", "exec"),
        namespace,
    )


@pytest.mark.parametrize("use_browser_result", [False, True])
def test_handoff_uses_the_requested_curation(
    monkeypatch, capsys, use_browser_result
):
    automatic, reviewed = object(), object()
    review = Mock()
    review.result.return_value = reviewed
    receipt = Mock()
    select = Mock(return_value=receipt)
    namespace = {
        "auto_labeled": automatic,
        "review": review,
        "use_browser_result": use_browser_result,
        "display": Mock(),
    }

    _execute_handoff(monkeypatch, namespace, select)

    expected = reviewed if use_browser_result else automatic
    policy = (
        "v2_accepted_single_units"
        if use_browser_result
        else "v2_unflagged_units"
    )
    assert namespace["final_curation"] is expected
    assert namespace["receipt"] is receipt
    select.assert_called_once_with(expected, policy=policy)
    if use_browser_result:
        review.result.assert_called_once_with()
        assert "completed browser review" in capsys.readouterr().out
    else:
        review.result.assert_not_called()
        assert "WITHOUT human review" in capsys.readouterr().out


@pytest.mark.parametrize("review_defined", [False, True])
def test_missing_browser_review_explains_how_to_recover(
    monkeypatch, review_defined
):
    select = Mock()
    namespace = {"use_browser_result": True}
    if review_defined:
        namespace["review"] = None  # optional FigPack cell was skipped

    with pytest.raises(RuntimeError, match="FigPackReview.resume") as error:
        _execute_handoff(monkeypatch, namespace, select)

    assert "spyglass-neuro[spikesorting-v2-curation]" in str(error.value)
    assert "section 8" in str(error.value)
    select.assert_not_called()


def test_unfinished_review_cannot_fall_back_to_automatic_labels(monkeypatch):
    review = Mock()
    review.result.side_effect = ValueError(
        "This browser review has not completed."
    )
    select = Mock()
    namespace = {
        "auto_labeled": object(),
        "review": review,
        "use_browser_result": True,
    }

    with pytest.raises(ValueError, match="not completed"):
        _execute_handoff(monkeypatch, namespace, select)

    select.assert_not_called()
