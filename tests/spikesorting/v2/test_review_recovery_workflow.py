"""Execute the published recovery steps with database/browser stand-ins.

The browser journey exercises real edits and commits. These tests exercise
the notebook control flow that chooses which branch is handed to
verification and analysis, including keeping a valid merge during recovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _workflow():
    notebook = json.loads(
        (_ROOT / "notebooks/10_Spike_SortingV2_Curation.ipynb").read_text()
    )
    recovery = []
    in_recovery = False
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "### 3-recover." in text:
            in_recovery = True
        if "### 3-hand-label." in text:
            break
        if in_recovery and cell["cell_type"] == "code":
            recovery.append(text)
    # The user selects their receipt in the first recovery cell.
    recovery[0] = recovery[0].replace(
        "bad_merge_receipt = None", "bad_merge_receipt = receipt", 1
    )
    verification = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "if pending_verification is not None and commit_merge_verification:"
        in "".join(cell["source"])
    )
    return recovery, verification


@pytest.mark.parametrize(
    "remaining_groups,has_changes",
    [
        ((), False),
        ((), True),
        (((3, 4),), True),
    ],
    ids=["complete-undo", "keep-label-edits", "keep-valid-merge"],
)
def test_recovery_hands_off_the_replacement(remaining_groups, has_changes):
    recovery, verification = _workflow()
    open_cell, preview_cell, commit_cell = recovery
    parent = SimpleNamespace(curation_id=1)
    abandoned = SimpleNamespace(curation_id=2, parent=parent)
    replacement = SimpleNamespace(curation_id=3, parent=parent)
    verified = SimpleNamespace(curation_id=4, parent=replacement)
    pending = Mock(review_id="replacement-review")
    verification_changes = Mock(has_changes=False)
    verification_changes.commit.return_value = Mock(
        curation=verified, needs_merge_verification=False
    )
    pending.preview_import.return_value = verification_changes
    replacement_receipt = Mock(
        curation=replacement, needs_merge_verification=bool(remaining_groups)
    )
    replacement_receipt.continue_review.return_value = pending
    changes = Mock(merge_groups=remaining_groups, has_changes=has_changes)
    changes.commit.return_value = replacement_receipt
    review = Mock()
    review.preview_import.return_value = changes
    receipt = SimpleNamespace(
        curation=abandoned,
        changes=SimpleNamespace(review=review, merge_groups=((1, 2), (3, 4))),
    )
    old_pending = Mock(review_id="abandoned-review")
    namespace = dict(
        receipt=receipt,
        open_review_in_browser=False,
        commit_recovery=False,
        pending_verification=old_pending,
        final_curation=None,
        display=Mock(),
    )

    exec(open_cell, namespace)
    review.open.assert_called_once()
    review.preview_import.assert_not_called()
    changes.commit.assert_not_called()

    exec(preview_cell, namespace)
    review.preview_import.assert_called_once()
    changes.commit.assert_not_called()
    exec(commit_cell, namespace)
    changes.commit.assert_not_called()
    assert namespace["pending_verification"] is old_pending
    namespace["commit_recovery"] = True
    exec(commit_cell, namespace)
    changes.commit.assert_called_once_with(
        conflict_resolutions={}, confirm_no_changes=not has_changes
    )

    if not remaining_groups:
        assert namespace["pending_verification"] is None
        assert namespace["final_curation"] is replacement
        replacement_receipt.continue_review.assert_not_called()
        return

    assert namespace["pending_verification"] is pending
    assert namespace["final_curation"] is None
    pending.open.assert_called_once()
    verification_changes.commit.assert_not_called()

    # After the user inspects the retained merge, the published verification
    # step must release this branch, never resume the abandoned review.
    api = Mock()
    api.resume.return_value = pending
    namespace.update(FigPackReview=api, commit_merge_verification=True)
    exec(verification, namespace)
    api.resume.assert_called_once_with("replacement-review")
    verification_changes.commit.assert_called_once_with(confirm_no_changes=True)
    assert namespace["pending_verification"] is None
    assert namespace["final_curation"] is verified
