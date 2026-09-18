"""Connected local review operations, using a separate Python/DB process.

HTTP threads never use the notebook's DataJoint connection. One operation at a
time runs per review, through the same facade used by notebooks. The journal
contains identities/results only; credentials travel to the worker over stdin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from spyglass.spikesorting.v2._json_io import read_json, write_json

OPERATION_FILE = "spyglass_review_operation.json"
RESULT_FILE = "spyglass_review_result.json"
ORIGIN_FILE = "spyglass_review_origin.json"
LOCK_FILE = "spyglass_review_operation.lock"

_WORKER = """
import json, sys
import datajoint as dj
request = json.load(sys.stdin)
dj.config.update(request.pop('config'))
from spyglass.spikesorting.v2._review_operations import run_operation
run_operation(**request)
"""


def _acquire_operation_lock(bundle):
    """Own a bundle until both the launcher and its worker close the file.

    The descriptor is inherited by the worker on our supported Linux/macOS
    hosts. Closing a notebook or restarting its HTTP server therefore cannot
    release ownership while the scientific operation is still running.
    """
    import fcntl

    lock = (Path(bundle) / LOCK_FILE).open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise
    return lock


class ReviewOperationService:
    def __init__(self, review_id, bundle, config):
        self.review_id = str(review_id)
        self.bundle = Path(bundle)
        self.config = config
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="review-operation"
        )
        self.future = None

    @classmethod
    def for_review(cls, review):
        import datajoint as dj

        return cls(review.review_id, review.uri, dict(dj.config))

    def start(self, request):
        if request.get("action") not in {
            "preview",
            "commit",
            "inspect",
            "parent",
        }:
            raise ValueError("Choose preview, commit, inspect, or parent.")
        try:
            operation_lock = _acquire_operation_lock(self.bundle)
        except BlockingIOError as exc:
            raise ValueError(
                "This review is still processing its previous action."
            ) from exc
        try:
            operation_id = str(uuid.uuid4())
            write_json(
                self.bundle / OPERATION_FILE,
                {
                    "status": "running",
                    "operation_id": operation_id,
                    "action": request["action"],
                    "message": "Preparing review operation…",
                },
            )
            self.future = self.executor.submit(
                self._execute_locked, request, operation_id, operation_lock
            )
        except BaseException:
            operation_lock.close()
            raise
        return {"operation_id": operation_id, "status": "running"}

    def _execute_locked(self, request, operation_id, operation_lock):
        with operation_lock:
            self._execute(request, operation_id, operation_lock.fileno())

    def _execute(self, request, operation_id, lock_fd):
        payload = {
            "config": self.config,
            "review_id": self.review_id,
            "bundle": str(self.bundle),
            "request": request,
            "operation_id": operation_id,
            "lock_fd": lock_fd,
        }
        try:
            outcome = subprocess.run(
                [sys.executable, "-c", _WORKER],
                input=json.dumps(payload, default=str),
                text=True,
                capture_output=True,
                check=False,
                pass_fds=(lock_fd,),
            )
            error = (
                outcome.stderr[-3000:]
                or "Review worker stopped. Retry to resume the saved operation."
            )
        except OSError as exc:
            error = f"Could not start the review worker: {exc}"
        journal = self.bundle / OPERATION_FILE
        state = read_json(journal)
        if (
            state.get("status") == "running"
            and state.get("operation_id") == operation_id
        ):
            # Worker bootstrap/process failures have no scientific exception
            # handler; keep the last journaled child, if one was committed.
            state.update(status="failed", message=error)
            write_json(journal, state)

    def status(self):
        try:
            operation_lock = _acquire_operation_lock(self.bundle)
        except BlockingIOError:
            state = read_json(self.bundle / OPERATION_FILE)
            # The worker journals its result before exiting. Keep actions
            # disabled until ownership is released, or an immediate commit
            # after preview can fail on the previous worker's lock.
            if state.get("status") != "running":
                state.update(
                    status="running", message="Finishing review operation…"
                )
        else:
            with operation_lock:
                state = read_json(self.bundle / OPERATION_FILE)
                if state.get("status") == "running":
                    state.update(
                        status="interrupted",
                        message="The worker stopped. Retry the action to resume its saved result.",
                    )
        # Filesystem destinations are internal details, never browser input.
        destination = state.pop("destination", None)
        if state.get("status") == "complete" and destination:
            from spyglass.spikesorting.v2._review_delivery import (
                serve_review_bundle,
            )

            child_id = destination.get("review_id")
            try:
                url = serve_review_bundle(
                    destination["bundle"],
                    operation_factory=(
                        (
                            lambda: type(self)(
                                child_id, destination["bundle"], self.config
                            )
                        )
                        if child_id
                        else None
                    ),
                )
            except OSError as exc:
                state.update(
                    status="failed",
                    message=f"The saved result could not be opened: {exc}. Retry the action; the scientific result is retained.",
                )
                return state
            if destination.get("focus_unit_ids"):
                from urllib.parse import urlencode

                url += "?" + urlencode(
                    {
                        "spyglass_units": ",".join(
                            map(str, destination["focus_unit_ids"])
                        )
                    }
                )
            state["url"] = url
        return state

    def close(self):
        self.executor.shutdown(wait=False)


def _curation_identity(ref):
    return {
        "sorting_id": str(ref.sorting_id),
        "curation_id": ref.curation_id,
        "curation_uuid": str(ref.curation_uuid),
    }


def run_operation(*, review_id, bundle, request, operation_id, lock_fd=None):
    """Run under bundle ownership, including across launcher process death."""
    lock = (
        _acquire_operation_lock(bundle)
        if lock_fd is None
        else os.fdopen(lock_fd, "a+")
    )
    with lock:
        write_json(
            Path(bundle) / OPERATION_FILE,
            {
                "status": "running",
                "operation_id": operation_id,
                "action": request["action"],
            },
        )
        _run_operation(
            review_id=review_id,
            bundle=bundle,
            request=request,
            operation_id=operation_id,
        )


def _run_operation(*, review_id, bundle, request, operation_id):
    """Worker entry point: all schema imports and DB work stay in this process."""
    root = Path(bundle)
    state = {
        "status": "running",
        "action": request["action"],
        "operation_id": operation_id,
    }

    def report(message, **values):
        state.update(message=message, **values)
        write_json(root / OPERATION_FILE, state)

    try:
        from spyglass.spikesorting.v2.review_api import FigPackReview

        review = FigPackReview.resume(review_id)
        if Path(review.uri).resolve() != root.resolve():
            raise ValueError(
                "The served bundle does not belong to this review."
            )
        action = request["action"]
        if action in {"preview", "commit"}:
            changes = review.preview_import()
            if action == "preview":
                report(
                    "Review the saved changes before committing.",
                    preview={
                        "annotations_hash": changes.annotations_hash,
                        "summary": changes.summary(),
                        "changed_units": changes.changed_units().to_dict(
                            orient="records"
                        ),
                        "has_changes": changes.has_changes,
                        "has_merges": bool(changes.merge_groups),
                        "conflicts": [
                            {
                                "merged_unit_id": conflict.merged_unit_id,
                                "contributors": {
                                    str(unit): list(labels)
                                    for unit, labels in conflict.contributor_labels.items()
                                },
                                "choices": list(
                                    dict.fromkeys(
                                        [
                                            *review.profile.label_options,
                                            *(
                                                label
                                                for labels in conflict.contributor_labels.values()
                                                for label in labels
                                            ),
                                        ]
                                    )
                                ),
                            }
                            for conflict in changes.label_conflicts
                        ],
                    },
                )
            else:
                if changes.annotations_hash != request.get("annotations_hash"):
                    raise ValueError(
                        "Saved draft changed after preview. Preview the current draft before committing."
                    )

                def committed(child):
                    identity = _curation_identity(child)
                    write_json(
                        root / RESULT_FILE,
                        {
                            "curation": identity,
                            "needs_merge_verification": bool(
                                changes.merge_groups
                            ),
                            "pending": True,
                        },
                    )
                    report(
                        (
                            "Curation committed. Reevaluating merged units…"
                            if changes.merge_groups
                            else "Curation committed."
                        ),
                        committed_curation=identity,
                    )

                receipt = changes.commit(
                    conflict_resolutions={
                        int(unit): tuple(labels)
                        for unit, labels in request.get(
                            "conflict_resolutions", {}
                        ).items()
                    },
                    confirm_no_changes=bool(
                        request.get("confirm_no_changes", False)
                    ),
                    on_commit=committed,
                )
                result = {
                    "curation": _curation_identity(receipt.curation),
                    "needs_merge_verification": receipt.needs_merge_verification,
                    "pending": False,
                }
                if receipt.needs_merge_verification:
                    report("Preparing the merged-unit verification review…")
                    child_review = receipt.continue_review()
                    from spyglass.spikesorting.v2._curation_transforms import (
                        allocate_merged_unit_ids,
                    )
                    from spyglass.spikesorting.v2.curation import CurationV2

                    unit_ids = (CurationV2.Unit & review.parent.as_key()).fetch(
                        "unit_id"
                    )
                    focus = list(
                        allocate_merged_unit_ids(unit_ids, changes.merge_groups)
                    )
                    result["verification_review_id"] = str(
                        child_review.review_id
                    )
                    write_json(
                        Path(child_review.uri) / ORIGIN_FILE,
                        {"review_id": str(review.review_id)},
                    )
                    state["destination"] = {
                        "bundle": child_review.uri,
                        "review_id": str(child_review.review_id),
                        "focus_unit_ids": focus,
                    }
                write_json(root / RESULT_FILE, result)
                report(
                    (
                        "Inspect the reevaluated merged units and record their review."
                        if receipt.needs_merge_verification
                        else "Review complete. This curation is ready for analysis."
                    ),
                    result=result,
                )
        elif action == "parent":
            origin = read_json(root / ORIGIN_FILE)
            if origin:
                parent_review = FigPackReview.resume(origin["review_id"])
            else:
                parent = review.parent.parent
                if parent is None:
                    raise ValueError(
                        "This is the root curation; there is no earlier parent to review."
                    )
                parent_review = parent.start_review(
                    review.profile, display_options=review.display_options
                )
            state["destination"] = {
                "bundle": parent_review.uri,
                "review_id": str(parent_review.review_id),
            }
            report(
                "Reviewing the earlier branch; undo its draft merge to create a replacement."
            )
        elif action == "inspect":
            report("Preparing selected-unit inspection…")
            units = request.get("unit_ids", [])
            time_range = request.get("time_range")
            view = review.inspect_units(
                units,
                time_range=time_range,
                include_traces=bool(
                    time_range and time_range[1] - time_range[0] <= 10
                ),
            )
            destination = root / "inspection" / operation_id
            view.save(str(destination), title="Selected-unit inspection")
            state["destination"] = {"bundle": str(destination)}
            report(
                "Selected-unit inspection is ready; the curation draft is unchanged."
            )
        report(state.get("message", "Complete"), status="complete")
    # Retain the scientific receipt even when a worker's operation fails.
    except Exception as exc:  # noqa: BLE001
        recovery = (
            " Use Preview and commit again to retry; any committed child is retained."
            if request["action"] == "commit"
            else ""
        )
        report(f"{type(exc).__name__}: {exc}{recovery}", status="failed")


def committed_review_result(review):
    """Follow this review's explicit verification chain, never a latest child."""
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.review_api import FigPackReview

    visited = set()
    while True:
        if review.review_id in visited:
            raise ValueError("Review result contains a verification cycle.")
        visited.add(review.review_id)
        value = read_json(Path(review.uri) / RESULT_FILE)
        if not value or value.get("pending"):
            raise ValueError(
                "This browser review has not completed. Resume its commit or verification action."
            )
        if value.get("needs_merge_verification"):
            review = FigPackReview.resume(value["verification_review_id"])
            continue
        identity = value["curation"]
        ref = CurationRef.from_key(identity)
        if str(ref.curation_uuid) != identity["curation_uuid"]:
            raise ValueError(
                "The committed curation generation no longer exists."
            )
        return ref
