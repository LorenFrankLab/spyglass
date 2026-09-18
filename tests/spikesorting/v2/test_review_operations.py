"""Operation transport retains scientific identity across partial failures."""

import json
from types import SimpleNamespace

import pytest


def test_worker_failure_retains_receipt_and_retry_completes_journal(
    tmp_path, monkeypatch
):
    from spyglass.spikesorting.v2 import review_api
    from spyglass.spikesorting.v2._review_operations import (
        OPERATION_FILE,
        RESULT_FILE,
        run_operation,
    )

    child = SimpleNamespace(
        sorting_id="sort", curation_id=2, curation_uuid="generation"
    )
    calls = []

    def commit(**kwargs):
        calls.append(kwargs)
        kwargs["on_commit"](child)
        if len(calls) == 1:
            raise RuntimeError("Evaluation failed; child was committed")
        return SimpleNamespace(curation=child, needs_merge_verification=False)

    changes = SimpleNamespace(
        annotations_hash="saved", merge_groups=(), commit=commit
    )
    review = SimpleNamespace(uri=str(tmp_path), preview_import=lambda: changes)
    monkeypatch.setattr(
        review_api.FigPackReview, "resume", lambda review_id: review
    )
    request = {
        "action": "commit",
        "annotations_hash": "saved",
        "confirm_no_changes": True,
    }
    run_operation(
        review_id="review",
        bundle=str(tmp_path),
        request=request,
        operation_id="first",
    )
    pending = json.loads((tmp_path / RESULT_FILE).read_text())
    assert pending["pending"]
    assert pending["curation"]["curation_uuid"] == "generation"
    assert (
        json.loads((tmp_path / OPERATION_FILE).read_text())["status"]
        == "failed"
    )
    run_operation(
        review_id="review",
        bundle=str(tmp_path),
        request=request,
        operation_id="retry",
    )
    completed = json.loads((tmp_path / RESULT_FILE).read_text())
    assert not completed["pending"]
    assert completed["curation"] == pending["curation"]
    assert len(calls) == 2

    # A stale browser preview never reaches the scientific commit.
    run_operation(
        review_id="review",
        bundle=str(tmp_path),
        request={**request, "annotations_hash": "stale"},
        operation_id="stale",
    )
    assert len(calls) == 2
    assert (
        json.loads((tmp_path / OPERATION_FILE).read_text())["status"]
        == "failed"
    )


def test_service_reports_interrupted_work_and_serializes_actions(
    tmp_path, monkeypatch
):
    import threading

    from spyglass.spikesorting.v2._review_operations import (
        OPERATION_FILE,
        ReviewOperationService,
    )

    (tmp_path / OPERATION_FILE).write_text(json.dumps({"status": "running"}))
    service = ReviewOperationService("review", tmp_path, {})
    assert service.status()["status"] == "interrupted"
    entered, finish = threading.Event(), threading.Event()

    def execute(request, operation_id, lock_fd):
        entered.set()
        finish.wait(5)

    monkeypatch.setattr(service, "_execute", execute)
    try:
        service.start({"action": "preview"})
        assert entered.wait(2)
        with pytest.raises(ValueError, match="previous action"):
            service.start({"action": "commit"})
    finally:
        finish.set()
        service.executor.shutdown(wait=True)


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_service_waits_for_worker_release_before_reporting_result(
    tmp_path, status
):
    from spyglass.spikesorting.v2._review_operations import (
        OPERATION_FILE,
        ReviewOperationService,
        _acquire_operation_lock,
    )

    state = {"status": status, "action": "preview", "operation_id": "finishing"}
    journal = tmp_path / OPERATION_FILE
    journal.write_text(json.dumps(state))
    service = ReviewOperationService("review", tmp_path, {})
    try:
        with _acquire_operation_lock(tmp_path):
            observed = service.status()
            assert observed["status"] == "running"
            assert observed["operation_id"] == state["operation_id"]
            assert json.loads(journal.read_text()) == state
        assert service.status() == state
    finally:
        service.close()


def test_delivery_failure_keeps_the_scientific_result(tmp_path, monkeypatch):
    from spyglass.spikesorting.v2 import _review_delivery
    from spyglass.spikesorting.v2._review_operations import (
        OPERATION_FILE,
        ReviewOperationService,
    )

    state = {
        "status": "complete",
        "committed_curation": {"curation_id": 2},
        "destination": {
            "bundle": str(tmp_path / "child"),
            "review_id": "child",
        },
    }
    (tmp_path / OPERATION_FILE).write_text(json.dumps(state))

    def unavailable(*args, **kwargs):
        raise OSError("Cannot bind a loopback port")

    monkeypatch.setattr(_review_delivery, "serve_review_bundle", unavailable)
    service = ReviewOperationService("review", tmp_path, {})
    try:
        status = service.status()
        assert status["status"] == "failed"
        assert status["committed_curation"] == {"curation_id": 2}
        assert "retained" in status["message"]
        assert json.loads((tmp_path / OPERATION_FILE).read_text()) == state
    finally:
        service.close()


def test_reopened_service_cannot_overlap_a_live_worker(tmp_path, monkeypatch):
    import threading

    from spyglass.spikesorting.v2._review_operations import (
        ReviewOperationService,
    )

    first = ReviewOperationService("review", tmp_path, {})
    second = ReviewOperationService("review", tmp_path, {})
    entered, finish = threading.Event(), threading.Event()

    def execute(request, operation_id, lock_fd):
        entered.set()
        assert finish.wait(5)

    monkeypatch.setattr(first, "_execute", execute)
    try:
        started = first.start({"action": "commit"})
        assert entered.wait(2)
        first.close()
        assert second.status()["status"] == "running"
        assert second.status()["operation_id"] == started["operation_id"]
        with pytest.raises(ValueError, match="previous action"):
            second.start({"action": "inspect"})
    finally:
        finish.set()
        first.executor.shutdown(wait=True)
        second.close()
    assert second.status()["status"] == "interrupted"


def test_bundle_ownership_survives_launcher_exit(tmp_path):
    import subprocess
    import sys
    import time

    from spyglass.spikesorting.v2._review_operations import (
        _acquire_operation_lock,
    )

    # The child retains the inherited lock even when its notebook process dies.
    worker = """import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
with os.fdopen(int(sys.argv[2]), "a+"):
    (root / "started").touch()
    deadline = time.monotonic() + 10
    while not (root / "finish").exists() and time.monotonic() < deadline:
        time.sleep(.02)
"""
    launcher = """import os, subprocess, sys
from spyglass.spikesorting.v2._review_operations import _acquire_operation_lock
lock = _acquire_operation_lock(sys.argv[1])
subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1], str(lock.fileno())], pass_fds=(lock.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", launcher, str(tmp_path), worker],
        check=True,
        timeout=5,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not (tmp_path / "started").exists() and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert (tmp_path / "started").exists()
        with pytest.raises(BlockingIOError):
            _acquire_operation_lock(tmp_path)
    finally:
        (tmp_path / "finish").touch()
    deadline = time.monotonic() + 5
    while True:
        try:
            with _acquire_operation_lock(tmp_path):
                break
        except BlockingIOError:
            assert time.monotonic() < deadline
            time.sleep(0.02)
