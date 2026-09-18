"""Local HTTP delivery of a saved review bundle (DB-free).

Serves a stand-in bundle directory and exercises the contract the browser
relies on: the bundle is readable at ``http://localhost:<port>/bundles/<id>/``, only its
``annotations.json`` accepts a PUT (so a browser save lands in the exact
durable file the importer reads), every bundle shares one server port,
and a missing bundle fails with a recovery hint.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("figpack") is None,
    reason="requires the spikesorting-v2-curation extra (figpack)",
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "review.figpack"
    (root / "data.zarr").mkdir(parents=True)
    (root / "index.html").write_text("<html><body>review</body></html>")
    (root / "data.zarr" / ".zmetadata").write_text("{}")
    (root / "annotations.json").write_text(json.dumps({"annotations": {}}))
    (root / "spyglass_curation.json").write_text(json.dumps({"id": 1}))
    yield root
    from spyglass.spikesorting.v2._review_delivery import stop_review_servers

    stop_review_servers()


def _revision(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.headers["ETag"]


def _put(url: str, body: bytes, *, revision=None) -> int:
    headers = {"Content-Type": "application/json"}
    if revision is not None:
        headers["If-Match"] = revision
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_serves_bundle_and_writes_only_annotations(bundle):
    from spyglass.spikesorting.v2._review_delivery import (
        serve_review_bundle,
        served_review_bundles,
    )

    url = serve_review_bundle(bundle)
    assert url.startswith("http://localhost:")  # the frontend's local check
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200
        assert b"review" in response.read()
    with urllib.request.urlopen(url + "data.zarr/.zmetadata", timeout=5) as r:
        assert r.status == 200

    edited = json.dumps({"annotations": {"labelsByUnit": {"1": ["accept"]}}})
    target = url + "annotations.json"
    assert _put(target, edited.encode(), revision=_revision(target)) == 200
    assert json.loads((bundle / "annotations.json").read_text()) == json.loads(
        edited
    )
    # Scientific data, assets and the identity sidecar stay read-only.
    for path in ("spyglass_curation.json", "data.zarr/.zmetadata", "new.js"):
        assert _put(url + path, b"{}") == 403
    assert (bundle / "spyglass_curation.json").read_text() == json.dumps(
        {"id": 1}
    )
    assert not (bundle / "new.js").exists()
    assert served_review_bundles() == {bundle.resolve(): url}


def test_repeated_open_reuses_server_and_stop_frees_it(bundle):
    from spyglass.spikesorting.v2._review_delivery import (
        serve_review_bundle,
        served_review_bundles,
        stop_review_servers,
    )

    first = serve_review_bundle(bundle)
    assert serve_review_bundle(bundle) == first
    assert serve_review_bundle(bundle, port=1) == first  # port ignored
    stop_review_servers(bundle)
    assert bundle.resolve() not in served_review_bundles()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(first, timeout=2)
    second = serve_review_bundle(bundle)
    with urllib.request.urlopen(second, timeout=5) as response:
        assert response.status == 200


def test_missing_bundle_names_recovery(tmp_path):
    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    with pytest.raises(FileNotFoundError, match="start_review"):
        serve_review_bundle(tmp_path / "gone")


def test_connected_actions_require_matching_origin_and_review(bundle):
    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    class Operations:
        review_id = "pinned-review"

        def start(self, request):
            return {"status": "running"}

        def close(self):
            pass

    url = serve_review_bundle(bundle, operation_factory=Operations)
    local_origin = urllib.parse.urlsplit(url)._replace(path="").geturl()
    for origin, identity, expected in (
        ("https://unrelated.example", "pinned-review", 403),
        (local_origin, "another-review", 403),
        (local_origin, "pinned-review", 202),
    ):
        request = urllib.request.Request(
            url + "api/operation",
            data=b'{"action":"preview"}',
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "X-Spyglass-Review": identity,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == expected


@pytest.mark.parametrize("action", ["inspect", "commit", "parent"])
def test_operation_destinations_share_port_and_isolate_bundles(bundle, action):
    """The one forwarded origin serves child assets, drafts, and operations."""
    from spyglass.spikesorting.v2._review_delivery import (
        serve_review_bundle,
        stop_review_servers,
    )
    from spyglass.spikesorting.v2._review_operations import (
        OPERATION_FILE,
        ReviewOperationService,
    )

    child = bundle.parent / "child.figpack"
    child.mkdir()
    (child / "index.html").write_text("<html>child review</html>")
    (child / "annotations.json").write_text("{}")
    root_url = serve_review_bundle(
        bundle,
        operation_factory=lambda: ReviewOperationService("root", bundle, {}),
    )
    destination = {"bundle": str(child), "focus_unit_ids": [3]}
    if action != "inspect":
        destination["review_id"] = "child"
    (bundle / OPERATION_FILE).write_text(
        json.dumps(
            {"status": "complete", "action": action, "destination": destination}
        )
    )
    with urllib.request.urlopen(root_url + "api/operation") as response:
        child_url = json.load(response)["url"]
    assert (
        urllib.parse.urlsplit(child_url).netloc
        == urllib.parse.urlsplit(root_url).netloc
    )
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(child_url).query) == {
        "spyglass_units": ["3"]
    }
    child_base = urllib.parse.urlsplit(child_url)._replace(query="").geturl()
    assert serve_review_bundle(child, port=1) == child_base
    with urllib.request.urlopen(child_url) as response:
        assert b"child review" in response.read()
    with urllib.request.urlopen(child_base + "api/capabilities") as response:
        assert json.load(response) == {
            "connected": action != "inspect",
            "review_id": "child" if action != "inspect" else None,
        }
    original = (bundle / "annotations.json").read_bytes()
    target = child_base + "annotations.json"
    assert _put(target, b'{"child":true}', revision=_revision(target)) == 200
    assert (bundle / "annotations.json").read_bytes() == original
    assert json.loads((child / "annotations.json").read_text()) == {
        "child": True
    }
    assert _put(child_base + "../annotations.json", b"{}") == 403
    assert _put(child_base + "spyglass_curation.json", b"{}") == 403
    # Closing the parent must not close an inspection or verification review.
    stop_review_servers(bundle)
    with urllib.request.urlopen(child_url) as response:
        assert response.status == 200
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(root_url)
    assert error.value.code == 404


def test_stale_and_unconditional_draft_saves_preserve_current_edits(bundle):
    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    target = serve_review_bundle(bundle) + "annotations.json"
    baseline = _revision(target)
    assert _put(target, b'{"first":true}', revision=baseline) == 200
    saved = (bundle / "annotations.json").read_bytes()
    assert _put(target, b'{"second":true}', revision=baseline) == 409
    assert _put(target, b'{"second":true}') == 428
    assert (bundle / "annotations.json").read_bytes() == saved
    assert _put(target, b'{"both":true}', revision=_revision(target)) == 200


def test_reopened_bundle_uses_current_save_protocol_without_rewriting_draft(
    bundle,
):
    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    script = bundle / "extension-spyglass-review.js"
    script.write_text("// old controls without a revision header")
    original = (bundle / "annotations.json").read_bytes()
    url = serve_review_bundle(bundle)
    with urllib.request.urlopen(url + script.name) as response:
        assert b'"If-Match"' in response.read()
    assert script.read_text() == "// old controls without a revision header"
    assert (bundle / "annotations.json").read_bytes() == original


def test_competing_processes_cannot_both_replace_the_same_draft(bundle):
    """Separate review servers must coordinate through shared bundle storage."""
    import concurrent.futures
    import subprocess
    import sys

    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    target = serve_review_bundle(bundle) + "annotations.json"
    code = """
import sys
from spyglass.spikesorting.v2._review_delivery import serve_review_bundle
print(serve_review_bundle(sys.argv[1]), flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(bundle)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        other = process.stdout.readline().strip() + "annotations.json"
        baseline = _revision(target)
        assert _revision(other) == baseline
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            pending = [
                pool.submit(_put, endpoint, body, revision=baseline)
                for endpoint, body in (
                    (target, b'{"writer":1}'),
                    (other, b'{"writer":2}'),
                )
            ]
            assert sorted(f.result() for f in pending) == [200, 409]
        assert json.loads((bundle / "annotations.json").read_text())[
            "writer"
        ] in (1, 2)
    finally:
        process.communicate(timeout=10)


def test_review_open_returns_url_without_launching_browser(bundle, monkeypatch):
    """``FigPackReview.open`` serves a local bundle (and only launches a
    browser when asked); a hosted review returns its URL untouched."""
    import uuid

    from spyglass.spikesorting.v2 import review_api

    launched: list[str] = []
    monkeypatch.setattr(review_api.webbrowser, "open", launched.append)
    review = review_api.FigPackReview.__new__(review_api.FigPackReview)
    object.__setattr__(review, "uri", str(bundle))
    object.__setattr__(review, "review_id", uuid.uuid4())
    url = review.open(open_browser=False)
    assert url.startswith("http://localhost:") and launched == []
    assert review.open() == url and launched == [url]

    object.__setattr__(review, "uri", "https://figpack.org/f/abc")
    assert review.is_hosted
    assert review.open(open_browser=False) == "https://figpack.org/f/abc"
    assert launched == [url]
