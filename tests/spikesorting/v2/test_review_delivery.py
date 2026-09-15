"""Local HTTP delivery of a saved review bundle (DB-free).

Serves a stand-in bundle directory and exercises the contract the browser
relies on: the bundle is readable at ``http://localhost:<port>/``, only its
``annotations.json`` accepts a PUT (so a browser save lands in the exact
durable file the importer reads), repeated opens reuse one server per
bundle, and a missing bundle fails with a recovery hint.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
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

    stop_review_servers(root)


def _put(url: str, body: bytes) -> int:
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={"Content-Type": "application/json"},
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
    assert _put(url + "annotations.json", edited.encode()) == 200
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
