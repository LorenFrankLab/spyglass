"""Local HTTP delivery of a saved FigPack review bundle.

A FigPack bundle opened as a ``file://`` URL shows a directory listing (or,
for its ``index.html``, an empty page whose module scripts fail CORS), and
the local draft control requires a loopback HTTP origin to write
``annotations.json``. This module serves the exact persisted bundle so **Save
draft** writes the same file that :meth:`FigPackReview.preview_import` reads.
Nothing is copied or rebuilt to open a review.

One server lives in this Python process (a background daemon thread, reused
across bundles, stopped at interpreter exit); the port is
never persisted -- ``review.uri`` stays the durable filesystem location and a
resume after a kernel restart simply starts delivery again over the same
files.

Security posture of the writable endpoint: bind loopback only, serve only the
requested bundle directory (FigPack's handler already refuses paths outside
it), and accept ``PUT`` for the bundle's ``annotations.json`` alone -- the
scientific data (``data.zarr``), the frontend assets and the Spyglass identity
sidecar stay read-only. An optional operation adapter accepts same-origin,
review-scoped POST requests for scientific actions; the handler itself has no
database connection or scientific policy.

Draft saves require the revision returned by the last read. A filesystem lock
serializes revision checks and atomic replacement across review processes.
DB-free; FigPack is imported when starting delivery, filelock when saving a draft.
"""

from __future__ import annotations

import atexit
import hashlib
import threading
from dataclasses import dataclass
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

#: The only file the browser may write: the FigPack annotation sidecar.
from spyglass.spikesorting.v2._review_http import WRITABLE_BUNDLE_FILES


@dataclass
class _Bundle:
    root: Path
    operations: object = None


_SERVER: tuple[ThreadingHTTPServer, threading.Thread] | None = None
_LOCK = threading.Lock()


def _handler_class():
    """Combine the HTTP adapter with FigPack's lazy static-file handler."""
    from figpack.core._file_handler import FileUploadCORSRequestHandler

    from spyglass.spikesorting.v2._review_http import ReviewBundleRequestMixin

    class ReviewBundleRequestHandler(
        ReviewBundleRequestMixin, FileUploadCORSRequestHandler
    ):
        pass

    return ReviewBundleRequestHandler


def review_bundle_url(port: int, bundle_id: str) -> str:
    """The browser URL for a bundle served on ``port``.

    Spelled with ``localhost`` (not ``127.0.0.1``): the FigPack frontend
    enables local in-place editing only for ``http://localhost:<port>/``.
    """
    return f"http://localhost:{port}/bundles/{bundle_id}/"


def serve_review_bundle(
    bundle, *, port: int | None = None, operation_factory=None
) -> str:
    """Serve ``bundle`` on this process's shared loopback server; return its URL.

    Parameters
    ----------
    bundle : str or Path
        The saved FigPack bundle directory (``review.uri`` for a local
        review). Must contain ``index.html``.
    port : int, optional
        Loopback port to bind on a fresh start (``None`` picks a free one).
        Ignored when this process already serves any bundle. All reviews and
        inspections share that port, so one SSH tunnel supports navigation.
    operation_factory : callable, optional
        Create the scientific operation adapter when attaching a connected
        review. Omit for DB-free static/read-only inspection delivery.

    Returns
    -------
    str
        ``http://localhost:<port>/bundles/<id>/``.

    Raises
    ------
    FileNotFoundError
        If the bundle directory or its ``index.html`` is missing (the
        message names the recovery: rebuild the review).
    """
    global _SERVER
    root = Path(bundle).resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(
            f"No FigPack review bundle at {root} (missing index.html). "
            "It was moved or deleted; rebuild it by starting the review "
            "again from its curation (start_review(...) recreates the bundle "
            "-- edits saved into the missing bundle are gone)."
        )
    bundle_id = hashlib.sha256(str(root).encode()).hexdigest()
    with _LOCK:
        if _SERVER is None:
            handler = partial(_handler_class(), enable_file_upload=True)
            server = ThreadingHTTPServer(("127.0.0.1", port or 0), handler)
            server.daemon_threads = True
            server.bundles = {}
            server.bundle_lock = _LOCK
            thread = threading.Thread(
                target=server.serve_forever,
                name="spyglass-review-server",
                daemon=True,
            )
            thread.start()
            _SERVER = server, thread
        server, _ = _SERVER
        served = server.bundles.setdefault(bundle_id, _Bundle(root))
        if served.operations is None and operation_factory is not None:
            served.operations = operation_factory()
        return review_bundle_url(server.server_port, bundle_id)


def served_review_bundles() -> dict[Path, str]:
    """``{bundle: url}`` for every bundle this process is serving."""
    with _LOCK:
        if _SERVER is None:
            return {}
        server, _ = _SERVER
        return {
            bundle.root: review_bundle_url(server.server_port, bundle_id)
            for bundle_id, bundle in server.bundles.items()
        }


def stop_review_servers(bundle=None) -> None:
    """Unregister a bundle; stop delivery when none remain (or stop all)."""
    global _SERVER
    root = Path(bundle).resolve() if bundle is not None else None
    with _LOCK:
        if _SERVER is None:
            return
        server, thread = _SERVER
        for bundle_id, served in list(server.bundles.items()):
            if root is None or served.root == root:
                del server.bundles[bundle_id]
                if served.operations is not None:
                    served.operations.close()
        stop = not server.bundles
        if stop:
            _SERVER = None
    if stop:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


atexit.register(stop_review_servers)
