"""Local HTTP delivery of a saved FigPack review bundle.

A FigPack bundle opened as a ``file://`` URL shows a directory listing (or,
for its ``index.html``, an empty page whose module scripts fail CORS), and
the FigPack frontend only enables in-place editing -- **Curate Figure** /
**Save Annotations**, which ``PUT`` the figure's ``annotations.json`` --
for a figure served from ``http://localhost:<port>/``. This module serves
the exact persisted bundle over a loopback server so the browser's saves
land in the same ``annotations.json`` that :meth:`FigPackReview.preview_import`
reads. Nothing is copied or rebuilt to open a review.

Servers live only in this Python process (a background daemon thread per
bundle, reused on repeated opens, stopped at interpreter exit); the port is
never persisted -- ``review.uri`` stays the durable filesystem location and a
resume after a kernel restart simply starts delivery again over the same
files.

Security posture of the writable endpoint: bind loopback only, serve only the
requested bundle directory (FigPack's handler already refuses paths outside
it), and accept ``PUT`` for the bundle's ``annotations.json`` alone -- the
scientific data (``data.zarr``), the frontend assets and the Spyglass identity
sidecar stay read-only.

DB-free: imports only the standard library and, lazily, FigPack's request
handler.
"""

from __future__ import annotations

import atexit
import threading
import urllib.parse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

#: The only file the browser may write: the FigPack annotation sidecar.
WRITABLE_BUNDLE_FILES = frozenset({"annotations.json"})

_SERVERS: dict[Path, tuple[ThreadingHTTPServer, threading.Thread]] = {}
_LOCK = threading.Lock()


def _handler_class():
    """Build the request-handler class (FigPack import kept lazy)."""
    from figpack.core._file_handler import FileUploadCORSRequestHandler

    class ReviewBundleRequestHandler(FileUploadCORSRequestHandler):
        """FigPack's upload-capable handler, writes limited to annotations."""

        def do_PUT(self):
            relative = urllib.parse.unquote(
                urllib.parse.urlparse(self.path).path.lstrip("/")
            )
            if relative not in WRITABLE_BUNDLE_FILES:
                self.send_error(
                    403,
                    "Forbidden: only the review's annotations.json may be "
                    "written through the local review server",
                )
                return
            super().do_PUT()

    return ReviewBundleRequestHandler


def review_bundle_url(port: int) -> str:
    """The browser URL for a bundle served on ``port``.

    Spelled with ``localhost`` (not ``127.0.0.1``): the FigPack frontend
    enables local in-place editing only for ``http://localhost:<port>/``.
    """
    return f"http://localhost:{port}/"


def serve_review_bundle(bundle, *, port: int | None = None) -> str:
    """Serve ``bundle`` over loopback (starting or reusing a server); return its URL.

    Parameters
    ----------
    bundle : str or Path
        The saved FigPack bundle directory (``review.uri`` for a local
        review). Must contain ``index.html``.
    port : int, optional
        Loopback port to bind on a fresh start (``None`` picks a free one).
        Ignored when this process already serves the bundle; the running
        server's URL is returned instead.

    Returns
    -------
    str
        ``http://localhost:<port>/``.

    Raises
    ------
    FileNotFoundError
        If the bundle directory or its ``index.html`` is missing (the
        message names the recovery: rebuild the review).
    """
    root = Path(bundle).resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(
            f"No FigPack review bundle at {root} (missing index.html). "
            "It was moved or deleted; rebuild it by starting the review "
            "again from its curation (start_review(...) recreates the bundle "
            "-- edits saved into the missing bundle are gone)."
        )
    with _LOCK:
        running = _SERVERS.get(root)
        if running is not None and running[1].is_alive():
            return review_bundle_url(running[0].server_port)
        handler = partial(
            _handler_class(),
            directory=str(root),
            enable_file_upload=True,
        )
        server = ThreadingHTTPServer(("127.0.0.1", port or 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name=f"spyglass-review-server:{root.name}",
            daemon=True,
        )
        thread.start()
        _SERVERS[root] = (server, thread)
        return review_bundle_url(server.server_port)


def served_review_bundles() -> dict[Path, str]:
    """``{bundle: url}`` for every bundle this process is serving."""
    with _LOCK:
        return {
            root: review_bundle_url(server.server_port)
            for root, (server, thread) in _SERVERS.items()
            if thread.is_alive()
        }


def stop_review_servers(bundle=None) -> None:
    """Stop the server for ``bundle`` (or every server when ``None``)."""
    with _LOCK:
        roots = list(_SERVERS) if bundle is None else [Path(bundle).resolve()]
        for root in roots:
            entry = _SERVERS.pop(root, None)
            if entry is None:
                continue
            server, thread = entry
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


atexit.register(stop_review_servers)
