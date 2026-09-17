"""Local HTTP delivery of a saved FigPack review bundle.

A FigPack bundle opened as a ``file://`` URL shows a directory listing (or,
for its ``index.html``, an empty page whose module scripts fail CORS), and
the local draft control requires a loopback HTTP origin to write
``annotations.json``. This module serves the exact persisted bundle so **Save
draft** writes the same file that :meth:`FigPackReview.preview_import` reads.
Nothing is copied or rebuilt to open a review.

Servers live only in this Python process (a background daemon thread per
bundle, reused on repeated opens, stopped at interpreter exit); the port is
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

DB-free: imports only the standard library and, lazily, FigPack's request
handler.
"""

from __future__ import annotations

import atexit
import json
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
        """Bundle delivery and review-scoped operations through an adapter."""

        def do_PUT(self):
            if not self._same_origin():
                return
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

        def _same_origin(self):
            hosts = {
                f"localhost:{self.server.server_port}",
                f"127.0.0.1:{self.server.server_port}",
            }
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in hosts or (
                origin is not None and origin != f"http://{host}"
            ):
                self.send_error(
                    403, "Review actions require this local review's origin."
                )
                return False
            return True

        def _json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if not path.startswith("/api/"):
                return super().do_GET()
            if not self._same_origin():
                return
            service = self.server.review_operations
            if path == "/api/capabilities":
                self._json(
                    {
                        "connected": service is not None,
                        "review_id": service.review_id if service else None,
                    }
                )
            elif path == "/api/operation" and service is not None:
                self._json(service.status())
            else:
                self._json(
                    {"error": "This bundle has no connected review operation."},
                    404,
                )

        def do_POST(self):
            if not self._same_origin():
                return
            service = self.server.review_operations
            if (
                urllib.parse.urlsplit(self.path).path != "/api/operation"
                or service is None
            ):
                self._json(
                    {"error": "This bundle has no connected review operation."},
                    404,
                )
                return
            if self.headers.get("X-Spyglass-Review") != service.review_id:
                self._json(
                    {"error": "The request does not identify this review."},
                    403,
                )
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 1_000_000:
                    raise ValueError(
                        "Review request must be a JSON object smaller than 1 MB."
                    )
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise TypeError("Review request must be an object.")
                self._json(service.start(request), 202)
            except (ValueError, TypeError) as exc:
                self._json({"error": str(exc)}, 400)

    return ReviewBundleRequestHandler


def review_bundle_url(port: int) -> str:
    """The browser URL for a bundle served on ``port``.

    Spelled with ``localhost`` (not ``127.0.0.1``): the FigPack frontend
    enables local in-place editing only for ``http://localhost:<port>/``.
    """
    return f"http://localhost:{port}/"


def serve_review_bundle(
    bundle, *, port: int | None = None, operation_factory=None
) -> str:
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
    operation_factory : callable, optional
        Create the scientific operation adapter when attaching a connected
        review. Omit for DB-free static/read-only inspection delivery.

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
            if (
                running[0].review_operations is None
                and operation_factory is not None
            ):
                running[0].review_operations = operation_factory()
            return review_bundle_url(running[0].server_port)
        handler = partial(
            _handler_class(),
            directory=str(root),
            enable_file_upload=True,
        )
        server = ThreadingHTTPServer(("127.0.0.1", port or 0), handler)
        server.daemon_threads = True
        server.review_operations = (
            operation_factory() if operation_factory else None
        )
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
            if server.review_operations is not None:
                server.review_operations.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


atexit.register(stop_review_servers)
