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
DB-free; FigPack and filelock are imported only when starting delivery.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import threading
import urllib.parse
from dataclasses import dataclass
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

#: The only file the browser may write: the FigPack annotation sidecar.
WRITABLE_BUNDLE_FILES = frozenset({"annotations.json"})


@dataclass
class _Bundle:
    root: Path
    operations: object = None


_SERVER: tuple[ThreadingHTTPServer, threading.Thread] | None = None
_LOCK = threading.Lock()


def _handler_class():
    """Build the request-handler class (FigPack import kept lazy)."""
    from figpack.core._file_handler import FileUploadCORSRequestHandler
    from filelock import FileLock

    from spyglass.spikesorting.v2._json_io import write_json

    def draft_revision(path):
        data = path.read_bytes() if path.exists() else b""
        return '"' + hashlib.sha256(data).hexdigest() + '"'

    class ReviewBundleRequestHandler(FileUploadCORSRequestHandler):
        """Bundle delivery and review-scoped operations through an adapter."""

        def parse_request(self):
            if not super().parse_request():
                return False
            parsed = urllib.parse.urlsplit(self.path)
            parts = parsed.path.split("/", 3)
            if len(parts) != 4 or parts[1] != "bundles":
                self.send_error(404, "Unknown review bundle")
                return False
            with _LOCK:
                self.bundle = self.server.bundles.get(parts[2])
            if self.bundle is None:
                self.send_error(404, "Unknown review bundle")
                return False
            self.directory = str(self.bundle.root)
            self.bundle_prefix = f"/bundles/{parts[2]}"
            self.bundle_path = parsed._replace(path="/" + parts[3]).geturl()
            return True

        def translate_path(self, path):
            # Keep self.path intact for directory redirects and browser URLs;
            # only filesystem resolution strips the registered bundle prefix.
            return super().translate_path(path.removeprefix(self.bundle_prefix))

        def _get_safe_file_path(self):
            # do_PUT admits exactly this sidecar, never a client-supplied path.
            path = (self.bundle.root / "annotations.json").resolve()
            if not path.is_relative_to(self.bundle.root):
                self.send_error(403, "Forbidden: path outside review bundle")
                return None
            return path

        def do_PUT(self):
            if not self._same_origin():
                return
            relative = urllib.parse.unquote(
                urllib.parse.urlparse(self.bundle_path).path.lstrip("/")
            )
            if relative not in WRITABLE_BUNDLE_FILES:
                self.send_error(
                    403,
                    "Forbidden: only the review's annotations.json may be "
                    "written through the local review server",
                )
                return
            path = self._get_safe_file_path()
            if path is None:
                return
            revision = self.headers.get("If-Match")
            if revision is None:
                self._json(
                    {"error": "Reload the review before saving a draft."}, 428
                )
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 10_000_000:
                    raise ValueError("Draft must be smaller than 10 MB.")
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict):
                    raise TypeError("Draft must be a JSON object.")
            except (ValueError, TypeError, UnicodeError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            # Atomic replacement keeps readers from seeing partial JSON. The
            # lock covers comparison AND write, including other processes on
            # shared storage; a thread lock alone cannot protect this draft.
            with FileLock(str(path) + ".lock"):
                if revision != draft_revision(path):
                    self._json(
                        {"error": "Another tab changed this draft."}, 409
                    )
                    return
                write_json(path, value)
                self._json({"saved": True}, etag=draft_revision(path))

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

        def _json(self, value, status=200, *, etag=None):
            self._respond(
                json.dumps(value).encode(),
                "application/json",
                status,
                etag=etag,
            )

        def _respond(self, data, content_type, status=200, *, etag=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            if etag is not None:
                self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlsplit(self.bundle_path).path
            if path == "/extension-spyglass-review.js":
                # Controls and the local save protocol upgrade together, even
                # when reopening a saved bundle from an earlier installation.
                self._respond(
                    Path(__file__)
                    .with_name("_review_controls.js")
                    .read_bytes(),
                    "text/javascript",
                )
                return
            if path == "/annotations.json":
                target = self._get_safe_file_path()
                if target is None:
                    return
                # Read payload and revision from the SAME bytes. A writer may
                # replace the file immediately afterwards; If-Match catches it.
                try:
                    data = target.read_bytes()
                except FileNotFoundError:
                    data = b""
                etag = '"' + hashlib.sha256(data).hexdigest() + '"'
                self._json(
                    json.loads(data) if data else {},
                    200 if data else 404,
                    etag=etag,
                )
                return
            if not path.startswith("/api/"):
                return super().do_GET()
            if not self._same_origin():
                return
            service = self.bundle.operations
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
            service = self.bundle.operations
            if (
                urllib.parse.urlsplit(self.bundle_path).path != "/api/operation"
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
