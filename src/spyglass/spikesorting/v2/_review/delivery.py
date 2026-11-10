"""Local HTTP delivery of a saved FigPack review bundle.

``file://`` breaks the bundle's module scripts and draft saving, so the
persisted bundle is served over loopback; **Save draft** writes the
``annotations.json`` that :meth:`FigPackReview.preview_import` reads.

- One daemon server per process, reused across bundles; the port is never
  persisted. Each bundle's URL id is random, so only its holder can reach it.
- Only ``annotations.json`` accepts ``PUT``, with the last-read revision.
- An optional operation adapter accepts same-origin, review-scoped POSTs.
"""

from __future__ import annotations

import atexit
import secrets
import threading
from dataclasses import dataclass
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


@dataclass
class _Bundle:
    root: Path
    operations: object = None


_SERVER: tuple[ThreadingHTTPServer, threading.Thread] | None = None
_LOCK = threading.Lock()


def _handler_class():
    """Combine the HTTP adapter with FigPack's lazy static-file handler."""
    from figpack.core._file_handler import FileUploadCORSRequestHandler

    from spyglass.spikesorting.v2._review.http import ReviewBundleRequestMixin

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


def with_focus_units(url: str, unit_ids) -> str:
    """Return ``url`` with ``spyglass_units`` set to ``unit_ids`` (if any)."""
    if not len(unit_ids):
        return url

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["spyglass_units"] = ",".join(str(int(unit)) for unit in unit_ids)
    return urlunsplit(parts._replace(query=urlencode(query)))


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
        If the bundle directory or its ``index.html`` is missing.
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
        # A random id keeps other local users who know the path out.
        ids = {served.root: key for key, served in server.bundles.items()}
        bundle_id = ids.get(root) or secrets.token_urlsafe(32)
        served = server.bundles.setdefault(bundle_id, _Bundle(root))
        if served.operations is None and operation_factory is not None:
            served.operations = operation_factory()
        return review_bundle_url(server.server_port, bundle_id)


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
