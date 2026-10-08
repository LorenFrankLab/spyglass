"""HTTP routing for static review assets, draft storage and operation adapters."""

import json
import urllib.parse
from pathlib import Path

from spyglass.spikesorting.v2._review.drafts import DraftError, ReviewDraftStore

WRITABLE_BUNDLE_FILES = frozenset({"annotations.json"})


class ReviewBundleRequestMixin:
    """Adapt review services to FigPack's static-file request handler."""

    def parse_request(self):
        if not super().parse_request():
            return False
        parsed = urllib.parse.urlsplit(self.path)
        parts = parsed.path.split("/", 3)
        if len(parts) != 4 or parts[1] != "bundles":
            self.send_error(404, "Unknown review bundle")
            return False
        with self.server.bundle_lock:
            self.bundle = self.server.bundles.get(parts[2])
        if self.bundle is None:
            self.send_error(404, "Unknown review bundle")
            return False
        self.directory = str(self.bundle.root)
        self.bundle_prefix = f"/bundles/{parts[2]}"
        self.bundle_path = parsed._replace(path="/" + parts[3]).geturl()
        self.drafts = ReviewDraftStore(self.bundle.root)
        return True

    def translate_path(self, path):
        # Preserve the original URL for redirects; strip its prefix only for I/O.
        return super().translate_path(path.removeprefix(self.bundle_prefix))

    def _get_safe_file_path(self):
        try:
            return self.drafts.path
        except DraftError as exc:
            self.send_error(exc.status, str(exc))
            return None

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

    def _read_json(self, limit, size_message, type_message):
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= limit:
            raise ValueError(size_message)
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise TypeError(type_message)
        return value

    def _json(self, value, status=200, *, etag=None):
        self._respond(
            json.dumps(value).encode(), "application/json", status, etag=etag
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
        if self._get_safe_file_path() is None:
            return
        revision = self.headers.get("If-Match")
        if revision is None:
            self._json(
                {"error": "Reload the review before saving a draft."}, 428
            )
            return
        try:
            value = self._read_json(
                10_000_000,
                "Draft must be smaller than 10 MB.",
                "Draft must be a JSON object.",
            )
            etag = self.drafts.save(value, revision)
        except DraftError as exc:
            self._json({"error": str(exc)}, exc.status)
            return
        except (ValueError, TypeError, UnicodeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        self._json({"saved": True}, etag=etag)

    def _serve_controls(self):
        self._respond(
            Path(__file__).with_name("controls.js").read_bytes(),
            "text/javascript",
        )

    def _serve_annotations(self):
        try:
            value, revision, present = self.drafts.read()
        except DraftError as exc:
            self.send_error(exc.status, str(exc))
            return
        self._json(value, 200 if present else 404, etag=revision)

    def _serve_api(self, path):
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
                {"error": "This bundle has no connected review operation."}, 404
            )

    def do_GET(self):
        path = urllib.parse.urlsplit(self.bundle_path).path
        route = {
            "/extension-spyglass-review.js": self._serve_controls,
            "/annotations.json": self._serve_annotations,
        }.get(path)
        if route is not None:
            return route()
        if not path.startswith("/api/"):
            return super().do_GET()
        if self._same_origin():
            self._serve_api(path)

    def do_POST(self):
        if not self._same_origin():
            return
        service = self.bundle.operations
        if (
            urllib.parse.urlsplit(self.bundle_path).path != "/api/operation"
            or service is None
        ):
            self._json(
                {"error": "This bundle has no connected review operation."}, 404
            )
            return
        if self.headers.get("X-Spyglass-Review") != service.review_id:
            self._json(
                {"error": "The request does not identify this review."}, 403
            )
            return
        try:
            request = self._read_json(
                1_000_000,
                "Review request must be a JSON object smaller than 1 MB.",
                "Review request must be an object.",
            )
            self._json(service.start(request), 202)
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)
