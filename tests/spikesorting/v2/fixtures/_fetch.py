"""Fetch canonical MEArec ground-truth fixtures from remote storage.

Step 2 of the fixture strategy (see ``README.md``). The MEArec fixtures are
biophysical-simulation output and are **not byte-reproducible across OS/arch**
(see the ``pytest-v2`` CI comments in ``.github/workflows/test-conda.yml`` and
``fixtures_manifest.json``), so regenerating them on every CI run is both slow
and impossible to hash-gate reliably. Instead the canonical NWBs are generated
once (``generate_mearec.py``), uploaded, and *downloaded* on demand here.
Because the downloaded bytes are fixed, ``nwb_sha256`` from
``fixtures_manifest.json`` becomes a real integrity gate again.

Usage
-----
As a library (e.g. from ``conftest.py``)::

    from tests.spikesorting.v2.fixtures._fetch import ensure_fixture
    path = ensure_fixture("mearec_polymer_smoke")   # downloads if missing, else None

As a CLI (e.g. in CI -- exits non-zero if a requested fixture can't be
produced)::

    python tests/spikesorting/v2/fixtures/_fetch.py mearec_polymer_smoke

Configuring URLs
----------------
Each value is a Box **direct-download** URL -- *not* the ``/s/<token>`` web
share link, which returns an HTML preview page rather than the file. The shared
file must be set to "people with the link" with download allowed.

Given a Box share link ``https://ucsf.box.com/s/<token>``, the direct-download
URL is::

    https://ucsf.box.com/index.php?rm=box_download_shared_file&shared_name=<token>&file_id=f_<file-id>

where ``<file-id>`` is the numeric Box file id (on the file's info panel, or in
the share page's HTML). ``urllib`` follows Box's 302 to the actual file. Leave
an entry as ``None`` to keep the generate-or-skip behaviour for that fixture
(tests skip when it is absent and no URL is set).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Transient-download retry policy (network blips / 5xx). A hash mismatch is
# retried too (covers a corrupt transfer) but ultimately fails loudly; a 4xx
# (bad/rotated link) fails fast with no retry.
_RETRIES = 3
_BACKOFF_S = 2

_THIS_DIR = Path(__file__).resolve().parent
_MANIFEST = _THIS_DIR / "fixtures_manifest.json"

# Box direct-download URLs (see module docstring for how to derive one from a
# ``/s/<token>`` share link). ``mearec_polymer_smoke`` is the only fixture the
# per-PR CI job consumes; the rest are used by the nightly/manual job and local
# full runs -- fill them in as their share links are added.
FIXTURE_URLS: dict[str, str | None] = {
    "mearec_polymer_smoke": (  # per-PR fixture
        "https://ucsf.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=wlrdwtaj7gmuarhh7s7boniz3owkl88h"
        "&file_id=f_2275378612327"
    ),
    "mearec_polymer_128ch_60s": (  # nightly / manual
        "https://ucsf.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=97co1vzih0u4ybkd7efy8yu2s6a2nv6l"
        "&file_id=f_2275441530141"
    ),
    "mearec_neuropixels_60s": (  # nightly / manual
        "https://ucsf.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=zb391me083j7nbd9rsso9osrtmigmcj0"
        "&file_id=f_2275442214033"
    ),
    "mearec_polymer_128ch_drift_120s": (  # nightly / manual
        "https://ucsf.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=4968ma26dq71cgceqsrfgavejacmyhca"
        "&file_id=f_2275470926183"
    ),
    "mearec_tetrode_60s": (  # nightly / manual
        "https://ucsf.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=5i8e4dvjx2x1i4twc0rw09cj8dv6cjh3"
        "&file_id=f_2275379903075"
    ),
    # Cross-session matcher gate pair (generate-or-skip until uploaded). The
    # AUC gate references both stems by path and skips when either is absent.
    "mearec_polymer_128ch_2sessions_s1": None,  # nightly / manual
    "mearec_polymer_128ch_2sessions_s2": None,  # nightly / manual
}


class FixtureFetchError(RuntimeError):
    """A required fixture could not be downloaded or failed verification."""


def _manifest_nwb_sha256(name: str) -> str | None:
    """Return the committed ``nwb_sha256`` for ``name``, or None if unknown."""
    if not _MANIFEST.exists():
        return None
    fixtures = json.loads(_MANIFEST.read_text()).get("fixtures", {})
    return fixtures.get(name, {}).get("nwb_sha256")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_http(url: str, dest_tmp: Path) -> None:
    """Download an HTTP(S) URL to ``dest_tmp`` (follows redirects)."""
    # A User-Agent avoids hosts (incl. Box) that reject default urllib clients.
    request = urllib.request.Request(
        url, headers={"User-Agent": "spyglass-tests"}
    )
    with (
        urllib.request.urlopen(request) as response,
        dest_tmp.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)


def ensure_fixture(name: str, *, required: bool = False) -> Path | None:
    """Return the local path to fixture ``name``, downloading it if missing.

    Parameters
    ----------
    name : str
        Fixture stem, e.g. ``"mearec_polymer_smoke"`` (no ``.nwb``).
    required : bool, optional
        When True, raise :class:`FixtureFetchError` if the fixture is absent and
        cannot be downloaded (no URL configured). When False (default), return
        ``None`` instead so callers can ``pytest.skip`` -- matching the
        pre-step-2 generate-or-skip behaviour.

    Returns
    -------
    pathlib.Path or None
        Path to the local ``<name>.nwb``; ``None`` if absent and not required.

    Raises
    ------
    FixtureFetchError
        On a configured-but-failed download, a sha256 mismatch against the
        manifest, or an absent ``required`` fixture with no URL.
    """
    if name not in FIXTURE_URLS:
        raise FixtureFetchError(
            f"unknown fixture {name!r}; known: {sorted(FIXTURE_URLS)}"
        )

    dest = _THIS_DIR / f"{name}.nwb"
    if dest.exists():
        return dest

    url = FIXTURE_URLS.get(name)
    if not url:
        if required:
            raise FixtureFetchError(
                f"fixture {name!r} is absent and no download URL is configured "
                f"in {Path(__file__).name}. Either set FIXTURE_URLS[{name!r}] "
                "to the uploaded artifact, or generate it locally with "
                "generate_mearec.py."
            )
        return None

    expected = _manifest_nwb_sha256(name)
    tmp = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        tmp.unlink(missing_ok=True)
        try:
            _download_http(url, tmp)
            if expected:
                actual = _sha256(tmp)
                if actual != expected:
                    # Not silently tolerated: a mismatch means a corrupt
                    # transfer or a rotated/stale artifact. Retry (transient
                    # corruption) but never fall back to anything.
                    raise FixtureFetchError(
                        f"{name}: downloaded nwb_sha256 {actual} != manifest "
                        f"{expected}. Corrupt download, or the manifest/URL is "
                        "stale (re-upload and re-commit fixtures_manifest.json)."
                    )
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if 400 <= exc.code < 500:
                # Permanent: bad/rotated link or sharing disabled -- fail fast.
                raise FixtureFetchError(
                    f"{name}: HTTP {exc.code} for its Box URL (permanent). "
                    "Check the share is 'people with the link' + download "
                    "allowed, and that the link in _fetch.py is current."
                ) from exc
            last_err = exc  # 5xx -- retry
        except (urllib.error.URLError, TimeoutError, FixtureFetchError) as exc:
            tmp.unlink(missing_ok=True)
            last_err = exc  # network blip or hash mismatch -- retry
        if attempt < _RETRIES:
            time.sleep(_BACKOFF_S * attempt)

    raise FixtureFetchError(
        f"{name}: download failed after {_RETRIES} attempts: {last_err}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ensure each named fixture is present (download required)."""
    names = list(argv) if argv is not None else sys.argv[1:]
    if not names:
        names = ["mearec_polymer_smoke"]

    exit_code = 0
    for name in names:
        try:
            path = ensure_fixture(name, required=True)
            print(f"[_fetch] {name}: ready at {path}")
        except FixtureFetchError as exc:
            print(f"[_fetch] ERROR: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
