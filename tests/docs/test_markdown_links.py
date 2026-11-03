"""Guard tests for markdown links on the rendered documentation site.

``mkdocs`` resolves a relative link against the page's location *inside*
``docs/src``, not against the file's location in the repository. Several
repo-root files are published by way of committed symlinks -- for example
``QUICKSTART.md`` is served as ``GettingStarted/QUICKSTART.md`` -- so a
link that resolves on GitHub can still 404 on the site (see issue #944).

These tests walk the markdown that mkdocs publishes, resolve each relative
link the way mkdocs would, and assert the target exists. A companion test
re-checks ``QUICKSTART.md`` from the repository root, because that file is
rendered from two places at once and a path that satisfies one rendering
can break the other. Only a site-absolute URL satisfies both.
"""

import posixpath
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
DOCS_SRC = REPO / "docs" / "src"
NB_DIR = REPO / "notebooks"
PKG_DIR = REPO / "src" / "spyglass"
QUICKSTART = REPO / "QUICKSTART.md"

SITE_URL = "https://lorenfranklab.github.io/spyglass"

# Fenced code blocks and inline code spans hold shell text, not links.
FENCE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
CODE_SPAN_RE = re.compile(r"`[^`\n]*`")
# Inline markdown link, e.g. `[Concepts](../notebooks/01_Concepts.ipynb)`
LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?(?P<target>[^\s)>]+)>?[^)]*\)")
# Tutorial page on the published site, e.g. `.../notebooks/01_Concepts/`
SITE_NB_RE = re.compile(
    rf"{re.escape(SITE_URL)}/[^/]+/notebooks/(?P<slug>[^/#?]+)/"
)

# Broken today, but owned elsewhere -- do not fail this suite on them.
KNOWN_BROKEN = {
    # `CustomAnalysisFiles.md` has never been written. The same dead path
    # is advertised to users by `src/spyglass/utils/dj_mixin.py`; both are
    # tracked separately from the QUICKSTART fix.
    ("ForDevelopers/Management.md", "./CustomAnalysisFiles.md"),
}

CHECKED_SUFFIXES = (".md", ".ipynb")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "//")


def _prose(path):
    """Return file text with code blocks and code spans removed.

    Parameters
    ----------
    path : pathlib.Path
        Markdown file to read.

    Returns
    -------
    str
        The markdown text, minus anything a reader cannot click.
    """
    text = path.read_text(encoding="utf-8")
    text = FENCE_RE.sub("", text)
    return CODE_SPAN_RE.sub("", text)


def _relative_targets(text):
    """Yield relative link targets worth resolving on disk.

    Absolute URLs, bare anchors and mailto links are not ours to check,
    and only ``.md``/``.ipynb`` targets name a documentation page.

    Parameters
    ----------
    text : str
        Markdown prose, already stripped of code.

    Yields
    ------
    str
        A link target with any ``#anchor`` or ``?query`` removed.
    """
    for match in LINK_RE.finditer(text):
        target = match.group("target")
        if target.startswith(SKIP_PREFIXES) or "://" in target:
            continue
        path = re.split(r"[#?]", target, maxsplit=1)[0]
        if path and path.endswith(CHECKED_SUFFIXES):
            yield path


def _mkdocs_pages():
    """Return every markdown page mkdocs publishes, as docs-dir URIs.

    Symlinked pages are reported at their location *in the docs tree*,
    which is the location mkdocs resolves their links against.

    Returns
    -------
    list of str
        Sorted ``docs/src``-relative POSIX paths.
    """
    return sorted(
        p.relative_to(DOCS_SRC).as_posix() for p in DOCS_SRC.rglob("*.md")
    )


def _api_page_exists(uri):
    """Return whether ``docs/src/api/make_pages.py`` will emit ``uri``.

    The api tree is written at build time by the ``gen-files`` plugin --
    one page per module under ``src/spyglass`` -- so those targets never
    exist on disk, yet mkdocs resolves them happily.

    Parameters
    ----------
    uri : str
        Docs-dir URI beginning with ``api/``.

    Returns
    -------
    bool
        True if the generator produces a page at that URI.
    """
    tail = uri[len("api/") :]
    if tail in ("index.md", "navigation.md"):  # hand-written / nav file
        return True
    return (PKG_DIR / tail).with_suffix(".py").exists()


def _mkdocs_broken(page_uri):
    """Return unresolved links for one published page.

    Parameters
    ----------
    page_uri : str
        Page path relative to ``docs/src``, e.g.
        ``GettingStarted/QUICKSTART.md``.

    Returns
    -------
    list of tuple
        One ``(page_uri, target)`` per link that mkdocs cannot resolve.
    """
    broken = []
    for target in _relative_targets(_prose(DOCS_SRC / page_uri)):
        if (page_uri, target) in KNOWN_BROKEN:
            continue
        # Normalize lexically, as mkdocs does on the page's src_uri --
        # not via `Path.resolve`, whose `..` handling differs inside
        # symlinked directories.
        uri = posixpath.normpath(
            posixpath.join(posixpath.dirname(page_uri), target)
        )
        if uri.startswith("api/"):
            if not _api_page_exists(uri):
                broken.append((page_uri, target))
        elif uri.startswith("..") or not (DOCS_SRC / uri).exists():
            broken.append((page_uri, target))
    return broken


def test_quickstart_is_published_via_symlink():
    """QUICKSTART is served from `GettingStarted/`, not the docs root."""
    published = DOCS_SRC / "GettingStarted" / "QUICKSTART.md"
    assert published.is_symlink(), f"Not a symlink: {published}"
    assert published.resolve() == QUICKSTART.resolve()


def test_mkdocs_links_resolve():
    """Every relative page link resolves from its docs-tree location."""
    broken = [item for uri in _mkdocs_pages() for item in _mkdocs_broken(uri)]
    if broken:
        listing = "\n".join(f"  {uri}: {tgt}" for uri, tgt in sorted(broken))
        n_pages = len({uri for uri, _ in broken})
        pytest.fail(
            f"{len(broken)} link(s) broken on the docs site, "
            f"in {n_pages} page(s):\n" + listing
        )


def test_quickstart_links_resolve_from_repo_root():
    """The same links also resolve for GitHub's repo-root rendering.

    ``QUICKSTART.md`` is rendered both at the repository root and, via
    symlink, from ``docs/src/GettingStarted/``. Prefixing a link with
    ``../`` fixes the site while breaking GitHub, so this test pins the
    other half of that trade-off.
    """
    broken = []
    for target in _relative_targets(_prose(QUICKSTART)):
        resolved = posixpath.normpath(target)
        if resolved.startswith("..") or not (REPO / resolved).exists():
            broken.append(target)
    assert not broken, (
        "QUICKSTART.md link(s) unresolvable from the repo root: "
        f"{sorted(broken)}"
    )


@pytest.mark.parametrize("page", _mkdocs_pages())
def test_site_notebook_urls_name_real_notebooks(page):
    """Absolute tutorial URLs point at a notebook that still exists.

    Renumbering a notebook silently invalidates hard-coded
    ``lorenfranklab.github.io/.../notebooks/<name>/`` URLs, which no
    relative-path check would catch.
    """
    stale = []
    for match in SITE_NB_RE.finditer(_prose(DOCS_SRC / page)):
        slug = match.group("slug")
        if slug == "index":  # tutorial overview page, not a notebook
            continue
        if not (NB_DIR / f"{slug}.ipynb").exists():
            stale.append(slug)
    assert not stale, f"{page} links to missing notebook(s): {sorted(stale)}"
