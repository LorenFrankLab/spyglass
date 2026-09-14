"""Guard tests for cross-references between tutorial notebooks.

Notebooks under ``notebooks/`` and their paired jupytext scripts under
``notebooks/py_scripts/`` link to one another with markdown links whose
target ends in ``.ipynb``. Renumbering a notebook silently breaks those
links on the rendered documentation site (see issue #944). These tests
walk every notebook and paired script, and assert each link target
exists.
"""

import re
from pathlib import Path

import pytest

NB_DIR = Path(__file__).parents[2] / "notebooks"
PY_DIR = NB_DIR / "py_scripts"

# Markdown link whose target ends in `.ipynb`, e.g. `[text](./00_Setup.ipynb)`
LINK_RE = re.compile(r"\]\((?P<target>[^)\s]+\.ipynb)\)")


def _source_files():
    """Yield every notebook and paired jupytext script, sorted by name."""
    return sorted(NB_DIR.glob("*.ipynb")) + sorted(PY_DIR.glob("*.py"))


def _broken_links(path):
    """Return list of (source, target) for links that do not resolve.

    Parameters
    ----------
    path : pathlib.Path
        Notebook or jupytext script to scan.

    Returns
    -------
    list of tuple
        One ``(relative_source_path, target)`` per unresolved link.
    """
    text = path.read_text(encoding="utf-8")
    broken = []
    for match in LINK_RE.finditer(text):
        target = match.group("target")
        if "://" in target:  # external link, not ours to validate
            continue
        # All links are authored relative to the `notebooks/` directory,
        # including those in the paired `py_scripts/` twins.
        resolved = (NB_DIR / target).resolve()
        if not resolved.is_file():
            broken.append((path.name, target))
    return broken


def test_notebook_dirs_exist():
    """The notebook and paired-script directories are where we expect."""
    assert NB_DIR.is_dir(), f"Missing notebook dir: {NB_DIR}"
    assert PY_DIR.is_dir(), f"Missing py_scripts dir: {PY_DIR}"


def test_notebook_links_resolve():
    """Every `.ipynb` markdown link resolves to a file on disk."""
    broken = [item for path in _source_files() for item in _broken_links(path)]
    if broken:
        listing = "\n".join(f"  {src}: {tgt}" for src, tgt in sorted(broken))
        n_files = len({src for src, _ in broken})
        pytest.fail(
            f"{len(broken)} broken notebook link(s) in {n_files} file(s):\n"
            + listing
        )


@pytest.mark.parametrize(
    "nb", sorted(NB_DIR.glob("*.ipynb")), ids=lambda p: p.name
)
def test_paired_script_links_match(nb):
    """A notebook and its `py_scripts` twin reference the same targets."""
    twin = PY_DIR / f"{nb.stem}.py"
    if not twin.is_file():
        pytest.skip(f"No paired script for {nb.name}")
    nb_links = sorted(LINK_RE.findall(nb.read_text(encoding="utf-8")))
    py_links = sorted(LINK_RE.findall(twin.read_text(encoding="utf-8")))
    assert nb_links == py_links, (
        f"{nb.name} and {twin.name} disagree on notebook links:\n"
        f"  {nb.name}: {nb_links}\n"
        f"  {twin.name}: {py_links}"
    )
