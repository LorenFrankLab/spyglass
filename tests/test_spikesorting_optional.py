"""The non-sorting pipelines work without the ``spikesorting`` extra.

The extra is simulated as missing with a ``sys.meta_path`` finder. Schema
modules are checked from source, so no test here needs a database.
"""

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "tests/decoding/scripts/smoke_installed_stack.py"


def _extra_packages() -> set[str]:
    """Distribution names in the ``spikesorting`` extra."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    specs = project["project"]["optional-dependencies"]["spikesorting"]
    return {Requirement(spec).name.lower() for spec in specs}


class _FailImport:
    """Meta path finder: importing ``module`` raises for ``missing``."""

    def __init__(self, module, missing):
        self.module = module
        self.missing = missing

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.module:
            raise ModuleNotFoundError(
                f"No module named {self.missing!r}", name=self.missing
            )
        return None


def test_extra_modules_match_pyproject():
    """The modules spyglass treats as the extra are the extra's packages."""
    from spyglass.utils.dj_helper_fn import SPIKESORTING_EXTRA_MODULES

    assert SPIKESORTING_EXTRA_MODULES == _extra_packages()


def test_non_sorting_pipelines_import_without_extra():
    """Core, LFP, position, linearization and ripple need no sorting package.

    Runs the package-build smoke check in a fresh interpreter, so modules this
    session already imported cannot mask a module-level import.
    """
    code = textwrap.dedent(f"""
        import runpy, sys

        class HideExtra:
            def find_spec(self, fullname, path=None, target=None):
                top = fullname.split(".")[0]
                if top in {sorted(_extra_packages())!r}:
                    raise ModuleNotFoundError(f"No module named {{top!r}}", name=top)

        sys.meta_path.insert(0, HideExtra())
        runpy.run_path({str(SMOKE_SCRIPT)!r})["check_without_spikesorting"]()
        """)
    # Unresolvable host: a schema import would fail instead of connecting.
    env = {**os.environ, "DJ_HOST": "no-db.invalid"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "spikesorting extra absent" in result.stdout
    assert "entry points import no extra" in result.stdout


def test_find_extra_imports_reports_sorting_modules():
    """The smoke check finds a module-level import it is meant to reject."""
    import runpy

    find = runpy.run_path(str(SMOKE_SCRIPT))["find_extra_imports"]
    hits = find(("spyglass.mua.v1",), frozenset({"sortingview"}))
    assert any(hit.startswith("spyglass.mua.v1.mua:") for hit in hits), hits


@pytest.fixture
def stub_core_merge_tables(monkeypatch):
    """Replace the LFP and position merge modules, which need a database."""
    for name, table in (
        ("spyglass.lfp.lfp_merge", "LFPOutput"),
        ("spyglass.position.position_merge", "PositionOutput"),
    ):
        module = types.ModuleType(name)
        setattr(module, table, table)
        monkeypatch.setitem(sys.modules, name, module)
    for name in list(sys.modules):
        if name.startswith(
            ("spyglass.decoding", "spyglass.spikesorting.spikesorting_merge")
        ):
            monkeypatch.delitem(sys.modules, name)


def test_declare_merge_tables_skips_sorting_without_extra(
    stub_core_merge_tables, monkeypatch
):
    """Session ingestion declares the core merge tables without the extra."""
    from spyglass.utils.dj_helper_fn import declare_all_merge_tables

    finder = _FailImport("spyglass.decoding", missing="spikeinterface")
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    assert declare_all_merge_tables() == ("LFPOutput", "PositionOutput")


def test_declare_merge_tables_raises_other_missing_modules(
    stub_core_merge_tables, monkeypatch
):
    """A missing module outside the extra is a bug, not a missing extra."""
    from spyglass.utils.dj_helper_fn import declare_all_merge_tables

    finder = _FailImport("spyglass.decoding", "spyglass.decoding.gone")
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    with pytest.raises(ModuleNotFoundError, match="spyglass.decoding.gone"):
        declare_all_merge_tables()
