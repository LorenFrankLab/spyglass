"""Static dependency-contract checks for the v2 stack.

These are pure-text checks (no DB, no SpikeInterface import): they parse
``pyproject.toml``, every conda environment file, and the ``pytest-legacy`` CI
job and assert the declared pins agree with each other.

Three coupled invariants live here:

1. The base ``numpy`` requirement floors at 1.26 so the numpy<2 pipelines
   still install, while the ``spikesorting-v2`` and
   ``spikesorting-v2-matching`` extras carry the ``>=2,<3`` baseline the SI
   0.104 stack needs, and the v2 conda env's SpikeInterface spec matches the
   ``pyproject`` hard pin. A bare ``numpy`` or a drifting SI range silently
   shifts the resolved stack.
2. The legacy (v0/v1) lane resolves the SI-0.99 stack by **sed-rewriting**
   the committed SpikeInterface and probeinterface dependency strings in an
   ephemeral checkout (``.github/workflows/test-conda.yml`` ``pytest-legacy``
   job and the local ``environment_spikesorting_legacy.yml`` doc). If that
   committed pin text changes, those seds silently no-op and the legacy env
   resolves the 0.104 line instead, so the sed source patterns must keep
   matching the current committed strings.
3. Every conda environment either declares the modern SciPy >=1.13 contract
   or documents the complete legacy pre-install pin-relaxation recipe.
"""

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
V2_ENV = REPO_ROOT / "environments" / "environment_spikesorting_v2.yml"
LEGACY_ENV = REPO_ROOT / "environments" / "environment_spikesorting_legacy.yml"
CONDA_CI = REPO_ROOT / ".github" / "workflows" / "test-conda.yml"
ENVIRONMENTS = REPO_ROOT / "environments"


def _raw_dependencies() -> list[str]:
    """The literal ``[project].dependencies`` strings, as written."""
    return tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]


def _base_requirements() -> dict[str, Requirement]:
    """Parse ``[project].dependencies`` into ``{name: Requirement}``."""
    reqs = {}
    for spec in _raw_dependencies():
        req = Requirement(spec)
        reqs[req.name.lower()] = req
    return reqs


def _raw_dependency(name: str) -> str:
    """The literal dependency string for ``name`` (e.g. ``"numpy>=2,<3"``).

    Uses the verbatim text rather than ``str(Requirement(...))`` because
    ``packaging`` canonicalizes/reorders specifiers (``numpy>=2,<3`` ->
    ``numpy<3,>=2``), which would not match the literal sed source text.
    """
    for spec in _raw_dependencies():
        if Requirement(spec).name.lower() == name.lower():
            return spec
    raise AssertionError(f"{name} not found in base dependencies")


def _extra_requirements(extra: str) -> dict[str, Requirement]:
    """Parse one ``[project.optional-dependencies]`` group."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    specs = project["optional-dependencies"][extra]
    return {Requirement(s).name.lower(): Requirement(s) for s in specs}


def _env_spikeinterface_spec(env_file: Path) -> str:
    """Return the SpikeInterface spec the env file pins in its pip section."""
    for line in env_file.read_text().splitlines():
        stripped = line.strip().lstrip("-").strip().strip('"')
        if stripped.lower().startswith("spikeinterface"):
            return stripped
    raise AssertionError(f"no spikeinterface line found in {env_file}")


def test_base_numpy_floor_allows_numpy_1x():
    """The base numpy floor admits the 1.x line the numpy<2 pipelines
    (DeepLabCut 3.x, keypoint-moseq 0.6) need, while keeping the upper bound
    that holds the resolver off a future numpy 3."""
    numpy = _base_requirements()["numpy"].specifier
    assert numpy.contains("1.26.4"), (
        f"base numpy is pinned {str(numpy)!r}; it must admit numpy 1.x so "
        "deeplabcut / keypoint-moseq can resolve."
    )
    assert numpy.contains("2.4.0"), (
        f"base numpy is pinned {str(numpy)!r}; it must still admit numpy 2 "
        "for the v2 SpikeInterface stack."
    )
    assert not numpy.contains("3.0.0"), (
        f"base numpy is pinned {str(numpy)!r}; the <3 cap must stay. A bare "
        "or uncapped numpy lets the resolver float onto a future major line."
    )


def test_v2_extras_pin_numpy_2():
    """The v2 extras carry the numpy-2 baseline the SI 0.104 + torch stack
    resolves on, now that the base requirement no longer does."""
    for extra in ("spikesorting-v2", "spikesorting-v2-matching"):
        reqs = _extra_requirements(extra)
        assert "numpy" in reqs, f"the {extra} extra does not declare numpy"
        specs = {(s.operator, s.version) for s in reqs["numpy"].specifier}
        assert specs == {(">=", "2"), ("<", "3")}, (
            f"the {extra} extra must pin numpy>=2,<3; found "
            f"{str(reqs['numpy'].specifier)!r}"
        )


def test_spikeinterface_hard_pin_matches_v2_env():
    """SpikeInterface is hard-pinned in pyproject and the v2 conda env spec
    matches that pin (not a looser range that could drift)."""
    reqs = _base_requirements()

    assert "spikeinterface" in reqs, "spikeinterface missing from base deps"
    si_specifiers = list(reqs["spikeinterface"].specifier)
    assert len(si_specifiers) == 1 and si_specifiers[0].operator == "==", (
        f"spikeinterface must be hard-pinned (==X.Y.Z); found "
        f"{str(reqs['spikeinterface'].specifier)!r}"
    )
    pinned_version = si_specifiers[0].version

    env_si = _env_spikeinterface_spec(V2_ENV)
    env_specifiers = list(Requirement(env_si).specifier)
    assert (
        len(env_specifiers) == 1
        and env_specifiers[0].operator == "=="
        and env_specifiers[0].version == pinned_version
    ), (
        f"environment_spikesorting_v2.yml pins SpikeInterface as {env_si!r} "
        f"but pyproject hard-pins =={pinned_version}. Make the env match the "
        f"hard pin (or document the intentional divergence in one place)."
    )


def test_all_conda_envs_are_modern_scipy_or_document_legacy_install():
    """Every environment makes its SciPy/runtime generation explicit.

    Modern Spyglass installs need SciPy >=1.13 alongside the NumPy 2 / SI 0.104
    package contract. The one deliberate exception is the SI-0.99 legacy
    environment, whose header must carry the exact pre-install pin-relaxation
    recipe; a stale ``scipy<1.13`` line without that recipe is not sufficient.
    """
    modern_scipy = re.compile(
        r'^\s*-\s*["\']?scipy>=1\.13(?:["\']|\s|$)', re.MULTILINE
    )
    legacy_markers = (
        "sed -i",
        "spikeinterface>=0.99.1,<0.100",
        "numpy>=1.23,<2",
    )

    for path in sorted(ENVIRONMENTS.glob("*.yml")):
        contents = path.read_text()
        is_modern = bool(modern_scipy.search(contents))
        documents_legacy_install = all(
            marker in contents for marker in legacy_markers
        )
        assert is_modern or documents_legacy_install, (
            f"{path} must pin scipy>=1.13 or document the complete legacy "
            "SI-0.99 pre-install sed recipe in its header"
        )


def test_pyproject_carries_no_si099_dependency_caps():
    """pyproject may not cap a dependency below what its own numpy pin needs.

    The ``spikesorting-v2`` extra pins ``numpy>=2`` for the SI 0.104 stack,
    and the conda environments pin ``scipy>=1.13`` to match. A ``scipy<1.13``
    or ``jax<0.7.2`` cap in the base dependencies (both SI-0.99 compatibility
    bounds) contradicts that: every scipy release capping numpy below 2 is
    excluded, so the resolver falls back to pre-numpy-2 ancients instead of
    failing loudly. The conda-env contract test above cannot catch it -- it
    reads only ``environments/*.yml``, so pyproject can disagree with every
    one of them while that test stays green.
    """
    requirements = _base_requirements()
    numpy = _extra_requirements("spikesorting-v2")["numpy"].specifier
    numpy_spec = str(numpy)
    assert numpy.contains("2.0.0") and not numpy.contains("1.26.0"), (
        f"the spikesorting-v2 extra pins numpy {numpy_spec!r}; this guard "
        "assumes the numpy-2 line and must be revisited if the v2 baseline "
        "moves."
    )
    for package, cap in (("scipy", "<1.13"), ("jax", "<0.7.2")):
        req = requirements.get(package)
        spec = str(req.specifier) if req else ""
        assert cap not in spec, (
            f"pyproject pins {package}{spec}, an SI-0.99 compatibility cap "
            f"that cannot co-resolve with numpy{numpy_spec}. Drop the cap, or "
            "move the whole package back to the SI-0.99 baseline."
        )
