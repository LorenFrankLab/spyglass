"""Static checks that pyproject, conda env files and the ``pytest-legacy`` CI
job declare consistent dependency pins (no DB, no SpikeInterface import).

- numpy floors at 1.26 in base; the v2 extras pin ``>=2,<3``; the v2 env's SI
  spec matches the pyproject pin.
- The legacy lane sed-rewrites the committed SI/probeinterface strings, so
  the sed patterns must match them.
- Every conda env declares SciPy >=1.13, except the test-only legacy env.
"""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
V2_ENV = REPO_ROOT / "environments" / "environment_spikesorting_v2.yml"
LEGACY_ENV = REPO_ROOT / "environments" / "environment_spikesorting_legacy.yml"
CONDA_CI = REPO_ROOT / ".github" / "workflows" / "test-conda.yml"
ENVIRONMENTS = REPO_ROOT / "environments"

# The pins the legacy (SI-0.99) recipes rewrite, as literal sed source text
# and its legacy replacement.
LEGACY_SED_REWRITES = (
    ('"spikeinterface==0.104.3"', '"spikeinterface>=0.99.1,<0.100"'),
    ('"probeinterface>=0.3.2"', '"probeinterface>=0.2.19,<0.3"'),
)

# A sed expression whose source pattern mentions numpy.
NUMPY_SED = re.compile(r"-e\s+'s/[^']*numpy")


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


def test_v2_extra_installs_native_mountainsort4():
    """The supported v2 installation includes the real MS4 algorithm."""
    requirements = _extra_requirements("spikesorting-v2")
    assert str(requirements["mountainsort4"].specifier) == "==1.0.7"
    assert "ml-ms4alg" not in requirements
    assert "ml_ms4alg" not in requirements


def test_v2_runtime_extras_declare_reentrant_analyzer_locks():
    """Standalone extensions need the same FileLock singleton behavior."""
    for extra in (
        "spikesorting-v2",
        "spikesorting-v2-matching",
        "spikesorting-v2-curation",
    ):
        requirements = _extra_requirements(extra)
        filelock = requirements["filelock"].specifier
        assert filelock.contains("3.15.0"), extra
        assert not filelock.contains("3.14.0"), extra


def test_base_numpy_floor_allows_numpy_1x():
    """The base numpy floor admits 1.x (DeepLabCut, keypoint-moseq 0.6) and
    the upper bound excludes numpy 3."""
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
    resolves on; the base requirement only floors at 1.26."""
    for extra in ("spikesorting-v2", "spikesorting-v2-matching"):
        reqs = _extra_requirements(extra)
        assert "numpy" in reqs, f"the {extra} extra does not declare numpy"
        specs = {(s.operator, s.version) for s in reqs["numpy"].specifier}
        assert specs == {(">=", "2"), ("<", "3")}, (
            f"the {extra} extra must pin numpy>=2,<3; found "
            f"{str(reqs['numpy'].specifier)!r}"
        )


def test_scipy_declared_in_base():
    """scipy is a direct dependency with the modern >=1.13 floor: modules
    outside the v2 stack import it at top level, so it may not arrive only
    as a transitive pull from some other package."""
    reqs = _base_requirements()
    assert "scipy" in reqs, "scipy missing from base dependencies"
    scipy = reqs["scipy"].specifier
    assert scipy.contains("1.13.0") and not scipy.contains("1.12.0"), (
        f"scipy is pinned {str(scipy)!r}; the base floor must be >=1.13 to "
        "match the conda environments."
    )


def test_spikesorting_extra_owns_sorting_dependencies():
    """The SpikeInterface stack lives in the ``spikesorting`` extra, not the
    base install, and every v2 runtime extra pulls that extra in."""
    sorting = _extra_requirements("spikesorting")
    assert not set(sorting) & set(_base_requirements()), (
        "base dependencies repeat spikesorting-extra packages: "
        f"{sorted(set(sorting) & set(_base_requirements()))}"
    )
    for extra in (
        "spikesorting-v2",
        "spikesorting-v2-matching",
        "spikesorting-v2-curation",
    ):
        requirements = _extra_requirements(extra)
        assert "spikesorting" in requirements["spyglass-neuro"].extras, extra


def test_spikeinterface_hard_pin_matches_v2_env():
    """SpikeInterface is hard-pinned in pyproject and the v2 conda env spec
    matches that pin (not a looser range that could drift)."""
    reqs = _extra_requirements("spikesorting")

    assert "spikeinterface" in reqs, "spikeinterface missing from the extra"
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


def test_legacy_sed_targets_present():
    """The legacy CI recipe rewrites exactly the SpikeInterface and
    probeinterface strings pyproject commits, and nothing about numpy (the
    legacy env file pins numpy<2 itself)."""
    pyproject = PYPROJECT.read_text()
    recipes = {CONDA_CI: CONDA_CI.read_text()}

    for source, legacy in LEGACY_SED_REWRITES:
        assert source in pyproject, (
            f"pyproject does not declare {source}, so the legacy sed "
            "source pattern silently no-ops and the env resolves the v2 line."
        )
        fragment = f"'s/{source}/{legacy}/'"
        for path, contents in recipes.items():
            assert (
                fragment in contents
            ), f"{path} has no sed rewriting {source} -> {legacy}."

    for path, contents in recipes.items():
        assert not NUMPY_SED.search(contents), (
            f"{path} still sed-rewrites the numpy pin. The legacy lane takes "
            "numpy<2 from environment_spikesorting_legacy.yml instead."
        )


def test_all_conda_envs_are_modern_scipy_or_document_legacy_install():
    """Every environment makes its SciPy/runtime generation explicit.

    Modern envs need SciPy >=1.13. Only the test-only SI-0.99 legacy env may
    skip it, and it must pin ``numpy<2`` in a dependency list.
    """
    modern_scipy = re.compile(
        r'^\s*-\s*["\']?scipy>=1\.13(?:["\']|\s|$)', re.MULTILINE
    )
    # A real conda or pip dependency list item, not prose that names the pin.
    legacy_numpy_pin = re.compile(r'^\s*-\s*["\']?numpy<2', re.MULTILINE)

    for path in sorted(ENVIRONMENTS.glob("*.yml")):
        contents = path.read_text()
        is_modern = bool(modern_scipy.search(contents))
        is_legacy = path == LEGACY_ENV and bool(
            legacy_numpy_pin.search(contents)
        )
        assert is_modern or is_legacy, (
            f"{path} must pin scipy>=1.13; only {LEGACY_ENV.name} may skip it "
            "and it must carry a real `- numpy<2` dependency entry"
        )


def test_pyproject_carries_no_si099_dependency_caps():
    """pyproject may not cap a dependency below what its own numpy pin needs.

    A ``scipy<1.13`` or ``jax<0.7.2`` cap (SI-0.99 bounds) conflicts with
    ``numpy>=2`` and makes the resolver fall back to old releases silently.
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


def test_ripple_detection_floor_covers_normalization_api():
    """The ripple-detection floor admits only releases carrying the
    normalization API ``spyglass.mua`` imports.

    ``src/spyglass/mua/v1/_detection.py`` imports ``normalize_signal`` and
    ``_validate_normalization_params`` from ``ripple_detection.core`` and
    ``_get_event_stats`` from ``ripple_detection.detectors``; the 1.6 line
    lacks them.
    """
    from ripple_detection import core, detectors

    specifier = _base_requirements()["ripple-detection"].specifier
    assert specifier.contains("1.7.0"), (
        f"ripple-detection is pinned {str(specifier)!r}; it must admit 1.7, "
        "the first release carrying the multiunit normalization API."
    )
    assert not specifier.contains("1.6.0"), (
        f"ripple-detection is pinned {str(specifier)!r}; 1.6 lacks "
        "core.normalize_signal and the detector's normalization parameters, "
        "so spyglass.mua fails to import against it."
    )

    for module, name in (
        (core, "normalize_signal"),
        (core, "_validate_normalization_params"),
        (detectors, "_get_event_stats"),
    ):
        assert hasattr(module, name), (
            f"ripple_detection.{module.__name__.split('.')[-1]} has no "
            f"{name!r}, which spyglass/mua/v1/_detection.py imports. Move "
            "the import and the pyproject floor together."
        )
