"""Pre-flight environment checks for the Position V2 pipeline.

Most checks use only the standard library (package *metadata*, never real
imports of the heavy, GPU-touching pose tools), so they can never trigger the
very import-time conflict they detect. The one exception is the BLAS/LAPACK
self-test, which does import numpy/scipy for real -- both are lightweight
core dependencies, not part of the TensorFlow/jax GPU collision this module
otherwise avoids triggering. Run this early -- right after importing
spyglass, before training a model -- to catch a misconfigured environment
before the heavy pose-tool imports crash on the GPU.

For a truly stand-alone check (e.g. an environment with no database configured),
run this file directly -- it imports nothing from spyglass, so it never opens a
database connection::

    python src/spyglass/position/v2/env_check.py
"""

import importlib.metadata as _md
import importlib.util as _iu
import os
import shutil
import sys
from pathlib import Path

# Packages that make up the legacy TensorFlow DeepLabCut backend. Position V1's
# DeepLabCut used TensorFlow; V2's DeepLabCut 3.x uses PyTorch and does not need
# any of these. Listed here so the guidance can name exactly what to remove.
_TF_STACK = (
    "tensorflow",
    "tensorflow-estimator",
    "tensorflow-io-gcs-filesystem",
    "keras",
    "tf-keras",
    "tf-slim",
    "tensorpack",
)


def _installed(name: str):
    """Return the installed version of *name*, or None if absent.

    Uses package metadata only -- it does not import the package, so it is fast
    and cannot trigger the very import-time conflict we are checking for.
    """
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _blas_lapack_error():
    """Return a LAPACK error message from a trivial eigh call, or None.

    Returns None if numpy/scipy are missing (nothing to check) or if the
    call succeeds. A broken BLAS/LAPACK build (seen with a mismatched
    conda-forge ``liblapack`` netlib build + ``blas=openblas``, e.g. pulled
    in by a ``defaults``-channel ``numpy-base``) fails on a 1x1 matrix --
    the ``dsyevr`` LAPACK workspace-size query hits a ``liwork=1`` edge case
    -- surfacing as a cryptic ``_flapack.error`` deep inside an unrelated
    import chain (e.g. deeplabcut -> filterpy -> scipy.stats, which calls
    ``eigh`` on a 1x1 matrix as an import-time side effect). A 2x2 matrix
    does NOT reproduce this; the check must use a 1x1 matrix.
    """
    try:
        import numpy
        import scipy.linalg
    except ImportError:
        return None
    try:
        scipy.linalg.eigh(numpy.eye(1))
    except Exception as exc:
        return str(exc)
    return None


def _qt_plugin_dirs():
    """Candidate directories holding Qt platform plugins.

    Located without importing a Qt binding: ``find_spec`` on a top-level
    package reads its path without executing it, and conda keeps the plugins
    under the environment prefix.

    Returns
    -------
    list[Path]
        Existing directories, in no particular order.
    """
    roots = [
        Path(sys.prefix) / "plugins" / "platforms",
        Path(sys.prefix) / "lib" / "qt6" / "plugins" / "platforms",
        Path(sys.prefix) / "lib" / "qt5" / "plugins" / "platforms",
    ]
    for pkg, sub in (
        ("PyQt5", "Qt5"),
        ("PyQt6", "Qt6"),
        ("PySide2", "Qt"),
        ("PySide6", "Qt"),
    ):
        try:
            spec = _iu.find_spec(pkg)
        except (ImportError, ValueError):  # pragma: no cover
            continue
        for location in getattr(spec, "submodule_search_locations", None) or []:
            roots.append(Path(location) / sub / "plugins" / "platforms")
    return [r for r in roots if r.is_dir()]


def _qt_platform_problem():
    """Report a session whose Qt platform plugin is missing.

    DeepLabCut's frame-labelling GUI is a Qt application. Qt picks a platform
    plugin from the session type, and a Wayland session needs the
    ``qwayland`` plugin, which the Qt libraries do not carry on their own.
    Without it the GUI aborts before drawing.

    A headless environment is left alone: a compute node has no display and
    needs none, so the absence of one is not a misconfiguration. An explicit
    ``QT_QPA_PLATFORM`` is also honoured rather than second-guessed.

    Returns
    -------
    str or None
        The message, or None when a GUI can start or is not expected.
    """
    wayland = os.environ.get("WAYLAND_DISPLAY")
    display = os.environ.get("DISPLAY")
    requested = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()

    if not wayland and not display:
        return None  # headless
    if requested and requested != "wayland":
        return None  # a platform was named; Qt will use it
    if not wayland:
        return None  # X11 session; the xcb plugin ships with Qt

    searched = _qt_plugin_dirs()
    # The inner any() consumes each generator. Testing the generators
    # themselves is always true, so the check could never fail
    # wherever a plugin directory exists.
    if any(any(d.glob("libqwayland*.so")) for d in searched):
        return None

    where = "\n      ".join(str(d) for d in searched) or "(no plugin dir found)"
    return (
        "This is a Wayland session (WAYLAND_DISPLAY is set) and no Qt "
        "wayland platform plugin was found, so DeepLabCut's frame-labelling "
        "GUI will abort with 'Could not load the Qt platform plugin "
        '"wayland"\'.\n'
        "    Fix: run the GUI through XWayland, which needs no new package:\n"
        "      QT_QPA_PLATFORM=xcb python -c "
        '"from deeplabcut import label_frames; ..."\n'
        "    Or install the plugin for a native Wayland window:\n"
        "      conda install -c conda-forge qt-wayland\n"
        f"    Searched:\n      {where}"
    )


def check_environment(raise_on_error: bool = False, verbose: bool = True):
    """Check for known Position V2 dependency conflicts.

    The most common issue affects users migrating from Position V1: a leftover
    TensorFlow install (V1's DeepLabCut backend) coexisting with V2's jax stack
    (pulled in by ``non_local_detector``). Both bundle XLA and collide on the
    GPU -- the tell-tale ``Unable to register cuDNN factory ... already
    registered`` -- and TensorFlow's ``ml-dtypes`` pin holds jax at an old
    version. DeepLabCut 3.x runs on PyTorch, so TensorFlow is not needed.

    This also checks for the ``ffmpeg`` CLI binary: Position V2 shells out to
    it (video clipping, frame extraction/stitching, NWB re-encoding) rather
    than using a pip package, so it is easy to miss -- pip metadata never
    reflects whether the binary is on ``PATH``.

    It also runs a trivial ``scipy.linalg.eigh`` call to catch a broken
    BLAS/LAPACK build early. A bad ``libopenblas`` build otherwise surfaces
    as a cryptic LAPACK error deep inside an unrelated import (e.g.
    deeplabcut's ``filterpy`` dependency calls ``eigh`` as an import-time
    side effect), which looks like a DeepLabCut or filterpy bug but is not.

    Parameters
    ----------
    raise_on_error : bool, optional
        Raise ``RuntimeError`` if any problem is found, by default False.
    verbose : bool, optional
        Print a human-readable report, by default True.

    Returns
    -------
    list[str]
        One message per detected problem; empty when the environment is clean.
    """
    problems = []

    tf_version = _installed("tensorflow")
    jax_version = _installed("jax")
    if tf_version and jax_version:
        present = [p for p in _TF_STACK if _installed(p)]
        problems.append(
            f"TensorFlow ({tf_version}) is installed alongside jax "
            f"({jax_version}). This is almost always a leftover from a "
            "Position V1 environment: V1's DeepLabCut used the TensorFlow "
            "backend, but V2's DeepLabCut 3.x uses PyTorch and does not need "
            "TensorFlow. The two XLA runtimes collide on the GPU and "
            "TensorFlow's ml-dtypes pin holds jax at an old version.\n"
            "    Fix -- remove the TensorFlow stack from this environment:\n"
            f"      pip uninstall -y {' '.join(present)}\n"
            "    Or build a clean environment from "
            "environments/environment_dlc.yml."
        )

    # PyTorch is the V2 backend for both DeepLabCut 3.x and SLEAP (sleap-nn).
    if (_installed("deeplabcut") or _installed("sleap-nn")) and not _installed(
        "torch"
    ):
        problems.append(
            "A pose tool (DeepLabCut/SLEAP) is installed but PyTorch (torch) "
            "is not. Position V2 uses the PyTorch backend for both. Install "
            "from environments/environment_dlc.yml or environment_sleap.yml."
        )

    if not shutil.which("ffmpeg"):
        problems.append(
            "The ffmpeg CLI binary was not found on PATH. Position V2 calls "
            "it via subprocess for video clipping/stitching and NWB "
            "re-encoding. It is not a pip package.\n"
            "    Fix -- install it via conda (already listed in "
            "environments/environment_dlc.yml and environment_sleap.yml):\n"
            "      conda install -c conda-forge ffmpeg"
        )

    qt_problem = _qt_platform_problem()
    if qt_problem:
        problems.append(qt_problem)

    blas_error = _blas_lapack_error()
    if blas_error:
        problems.append(
            "scipy.linalg.eigh failed on a trivial 1x1 matrix -- this "
            f"environment's BLAS/LAPACK build is broken ({blas_error}). "
            "This crashes any import that triggers it transitively (e.g. "
            "deeplabcut -> filterpy -> scipy.stats calls eigh at import "
            "time), often deep in an unrelated traceback.\n"
            "    Most often caused by a `defaults`-channel numpy install: "
            "its `numpy-base` companion package hard-depends on "
            "`blas * openblas`, which combines with conda-forge's "
            "`liblapack` into a mismatched, broken netlib/openblas build. "
            "Pinning a specific libopenblas version is NOT a reliable fix -- "
            "the mismatch can recur on other point releases. Fix -- drop "
            "the `defaults` channel and reinstall numpy/scipy from "
            "conda-forge only (see environments/environment_dlc.yml):\n"
            "      conda install -c conda-forge --override-channels "
            '"numpy" "scipy"\n'
            "    Or force the MKL backend explicitly:\n"
            '      conda install -c conda-forge "blas=*=*mkl"'
        )

    if verbose:
        if problems:
            print(
                f"Position V2 environment check found {len(problems)} "
                "issue(s):\n"
            )
            for i, problem in enumerate(problems, 1):
                print(f"  {i}. {problem}\n")
        else:
            print(
                "Position V2 environment check: OK "
                "(no known dependency conflicts)."
            )

    if raise_on_error and problems:
        raise RuntimeError(" ".join(problems))

    return problems


if __name__ == "__main__":
    import sys

    sys.exit(1 if check_environment() else 0)
