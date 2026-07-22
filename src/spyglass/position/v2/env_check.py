"""Pre-flight environment checks for the Position V2 pipeline.

The check logic uses only the standard library (package *metadata*, never real
imports), so it can never trigger the very import-time conflict it detects. Run
it early -- right after importing spyglass, before training a model -- to catch
a misconfigured environment before the heavy pose-tool imports crash on the GPU.

For a truly stand-alone check (e.g. an environment with no database configured),
run this file directly -- it imports nothing from spyglass, so it never opens a
database connection::

    python src/spyglass/position/v2/env_check.py
"""

import importlib.metadata as _md

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


def check_environment(raise_on_error: bool = False, verbose: bool = True):
    """Check for known Position V2 dependency conflicts.

    The most common issue affects users migrating from Position V1: a leftover
    TensorFlow install (V1's DeepLabCut backend) coexisting with V2's jax stack
    (pulled in by ``non_local_detector``). Both bundle XLA and collide on the
    GPU -- the tell-tale ``Unable to register cuDNN factory ... already
    registered`` -- and TensorFlow's ``ml-dtypes`` pin holds jax at an old
    version. DeepLabCut 3.x runs on PyTorch, so TensorFlow is not needed.

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
