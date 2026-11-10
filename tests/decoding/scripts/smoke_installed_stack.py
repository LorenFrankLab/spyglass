"""Check that the installed jax/numpy/spikeinterface stack runs the detector.

Run by the package-build workflow against the installed wheel or sdist. The
resolver can install a combination that imports individually but not together.

With ``--without-spikesorting`` it checks an install without the
``spikesorting`` extra: those packages are absent, the non-sorting pipelines
import none of them at module level, and the detector still runs. Schema
modules are checked from source, not imported, so no database is needed.
"""

import argparse
import ast
import importlib
import importlib.util
import tempfile
from pathlib import Path

# Public non-sorting entry points, plus what ``insert_sessions`` imports.
NON_SORTING_ENTRY_POINTS = (
    "spyglass",
    "spyglass.settings",
    "spyglass.utils",
    "spyglass.sharing",
    "spyglass.common",
    "spyglass.lfp",
    "spyglass.lfp.v1",
    "spyglass.lfp.analysis.v1",
    "spyglass.position",
    "spyglass.position.v1",
    "spyglass.linearization",
    "spyglass.linearization.v0",
    "spyglass.linearization.v1",
    "spyglass.ripple",
    "spyglass.ripple.v1",
    "spyglass.spikesorting.imported",
)

# Entry points that import without a database connection.
DB_FREE_ENTRY_POINTS = (
    "spyglass",
    "spyglass.settings",
    "spyglass.utils",
    "spyglass.sharing",
)


def _module_file(name: str, root: Path) -> Path | None:
    """Return the source file of a ``spyglass`` module, or None."""
    path = root.parent.joinpath(*name.split("."))
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    return None


def _guards_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    """Whether a ``try`` block tolerates a missing module."""
    for handler in handlers:
        if handler.type is None:
            return True
        types = getattr(handler.type, "elts", [handler.type])
        names = {getattr(t, "id", None) for t in types}
        if names & {"ImportError", "ModuleNotFoundError", "Exception"}:
            return True
    return False


def _module_level_imports(path: Path, name: str) -> list[tuple[str, int]]:
    """Imports that run when the module is imported, as (module, line).

    Skips function bodies, ``if TYPE_CHECKING:`` blocks and ``try`` bodies
    that catch ``ImportError``. ``from a import b`` yields both ``a`` and
    ``a.b``, since ``b`` may be a submodule.
    """
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found = []

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.Import):
                found.extend((alias.name, node.lineno) for alias in node.names)
                continue
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parts = package.split(".")
                    if node.level > 1:
                        parts = parts[: 1 - node.level]
                    module = ".".join(parts + ([module] if module else []))
                found.append((module, node.lineno))
                found.extend(
                    (f"{module}.{alias.name}", node.lineno)
                    for alias in node.names
                )
                continue
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(
                node.test
            ):
                visit(node.orelse)
                continue
            if isinstance(node, ast.Try):
                if not _guards_import_error(node.handlers):
                    visit(node.body)
                    for handler in node.handlers:
                        visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
                continue
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(node, field, None) or [])

    visit(ast.parse(path.read_text()).body)
    return found


def find_extra_imports(
    entry_points: tuple[str, ...], blocked: frozenset[str]
) -> list[str]:
    """Module-level imports of ``blocked`` packages reachable from entries.

    Parameters
    ----------
    entry_points : tuple of str
        Dotted ``spyglass`` modules to start from.
    blocked : frozenset of str
        Top-level module names that must not be imported.

    Returns
    -------
    list of str
        ``"module:line imports name (via parent)"`` for each hit.
    """
    root = Path(importlib.util.find_spec("spyglass").origin).parent
    parents = {}
    hits = []
    stack = [(entry, None) for entry in entry_points]
    while stack:
        name, parent = stack.pop()
        parts = name.split(".")
        # Importing a.b.c first imports a and a.b.
        for i in range(1, len(parts) + 1):
            module = ".".join(parts[:i])
            path = _module_file(module, root)
            if module in parents or path is None:
                continue
            parents[module] = parent
            for imported, line in _module_level_imports(path, module):
                top = imported.split(".")[0]
                if top == "spyglass":
                    stack.append((imported, module))
                elif top in blocked:
                    hits.append(
                        f"{module}:{line} imports {imported} "
                        f"(via {parents[module]})"
                    )
    return sorted(set(hits))


def _is_installed(name: str) -> bool:
    """Whether a top-level module can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


def check_without_spikesorting() -> None:
    """Check the non-sorting pipelines without the ``spikesorting`` extra."""
    from spyglass.utils.dj_helper_fn import SPIKESORTING_EXTRA_MODULES

    installed = sorted(
        m for m in SPIKESORTING_EXTRA_MODULES if _is_installed(m)
    )
    assert not installed, f"spikesorting extra packages installed: {installed}"
    print("spikesorting extra absent:", sorted(SPIKESORTING_EXTRA_MODULES))

    for name in DB_FREE_ENTRY_POINTS:
        importlib.import_module(name)
    print("imported", ", ".join(DB_FREE_ENTRY_POINTS))

    hits = find_extra_imports(
        NON_SORTING_ENTRY_POINTS, SPIKESORTING_EXTRA_MODULES
    )
    assert not hits, "non-sorting modules need the extra:\n" + "\n".join(hits)
    print(f"{len(NON_SORTING_ENTRY_POINTS)} entry points import no extra")


def check_detector_stack(spikesorting: bool = True) -> None:
    """Fit, predict, save and reload the real detector."""
    import jax
    import numpy
    import scipy

    print("jax", jax.__version__)
    print("numpy", numpy.__version__)
    print("scipy", scipy.__version__)
    if spikesorting:
        import spikeinterface

        print("spikeinterface", spikeinterface.__version__)

    import non_local_detector
    from non_local_detector import ContFragClusterlessClassifier

    print("non_local_detector", non_local_detector.__version__)
    ContFragClusterlessClassifier()
    print("classifier construction OK")

    # Construction alone misses detector/JAX runtime incompatibilities.
    from non_local_detector import ContFragSortedSpikesClassifier
    from non_local_detector.environment import Environment

    t = numpy.arange(0, 10, 0.05)
    data = dict(
        position_time=t,
        position=(50 + 40 * numpy.sin(t / 3))[:, None],
        spike_times=[numpy.arange(0.5, 9.5, 0.7)],
    )
    model = ContFragSortedSpikesClassifier(
        sampling_frequency=20,
        environments=Environment(place_bin_size=10),
        sorted_spikes_algorithm_params={
            "position_std": 8.0,
            "block_size": 1000,
        },
    )
    model.fit(**data)
    result = model.predict(**data, time=t)
    with tempfile.TemporaryDirectory() as directory:
        results_path = Path(directory) / "results.nc"
        model_path = Path(directory) / "model.pkl"
        model.save_results(result, results_path)
        model.save_model(model_path)
        restored = model.load_model(model_path)
        numpy.testing.assert_allclose(
            restored.predict(**data, time=t).acausal_posterior.sum(
                "state_bins"
            ),
            1,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            model.load_results(results_path).acausal_posterior,
            result.acausal_posterior,
        )
    print("real detector fit/predict/save/reload OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--without-spikesorting",
        action="store_true",
        help="check an install without the spikesorting extra",
    )
    args = parser.parse_args()
    if args.without_spikesorting:
        check_without_spikesorting()
    check_detector_stack(spikesorting=not args.without_spikesorting)


if __name__ == "__main__":
    main()
