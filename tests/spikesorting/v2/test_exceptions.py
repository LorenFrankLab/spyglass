"""Every v2 exception shares one base class. DB-free."""

import ast
from importlib import import_module
from pathlib import Path

import datajoint as dj

_V2 = Path(__file__).resolve().parents[3] / "src/spyglass/spikesorting/v2"


def _v2_exception_classes():
    for path in _V2.rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name.endswith("Error")
                and not node.name.startswith("_")
            ):
                relative = path.relative_to(_V2.parents[2]).with_suffix("")
                yield ".".join(relative.parts), node.name


def test_every_v2_exception_has_the_common_base():
    from spyglass.spikesorting.v2.exceptions import SpikeSortingV2Error

    classes = list(_v2_exception_classes())
    assert len(classes) > 40
    for module, name in classes:
        cls = getattr(import_module(module), name)
        assert issubclass(cls, SpikeSortingV2Error), f"{module}.{name}"


def test_direct_insert_refusal_is_one_error_type():
    """The selection guards and AutoCurationRules raise the same class, which
    stays a DataJointError for existing ``except`` clauses."""
    from spyglass.spikesorting.v2.exceptions import UnsupportedDirectInsertError

    assert issubclass(UnsupportedDirectInsertError, dj.errors.DataJointError)
    assert issubclass(UnsupportedDirectInsertError, RuntimeError)
