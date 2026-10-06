"""Unit tests for DecodingParameters serialization round-trip.

These load the real conversion helpers and DecodingParameters class with
in-memory DataJoint boundaries, so they touch no database tables or records.
The production serialization, insert, and fetch methods are exercised intact.
They would have caught the registry regression where every
detector/classifier class (an ``ABCMeta`` ``BaseEstimator`` subclass) was
silently excluded from ``_model_class_registry``. See PR #1618.
"""

import ast
import builtins
import copy
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_decoding_modules(monkeypatch):
    """Load production code while replacing only schema/table boundaries."""

    class Lookup:
        row_count = 4

        def __len__(self):
            return self.row_count

        def insert(self, rows, *args, **kwargs):
            self.inserted_rows = list(rows)
            self.insert_args = args
            self.insert_kwargs = kwargs
            type(self).last_insert = self

        def fetch1(self, *args, **kwargs):
            row = copy.deepcopy(self.inserted_rows[0])
            if not args:
                return row
            values = tuple(row[name] for name in args)
            return values[0] if len(args) == 1 else values

    class SpyglassMixin:
        pass

    class SpyglassMixinPart:
        pass

    dj = ModuleType("datajoint")
    dj.Lookup = Lookup
    dj.Manual = type("Manual", (), {})
    dj.schema = lambda name: lambda cls: cls
    monkeypatch.setitem(sys.modules, "datajoint", dj)

    for name in (
        "spyglass",
        "spyglass.common",
        "spyglass.common.common_session",
        "spyglass.decoding",
        "spyglass.decoding.v1",
        "spyglass.position",
        "spyglass.position.position_merge",
        "spyglass.utils",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules["spyglass.common.common_session"].Session = object
    sys.modules["spyglass.position.position_merge"].PositionOutput = object
    utils = sys.modules["spyglass.utils"]
    utils.SpyglassMixin = SpyglassMixin
    utils.SpyglassMixinPart = SpyglassMixinPart
    utils.logger = logging.getLogger(__name__)

    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "spyglass"
        / "decoding"
        / "v1"
    )
    modules = {}
    for name in ("dj_decoder_conversion", "core"):
        full_name = f"spyglass.decoding.v1.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, source / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


@pytest.fixture(autouse=True)
def decoding_modules(monkeypatch):
    return _load_decoding_modules(monkeypatch)


DEFAULT_MODEL_CLASSES = [
    "ContFragClusterlessClassifier",
    "NonLocalClusterlessDetector",
    "ContFragSortedSpikesClassifier",
    "NonLocalSortedSpikesDetector",
]

BASE_DETECTOR_CLASSES = ["ClusterlessDetector", "SortedSpikesDetector"]

# Non-default subclass-only values for the NonLocal* models, set so the
# round-trip is asserted by VALUE (not just key presence). Key-set equality is
# guaranteed by the sklearn get_params contract regardless of correctness, so
# it cannot detect a corrupted value -- and preserving these subclass-only
# parameters is the whole point of reconstructing via the concrete class.
NONLOCAL_PARAM_OVERRIDES = {
    "non_local_position_penalty": 2.5,
    "non_local_penalty_std": 0.75,
}


def _serialize_like_insert(model):
    """Mirror DecodingParameters.insert serialization of a detector instance."""
    from spyglass.decoding.v1.dj_decoder_conversion import (
        convert_classes_to_dict,
    )

    params = model.get_params(deep=False)
    params["class_name"] = type(model).__name__
    return convert_classes_to_dict(params)


def test_model_class_registry_contains_detectors():
    """All concrete detectors plus the two base classes must be resolvable.

    Pins the metaclass-filtering regression directly: detectors use
    ``ABCMeta``, so a ``__class__.__name__ == "type"`` filter drops them.
    """
    from spyglass.decoding.v1.dj_decoder_conversion import (
        _model_class_registry,
    )

    registry = _model_class_registry()
    for name in DEFAULT_MODEL_CLASSES + BASE_DETECTOR_CLASSES:
        assert name in registry, f"{name} missing from model class registry"


@pytest.mark.parametrize("class_name", DEFAULT_MODEL_CLASSES)
def test_decoding_params_roundtrip(class_name):
    """Serialize -> restore yields the concrete subclass with values intact."""
    import non_local_detector as nld

    from spyglass.decoding.v1.dj_decoder_conversion import restore_classes

    cls = getattr(nld, class_name)
    # NonLocal* models carry subclass-only penalty parameters in recent
    # non_local_detector versions; set non-default values so the round-trip is
    # checked by value, not just key presence. Gate on the params the installed
    # version actually accepts -- they were added after 0.6.9 -- so older
    # versions still exercise the isinstance/keys assertions without erroring.
    available = set(cls().get_params(deep=False))
    overrides = {
        name: value
        for name, value in NONLOCAL_PARAM_OVERRIDES.items()
        if class_name.startswith("NonLocal") and name in available
    }
    model = cls(**overrides)

    restored = restore_classes(_serialize_like_insert(model))

    assert isinstance(restored, cls)
    restored_params = restored.get_params(deep=False)
    assert restored_params.keys() == model.get_params(deep=False).keys()
    # Subclass-only parameter values survive (the point of concrete-class
    # reconstruction); reconstructing via the base detector would drop them.
    for name, value in overrides.items():
        assert restored_params[name] == value


def test_restore_classes_legacy_dict_returns_dict():
    """Legacy rows (no ``class_name``) return a dict with nested classes restored."""
    from non_local_detector import ContFragClusterlessClassifier
    from non_local_detector.environment import Environment

    from spyglass.decoding.v1.dj_decoder_conversion import (
        convert_classes_to_dict,
        restore_classes,
    )

    model = ContFragClusterlessClassifier()
    # Old ``vars()``-style serialization carries no top-level ``class_name``.
    legacy = convert_classes_to_dict(dict(vars(model)))

    restored = restore_classes(legacy)

    assert isinstance(restored, dict)
    assert "class_name" not in restored
    # The dict path still rebuilds the nested classes the make() sites need.
    assert isinstance(restored["environments"][0], Environment)


def test_restore_classes_legacy_dict_strips_derived_attrs():
    """Legacy rows with a derived ``_``-prefixed attr rebuild via the base detector.

    Rows serialized via ``vars(model)`` before the ``get_params()`` switch could
    persist detector internals that are not constructor parameters -- notably
    ``_frozen_discrete_transition_rows_mask_``, a mask computed in ``__init__`` on
    some non_local_detector versions. Such rows carry no ``class_name``, so they
    take the dict fallback and are rebuilt with the base detector; without
    stripping, ``ClusterlessDetector(**restored)`` raises ``TypeError``. The
    attribute is injected explicitly so the test reproduces a poisoned legacy row
    regardless of whether the installed version still writes it in ``vars()``.
    """
    from non_local_detector import ContFragClusterlessClassifier
    from non_local_detector.models.base import ClusterlessDetector

    from spyglass.decoding.v1.dj_decoder_conversion import (
        convert_classes_to_dict,
        restore_classes,
    )

    model = ContFragClusterlessClassifier()
    # Legacy vars()-style serialization: no class_name, plus a derived internal.
    legacy = convert_classes_to_dict(dict(vars(model)))
    legacy["_frozen_discrete_transition_rows_mask_"] = None

    restored = restore_classes(legacy)

    assert isinstance(restored, dict)
    assert "_frozen_discrete_transition_rows_mask_" not in restored
    # The point of the strip: base-detector reconstruction no longer raises.
    assert isinstance(ClusterlessDetector(**restored), ClusterlessDetector)


@pytest.mark.parametrize(
    "class_name",
    ["NonLocalClusterlessDetector", "NonLocalSortedSpikesDetector"],
)
def test_restore_classes_legacy_nonlocal_upgrades_to_concrete_class(class_name):
    """Legacy NonLocal rows rebuild via the concrete class, preserving penalties.

    NonLocal* detectors accept ``non_local_position_penalty`` /
    ``non_local_penalty_std`` that the base detector rejects. A legacy row (no
    ``class_name``) carrying them must be rebuilt as the concrete NonLocal class
    rather than the base detector, or those parameters are lost / raise.
    Parametrized over both modalities (clusterless and sorted spikes) since each
    infers its family from a different marker key. Gated on the installed
    non_local_detector actually exposing the params (added after 0.6.9),
    mirroring ``test_decoding_params_roundtrip``.
    """
    import non_local_detector as nld

    from spyglass.decoding.v1.dj_decoder_conversion import (
        convert_classes_to_dict,
        restore_classes,
    )

    cls = getattr(nld, class_name)
    available = set(cls().get_params(deep=False))
    overrides = {
        name: value
        for name, value in NONLOCAL_PARAM_OVERRIDES.items()
        if name in available
    }
    if not overrides:
        pytest.skip("installed non_local_detector has no NonLocal-only params")

    model = cls(**overrides)
    # Legacy vars()-style serialization carries no top-level class_name.
    legacy = convert_classes_to_dict(dict(vars(model)))

    restored = restore_classes(legacy)

    assert isinstance(restored, cls)
    restored_params = restored.get_params(deep=False)
    for name, value in overrides.items():
        assert restored_params[name] == value


def test_convert_classes_to_dict_stringifies_sorted_algorithm_model():
    """Sorted-spikes algorithm params get the same model->name conversion.

    ``convert_classes_to_dict`` routes both ``clusterless_algorithm_params`` and
    ``sorted_spikes_algorithm_params`` through ``_convert_algorithm_params``, so a
    class stored under ``model`` becomes its name for datajoint storage
    symmetrically across modalities (previously only clusterless was routed).
    """
    from non_local_detector import ContFragSortedSpikesClassifier

    from spyglass.decoding.v1.dj_decoder_conversion import (
        convert_classes_to_dict,
    )

    class _DummyAlgorithmModel:
        pass

    params = dict(vars(ContFragSortedSpikesClassifier()))
    params["sorted_spikes_algorithm_params"] = {"model": _DummyAlgorithmModel}

    converted = convert_classes_to_dict(params)

    assert (
        converted["sorted_spikes_algorithm_params"]["model"]
        == "_DummyAlgorithmModel"
    )


def test_restore_classes_unknown_class_raises():
    """An unrecognized ``class_name`` fails loudly, listing known classes."""
    from non_local_detector import ContFragClusterlessClassifier

    from spyglass.decoding.v1.dj_decoder_conversion import restore_classes

    stored = _serialize_like_insert(ContFragClusterlessClassifier())
    stored["class_name"] = "NotARealDetector"

    with pytest.raises(ValueError, match="Unknown decoder model class"):
        restore_classes(stored)


@pytest.mark.parametrize("class_name", DEFAULT_MODEL_CLASSES)
def test_actual_insert_fetch_preserves_constructor_and_caller(
    decoding_modules, class_name
):
    """The merged insert uses get_params and permits reuse of caller rows."""
    import non_local_detector as nld

    cls = getattr(nld, class_name)
    available = set(cls().get_params(deep=False))
    overrides = {
        name: value
        for name, value in NONLOCAL_PARAM_OVERRIDES.items()
        if class_name.startswith("NonLocal") and name in available
    }
    model = cls(**overrides)
    # Newer NLD exposes this as a property, while old vars()-serialized rows
    # stored it in __dict__. Inject the historical state without its setter.
    vars(model)[
        "_frozen_discrete_transition_rows_mask_"
    ] = "derived legacy state"
    rows = [
        {
            "decoding_param_name": "repeatable_model",
            "decoding_params": model,
            "decoding_kwargs": {"max_iter": 3},
        }
    ]
    table = decoding_modules.core.DecodingParameters()

    for _ in range(2):
        table.insert(rows, skip_duplicates=True)
        stored = table.inserted_rows[0]
        assert stored is not rows[0]
        assert stored["decoding_params"]["class_name"] == class_name
        assert (
            "_frozen_discrete_transition_rows_mask_"
            not in stored["decoding_params"]
        )
        assert table.insert_kwargs == {"skip_duplicates": True}
        restored = table.fetch1("decoding_params")
        assert type(restored) is cls
        assert restored.get_params(deep=False).keys() == available
        for name, value in overrides.items():
            assert restored.get_params(deep=False)[name] == value

    assert rows[0]["decoding_params"] is model
    assert rows[0]["decoding_kwargs"] == {"max_iter": 3}
    assert vars(model)["_frozen_discrete_transition_rows_mask_"] == (
        "derived legacy state"
    )
    assert all(
        "class_name" not in vars(transition)
        for transition_row in model.continuous_transition_types
        for transition in transition_row
    )


def test_actual_insert_default_calls_instance_insert(decoding_modules):
    """Default insertion reaches the override and serializes every model."""
    table_cls = decoding_modules.core.DecodingParameters

    table_cls.insert_default()

    table = table_cls.last_insert
    assert table.insert_kwargs == {"skip_duplicates": True}
    assert {
        row["decoding_params"]["class_name"] for row in table.inserted_rows
    } == set(DEFAULT_MODEL_CLASSES)


def test_conversion_and_populated_core_import_without_nld(monkeypatch):
    """Model reconstruction alone may import the optional NLD dependency."""
    real_import = builtins.__import__
    attempted = []

    def import_without_nld(name, *args, **kwargs):
        if name == "non_local_detector" or name.startswith(
            "non_local_detector."
        ):
            attempted.append(name)
            raise AttributeError("broken optional non_local_detector")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_nld)
    modules = _load_decoding_modules(monkeypatch)
    table = modules.core.DecodingParameters()

    assert table.contents == []
    assert attempted == []

    with pytest.raises(AttributeError, match="broken optional"):
        modules.dj_decoder_conversion._model_class_registry()
    assert attempted == ["non_local_detector"]


def test_default_contents_recovers_from_broken_optional_import(
    decoding_modules, monkeypatch, caplog
):
    """A fresh table warns on import failure and retries after recovery."""
    table = decoding_modules.core.DecodingParameters()
    table.row_count = 0
    real_import = builtins.__import__

    def import_without_nld(name, *args, **kwargs):
        if name == "non_local_detector":
            raise AttributeError("broken optional non_local_detector")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as blocked:
        blocked.setattr(builtins, "__import__", import_without_nld)
        assert table.contents == []
    assert "Skipping default decoding paramsets" in caplog.text
    assert len(table.contents) == 4


@pytest.mark.parametrize(
    "tutorial_name, class_name",
    [
        ("41_Decoding_Clusterless", "ContFragClusterlessClassifier"),
        ("42_Decoding_SortedSpikes", "ContFragSortedSpikesClassifier"),
    ],
)
def test_paired_tutorial_model_insert_and_fetch(
    decoding_modules, tutorial_name, class_name
):
    """Execute tutorial model examples through the real serialization methods."""
    import non_local_detector as nld

    notebooks = Path(__file__).resolve().parents[2] / "notebooks"
    notebook = json.loads((notebooks / f"{tutorial_name}.ipynb").read_text())
    notebook_body = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        # IPython help/magic lines do not participate in Python model examples.
        source = "\n".join(
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith(("?", "%", "!"))
        )
        notebook_body.extend(ast.parse(source).body)
    notebook_tree = ast.Module(body=notebook_body, type_ignores=[])
    script_tree = ast.parse(
        (notebooks / "py_scripts" / f"{tutorial_name}.py").read_text()
    )
    assert ast.dump(notebook_tree) == ast.dump(
        script_tree
    ), "paired notebook and Python tutorial code differ"

    table = decoding_modules.core.DecodingParameters()

    class TutorialParameters:
        def insert1(self, row, **kwargs):
            table.insert([row], **kwargs)

        def __and__(self, restriction):
            return table

    scope = {
        "DecodingParameters": TutorialParameters(),
        class_name: getattr(nld, class_name),
    }
    inserts = [
        node
        for node in notebook_body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "DecodingParameters"
        and node.value.func.attr == "insert1"
    ]
    assert len(inserts) == 1
    exec(
        compile(
            ast.Module(body=inserts, type_ignores=[]), tutorial_name, "exec"
        ),
        scope,
    )
    restored = table.fetch1()["decoding_params"]
    assert type(restored) is getattr(nld, class_name)
    assert restored.get_params(deep=False)["sampling_frequency"] == 500

    if class_name == "ContFragClusterlessClassifier":
        # Execute the fetched-model inspection exactly as shown. The former
        # Classifier(**row["decoding_params"]) example failed on an instance.
        fetch_start = next(
            i
            for i, node in enumerate(notebook_body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "model_params"
                for target in node.targets
            )
        )
        inspection = []
        for node in notebook_body[fetch_start:]:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                break
            inspection.append(node)
        exec(
            compile(
                ast.Module(body=inspection, type_ignores=[]),
                tutorial_name,
                "exec",
            ),
            scope,
        )
        assert type(scope["model"]) is getattr(nld, class_name)
        assert scope["model"].sampling_frequency == 500
