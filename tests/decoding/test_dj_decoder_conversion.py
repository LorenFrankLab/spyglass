"""Unit tests for DecodingParameters serialization round-trip.

These exercise the pure conversion helpers in
``spyglass.decoding.v1.dj_decoder_conversion``. They touch no database
tables or records (no inserts or fetches), though importing the module
establishes a DataJoint schema connection like the rest of the suite.
They would have caught the registry regression where every
detector/classifier class (an ``ABCMeta`` ``BaseEstimator`` subclass) was
silently excluded from ``_model_class_registry``. See PR #1618.
"""

import pytest

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
