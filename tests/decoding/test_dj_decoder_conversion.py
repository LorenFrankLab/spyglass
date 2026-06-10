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
    # NonLocal* models carry subclass-only penalty parameters; set non-default
    # values so the round-trip is checked by value, not just by key presence.
    overrides = (
        NONLOCAL_PARAM_OVERRIDES if class_name.startswith("NonLocal") else {}
    )
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


def test_restore_classes_unknown_class_raises():
    """An unrecognized ``class_name`` fails loudly, listing known classes."""
    from non_local_detector import ContFragClusterlessClassifier

    from spyglass.decoding.v1.dj_decoder_conversion import restore_classes

    stored = _serialize_like_insert(ContFragClusterlessClassifier())
    stored["class_name"] = "NotARealDetector"

    with pytest.raises(ValueError, match="Unknown decoder model class"):
        restore_classes(stored)
