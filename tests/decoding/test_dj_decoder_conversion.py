"""Unit tests for DecodingParameters serialization round-trip.

These exercise the pure conversion helpers in
``spyglass.decoding.v1.dj_decoder_conversion`` and do not require a live
database -- they would have caught the registry regression where every
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
    """Serialize -> restore returns an instance of the *concrete* subclass."""
    import non_local_detector as nld

    from spyglass.decoding.v1.dj_decoder_conversion import restore_classes

    cls = getattr(nld, class_name)
    model = cls()

    restored = restore_classes(_serialize_like_insert(model))

    assert isinstance(restored, cls)
    # The public constructor contract round-trips key-for-key.
    assert (
        restored.get_params(deep=False).keys()
        == model.get_params(deep=False).keys()
    )


def test_restore_classes_legacy_dict_returns_dict():
    """Legacy rows (no ``class_name``) return the converted dict unchanged."""
    from non_local_detector import ContFragClusterlessClassifier

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


def test_restore_classes_unknown_class_raises():
    """An unrecognized ``class_name`` fails loudly, listing known classes."""
    from non_local_detector import ContFragClusterlessClassifier

    from spyglass.decoding.v1.dj_decoder_conversion import restore_classes

    stored = _serialize_like_insert(ContFragClusterlessClassifier())
    stored["class_name"] = "NotARealDetector"

    with pytest.raises(ValueError, match="Unknown decoder model class"):
        restore_classes(stored)
