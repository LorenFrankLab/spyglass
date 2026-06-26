"""Compatibility shim for a possibly-broken ``non_local_detector``/``jax``.

``non_local_detector`` imports ``jax`` at package init, which can raise
(e.g. ``AttributeError: module 'numpy.dtypes' has no attribute
'StringDType'`` when ``numpy``/``jax`` versions are incompatible). This
module centralizes the import of every ``non_local_detector`` symbol used
across ``spyglass.decoding``, so that the rest of the package can be
imported (and merge tables declared) even when ``non_local_detector`` is
unusable. Symbols are ``None`` when unavailable; check
``NON_LOCAL_DETECTOR_AVAILABLE`` before using them.
"""

from spyglass.utils import logger

NON_LOCAL_DETECTOR_AVAILABLE = True
NON_LOCAL_DETECTOR_IMPORT_ERROR = None

try:
    from non_local_detector import (
        ContFragClusterlessClassifier,
        ContFragSortedSpikesClassifier,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )
    from non_local_detector import __version__ as non_local_detector_version
    from non_local_detector import continuous_state_transitions as cst
    from non_local_detector import discrete_state_transitions as dst
    from non_local_detector import initial_conditions as ic
    from non_local_detector import analysis
    from non_local_detector.environment import Environment
    from non_local_detector.models.base import (
        ClusterlessDetector,
        SortedSpikesDetector,
    )
    from non_local_detector.observation_models import ObservationModel
    from non_local_detector.visualization.figurl_1D import (
        create_1D_decode_view,
    )
    from non_local_detector.visualization.figurl_2D import (
        create_2D_decode_view,
    )
except Exception as e:
    NON_LOCAL_DETECTOR_AVAILABLE = False
    NON_LOCAL_DETECTOR_IMPORT_ERROR = e

    non_local_detector_version = "unavailable"
    ContFragClusterlessClassifier = None
    ContFragSortedSpikesClassifier = None
    NonLocalClusterlessDetector = None
    NonLocalSortedSpikesDetector = None
    cst = None
    dst = None
    ic = None
    analysis = None
    Environment = None
    ClusterlessDetector = None
    SortedSpikesDetector = None
    ObservationModel = None
    create_1D_decode_view = None
    create_2D_decode_view = None

    logger.warning(
        "spyglass.decoding: 'non_local_detector' is unavailable, decoding "
        f"will not work ({e!r}). Import-time table declarations will "
        "proceed, but calling decoding methods will raise."
    )


def raise_if_unavailable():
    """Raise a clear error if non_local_detector could not be imported."""
    if not NON_LOCAL_DETECTOR_AVAILABLE:
        raise ImportError(
            "This operation requires 'non_local_detector', which failed to "
            f"import: {NON_LOCAL_DETECTOR_IMPORT_ERROR!r}"
        ) from NON_LOCAL_DETECTOR_IMPORT_ERROR
