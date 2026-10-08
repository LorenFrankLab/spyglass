"""Supported DataJoint lifecycle for the staged-output adapter."""

from importlib.metadata import version

STAGED_POPULATE_DATAJOINT_VERSION = "0.14.9"


def require_staged_populate_support() -> None:
    """Refuse an unverified lifecycle before declaring staged-output tables.

    StagedOutputCleanupMixin relies on the tri-part transaction and return
    contracts of AutoPopulate._populate1. Expanding this version requires
    exercising its real fetch/compute/insert, commit and rollback tests.
    """
    installed = version("datajoint")
    if installed != STAGED_POPULATE_DATAJOINT_VERSION:
        raise RuntimeError(
            "Spike sorting v2 staged outputs require DataJoint "
            f"{STAGED_POPULATE_DATAJOINT_VERSION}; installed {installed}. "
            'Install the tested stack with pip install "spyglass-neuro[spikesorting-v2]".'
        )
