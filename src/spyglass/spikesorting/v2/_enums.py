"""Stdlib-only enums for the spike-sorting v2 schema.

``CurationSource`` and ``CurationLabel`` live here -- rather than in
``utils`` -- so the dependency-light service modules
(``_curation_transforms`` et al.) can import the canonical label set
without taking a dependency on ``utils`` (which imports DataJoint /
SpikeInterface at module load). This module's own imports are limited to
the standard library (``enum``), so it adds no DataJoint / SpikeInterface
dependency to whatever imports it. (Importing it as a ``spyglass``
submodule still runs ``spyglass``'s package ``__init__``, which loads
DataJoint -- that is package-wide and unrelated to this module.)
``utils`` re-exports both names, so existing
``from ...v2.utils import CurationLabel`` and ``CurationSource`` call sites
are unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

# Bad-channel labels returned by SpikeInterface's ``detect_bad_channels``
# (``coherence+psd``). A ``Literal`` rather than an ``Enum`` because these
# values flow through plain result dicts as the raw SpikeInterface strings;
# naming the closed set here gives the wrapper / persist helper one source of
# truth for the vocabulary and lets a type checker catch a typo'd label.
BadChannelLabel = Literal["good", "dead", "noise", "out"]


class CurationSource(str, Enum):
    """Provenance of how a ``CurationV2`` row was created.

    Matches the enum on the table's ``curation_source`` column. Promoted
    from a runtime set check so a typo at insert time raises a clear
    ``ValueError`` instead of a DataJoint enum-mismatch error from
    MySQL.
    """

    manual = "manual"
    analyzer_curation = "analyzer_curation"
    figpack = "figpack"


class CurationLabel(str, Enum):
    """Curation labels recognized by ``CurationV2.insert_curation``.

    The members are a validated set so a typo raises at insert
    time. The backing ``CurationV2.UnitLabel.curation_label``
    column is a ``varchar(32)``, not a MySQL enum: DataJoint *can*
    declare an enum column (``curation_source`` on ``CurationV2`` is
    one), but v2 chooses varchar because the label set is open-ended --
    a lab adding a custom label later would otherwise need a forbidden
    ``ALTER TABLE`` under the zero-migration policy. The typo guard is
    enforced in Python on every insert path instead: both
    ``CurationV2.insert_curation`` and a direct
    ``CurationV2.UnitLabel.insert1`` / ``insert`` validate against this
    set (pass ``allow_custom_labels=True`` to opt out).
    """

    accept = "accept"
    mua = "mua"
    noise = "noise"
    artifact = "artifact"
    reject = "reject"

    @classmethod
    def normalize(cls, label) -> str:
        """Coerce a label to its canonical string value.

        Accepts a ``CurationLabel`` member (returns its ``.value``) or any
        other value (returns ``str(label)``). Single source of truth for the
        ``label.value if isinstance(...) else str(label)`` coercion the insert
        and accessor paths apply at every label boundary.
        """
        return label.value if isinstance(label, cls) else str(label)
