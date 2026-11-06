"""Stdlib-only enums for the spike-sorting v2 schema."""

from __future__ import annotations

from enum import Enum
from typing import Literal

# SpikeInterface ``detect_bad_channels`` labels. A ``Literal`` because the raw
# strings flow through plain result dicts.
BadChannelLabel = Literal["good", "dead", "noise", "out"]


class CurationSource(str, Enum):
    """Provenance of how a ``CurationV2`` row was created.

    Matches the enum on the table's ``curation_source`` column.
    """

    manual = "manual"
    analyzer_curation = "analyzer_curation"
    figpack = "figpack"
    curation_evaluation = "curation_evaluation"


class CurationLabel(str, Enum):
    """Curation labels recognized by ``CurationV2.insert_curation``.

    The backing column is ``varchar(32)``, not a MySQL enum, so a lab can add
    labels without altering a populated table. Insert paths validate against
    this set; pass ``allow_custom_labels=True`` to opt out.
    """

    accept = "accept"
    mua = "mua"
    noise = "noise"
    artifact = "artifact"
    reject = "reject"

    @classmethod
    def normalize(cls, label) -> str:
        """Coerce a label to its canonical string value.

        Returns a member's ``.value``, else ``str(label)``.
        """
        return label.value if isinstance(label, cls) else str(label)
