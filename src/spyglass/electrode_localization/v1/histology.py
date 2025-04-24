"""Histology-derived coordinates and region assignment for an electrode"""

import datajoint as dj

from spyglass.common import (
    BrainCoordinateSystem,
    BrainRegion,
    Electrode,
)  # noqa: F401
from spyglass.histology.v1.histology import (  # noqa: F401
    HistologyImages,
    HistologyRegistration,
)
from spyglass.utils import SpyglassMixin

schema = dj.schema("electrode_localization_v1")


@schema
class ChannelBrainLocationHistologyV1(SpyglassMixin, dj.Manual):
    definition = """
    # Histology-derived coordinates and region assignment for an electrode
    -> Electrode                   # Electrode being localized
    -> HistologyImages              # Source NWB file link for histology images
    -> HistologyRegistration       # Alignment parameters used
    ---
    -> BrainCoordinateSystem       # Defines the space for pos_x,y,z (e.g., Allen CCF RAS um)
    pos_x: float                   # (um) coordinate in the specified space
    pos_y: float                   # (um) coordinate in the specified space
    pos_z: float                   # (um) coordinate in the specified space
    -> BrainRegion                 # Assigned brain region
    """
