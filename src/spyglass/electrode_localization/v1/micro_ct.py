"""MicroCT-derived coordinates and region assignment for an electrode"""

import datajoint as dj

from spyglass.common import (
    BrainCoordinateSystem,
    BrainRegion,
    Electrode,
)  # noqa: F401
from spyglass.micro_ct.v1.micro_ct import (  # noqa: F401
    MicroCTImages,
    MicroCTRegistration,
)
from spyglass.utils import SpyglassMixin

schema = dj.schema("electrode_localization_v1")


@schema
class ChannelBrainLocationMicroCTV1(SpyglassMixin, dj.Manual):
    definition = """
    # MicroCT-derived coordinates and region assignment for an electrode
    -> Electrode                   # Electrode being localized
    -> MicroCTImages                # Source NWB file link for microCT data
    -> MicroCTRegistration             # Alignment parameters used
    ---
    -> BrainCoordinateSystem            # Defines the space for pos_x,y,z
    pos_x: float                   # (um) coordinate in the specified space
    pos_y: float                   # (um) coordinate in the specified space
    pos_z: float                   # (um) coordinate in the specified space
    -> BrainRegion                 # Assigned brain region
    """
