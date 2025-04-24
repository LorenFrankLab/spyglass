import datajoint as dj

from spyglass.common import Electrode  # noqa: F401
from spyglass.electrode_localization.v1.coordinate_system import (  # noqa: F401
    CoordinateSystem,
)
from spyglass.micro_ct.v1.micro_ct import MicroCTImage  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("electrode_localization_v1")


@schema
class AlignmentMicroCT(SpyglassMixin, dj.Manual):  # Or dj.Computed
    definition = """
     # Stores results and parameters of aligning microCT image data to a target coordinate system
     -> MicroCTImage           # Link to the source microCT NWB file info
     alignment_id: varchar(32) # Unique ID for this specific alignment instance/parameters
     ---
     -> CoordinateSystem       # The TARGET coordinate system achieved by this alignment
     # -> BrainAtlas          # Optional: Link to the specific target atlas instance
     alignment_method: varchar(128) # Name of algorithm/tool used
     alignment_params = NULL: blob   # Store parameters dict/json
     transformation_matrix = NULL: blob # Store affine matrix if applicable
     warp_field_path = NULL: varchar(512) # Store path to warp field file if non-linear
     alignment_quality = NULL: float   # Optional QC metric for the alignment
     alignment_time = CURRENT_TIMESTAMP: timestamp
     alignment_notes = "": varchar(2048)
     """


@schema
class ChannelBrainLocationMicroCTV1(SpyglassMixin, dj.Manual):
    definition = """
    # MicroCT-derived coordinates and region assignment for an electrode
    -> Electrode                   # Electrode being localized
    -> MicroCTImage                # Source NWB file link for microCT data
    -> AlignmentMicroCT             # Alignment parameters used
    ---
    -> CoordinateSystem            # Defines the space for pos_x,y,z
    pos_x: float                   # (um) coordinate in the specified space
    pos_y: float                   # (um) coordinate in the specified space
    pos_z: float                   # (um) coordinate in the specified space
    -> BrainRegion                 # Assigned brain region (controlled vocabulary)
    """
