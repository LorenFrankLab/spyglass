import datajoint as dj

from spyglass.common import Electrode  # noqa: F401
from spyglass.electrode_localization.v1.coordinate_system import (  # noqa: F401
    CoordinateSystem,
)
from spyglass.histology.v1.histology import HistologyImage  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("electrode_localization_v1")


class AlignmentHistology(SpyglassMixin, dj.Manual):
    definition = """
     # Stores results and parameters of aligning histology image data to a target coordinate system
     -> HistologyImage         # Link to the source histology NWB file info
     alignment_id: varchar(32) # Unique ID for this specific alignment instance/parameters (e.g., 'Elastix_Default_v1')
     ---
     -> CoordinateSystem       # The TARGET coordinate system achieved by this alignment (e.g., 'allen_ccf_v3_ras_um')
     # -> BrainAtlas          # Optional: Link to the specific target atlas instance if needed
     alignment_method: varchar(128) # Name of algorithm/tool used (e.g., 'Manual Landmark', 'Elastix', 'BRAINRegistration', 'ABBA', 'QuickNII')
     alignment_params = NULL: blob   # Store parameters as dict/json blob, or link -> AlignmentParams table if complex/reused
     transformation_matrix = NULL: blob # Store affine matrix if computed/applicable (e.g., 4x4 np.array.tobytes())
     warp_field_path = NULL: varchar(512) # Store path to warp field file if non-linear (@external store might be better long term)
     alignment_quality = NULL: float   # Optional QC metric for the alignment (e.g., Dice score, landmark error)
     alignment_time = CURRENT_TIMESTAMP: timestamp  # Time this alignment entry was created/run
     alignment_notes = "": varchar(2048) # Any specific notes about this alignment run
     """


@schema
class ChannelBrainLocationHistologyV1(SpyglassMixin, dj.Manual):
    definition = """
    # Histology-derived coordinates and region assignment for an electrode
    -> Electrode                   # Electrode being localized
    -> HistologyImage              # Source NWB file link for histology images
    -> AlignmentHistology             # Alignment parameters used
    ---
    -> CoordinateSystem            # Defines the space for pos_x,y,z (e.g., Allen CCF RAS um)
    pos_x: float                   # (um) coordinate in the specified space
    pos_y: float                   # (um) coordinate in the specified space
    pos_z: float                   # (um) coordinate in the specified space
    -> BrainRegion                 # Assigned brain region
    """
