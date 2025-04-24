import datajoint as dj

from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_merge_tables import _Merge

schema = dj.schema("electrode_localization_v1")


@schema
class CoordinateSystem(dj.Lookup):
    definition = """
    # Defines standard coordinate systems used for spatial data
    coordinate_system_id: varchar(32) # e.g., 'Allen_CCF_v3_RAS_um'
    ---
    description: varchar(255)
    """
    # Maybe use https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html
    contents = [
        [
            "Allen_CCF_v3_RAS_um",
            "Allen CCF v3 Reference Atlas Space, RAS orientation, unit um",
        ],
        ["Histology_Image_Pixels", "2D Pixels from processed histology image"],
        ["MicroCT_Voxel", "3D Voxel space from microCT scan"],
        ["whs_sd_rat_39um", "3D Voxel space from Waxholm atlas"],
    ]
