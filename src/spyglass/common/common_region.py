import datajoint as dj

from spyglass.utils.dj_mixin import SpyglassMixin, logger

schema = dj.schema("common_region")


@schema
class BrainRegion(SpyglassMixin, dj.Lookup):
    definition = """
    region_id: smallint auto_increment
    ---
    region_name: varchar(200)       # Name of the region (e.g., 'Hippocampal formation')
    region_abbr=NULL: varchar(64)   # Standard abbreviation (e.g., 'HPF')
    subregion_name=NULL: varchar(200) # Subregion name (e.g., 'Cornu Ammonis 1')
    subregion_abbr=NULL: varchar(64)  # Subregion abbreviation (e.g., 'CA1')
    subsubregion_name=NULL: varchar(200) # Sub-subregion name (e.g., 'stratum pyramidale')
    subsubregion_abbr=NULL: varchar(64) # Sub-subregion abbreviation (e.g., 'sp')
    atlas_source: varchar(128)      # Source atlas (e.g., 'Allen CCF v3', 'Paxinos Rat 6th Ed')
    """

    # TODO consider making (region_name, subregion_name, subsubregion_name) a
    # primary key subregion_name='' and subsubregion_name='' will be necessary
    # but that seems OK

    @classmethod
    def fetch_add(
        cls,
        region_name: str,
        subregion_name: str = None,
        subsubregion_name: str = None,
    ):
        """Return the region ID for names. If no match, add to the BrainRegion.

        The combination of (region_name, subregion_name, subsubregion_name) is
        effectively unique, then.

        Parameters
        ----------
        region_name : str
            The name of the brain region.
        subregion_name : str, optional
            The name of the subregion within the brain region.
        subsubregion_name : str, optional
            The name of the subregion within the subregion.

        Returns
        -------
        region_id : int
            The index of the region in the BrainRegion table.
        """
        key = dict(
            region_name=region_name,
            subregion_name=subregion_name,
            subsubregion_name=subsubregion_name,
        )
        query = BrainRegion & key
        if not query:
            logger.info(
                f"Brain region '{region_name}' not found. Adding to BrainRegion. "
                "Please make sure to check the spelling and format."
                "Remove any extra spaces or special characters."
            )
            cls.insert1(key)
            query = BrainRegion & key
        return query.fetch1("region_id")


@schema
class CoordinateSystem(dj.Lookup):
    definition = """
    # Defines standard coordinate systems used for spatial data.
    coordinate_system_id: varchar(64) # Primary key (e.g., 'Allen_CCFv3_RAS_um')
    ---
    description: varchar(255)         # Description of the coordinate system
    orientation: enum(                # Anatomical orientation convention
        "RAS", "LPS", "PIR", "ASL", "XYZ", "Other"
        )
    unit: enum(                       # Spatial unit
        "um", "mm", "pixels", "voxels"
        )
    atlas_source=NULL: varchar(128)   # Source if based on an atlas (e.g., 'Allen CCF v3', 'WHS Rat v4')
    """
    contents = [
        [
            "Allen_CCFv3_RAS_um",
            "Allen CCF v3 Mouse Atlas, RAS orientation, micrometers",
            "RAS",
            "um",
            "Allen CCF v3",
        ],
        [
            "Paxinos_Rat_6th_PIR_um",
            "Paxinos & Watson Rat Atlas 6th Ed, PIR orientation, micrometers",
            "PIR",
            "um",
            "Paxinos Rat 6th Ed",
        ],
        [
            "WHS_Rat_v4_RAS_um",
            "Waxholm Space Sprague Dawley Rat Atlas v4, RAS orientation, micrometers",
            "RAS",
            "um",
            "WHS Rat v4",
        ],
        [
            "Histology_Image_Pixels",
            "2D Pixels from processed histology image (Origin/Orientation Varies)",
            "Other",
            "pixels",
            None,
        ],
        [
            "MicroCT_Voxel_Scan",
            "3D Voxel space from raw microCT scan (Orientation relative to scanner)",
            "XYZ",  # Or 'Other' depending on context
            "voxels",
            None,
        ],
    ]
