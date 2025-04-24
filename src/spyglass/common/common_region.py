import datajoint as dj

from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("common_region")


@schema
class BrainRegion(SpyglassMixin, dj.Lookup):
    definition = """
    region_id: smallint auto_increment
    ---
    region_name: varchar(200)             # the name of the brain region
    subregion_name=NULL: varchar(200)     # subregion name
    subsubregion_name=NULL: varchar(200)  # subregion within subregion
    atlas_name=NULL: varchar(200)         # name of the atlas
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
            cls.insert1(key)
            query = BrainRegion & key
        return query.fetch1("region_id")


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
