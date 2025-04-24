import datajoint as dj

from spyglass.common import (  # noqa: F401
    AnalysisNwbfile,
    CoordinateSystem,
    LabMember,
    Subject,
)
from spyglass.utils import SpyglassMixin

schema = dj.schema("histology_v1")


@schema
class Histology(SpyglassMixin, dj.Manual):
    definition = """
    # Represents a single histology preparation for a subject
    -> Subject
    histology_id: varchar(32) # User-defined ID for this prep (e.g., 'probe_track_run1', 'anatomy_stain_seriesA')
    ---
    prep_date=NULL: date           # Optional: Date of tissue preparation
    slice_orientation: enum("coronal", "sagittal", "horizontal", "other") # Orientation of sections
    slice_thickness: float     # (um) Thickness of sections
    mounting_medium = NULL : varchar(128)
    experiment_purpose: varchar(1024) # e.g., 'Probe track recovery for Neuropixel P0', 'ChR2 expression check in mPFC', 'General anatomical reference'
    notes = "": varchar(2048)  # Optional general notes about the preparation
    -> [nullable] LabMember.proj(histology_experimenter='user_name') # Optional: who did the prep?
    """

    class HistologyStain(SpyglassMixin, dj.Part):
        definition = """
        # Details of specific stains used in a histology preparation
        -> Histology
        stain_index: tinyint unsigned # Use index for multiple stains per prep (0, 1, 2...)
        ---
        identified_feature: varchar(128) # Biological target, structure, or marker identified (e.g., 'GFAP+ Astrocytes', 'Nissl Bodies', 'ChR2-tdTomato+ Cells', 'ProbeTrack_DiI', 'Gad2 mRNA')
        visualization_agent: varchar(128)  # Method/molecule making feature visible (e.g., 'Alexa 488', 'Cresyl Violet', 'Native tdTomato Fluorescence', 'DiI', 'NBT/BCIP via ISH probe')
        stain_type : enum("immunohistochemistry", "genetic_marker", "tracer", "anatomical", "histochemical", "in_situ_hybridization", "other") = "other"
        stain_protocol_name = NULL : varchar(128) # Optional: name of the protocol used for this stain
        antibody_details = NULL : varchar(255) # Optional: specific antibody info (e.g. company, cat#, lot#)
        stain_notes = "": varchar(1024) # Optional notes about this specific stain (e.g., concentration, incubation)
        """


class HistologyImages(SpyglassMixin, dj.Computed):
    definition = """
    # Links Histology info to the Analysis NWB file containing the image volume data/links
   -> Histology
   images_id: varchar(32) # User-defined ID for these images (e.g., 'probe_track_run1', 'anatomy_stain_seriesA')
   ---
   -> AnalysisNwbfile
   color_to_stain = NULL: blob # Mapping of color channels to stains (e.g., {'DAPI': 'blue', 'GFAP': 'green'})
   pixel_size_x: float # (um) Pixel size in X direction
   pixel_size_y: float # (um) Pixel size in Y direction
   pixel_size_z: float # (um) Pixel size in Z direction
   objective_magnification: float # Magnification of the objective lens used (e.g., 20x, 40x)
   scale: float # Scale factor for the image (e.g., 1.0 for no scaling, 2.0 for double size)
   image_modality: enum("fluorescence", "brightfield", "other") # Modality of the image (e.g., 'fluorescence', 'brightfield')
   """

    def make(self, key: dict) -> None:
        """Populate HistologyImage table with NWB file links and metadata"""
        # Placeholder for actual implementation
        # This function should create an AnalysisNwbfile entry and link it to the Histology entry
        # It should also populate the color_to_stain and scale fields based on the image data
        pass


class HistologyRegistration(SpyglassMixin, dj.Manual):
    definition = """
     # Stores results and parameters of aligning histology image data to a target coordinate system
     -> HistologyImages         # Link to the source histology NWB file info
     registration_id: varchar(32) # Unique ID for this specific registration instance/parameters (e.g., 'Elastix_Default_v1')
     ---
     -> CoordinateSystem       # The TARGET coordinate system achieved by this registration (e.g., 'allen_ccf_v3_ras_um')
     registration_method       : varchar(128)    # algorithmic approach, e.g. 'affine+bspline'
     registration_software     : varchar(128)    # e.g. 'ANTs', 'elastix', 'SimpleITK'
     registration_software_version : varchar(64)  # e.g. '2.3.5', '1.3.0'
     transformation_matrix = NULL: blob # Store affine matrix if computed/applicable (e.g., 4x4 np.array.tobytes())
     warp_field_path = NULL: varchar(512) # Store path to warp field file if non-linear
     registration_quality = NULL: float   # Optional QC metric for the registration (e.g., Dice score, landmark error)
     registration_time = CURRENT_TIMESTAMP: timestamp  # Time this registration entry was created/run
     registration_notes = "": varchar(2048) # Any specific notes about this registration run
     """
