import datajoint as dj

from spyglass.common import (  # noqa: F401
    AnalysisNwbfile,
    CoordinateSystem,
    LabMember,
    Subject,
)
from spyglass.utils import SpyglassMixin

schema = dj.schema("microct_v1")


@schema
class MicroCTScan(SpyglassMixin, dj.Manual):
    definition = """
    # Metadata for a microCT scan of a subject's brain/tissue
    -> Subject
    scan_id: varchar(32) # User-defined ID for this scan (e.g., 'SubjX_OsO4_Scan1')
    ---
    # Preparation Details (can be brief if standardized, use notes for variations)
    stain_reagent = 'Aqueous 2% OsO4': varchar(128) # Osmium Tetroxide details
    stain_duration_days = 14.0 : float          # Duration of staining
    embedding_resin = 'Durcupan ACM': varchar(128) # Resin used for embedding
    prep_protocol_notes = "": varchar(2048)    # Notes on staining, dehydration, embedding variations

    # Scan Details
    scan_date=NULL: date                         # Date of the scan itself
    scanner_details: varchar(255)                # e.g., 'Nikon Metrology X-Tek HMX ST 225 @ HCNS'
    source_target_type = 'Molybdenum': varchar(64) # X-ray source target material
    filter_material = 'None': varchar(64)        # Filter material used (e.g., None, Copper)
    filter_thickness_mm = 0.0: float             # (mm) Filter thickness
    voltage_kv: float                            # (kV) Scan voltage (e.g., 100 or 130)
    current_ua: float                            # (uA) Scan current (e.g., 105 or 135)
    exposure_time_s: float                       # (s) Exposure time per projection (e.g., 1.0)
    num_projections: int                         # Number of projections acquired (e.g., 3184)
    frames_per_projection = 1: int               # Number of frames averaged per projection (e.g., 4)

    # Reconstruction Details (often from scanner software)
    reconstruction_software = NULL: varchar(128) # e.g., 'CT Pro (Nikon)'
    voxel_size_x: float                          # (um) Reconstructed voxel size X
    voxel_size_y: float                          # (um) Reconstructed voxel size Y
    voxel_size_z: float                          # (um) Reconstructed voxel size Z
    output_format = 'TIFF stack': varchar(64)    # Format of raw reconstructed data

    # Raw Data Location (Essential Input for downstream processing)
    raw_scan_path: varchar(512)                  # Path to the raw output (e.g., folder containing TIFF stack)

    scan_notes = "": varchar(2048)               # General notes about the scan itself
    -> [nullable] LabMember.proj(scanner_operator='user_name') # Optional
    """


class MicroCTImages(SpyglassMixin, dj.Computed):
    definition = """
    # Links MicroCTScan info to the Analysis NWB file containing the image volume data/links
    -> MicroCTScan
    images_id: varchar(32) # User-defined ID for these images (e.g., 'SubjX_OsO4_Scan1')
    ---
    -> AnalysisNwbfile
    processing_time=CURRENT_TIMESTAMP: timestamp
    processing_notes = "": varchar(1024)
    """

    def make(self, key: dict) -> None:
        """Populate MicroCTImage table with NWB file links and metadata"""
        # Placeholder for actual implementation
        # This function should create an AnalysisNwbfile entry and link it to the MicroCTScan entry
        # It should also populate the processing_time and processing_notes fields based on the image data
        pass


@schema
class MicroCTRegistration(SpyglassMixin, dj.Manual):
    definition = """
     # Stores results and parameters of aligning microCT image data to a target coordinate system
     -> MicroCTImages          # Link to the source microCT NWB file info
     registration_id: varchar(32) # Unique ID for this specific registration instance/parameters
     ---
     -> CoordinateSystem       # The TARGET coordinate system achieved by this registration
     registration_method       : varchar(128)    # algorithmic approach, e.g. 'affine+bspline'
     registration_software     : varchar(128)    # e.g. 'ANTs', 'elastix', 'SimpleITK'
     registration_software_version : varchar(64)  # e.g. '2.3.5', '1.3.0'
     registration_params = NULL: blob   # Store parameters dict/json
     transformation_matrix = NULL: blob # Store affine matrix if applicable
     warp_field_path = NULL: varchar(512) # Store path to warp field file if non-linear
     registration_quality = NULL: float   # Optional QC metric for the registration
     registration_time = CURRENT_TIMESTAMP: timestamp
     registration_notes = "": varchar(2048)
     """
