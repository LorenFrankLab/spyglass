import datajoint as dj

from spyglass.common import (  # noqa: F401
    AnalysisNwbfile,
    CoordinateSystem,
    LabMember,
    Subject,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("microct_v1")


@schema
class MicroCTScan(SpyglassMixin, dj.Manual):
    definition = """
    # Metadata for a microCT scan of a subject's brain/tissue
    -> Subject
    scan_id: varchar(32)                # User-defined ID (e.g., 'SubjX_OsO4_Scan1')
    ---
    # --- Preparation Details ---
    stain_reagent='Aqueous 2% OsO4': varchar(128) # Staining reagent details (e.g., 'OsO4', 'I2E')
    stain_duration_days=14.0: float     # Duration of staining in days
    embedding_resin='Durcupan ACM': varchar(128) # Resin used for embedding
    prep_protocol_notes="": varchar(2048) # Notes on staining, dehydration, embedding variations

    # --- Scan Details ---
    scan_date=NULL: date                # Date of the scan itself
    scanner_details: varchar(255)       # e.g., 'Nikon Metrology X-Tek HMX ST 225 @ HCNS'
    source_target_type='Molybdenum': varchar(64) # X-ray source target material
    filter_material='None': varchar(64) # Filter material used (e.g., 'None', 'Copper')
    filter_thickness_mm=0.0: float      # (mm) Filter thickness
    voltage_kv: float                   # (kV) Scan voltage (e.g., 100, 130)
    current_ua: float                   # (uA) Scan current (e.g., 105, 135)
    exposure_time_s: float              # (s) Exposure time per projection (e.g., 1.0)
    num_projections: int                # Number of projections acquired (e.g., 3184)
    frames_per_projection=1: int        # Frames averaged per projection (e.g., 1, 4)

    # --- Reconstruction Details ---
    reconstruction_software=NULL: varchar(128) # e.g., 'CT Pro (Nikon)', 'NRecon (Bruker)'
    voxel_size_x: float                 # (um) Reconstructed voxel size X
    voxel_size_y: float                 # (um) Reconstructed voxel size Y
    voxel_size_z: float                 # (um) Reconstructed voxel size Z
    output_format='TIFF stack': varchar(64) # Format of raw reconstructed data

    # --- Data Location & Notes ---
    raw_scan_path: varchar(512)         # Path to raw output (e.g., folder containing TIFF stack)
    scan_notes="": varchar(2048)        # General notes about the scan itself
    -> [nullable] LabMember.proj(scanner_operator='user_name') # Optional: Who operated the scanner?
    """


@schema
class MicroCTImages(SpyglassMixin, dj.Computed):
    definition = """
    # Links MicroCTScan info to the Analysis NWB file containing the image volume
    -> MicroCTScan
    images_id: varchar(32)              # User-defined ID for these images (e.g., scan_id)
    ---
    -> AnalysisNwbfile                  # Link to the NWB file storing image data
    processing_time=CURRENT_TIMESTAMP: timestamp # Timestamp of NWB file creation
    processing_notes="": varchar(1024)  # Notes on image processing applied before/during NWB creation
    """

    # Ensure this key is unique for MicroCTImages entries
    # key_source = MicroCTScan

    def make(self, key: dict) -> None:
        """
        Populate MicroCTImages table.
        This method should:
        1. Find the raw reconstructed image data using `raw_scan_path` from `MicroCTScan`.
        2. Process the images if necessary (e.g., format conversion, cropping).
        3. Create an NWB file containing the image volume (e.g., using `pynwb.image.ImageSeries`).
           - Store relevant metadata (voxel sizes from MicroCTScan, etc.) within the NWB file.
        4. Create an `AnalysisNwbfile` entry for the new NWB file.
        5. Insert the key, including the `analysis_file_name` from AnalysisNwbfile,
           and any processing notes into this table.
        """
        logger.info(f"Populating MicroCTImages for key: {key}")
        # Placeholder: Replace with actual NWB creation and insertion logic
        # Example steps (conceptual):
        # 1. scan_entry = (MicroCTScan & key).fetch1()
        # 2. raw_path = scan_entry['raw_scan_path']
        # 3. image_data, metadata = process_microct_images(raw_path, scan_entry) # Your function
        # 4. nwb_file_name = f"{key['subject_id']}_{key['scan_id']}_images.nwb"
        # 5. nwb_file_path = AnalysisNwbfile().create(nwb_file_name)
        # 6. create_microct_nwb(nwb_file_path, image_data, metadata) # Your function
        # 7. AnalysisNwbfile().add(key['subject_id'], nwb_file_name)
        # 8. self.insert1({
        #     **key,
        #     'images_id': key['scan_id'], # Or generate a new unique ID if needed
        #     'analysis_file_name': nwb_file_name,
        #     'processing_notes': '...',
        # })
        pass


@schema
class MicroCTRegistration(SpyglassMixin, dj.Manual):
    definition = """
    # Stores results/params of aligning microCT image data to a target coordinate system
    -> MicroCTImages                    # Link to the source microCT NWB file info
    registration_id: varchar(32)        # Unique ID for this registration instance/parameters
    ---
    -> CoordinateSystem                 # The TARGET coordinate system (e.g., 'allen_ccf_v3_ras_um')

    # --- Registration Parameters ---
    registration_method: varchar(128)   # Algorithmic approach (e.g. 'affine+bspline', 'manual_landmark')
    registration_software: varchar(128) # Software used (e.g. 'ANTs', 'elastix', 'SimpleITK', 'CloudCompare')
    registration_software_version: varchar(64) # Software version (e.g. '2.3.5', '5.0.1')
    registration_params=NULL: blob      # Store detailed parameters (e.g., dict/JSON/YAML content)

    # --- Registration Results ---
    transformation_matrix=NULL: blob    # Store affine matrix (e.g., 4x4 np.array.tobytes())
    warp_field_path=NULL: varchar(512)  # Path to non-linear warp field file (e.g., .nii.gz, .mha)
    registration_quality=NULL: float    # Optional QC metric (e.g., Dice score, landmark error in um)
    registration_time=CURRENT_TIMESTAMP: timestamp # Time this registration entry was created/run
    registration_notes="": varchar(2048) # Specific notes about this registration run
    """
