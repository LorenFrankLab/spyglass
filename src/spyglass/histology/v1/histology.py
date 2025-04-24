import datajoint as dj

from spyglass.common import (  # noqa: F401
    AnalysisNwbfile,
    CoordinateSystem,
    LabMember,
    Subject,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("histology_v1")


@schema
class Histology(SpyglassMixin, dj.Manual):
    definition = """
    # Represents a single histology preparation for a subject
    -> Subject
    histology_id: varchar(32)           # User-defined ID (e.g., 'probe_track_run1')
    ---
    prep_date=NULL: date                # Optional: Date of tissue preparation
    slice_orientation: enum(            # Orientation of sections
        "coronal",
        "sagittal",
        "horizontal",
        "other"
    )
    slice_thickness: float              # (um) Thickness of sections
    mounting_medium=NULL: varchar(128)  # e.g., 'DPX', 'Fluoromount-G'
    experiment_purpose: varchar(1024)   # e.g., 'Probe track recovery', 'Anatomical ref'
    notes="": varchar(2048)             # Optional general notes about the preparation
    -> [nullable] LabMember.proj(histology_experimenter='user_name') # Optional: who did the prep?

    # --- Data Source ---
    output_format='TIFF stack': varchar(64) # Format of raw image data from scanner
    raw_scan_path: varchar(512)         # Path to the raw output (e.g., folder containing TIFF stack)
    """

    class HistologyStain(SpyglassMixin, dj.Part):
        definition = """
        # Details of specific stains used in a histology preparation
        -> Histology
        stain_index: tinyint unsigned    # Use index for multiple stains (0, 1, 2...)
        ---
        identified_feature: varchar(128) # Biological target/marker (e.g., 'Nissl Bodies', 'ChR2-tdTomato+', 'ProbeTrack_DiI')
        visualization_agent: varchar(128)# Method/molecule making feature visible (e.g., 'Cresyl Violet', 'Native tdTomato', 'DiI', 'Alexa 488')
        stain_type: enum(                # Type of staining method used
            "immunohistochemistry",
            "genetic_marker",
            "tracer",
            "anatomical",
            "histochemical",
            "in_situ_hybridization",
            "other"
        ) = "other"
        stain_protocol_name=NULL: varchar(128) # Optional: name of the protocol used
        antibody_details=NULL: varchar(255)    # Optional: specific antibody info (e.g. company, cat#, lot#)
        stain_notes="": varchar(1024)          # Optional notes about this specific stain (e.g., concentration)
        """


@schema
class HistologyImages(SpyglassMixin, dj.Computed):
    definition = """
    # Links Histology info to the Analysis NWB file containing the image data
    -> Histology
    images_id: varchar(32)              # User-defined ID for these images (e.g., histology_id)
    ---
    -> AnalysisNwbfile                  # Link to the NWB file storing image data
    processing_time=CURRENT_TIMESTAMP: timestamp # Timestamp of NWB file creation
    # --- Image Acquisition/Processing Details ---
    color_to_stain=NULL: blob           # Mapping channel colors to stain features (e.g., {'DAPI': 'blue', 'GFAP': 'green'})
    pixel_size_x: float                 # (um) Pixel size in X dimension after processing/scaling
    pixel_size_y: float                 # (um) Pixel size in Y dimension after processing/scaling
    pixel_size_z: float                 # (um) Pixel size in Z dimension (often slice_thickness or scan step)
    objective_magnification: float      # Magnification of the objective lens (e.g., 20 for 20x)
    image_modality: enum(               # Modality of the microscopy
        "fluorescence",
        "brightfield",
        "other"
    )
    processing_notes="": varchar(1024)  # Notes on image processing applied before/during NWB creation
    """

    # Ensure this key is unique for HistologyImages entries
    # key_source = Histology

    def make(self, key: dict) -> None:
        """
        Populate HistologyImages table.
        This method should:
        1. Find the raw image data using `raw_scan_path` from the `Histology` table.
        2. Process the images as needed (e.g., stitching, scaling).
        3. Create an NWB file containing the processed image stack (e.g., using `pynwb.image.ImageSeries`).
           - Store relevant metadata (pixel sizes, objective, modality, etc.) within the NWB file.
        4. Create an `AnalysisNwbfile` entry for the new NWB file.
        5. Insert the key, including the `analysis_file_name` from AnalysisNwbfile,
           along with image metadata like `pixel_size_*`, `color_to_stain`, etc., into this table.
        """
        logger.info(f"Populating HistologyImages for key: {key}")
        # Placeholder: Replace with actual NWB creation and insertion logic
        # Example steps (conceptual):
        # 1. histology_entry = (Histology & key).fetch1()
        # 2. raw_path = histology_entry['raw_scan_path']
        # 3. image_data, metadata = process_histology_images(raw_path) # Your function
        # 4. nwb_file_name = f"{key['subject_id']}_{key['histology_id']}_images.nwb"
        # 5. nwb_file_path = AnalysisNwbfile().create(nwb_file_name)
        # 6. create_histology_nwb(nwb_file_path, image_data, metadata) # Your function
        # 7. AnalysisNwbfile().add(key['subject_id'], nwb_file_name)
        # 8. self.insert1({
        #     **key,
        #     'images_id': key['histology_id'], # Or generate a new unique ID if needed
        #     'analysis_file_name': nwb_file_name,
        #     'pixel_size_x': metadata['pixel_size_x'],
        #     # ... other metadata fields ...
        # })
        pass


@schema
class HistologyRegistration(SpyglassMixin, dj.Manual):
    definition = """
    # Stores results/params of aligning histology image data to a target coordinate system
    -> HistologyImages                  # Link to the source histology NWB file info
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
