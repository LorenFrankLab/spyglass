from pathlib import Path

import datajoint as dj

from spyglass.common.common_lab import LabMember  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_subject import Subject  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

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


# Store the image file path and modality?
# Store the scale and color_to_stain mapping?
# Store in analysis nwb file?
class HistologyImage(dj.Manual):
    definition = """
   -> Histology
   image_id: int unsigned auto_increment
   ---
   image_file_path: varchar(255) # Path to image file
   image_modality: enum("brightfield", "epifluorescence", "confocal", "slide_scanner", "other")

   """


# # Example: Electrode Localization Element might have:
# class ElectrodePosition(dj.Computed):
#     definition = """
#    -> Electrode
#    -> Histology
#    ---
#    # Coordinates in some defined space (histology image, atlas, etc.)
#    pos_x: float
#    pos_y: float
#    pos_z: float
#    -> BrainRegion # Final assigned region
#    """
