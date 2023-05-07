import spyglass as sg
import os

# ignore datajoint+jupyter async warnings
import warnings

os.environ["SPYGLASS_BASE_DIR"]="/stelmo/nwb"
bd = os.environ["SPYGLASS_BASE_DIR"]
os.environ["SPYGLASS_RECORDING_DIR"] = bd + "/recording"
os.environ["SPYGLASS_SORTING_DIR"]= bd + "/sorting"
os.environ["SPYGLASS_VIDEO_DIR"]= bd + "/video"
os.environ["SPYGLASS_WAVEFORMS_DIR"]= bd + "/waveforms"
os.environ["SPYGLASS_TEMP_DIR"]= bd + "/tmp/spyglass"
os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT']="True"


# import tables so that we can call them easily
from spyglass.common import AnalysisNwbfile


def main():
    AnalysisNwbfile().nightly_cleanup()

    
if __name__ == '__main__':
    main()

