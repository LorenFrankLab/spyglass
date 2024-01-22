#!/usr/bin/env python

import os

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# NOTE: "SPIKE_SORTING_STORAGE_DIR" -> "SPYGLASS_SORTING_DIR"
os.environ["SPYGLASS_SORTING_DIR"] = "/stelmo/nwb/spikesorting"


# import tables so that we can call them easily
from spyglass.common import AnalysisNwbfile
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.spikesorting import SpikeSorting


def main():
    AnalysisNwbfile().nightly_cleanup()
    SpikeSorting().nightly_cleanup()
    DecodingOutput().cleanup()


if __name__ == "__main__":
    main()
