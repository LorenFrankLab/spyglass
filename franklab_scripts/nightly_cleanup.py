#!/usr/bin/env python

import os

# ignore datajoint+jupyter async warnings
import warnings

import numpy as np
import spyglass as nd
from spyglass.decoding.clusterless import MarkParameters, UnitMarkParameters, UnitMarks

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
os.environ["SPIKE_SORTING_STORAGE_DIR"] = "/stelmo/nwb/spikesorting"


# import tables so that we can call them easily
from spyglass.common import (
    AnalysisNwbfile,
)
from spyglass.spikesorting import (
    SpikeSorting,
)


def main():

    AnalysisNwbfile().nightly_cleanup()
    SpikeSorting().nightly_cleanup()


if __name__ == "__main__":
    main()
