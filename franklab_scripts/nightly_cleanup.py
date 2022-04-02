#!/usr/bin/env python

import os
import numpy as np

import spyglass as nd

# ignore datajoint+jupyter async warnings
import warnings

from spyglass.decoding.clusterless import MarkParameters, UnitMarkParameters, UnitMarks
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=ResourceWarning)
os.environ['SPIKE_SORTING_STORAGE_DIR']="/stelmo/nwb/spikesorting"


# import tables so that we can call them easily
from spyglass.common import (RawPosition, HeadDir, Speed, LinPos, StateScriptFile, VideoFile,
                                  DataAcquisitionDevice, CameraDevice, Probe,
                                  DIOEvents,
                                  ElectrodeGroup, Electrode, Raw, SampleCount,
                                  LFPSelection, LFP, LFPBandSelection, LFPBand,
                                  IntervalList,
                                  Lab, LabMember, LabTeam, Institution,
                                  BrainRegion,
                                  SensorData,
                                  Session, ExperimenterList,
                                  Subject,
                                  Task, TaskEpoch,
                                  Nwbfile, AnalysisNwbfile, NwbfileKachery, AnalysisNwbfileKachery)
from spyglass.spikesorting import SortGroup, SortInterval, SpikeSorting, SpikeSorterParameters

def main():
    
    AnalysisNwbfile().nightly_cleanup()
    SpikeSorting().nightly_cleanup()

    
if __name__ == '__main__':
    main()

