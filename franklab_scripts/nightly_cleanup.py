!#/


import os
import numpy as np

import nwb_datajoint as nd

# ignore datajoint+jupyter async warnings
import warnings

from nwb_datajoint.common.common_spikesorting import AutomaticCurationSpikeSorting, UnitInclusionParameters
from nwb_datajoint.decoding.clusterless import MarkParameters, UnitMarkParameters, UnitMarks
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=ResourceWarning)
os.environ['SPIKE_SORTING_STORAGE_DIR']="/stelmo/nwb/spikesorting"


# import tables so that we can call them easily
from nwb_datajoint.common import (RawPosition, HeadDir, Speed, LinPos, StateScriptFile, VideoFile,
                                  DataAcquisitionDevice, CameraDevice, Probe,
                                  DIOEvents,
                                  ElectrodeGroup, Electrode, Raw, SampleCount,
                                  LFPSelection, LFP, LFPBandSelection, LFPBand,
                                  SortGroup, SpikeSorting, SpikeSorter, SpikeSorterParameters,
                                  SpikeSortingWaveformParameters, SpikeSortingArtifactParameters,
                                  SpikeSortingParameters,SpikeSortingMetrics, CuratedSpikeSorting,
                                  AutomaticCurationSpikeSorting, AutomaticCurationSpikeSortingParameters, AutomaticCurationParameters,
                                  FirFilter,
                                  IntervalList, SortInterval,
                                  Lab, LabMember, LabTeam, Institution,
                                  BrainRegion,
                                  SensorData,
                                  Session, ExperimenterList,
                                  Subject,
                                  Task, TaskEpoch,
                                  Nwbfile, AnalysisNwbfile, NwbfileKachery, AnalysisNwbfileKachery)


def main():
    
    AnalysisNwbfile().nightly_cleanup()
    SpikeSorting().nightly_cleanup()

    
if __name__ == '__main__':
    main()

