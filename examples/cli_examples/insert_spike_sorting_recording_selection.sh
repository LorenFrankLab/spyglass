#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-spike-sorting-recording-selection $thisdir/spikesortingrecordingselection.yaml

spyglass list-spike-sorting-recording-selections RN2_20191110_.nwb