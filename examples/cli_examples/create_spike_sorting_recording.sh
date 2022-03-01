#!/bin/bash

thisdir=`dirname "$0"`
spyglass create-spike-sorting-recording $thisdir/spikesortingrecordingselection.yaml

spyglass list-spike-sorting-recordings RN2_20191110_.nwb