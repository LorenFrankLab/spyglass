#!/bin/bash

thisdir=`dirname "$0"`
spyglass create-spike-sorting-recording-view $thisdir/parameters.yaml "$@"
