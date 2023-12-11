#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-spike-sorting-preprocessing-parameters $thisdir/spikesortingpreprocessingparameters.yaml

spyglass list-spike-sorting-preprocessing-parameters
