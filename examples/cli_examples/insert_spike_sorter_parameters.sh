#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-spike-sorter-parameters $thisdir/spikesorterparameters_default.yaml

spyglass list-spike-sorter-parameters
