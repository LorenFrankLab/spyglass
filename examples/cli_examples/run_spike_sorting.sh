#!/bin/bash

thisdir=`dirname "$0"`
spyglass run-spike-sorting $thisdir/parameters.yaml

spyglass list-spike-sortings RN2_20191110_.nwb
