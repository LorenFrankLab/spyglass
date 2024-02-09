#!/bin/bash

thisdir=`dirname "$0"`
spyglass create-spike-sorting-view $thisdir/parameters.yaml "$@"
