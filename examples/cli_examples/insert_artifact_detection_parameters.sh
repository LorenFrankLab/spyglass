#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-artifact-detection-parameters $thisdir/artifactdetectionparameters_default.yaml
spyglass insert-artifact-detection-parameters $thisdir/artifactdetectionparameters_none.yaml

spyglass list-artifact-detection-parameters
