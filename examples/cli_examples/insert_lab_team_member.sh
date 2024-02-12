#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-lab-team-member $thisdir/labteammember.yaml

spyglass list-lab-team-members LorenLab
