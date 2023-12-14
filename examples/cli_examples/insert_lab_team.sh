#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-lab-team $thisdir/team.yaml

spyglass list-lab-teams
