#!/bin/bash

thisdir=`dirname "$0"`
spyglass insert-lab-member $thisdir/labmember.yaml

spyglass list-lab-members
