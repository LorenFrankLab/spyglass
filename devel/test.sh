#!/bin/bash

# Instructions:
# * pip install pytest
# * You must have a fresh (empty) datajoint mysql database running on port 3306
#   with the below user/pass
# * If you are in a devcontainer, you probably want the mysql database running
#   outside the container, because the devcontainer does not have docker
# * Run this script from the root directory of nwb_datajoint: i.e., the command
#   should be devel/test.sh

export DJ_HOST=localhost:3306
export DJ_USER=root
export DJ_PASS=tutorial
pytest --pyargs nwb_datajoint -s