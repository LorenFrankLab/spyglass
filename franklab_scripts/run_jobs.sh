#!/bin/bash

# SETUP:
# 1. Create a conda environment with named 'spyglass' with datajoint installed.
# 2. Edit the path next to 'cd' to point to the root of the spyglass repository.
# 3. Set up the cron job (See README.md)

# Run from the root of the spyglass repository
cd /path/to/spyglass

# Update the spyglass repository
git pull https://github.com/LorenFrankLab/spyglass.git master

# Activate the conda environment
source activate spy || \
    { echo "Error: Could not activate the spyglass environment"; exit 1; }

# Test connection to the database
python -c "import datajoint as dj; dj.conn()" || \
    { echo "Error: Could not connect to the database"; exit 1; }

# Run cleanup script
python franklab_scrips/cleanup.py
