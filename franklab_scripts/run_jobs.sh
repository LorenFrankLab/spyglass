#!/bin/bash

# Run from the root of the spyglass repository
cd ..

# Update the spyglass repository
git pull https://github.com/LorenFrankLab/spyglass.git master

# Activate the conda environment
source activate spyglass || \
    { echo "Error: Could not activate the spyglass environment"; exit 1; }

# Test connection to the database
python -c "import datajoint as dj; dj.conn()" || \
    { echo "Error: Could not connect to the database"; exit 1; }

# Run cleanup script
python franklab_scrips/cleanup.py
