
# nwb_datajoint

This code allows for import of an NWB file into a matching set of DataJoint schema

Old code for importing older Frank Lab data is in the develop branch

Newer code for importing from new NWB data files is in the develop_nwbraw branch

## Setup - 
1. Clone this repository
2. Setup conda environment using the file in the root nwb_datajoint directory
```conda env create -f environment.yml```
3. Install this repo ```python setup.py develop``` 
4. Try running the notebook/populate_from_NWB.ipynb
