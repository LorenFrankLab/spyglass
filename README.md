# nwb_datajoint

## Setup 
to date notebooks 
1. Clone repository
2. Setup conda environment
```conda env create -f environment.yml```
3. Install the repo ```python setup.py develop``` 
4. [Use the datajoint docker installation](https://tutorials.datajoint.io/setting-up/local-database.html)
5. Run notebooks/populate_DJ_from_NWB_raw.ipynb to import one or more NWB files
into the datajoint database
6. As another example, after importing, run notebooks/NWB_DJ_LFP.ipynwb to extract LFP 
