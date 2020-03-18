# nwb_datajoint

## Setup
1. Clone repository
2. Setup conda environment
```conda env create -f environment.yml```
3. Install the repo ```python setup.py develop```
4. [Use the datajoint docker installation](https://tutorials.datajoint.io/setting-up/local-database.html)
5. Run the docker image.
6. Copy `bon_03.nwb` and `bon_04.nwb` into the `/data` folder
7. Start jupyter lab ```jupyter lab```
8. open `notebooks/run_pipeline.ipynb`

Note that there are two branches:
+ `master`: datajoint and nwb work using the path file store.
+ `use-object-ids`: datajoint and nwb work using object ids.
+ `develop`: Loren's datajoint code.
