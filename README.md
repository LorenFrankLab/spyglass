# nwb_datajoint

The Frank lab Datajoint pipeline is designed to facilitate data storage, analysis, and sharing.

## Setup

### Installing packages

1. Clone this repository:

   ```bash
   git clone https://github.com/LorenFrankLab/nwb_datajoint.git
   ```

2. Set up and activate a conda environment from `environment.yml`:

   ```bash
   cd nwb_datajoint
   conda env create -f environment.yml
   conda activate nwb_datajoint
   ```

3. Install this repository:

   ```bash
   # This is preferred to `python setup.py develop`
   # Note the dot (.) at the end
   pip install -e .
   ```

### Setting up database access

1. Ask Loren or Eric to set up an account for you on the Frank lab database (`lmf-db.cin.ucsf.edu`). Note that you have to be connected to UCSF LAN to access this server.

2. Add the following environment variables (e.g. in `~/.bashrc`). This example assumes that you are interacting with the database on a computer that has mounted `stelmo` at `/stelmo`.

     ```bash
     export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/" 
     export SPIKE_SORTING_STORAGE_DIR="/stelmo/nwb/spikesorting" # where output of spike sorting will be sorted
     export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
     export KACHERY_P2P_API_HOST="typhoon"
     export KACHERY_P2P_API_PORT="14747"
     export KACHERY_TEMP_DIR="/stelmo/nwb/tmp"
     ```

3. Configure DataJoint. When your account is created, you will be given a temporary password. You can [change your password](https://github.com/LorenFrankLab/nwb_datajoint/blob/master/franklab_scripts/franklab_dj_initial_setup.py) and [set up external stores](https://github.com/LorenFrankLab/nwb_datajoint/blob/master/franklab_scripts/franklab_dj_stores_setup.py). You should need to run these only once.

Finally, open up a python console and import `nwb_datajoint` to check that the setup has worked.

### Tutorials

The tutorials are in the form of Jupyter Notebooks and can be found in the `notebooks` directory. We strongly recommend opening them in the context of `jupyterlab`. Start with these tutorials:

* `0_intro.ipynb`: general introduction to the database
* `1_spikesorting.ipynb`: how to run spike sorting
