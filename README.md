[![Import test](https://github.com/LorenFrankLab/nwb_datajoint/actions/workflows/workflow.yml/badge.svg)](https://github.com/LorenFrankLab/nwb_datajoint/actions/workflows/workflow.yml)
[![Nightly Test](https://github.com/LorenFrankLab/nwb_datajoint/actions/workflows/nightly.yml/badge.svg)](https://github.com/LorenFrankLab/nwb_datajoint/actions/workflows/nightly.yml)

# nwb_datajoint

The Frank lab Datajoint pipeline facilitates the storage, analysis, and sharing of neuroscience data to support reproducible research. It is designed to be interoperable with the NWB format and integrates open-source tools into a coherent framework.

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
   pip install -e .
   ```

### Setting up database access

1. Ask Loren or Eric to set up an account for you on the Frank lab `datajoint` database. Note that you have to be connected to UCSF LAN to access this server.

   > If you're not affiliated with UCSF or if you are just looking to try out `nwb_datajoint`, then you will need to set up a different MySQL server. For example, you can set up your own local server with a Docker image of a MySQL server configured for Datajoint (see [instructions](https://tutorials.datajoint.io/setting-up/local-database.html) and/or [tutorial notebook](./notebooks/docker_mysql_tutorial.ipynb)).

2. Add the following environment variables (e.g. in `~/.bashrc`). We assumes that you are interacting with the database on a computer that has mounted `stelmo` at `/stelmo` (if the mount location is different, change accordingly). For this to take effect, log out and log back in, or run `source ~/.bashrc` in the terminal.

     ```bash
     export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/"
     export NWB_DATAJOINT_RECORDING_DIR="/stelmo/nwb/recording"
     export NWB_DATAJOINT_SORTING_DIR="/stelmo/nwb/sorting"
     export NWB_DATAJOINT_WAVEFORMS_DIR="/stelmo/nwb/waveforms"
     export NWB_DATAJOINT_TEMP_DIR="/stelmo/nwb/tmp"
     export KACHERY_DAEMON_HOST="typhoon"
     export KACHERY_DAEMON_PORT="14747"
     export KACHERY_STORAGE_DIR="/stelmo/nwb/kachery-storage"
     export KACHERY_TEMP_DIR="/stelmo/nwb/tmp"
     export FIGURL_CHANNEL="franklab2"
     export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
     ```

     Note that a local NWB_DATAJOINT_TEMP_DIR (e.g. one on your machine) will speed up spike sorting, but make sure it has enough free space (ideally at least 500GB)

3. Check if you have access to the `kachery` daemon. Open up a terminal, activate the conda environment, and type

   ```bash
   kachery-client info
   ```

   If it says you do not have access to the daemon, ask Kyu to add your username to the `kachery-users` group.
   > If you're running your own `kachery` daemon, then this doesn't apply.
  
4. Configure DataJoint. To connect to the Datajoint database, we have to specify information about it such as the hostname and the port. You should also change your password from the temporary one you were given. Go to the config directory, and run [`dj_config.py`](https://github.com/LorenFrankLab/nwb_datajoint/blob/master/config/dj_config.py) in the terminal with your username:

    ```bash
    cd config # change to the config directory
    python dj_config.py <username> # run the configuration script
    ```

   > Again, if you're using your own MySQL server, then you may need to change the other settings as well.

Finally, open up a python console (e.g. run `ipython` from terminal) and import `nwb_datajoint` to check that the installation has worked.

## Tutorials

The tutorials for `nwb_datajoint` is currently in the form of Jupyter Notebooks and can be found in the `notebooks` directory. We strongly recommend opening them in the context of `jupyterlab`.

* `0_intro.ipynb`: general introduction to the database
* `1_spikesorting.ipynb`: how to run spike sorting
* `2_curation.ipynb`: how to curate spike sorted data
