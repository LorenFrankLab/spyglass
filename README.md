[![Import test](https://github.com/LorenFrankLab/spyglass/actions/workflows/workflow.yml/badge.svg)](https://github.com/LorenFrankLab/spyglass/actions/workflows/workflow.yml)
[![Nightly Test](https://github.com/LorenFrankLab/spyglass/actions/workflows/nightly.yml/badge.svg)](https://github.com/LorenFrankLab/spyglass/actions/workflows/nightly.yml)

# spyglass

`spyglass` is a data analysis framework that facilitates the storage, analysis, visualization, and sharing of neuroscience data to support reproducible research. It is designed to be interoperable with the NWB format and integrates open-source tools into a coherent framework.

## Setup

### Installing packages

1. Currently `spyglass` is under active development. We encourage installation via cloning this repo. To do so:

   ```bash
   git clone https://github.com/LorenFrankLab/spyglass.git
   ```

2. Set up and activate a conda environment from `environment.yml`:

   ```bash
   cd spyglass
   conda env create -f environment.yml
   conda activate spyglass
   ```

3. Install this repository:

   ```bash
   pip install -e .
   ```

### Setting up Frank lab database access

Step 1 below only applies if you are a member of Frank lab at UCSF. If you're not affiliated with UCSF or if you are just looking to try out `spyglass`, then you will need to set up a different MySQL server and add your own account to it. For example, you can set up your own local server with a Docker image of a MySQL server configured for Datajoint (see [instructions](https://tutorials.datajoint.org/setting-up/local-database.html) and/or [tutorial notebook](./notebooks/docker_mysql_tutorial.ipynb)).

1. Ask Loren or Eric to set up an account for you on the Frank lab `DataJoint` database. Note that you have to be connected to UCSF LAN to access this server.

2. Add the following environment variables (e.g. in `~/.bashrc`). We assume that you are interacting with the database on a computer that has mounted `stelmo` at `/stelmo` (if the mount point is different, change accordingly). For this to take effect, log out and log back in, or run `source ~/.bashrc` in the terminal.

     ```bash
     export SPYGLASS_BASE_DIR="/stelmo/nwb/"
     export SPYGLASS_RECORDING_DIR="$SPYGLASS_BASE_DIR/recording"
     export SPYGLASS_SORTING_DIR="$SPYGLASS_BASE_DIR/sorting"
     export SPYGLASS_WAVEFORMS_DIR="$SPYGLASS_BASE_DIR/waveforms"
     export SPYGLASS_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
     export KACHERY_CLOUD_DIR="$SPYGLASS_BASE_DIR/.kachery-cloud"
     export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
     ```
     Note that a local SPYGLASS_TEMP_DIR (e.g. one on your machine) will speed up spike sorting, but make sure it has enough free space (ideally at least 500GB)
 
3. Configure `DataJoint`. To connect to the database, you need to specify information such as the hostname and the port. You should also change your password from the temporary one you were given. Go to the config directory, and run [`dj_config.py`](https://github.com/LorenFrankLab/spyglass/blob/master/config/dj_config.py) in the terminal with your username:

    ```bash
    cd config # change to the config directory
    python dj_config.py <username> # run the configuration script
    ```

Finally, open up a python console (e.g. run `ipython` from terminal) and import `spyglass` to check that the installation has worked.

## Tutorials

The tutorials for `spyglass` is currently in the form of Jupyter Notebooks and can be found in the `notebooks` directory. We strongly recommend opening them in the context of `jupyterlab`.

* `0_intro.ipynb`: general introduction to the database
* `1_spikesorting.ipynb`: how to run spike sorting
* `2_curation.ipynb`: how to curate spike sorted data
