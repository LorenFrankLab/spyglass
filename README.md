# nwb_datajoint

The Frank lab Datajoint pipeline facilitates the storage, analysis, and sharing of neuroscience data to support reproducible research. It integrates existing open-source projects into a coherent framework so that they can be easily used.

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
   # to use the package
   pip install nwb_datajoint
   # if you're a developer:
   pip install -e .
   ```

### Setting up database access

1. Ask Loren or Eric to set up an account for you on the Frank lab database. Note that you have to be connected to UCSF LAN to access this server.

   > If you're not affiliated with UCSF or if you are just looking to try out `nwb_datajoint`, then you will need to set up a different MySQL server. For example, you can set up your own local server with a Docker image of a MySQL server configured for Datajoint (see instructions [here](https://tutorials.datajoint.io/setting-up/local-database.html)

2. Add the following environment variables (e.g. in `~/.bashrc`). This example assumes that you are interacting with the database on a computer that has mounted `stelmo` at `/stelmo` (if the mount location is different, change accordingly).

     ```bash
     export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/" 
     export SPIKE_SORTING_STORAGE_DIR="/stelmo/nwb/spikesorting"
     export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
     export KACHERY_P2P_API_HOST="typhoon"
     export KACHERY_P2P_API_PORT="14747"
     export KACHERY_TEMP_DIR="/stelmo/nwb/tmp"
     ```

   > If you're not connected to UCSF network, then you will have to run your own `kachery-p2p` daemon for curating spike sorting. Consult the guide [here](https://github.com/flatironinstitute/kachery-p2p/blob/main/doc/setup_and_installation.md).

3. Configure DataJoint. To connect to the Datajoint database, we have to specify information about it such as the hostname and the port. You should also change your password from the temporary one you were given. To do so, open up [`dj_config.py`](https://github.com/LorenFrankLab/nwb_datajoint/blob/master/config/dj_config.py), change the user name, and run it.

   > Again, if you're using your own MySQL server, then you may need to change the other settings as well.

Finally, open up a python console and import `nwb_datajoint` to check that the installation has worked.

## Tutorials

The tutorials for `nwb_datajoint` is currently in the form of Jupyter Notebooks and can be found in the `notebooks` directory. We strongly recommend opening them in the context of `jupyterlab`.

* `0_intro.ipynb`: general introduction to the database
* `1_spikesorting.ipynb`: how to run spike sorting
