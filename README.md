# nwb_datajoint
The Frank lab Datajoint database is designed to facilitate data storage, analysis, and sharing.

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
   python setup.py develop
   ```
4. Install [Labbox-ephys](https://github.com/laboratorybox/labbox-ephys):
   * `labbox-ephys` is used for visualizing and curating spike sorting results. We will install the package as well as the `jupyterlab` widget.
   ```bash
   cd ..
   git clone https://github.com/laboratorybox/labbox-ephys.git
   cd labbox-ephys
   # Install python packages
   pip install -e ./python
   pip install -e jupyterlab/labbox_ephys_widgets_jp
   # Install jupyterlab extension
   jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
   jupyter labextension install jupyterlab/labbox_ephys_widgets_jp
   ```
5. Install latest versions of spikeextractors and spiketoolkit packages.
   * For now we will install these packages locally, because some of the pre-release features are useful for us.
   ```bash
   # Install spikeextractors
   cd ..
   git clone https://github.com/SpikeInterface/spikeextractors.git
   cd spikeextractors
   pip install -e .
   # Install spiketoolkit
   cd ..
   git clone https://github.com/SpikeInterface/spiketoolkit.git
   cd spiketoolkit
   pip install -e .
   ```

### Setting up database access
1. Ask Loren or Eric to set up an account for you on the Frank lab database (`lmf-db.cin.ucsf.edu`). Note that you have to be connected to UCSF LAN to access this server.
2. Add the following environment variables (e.g. in `~/.bashrc`). This example assumes that you are interacting with the database on a computer that has mounted `stelmo` at `/stelmo`.
     ```
     export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/"
     export LABBOX_EPHYS_DATA_DIR="/stelmo/nwb/"
     export KACHERY_STORAGE_DIR="/stelmo/nwb/kachery-storage"
     export KACHERY_P2P_CONFIG_DIR="/your-home-directory/.kachery-p2p"
     export KACHERY_P2P_API_PORT="some-port-number"`  # (optional)
     export DJ_SUPPORT_FILEPATH_MANAGEMENT=true
     ```
3. Configure DataJoint:
   * When your account is created, you will be given a temporary password. You can [change your password](https://github.com/LorenFrankLab/nwb_datajoint/blob/master/franklab_scripts/franklab_dj_initial_setup.py) and [set up external stores](https://github.com/LorenFrankLab/nwb_datajoint/blob/master/franklab_scripts/franklab_dj_stores_setup.py). You should need to run these only once.
4. Finally, open up a python console and import `nwb_datajoint` to check that the setup has worked.

### Tutorials
The tutorials are in the form of Jupyter Notebooks and can be found in the `notebooks` directory. We recommend opening them in the context of `jupyterlab` if you want to do curation. Some of the tutorials we recommend that you start with are:
* `0_intro.ipynb`: general introduction to the database
* `1_spikesorting.ipynb`: how to run spike sorting

### Notes
* For curation, you must be running `kachery-p2p` daemon in the background. This manages file storage an lookup. To run `kachery-p2p`, use the following command after activating the `nwb_datajoint` environment. Keep this running in the background using a tool like `tmux`.
  ```bash
  kachery-p2p-start-daemon --label franklab --config https://gist.githubusercontent.com/khl02007/b3a092ba3e590946480fb1267964a053/raw/f05eda4789e61980ce630b23ed38a7593f58a7d9/franklab_kachery-p2p_config.yaml
  ```
* If you want to use the web-GUI, you will run `labbox-ephys` using the launcher. To do so you need to first install [Docker](https://docs.docker.com/get-docker/). Make sure to enable running Docker without `sudo` (see [this](https://docs.docker.com/engine/install/linux-postinstall/)). Test Docker installation with:
  ```bash
  docker run --rm hello-world
  ```
  Then follow the instruction [here](https://github.com/laboratorybox/labbox-ephys) for launching `labbox-ephys` Docker container.

### Troubleshooting common problems
* If you have an error writing NWB files, then downgrade h5py to 2.10.0
* If Mountainsort4 stalls, then install [ml_ms4alg from our fork](https://github.com/LorenFrankLab/ml_ms4alg) and upgrade numpy to 1.19.4
* If you have an error installing labbox-ephys Jupyter widget, run `export NODE_OPTIONS="--max-old-space-size=8192` and then try again.
