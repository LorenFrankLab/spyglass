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
   conda env create -f environment.yml
   conda activate nwb_datajoint
   ```
3. Install this repository:
   ```bash
   python setup.py develop
   ```
4. Install [LorenFrankLab/ndx-franklab-novela](https://github.com/LorenFrankLab/ndx-franklab-novela) repository:
   * `ndx-franklab-novela` is needed to read NWB files that make use of the Frank lab extension.
   ```bash
   cd ..
   git clone https://github.com/LorenFrankLab/ndx-franklab-novela.git
   cd ndx_franklab_novela
   python setup.py develop
   ```
5. Install [Labbox-ephys](https://github.com/laboratorybox/labbox-ephys):
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

### Setting up database access
1. Ask Loren or Eric to set up an account for you on the Frank lab database (`lmf-db.cin.ucsf.edu`). Note that you have to be connected to UCSF LAN to access this server.
2. Add the following environment variables (e.g. in `~/.bashrc`). This example assumes that you are interacting with the database on a computer that has mounted `stelmo` at `/stelmo`.
     ```
     export LABBOX_EPHYS_DATA_DIR="/stelmo/nwb/"
     export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/"
     export KACHERY_STORAGE_DIR="/stelmo/nwb/kachery-storage"
     export KACHERY_P2P_CONFIG_DIR="/your-home-directory/.kachery-p2p"
     export DJ_SUPPORT_FILEPATH_MANAGEMENT=true
     export KACHERY_P2P_API_PORT="some-port-number"` (this is optional)
     ```
3. Configure Datajoint:
   * [Change your password](https://github.com/LorenFrankLab/nwb_datajoint/blob/develop_nwbraw/franklab_scripts/franklab_dj_initial_setup.py) for accessing the database and [set up external stores](https://github.com/LorenFrankLab/nwb_datajoint/blob/develop_nwbraw/franklab_scripts/franklab_dj_stores_setup.py). Should need to run these only once.
4. Finally, open up a python console and import `nwb_datajoint` to check that the setup has worked.

### Tutorials
The tutorials are in the form of jupyter notebooks and can be found in the `notebooks` directory. We recommend opening them in the context of `jupyterlab` if you want to do curation. Some of the tutorials we recommend that you start with are:
* `beans.ipynb`: latest and most detailed comments
* `populate_from_NWB_tutorial.ipynb`: for inserting data
* `nwbdj_lfp_tutorial.ipynwb`: to extract LFP  
* `nwbdj_spikeinterface.pynwb`: to run spikesorting

### Notes
* The above instruction assumes that you will be using the `jupyterlab` widget for curation. For curation with the web-based GUI, you must be running `kachery-p2p` and `labbox-ephys` daemons in the background.
  * To run `kachery-p2p`, use the following command after activating the `nwb_datajoint` environment. You should keep this running in the background using a tool like `tmux`.
  ```bash
  kachery-p2p-start-daemon --label franklab --config https://gist.githubusercontent.com/khl02007/b3a092ba3e590946480fb1267964a053/raw/f05eda4789e61980ce630b23ed38a7593f58a7d9/franklab_kachery-p2p_config.yaml
  ```
  * To run `labbox-ephys`, you need to first install [Docker](https://docs.docker.com/get-docker/). Make sure to enable running Docker without `sudo` (see [this](https://docs.docker.com/engine/install/linux-postinstall/)). Test Docker installation with:
  ```bash
  docker run --rm hello-world
  ```
  See instruction [here](https://github.com/laboratorybox/labbox-ephys) for launching `labbox-ephys` Docker container.

### Troubleshooting common problems
* If cannot connect to dj, then downgrade pymysql to 0.9.2
* If cannot write nwb, then downgrade h5py to 2.10.0
* If stall during spikesorting, then install [ml_ms4alg from our fork](https://github.com/LorenFrankLab/ml_ms4alg) and upgrade numpy to 1.19.4
