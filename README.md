
# nwb_datajoint

## Setup
### Instructions as of Jan 10, 2021

1. Clone this repository
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
4. Install [LorenFrankLab/ndx-franklab-novela](https://github.com/LorenFrankLab/ndx-franklab-novela) repo
   ```bash
   cd ..
   git clone https://github.com/LorenFrankLab/ndx-franklab-novela.git
   cd ndx_franklab_novela
   python setup.py develop
   ```
5. Install [Labbox-ephys](https://github.com/laboratorybox/labbox-ephys)
   ```bash
   cd ..
   git clone https://github.com/laboratorybox/labbox-ephys.git
   cd labbox-ephys/python
   python setup.py develop
   ```
6. Install [Docker](https://docs.docker.com/get-docker/)  
   * Enable running docker without `sudo` (see [this](https://docs.docker.com/engine/install/linux-postinstall/); if running on lab server and don't have `sudo` privilege, ask Loren)
   * Test with:
   ```bash
   docker run --rm hello-world
   ```
7. Database access
   * Ask Loren for mysql access to `lmf-db.cin.ucsf.edu`
   * If you're looking to practice (recommended if starting out), you can use your own database (see [this](https://tutorials.datajoint.io/setting-up/get-database.html))
8. Configure Datajoint
   * [Change your password](https://github.com/LorenFrankLab/nwb_datajoint/blob/develop_nwbraw/franklab_scripts/franklab_dj_initial_setup.py) for accessing Datajoint database and [set up external stores](https://github.com/LorenFrankLab/nwb_datajoint/blob/develop_nwbraw/franklab_scripts/franklab_dj_stores_setup.py). Should need to run these only once.
9. Open up a python console and import `nwb_datajoint` to check that installation has worked.
10. Run `populate_from_NWB_tutorial.ipynb` from the notebooks directory.
11. Other examples:  
	 * `nwbdj_lfp_tutorial.ipynwb` to extract LFP  
	 * `nwbdj_spikeinterface.pynwb` to setup and run spikesorting

### Notes
* Add environment variables (e.g. in `~/.bashrc`); note that these assume that you're interacting with DJ database from one of the lab's servers (e.g. virgas)
  ```
  export LABBOX_EPHYS_DATA_DIR="/stelmo/nwb/"
  export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/"
  export KACHERY_STORAGE_DIR="/stelmo/nwb/kachery-storage"
  export KACHERY_P2P_CONFIG_DIR="/your-home-directory/.kachery-p2p"
  export DJ_SUPPORT_FILEPATH_MANAGEMENT=true
  export KACHERY_P2P_API_PORT="some-port-number"` (this is optional)
  ```
* For curation with the web GUI, you must be running `kachery-p2p` and `labbox-ephys` daemons in the background
  * kachery-p2p:
  ```bash
  kachery-p2p-start-daemon --label franklab --config https://gist.githubusercontent.com/khl02007/b3a092ba3e590946480fb1267964a053/raw/f05eda4789e61980ce630b23ed38a7593f58a7d9/franklab_kachery-p2p_config.yaml
  ```
  * labbox: see instruction [here](https://github.com/laboratorybox/labbox-ephys) for launching labbox-ephys docker container
  * if running jupyter widget, see instruction [here](https://github.com/laboratorybox/labbox-ephys/blob/master/doc/labbox_ephys_widgets_jp.md)

### Troubleshooting common problems
* If cannot connect to dj, then downgrade pymysql to 0.9.2
* If cannot write nwb, then downgrade h5py to 2.10.0
* If stall during spikesorting, then install [ml_ms4alg from our fork](https://github.com/LorenFrankLab/ml_ms4alg) and upgrade numpy to 1.19.4
* If cannot get waveforms, downgrade following packages to these versions:  
spikecomparison           0.3.0        
spikeextractors           0.9.1        
spikefeatures             0.1.1        
spikeinterface            0.10.0       
spikemetrics              0.2.2        
spikesorters              0.4.2        
spiketoolkit              0.7.0        
spikewidgets              0.5.0        
