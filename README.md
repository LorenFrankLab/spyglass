# nwb_datajoint

## Setup

### Instructions as of 12-14-2020

1. Clone this repository

  ```bash
  git clone https://github.com/LorenFrankLab/nwb_datajoint.git
  ```

2. Set up and activate the conda environment called "nwb_datajoint"

  ```bash
  cd nwb_datajoint
  conda env create -f environment.yml
  conda activate nwb_datajoint
  ```

3. Install the repo

  ```bash
  python setup.py develop
  cd ..
  ```

4. [Use the datajoint docker installation](https://tutorials.datajoint.io/setting-up/local-database.html)

5. Temporary development package

  Clone the `ndx-franklab-novela` repo from GitHub (conda version too restrictive)

  ```bash
  git clone https://github.com/NovelaNeuro/ndx-franklab-novela.git
  cd ndx-franklab-novela
  ```

  Open `setup.py` in your favorite text editor and remove the line `hdmf==1.6.4` on line 36

  ```bash
  vim setup.py
  ```

  Install the repo

  ```bash
  python setup.py develop
  cd ..
  ```

6. Temporary for spike sorting: Install and run the development version of [labbox-ephys](https://github.com/laboratorybox/labbox-ephys/)

  ```bash
  git clone https://github.com/laboratorybox/labbox-ephys.git
  cd labbox-ephys/python
  python setup.py develop
  cd ../..
  ```

7. Install and run the development version of [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)

  ```bash
  pip uninstall pynwb
  git clone --recurse-submodules https://github.com/NeurodataWithoutBorders/pynwb.git
  cd pynwb
  python setup.py develop
  cd ..
  ```

8. Run `populate_from_NWB_tutorial.ipynb` from the `notebooks` directory

9. Other examples:
  - `nwbdj_lfp_tutorial.ipynb` to extract LFP
  - `nwbdj_spikeinterface_test.ipynb` to set up and run spike sorting
