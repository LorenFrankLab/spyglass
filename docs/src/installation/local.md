# Local Installation

## Clone Repository

For local development, first pull down the code base -

```bash
git clone https://github.com/LorenFrankLab/spyglass.git
```

Set up and activate a conda environment from environment.yml:

```bash
cd spyglass
conda env create -f environment.yml
conda activate spyglass
```

Install this repository:

```bash
pip install -e .
```

## Additional Packages

Some of the pipeline requires installation of additional packages. For example,
the spike sorting pipeline relies on `spikeinterface`. We recommend installing
it directly from the GitHub repo:

```bash
pip install spikeinterface[full,widgets]
```

You may also need to install individual sorting algorithms. For example, Loren
Frank's lab at UCSF typically uses `mountainsort4`:

```bash
pip install mountainsort4
```

WARNING: If you are on an M1 Mac, you need to install `pyfftw` via `conda`
BEFORE installing `ghostipy`:

```bash
conda install -c conda-forge pyfftw
```

The LFP pipeline uses `ghostipy`:

```bash
pip install ghostipy
```

## Setting up database access

1. To use `spyglass`, you need to have access to a MySQL database. If your lab
   already administers a database, connect to it by setting
   [DataJoint](https://www.datajoint.org/) configurations. If you want to run
   your own database, consult instructions in [datajoint tutorial](https://tutorials.datajoint.org/setting-up/get-database.html)
   and/or [our tutorial notebook](../notebooks/docker_mysql_tutorial.ipynb).

2. Add the following environment variables (e.g. in `~/.bashrc`). The following
   are specific to Frank lab so you may want to change `SPYGLASS_BASE_DIR`.

   ```bash
   export SPYGLASS_BASE_DIR="/stelmo/nwb"
   export SPYGLASS_RECORDING_DIR="$SPYGLASS_BASE_DIR/recording"
   export SPYGLASS_SORTING_DIR="$SPYGLASS_BASE_DIR/sorting"
   export SPYGLASS_VIDEO_DIR="$SPYGLASS_BASE_DIR/video"
   export SPYGLASS_WAVEFORMS_DIR="$SPYGLASS_BASE_DIR/waveforms"
   export SPYGLASS_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp/spyglass"
   export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
   ```

   Note that a local `SPYGLASS_TEMP_DIR` (e.g. one on your machine) will speed
   up spike sorting, but make sure it has enough free space (ideally at least
   500GB)

   Before proceeding, run -

   ```bash
   source ~/.bashrc
   ```

   in order to persist the changes.

3. Set up [`kachery-cloud`](https://github.com/flatironinstitute/kachery-cloud)
   (Frank Lab members only). Once you have initialized a `kachery-cloud`
   directory, add the following environment variables (again, shown for Frank
   lab).

   ```bash
   export KACHERY_CLOUD_DIR="$SPYGLASS_BASE_DIR/.kachery-cloud"
   export KACHERY_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
   ```

   Before proceeding, run -

   ```bash
   source ~/.bashrc
   ```

   in order to persist the changes.

4. Configure DataJoint. To connect to the
   [DataJoint](https://www.datajoint.org/) database, we have to specify
   information about it such as the hostname and the port. You should also
   change your password from the temporary one you were given. Go to the config
   directory, and run
   [`dj_config.py`](https://github.com/LorenFrankLab/spyglass/blob/master/config/dj_config.py)
   in the config folder. Then run
   [`dj_config.py`](https://github.com/LorenFrankLab/spyglass/blob/master/config/dj_config.py)
   terminal with your username:

   ```bash
   cd config # change to the config directory
   python dj_config.py <username> # run the configuration script
   ```

Finally, open up a python console (e.g. run `ipython` from terminal) and import
`spyglass` to check that the installation has worked.
