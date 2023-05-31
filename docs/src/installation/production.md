# Production Installation

## Virtual Environment

It is recommended you install a [virtual
environment](https://en.wikipedia.org/wiki/Virtual_environment_software). There
are many options like
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
and [venv](https://docs.python.org/3/library/venv.html). This installation
instruction will use
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

The instructions to install
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
can be found at
[https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
can be used both for installing packages and for creating a
[virtual environment](https://towardsdatascience.com/introduction-to-conda-virtual-environments-eaea4ac84e28).

To create the environment after
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
is installed, run -

```bash
conda create -n spyglass-env python=3.9.0
```

The name selected here is `spyglass-env`. However, a different name can be used
if desired or necessary.

Now, it is time to activate the virtual environment. To start, run -

```bash
conda activate spyglass-env
```

## Installing Spyglass

`spyglass` can be installed via
[pip](<https://en.wikipedia.org/wiki/Pip_(package_manager)>) off of
[pypi](https://pypi.org/project/spyglass-neuro/):

```bash
pip install spyglass-neuro
```

## Additional Packages

Some pipelines require installation of additional packages. For example, the
spike sorting pipeline relies on `spikeinterface`. We recommend installing it
directly from the GitHub repo:

```bash
pip install spikeinterface[full,widgets]
```

You may also need to install individual sorting algorithms. For example, Loren
Frank's lab at [UCSF](https://www.ucsf.edu/) typically uses `mountainsort4`:

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
   your own database, consult instructions in
   [datajoint tutorial](https://tutorials.datajoint.org/setting-up/get-database.html)
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
   (if you are in Frank lab, skip this step). Once you have initialized a
   `kachery-cloud` directory, add the following environment variables (again,
   shown for Frank lab).

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
   change your password from the temporary one you were given. Download
   [`dj_config.py`](https://github.com/LorenFrankLab/spyglass/blob/master/config/dj_config.py)
   from
   [https://github.com/LorenFrankLab/spyglass/blob/master/config/dj_config.py](https://github.com/LorenFrankLab/spyglass/blob/master/config/dj_config.py)
   and save locally as `dj_config.py`, to any folder. Instructions on how to
   download a single file from github can be found at
   [https://stackoverflow.com/a/13593430/178550](https://stackoverflow.com/a/13593430/178550).
   Then run `dj_config.py` in a terminal with your username -

   ```bash
   cd config # change to the config directory
   python dj_config.py <username> # run the configuration script
   ```

Finally, open up a python console (e.g. run `ipython` from terminal) and import
`spyglass` to check that the installation has worked.
