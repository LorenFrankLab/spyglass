# Installation

_Note:_ Developers, or those who wish to add features or otherwise work on the
codebase should follow the same steps below, but install Spyglass as editable
with the `-e` flag: `pip install -e /path/to/spyglass`

## Basic Installation

For basic installation steps, see the
[Setup notebook](./notebooks/00_Setup.ipynb) 'local installation' section,
including python, mamba (for managing a
[virtual environment](https://en.wikipedia.org/wiki/Virtual_environment_software)),
VSCode, Jupyter, and git. This notebook also covers
[database access](#database-access).

## Additional Packages

Some pipelines require installation of additional packages.

The spike sorting pipeline relies on `spikeinterface` and optionally
`mountainsort4`.

```bash
pip install spikeinterface[full,widgets]
pip install mountainsort4
```

__WARNING:__ If you are on an M1 Mac, you need to install `pyfftw` via `conda`
BEFORE installing `ghostipy`:

```bash
conda install -c conda-forge pyfftw
```

The LFP pipeline uses `ghostipy`:

```bash
pip install ghostipy
```

## Database access

For basic installation steps, see the
[Setup notebook](./notebooks/00_Setup.ipynb) 'database connection' section. For
additional details, see the
[DataJoint documentation](https://datajoint.com/docs/elements/user-guide/#relational-databases).

### Config

#### Via File (Recommended)

A `dj_local_conf.json` file in your current directory when launching python can
hold all the specifics needed to connect to a database. This can include
different directories for different pipelines. If only the Spyglass `base` is
specified, other subfolder names are assumed from defaults. See
`dj_local_conf_example.json` for the full set of options. This example can be
copied and saved as `dj_local_conf.json` to set the configuration for a given
folder. Alternatively, it can be saved as `.datajoint_config.json` in a user's
home directory to be accessed globally. See
[DataJoint docs](https://datajoint.com/docs/core/datajoint-python/0.14/quick-start/#connection)
for more details.

Note that raw and analysis folder locations should be specified under both
`stores` and `custom` sections of the config file. The `stores` section is used
by DataJoint to store the location of files referenced in database, while the
`custom` section is used by Spyglass. Spyglass will check that these sections
match on startup.

#### Via Environment Variables

Older versions of Spyglass relied exclusively on environment for config. If
`spyglass_dirs` is not found in the config file, Spyglass will look for
environment variables. These can be set either once in a terminal session, or
permanently in a unix settings file (e.g., `.bashrc` or `.bash_profile`) in your
home directory.

```bash
export SPYGLASS_BASE_DIR="/stelmo/nwb"
export SPYGLASS_RECORDING_DIR="$SPYGLASS_BASE_DIR/recording"
export SPYGLASS_SORTING_DIR="$SPYGLASS_BASE_DIR/sorting"
export SPYGLASS_VIDEO_DIR="$SPYGLASS_BASE_DIR/video"
export SPYGLASS_WAVEFORMS_DIR="$SPYGLASS_BASE_DIR/waveforms"
export SPYGLASS_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
```

To load variables from a `.bashrc` file, run `source ~/.bashrc` in a terminal.

#### Temporary directory

A temporary directory will speed up spike sorting. If unspecified by either
method above, it will be assumed as a `tmp` subfolder relative to the base path.
Be sure it has enough free space (ideally at least 500GB).

#### Subfolders

If subfolders do not exist, they will be created automatically. If unspecified
by either method above, they will be assumed as `recording`, `sorting`, `video`,
etc. subfolders relative to the base path.

## File manager

[`kachery-cloud`](https://github.com/flatironinstitute/kachery-cloud) is a file
manager for Frank Lab collaborators who do not have access to the lab's
production database.

To customize `kachery` file paths, see `dj_local_conf_example.json` or set the
following variables in your unix settings file (e.g., `.bashrc`). If
unspecified, the defaults below are assumed.

```bash
export KACHERY_CLOUD_DIR="$SPYGLASS_BASE_DIR/.kachery-cloud"
export KACHERY_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
```

Be sure to load these with `source ~/.bashrc` to persist changes.

## Test connection

Finally, open up a python console (e.g., run `ipython` from terminal) and import
`spyglass` to check that the installation has worked.

```python
from spyglass.common import Nwbfile

Nwbfile()
```
