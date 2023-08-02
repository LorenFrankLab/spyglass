# Installation

_Note:_ Developers, or those who wish to add features or otherwise work on the
codebase should follow the same steps below, but install Spyglass as editable
with the `-e` flag: `pip install -e /path/to/spyglass`

## Basic Installation

For basic installation steps, see the
[Setup notebook](../notebooks/00_Setup.ipynb) 'local installation' section,
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

WARNING: If you are on an M1 Mac, you need to install `pyfftw` via `conda`
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
[Setup notebook](../notebooks/00_Setup.ipynb) 'database connection' section. For
additional details, see the
[DataJoint documentation](https://datajoint.com/docs/elements/user-guide/#relational-databases).

### Config

A `dj_local_conf.json` file in your Spyglass directory (or wherever python is
launched) can hold all the specifics needed to connect to a database. This can
include different directories for different pipelines. If only the `base` is
specified, the subfolder names below are included as defaults.

```json
{
  "custom": {
    "database.prefix": "username_",
    "spyglass_dirs": {
      "base": "/your/base/path",
      "raw":"/your/base/path/raw",
      "analysis":"/your/base/path/analysis",
      "recording":"/your/base/path/recording",
      "spike_sorting_storage":"/your/base/path/spikesorting",
      "waveforms":"/your/base/path/waveforms",
      "temp":"/your/base/path/tmp",
    }
  }
}
```

For those who prefer environment variables, the following can pasted into a
file like `~/.bashrc`.

```bash
export SPYGLASS_BASE_DIR="/stelmo/nwb"
export SPYGLASS_RECORDING_DIR="$SPYGLASS_BASE_DIR/recording"
export SPYGLASS_SORTING_DIR="$SPYGLASS_BASE_DIR/sorting"
export SPYGLASS_VIDEO_DIR="$SPYGLASS_BASE_DIR/video"
export SPYGLASS_WAVEFORMS_DIR="$SPYGLASS_BASE_DIR/waveforms"
export SPYGLASS_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp/spyglass"
export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
```

And then loaded with `source ~/.bashrc`.

Note that a local `SPYGLASS_TEMP_DIR` (e.g., one on your machine) will speed
up spike sorting, but make sure it has enough free space (ideally at least
500GB)

## File manager

[`kachery-cloud`](https://github.com/flatironinstitute/kachery-cloud) is a file
manager for Frank Lab collaborators who do not have access to the lab's
production database.

To customize `kachery` file paths, the following can similarly be pasted into
your `.bashrc`. If unspecified, the defaults below are assumed.

```bash
export KACHERY_CLOUD_DIR="$SPYGLASS_BASE_DIR/.kachery-cloud"
export KACHERY_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
```

Be sure to load these with `source ~/.bashrc` to persist changes.

## Test connection

Finally, open up a python console (e.g., run `ipython` from terminal) and import
`spyglass` to check that the installation has worked.
