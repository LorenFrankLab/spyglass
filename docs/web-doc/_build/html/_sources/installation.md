# Installation

## Installing packages

1. Clone this repository:

   ```bash

      git clone https://github.com/LorenFrankLab/nwb_datajoint.git
   ```

2. Set up and activate a conda environment from ``environment.yml``:

   ```bash

      cd nwb_datajoint
      conda env create -f environment.yml
      conda activate nwb_datajoint
   ```

3. Install this repository:

   ```bash

      pip install -e .
   ```

## Setting up database access

1. Ask Loren or Eric to set up an account for you on the Frank lab
   database. Note that you have to be connected to UCSF LAN to access
   this server.

      If you’re not affiliated with UCSF or if you are just looking to
      try out ``nwb_datajoint``, then you will need to set up a
      different MySQL server. For example, you can set up your own local
      server with a Docker image of a MySQL server configured for
      Datajoint (see
      `instructions <https://tutorials.datajoint.io/setting-up/local-database.html>`__
      and/or `tutorial
      notebook <./notebooks/docker_mysql_tutorial.ipynb>`__).

2. Add the following environment variables (e.g. in ``~/.bashrc``). This
   example assumes that you are interacting with the database on a
   computer that has mounted ``stelmo`` at ``/stelmo`` (if the mount
   location is different, change accordingly). For this to take effect,
   log out and log back in, or run ``source ~/.bashrc`` in the terminal.

   ```bash

      export NWB_DATAJOINT_BASE_DIR="/stelmo/nwb/"
      export SPIKE_SORTING_STORAGE_DIR="/stelmo/nwb/spikesorting"
      export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
      export KACHERY_DAEMON_HOST="typhoon"
      export KACHERY_DAEMON_PORT="14747"
      export KACHERY_TEMP_DIR="/stelmo/nwb/tmp"
      export NWB_DATAJOINT_TEMP_DIR="/stelmo/nwb/tmp"
      export KACHERY_STORAGE_DIR="/stelmo/nwb/kachery-storage"
      export FIGURL_CHANNEL="franklab2"
   ```

   Note that a local NWB_DATAJOINT_TEMP_DIR (e.g. one on your machine)
   will speed up spikesorting, but make sure it has enough free space
   (ideally at least 500GB)

3. Check if you have access to the ``kachery`` daemon. Open up a
   terminal, activate the conda environment, and type

   ```bash

      kachery-client info
   ```

   If it says you do not have access to the daemon, ask Kyu to add your
   username to the ``kachery-users`` group. > If you’re running your own
   ``kachery`` daemon, then this doesn’t apply.

4. Configure DataJoint. To connect to the Datajoint database, we have to
   specify information about it such as the hostname and the port. You
   should also change your password from the temporary one you were
   given. Go to the config directory, and run
   ```dj_config.py`` <https://github.com/LorenFrankLab/nwb_datajoint/blob/master/config/dj_config.py>`__
   in the terminal with your username:

   ```bash

      cd config # change to the config directory
      python dj_config.py <username> # run the configuration script
   ```

   ..

      Again, if you’re using your own MySQL server, then you may need to
      change the other settings as well.

Finally, open up a python console (e.g. run ``ipython`` from terminal)
and import ``nwb_datajoint`` to check that the installation has worked.
