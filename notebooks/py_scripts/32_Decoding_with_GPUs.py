# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3.10.5 64-bit
#     language: python
#     name: python3
# ---

# # GPU Use

# ## Overview
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on decoding in Spyglass. To set up
# your Spyglass environment and database, see
# [the Setup notebook](./00_Setup.ipynb).
#
# In this tutorial, we'll set up GPU access for subsequent decoding analyses. While this notebook doesn't have any direct prerequisites, you will need
# [Spike Sorting](./02_Spike_Sorting.ipynb) data for the next step.
#

# ## GPU Clusters
#

# ### Connecting
#
# Members of the Frank Lab have access to two GPU cluster, `breeze` and `zephyr`.
# To access them, specify the cluster when you `ssh`, with the default port:
#
# > `ssh username@{breeze or zephyr}.cin.ucsf.edu`
#
# There are currently 10 available GPUs, each with 80 GB RAM, each referred to by their IDs (0 - 9).
#
# <!-- TODO: Use the position pipeline code for selecting GPU -->
#
# ### Selecting a GPU
#
# For decoding, we first install `cupy`. By doing so with conda, we're sure to
# install the correct cuda-toolkit:
#
# ```bash
# conda install cupy
# ```
#
# Next, we'll select a single GPU for decoding, using `cp.cuda.Device(GPU_ID)` in a context manager (i.e., `with`). Below, we'll select GPU #6 (ID = 5).
#
# _Warning:_ Omitting the context manager will cause cupy to default to using GPU 0.

# ### Which GPU?
#
# You can see which GPUs are occupied by running the command `nvidia-smi` in
# a terminal (or `!nvidia-smi` in a notebook). Pick a GPU with low memory usage.
#
# In the output below, GPUs 1, 4, 6, and 7 have low memory use and power draw (~42W), are probably not in use.

# !nvidia-smi

# We can monitor GPU use with the terminal command `watch -n 0.1 nvidia-smi`, will
# update `nvidia-smi` every 100 ms. This won't work in a notebook, as it won't
# display the updates.
#
# Other ways to monitor GPU usage are:
#
# - A
#   [jupyter widget by nvidia](https://github.com/rapidsai/jupyterlab-nvdashboard)
#   to monitor GPU usage in the notebook
# - A [terminal program](https://github.com/peci1/nvidia-htop) like nvidia-smi
#   with more information about  which GPUs are being utilized and by whom.

# ## Imports
#

# +
import os
import datajoint as dj

import cupy as cp
import numpy as np

import dask
import dask_cuda

import replay_trajectory_classification as rtc
from replay_trajectory_classification import (
    sorted_spikes_simulation as rtc_spike_sim,
    environment as rtc_env,
    continuous_state_transitions as rts,
)


# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import logging

# Set up logging message formatting
logging.basicConfig(
    level="INFO", format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
# -

# ## Simulated data
#

# First, we'll simulate some data.

# +
(
    time,
    position,
    sampling_frequency,
    spikes,
    place_fields,
) = rtc_spike_sim.make_simulated_run_data()

replay_time, test_spikes = rtc_spike_sim.make_continuous_replay()
# -

# ## Set up classifier

# +
movement_var = rts.estimate_movement_var(position, sampling_frequency)

environment = rtc_env.Environment(place_bin_size=np.sqrt(movement_var))

continuous_transition_types = [
    [
        rts.RandomWalk(movement_var=movement_var * 120),
        rts.Uniform(),
        rts.Identity(),
    ],
    [rts.Uniform(), rts.Uniform(), rts.Uniform()],
    [
        rts.RandomWalk(movement_var=movement_var * 120),
        rts.Uniform(),
        rts.Identity(),
    ],
]

classifier = rtc.SortedSpikesClassifier(
    environments=environment,
    continuous_transition_types=continuous_transition_types,
    # specify GPU enabled algorithm for the likelihood
    sorted_spikes_algorithm="spiking_likelihood_kde_gpu",
    sorted_spikes_algorithm_params={"position_std": 3.0},
)
state_names = ["continuous", "fragmented", "stationary"]
# -

# We can use a context manager to specify which GPU (device)
#

# +
GPU_ID = 5  # Use GPU #6

with cp.cuda.Device(GPU_ID):
    # Fit the model place fields
    classifier.fit(position, spikes)

    # Run the model on the simulated replay
    results = classifier.predict(
        test_spikes,
        time=replay_time,
        state_names=state_names,
        use_gpu=True,  # Use GPU for computation of causal/acausal posterior
    )
# -

# ## Multiple GPUs
#
# Using multiple GPUs requires the `dask_cuda`:
#
# ```bash
# conda install -c rapidsai -c nvidia -c conda-forge dask-cuda
# ```
#
# We will set up a client to select GPUs. By default, this is all available
# GPUs. Below, we select a subset using the `CUDA_VISIBLE_DEVICES`.

# +
cluster = dask_cuda.LocalCUDACluster(CUDA_VISIBLE_DEVICES=[4, 5, 6])
client = dask.distributed.Client(cluster)

client


# -

# To use this client, we declare a function we want to run on each GPU with the
# `dask.delayed` decorator.
#
# In the example below, we run `test_gpu` on each item of `data` where each item is processed on a different GPU.


# +
def setup_logger(name_logfile, path_logfile):
    """Sets up a logger for each function that outputs
    to the console and to a file"""
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    )
    fileHandler = logging.FileHandler(path_logfile, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


# This uses the dask.delayed decorator on the test_gpu function
@dask.delayed
def test_gpu(x, ind):
    # Create a log file for this run of the function
    logger = setup_logger(
        name_logfile=f"test_{ind}", path_logfile=f"test_{ind}.log"
    )

    # Test to see if these go into different log files
    logger.info(f"This is a test of {ind}")
    logger.info("This should be in a unique file")

    # Run a GPU computation
    return cp.asnumpy(cp.mean(x[:, None] @ x[:, None].T, axis=0))


# Make up 10 fake datasets
x = cp.random.normal(size=10_000, dtype=cp.float32)
data = [x + i for i in range(10)]

# Append the result of the computation into a results list
results = [test_gpu(x, ind) for ind, x in enumerate(data)]

# Run `dask.compute` on the results list for the code to run
dask.compute(*results)
# -

# This example also shows how to create a log file for each item in data with the `setup_logger` function.
