# Tutorial Notebooks

There are several paths one can take to these notebooks. The notebooks have
two-digits in their names, the first of which indicates its 'batch', as
described in the categories below.

## 0. Intro

Everyone should complete the [Setup](./00_Setup.ipynb) and
[Insert Data](./02_Insert_Data.ipynb) notebooks. The
[Concepts](./01_Concepts.ipynb) notebook offers additional information that will
help users understand the data structure and how to interact with it.

[Data Sync](./03_Data_Sync.ipynb) is an optional additional tool for
collaborators that want to share analysis files.

The [Merge Tables notebook](./04_Merge_Tables.ipynb) explains details on a
pipeline versioning technique unique to Spyglass. This is important for
understanding the later notebooks.

The [Export notebook](./05_Export.ipynb) shows how to export data from the
database.

## 1. Spike Sorting Pipeline

This series of notebooks covers the process of spike sorting, from automated
spike sorting to optional manual curation of the output of the automated
sorting.

Spikesorting results from any pipeline can then be organized and tracked using
tools in [Spikesorting Analysis](./11_Spike_Sorting_Analysis.ipynb).

## 2. Position Pipeline

This series of notebooks covers tracking the position(s) of the animal. The user
can employ two different methods:

1. The simple [Trodes](20_Position_Trodes.ipynb) methods of tracking LEDs on the
    animal's headstage
2. [DLC (DeepLabCut)](./21_DLC.ipynb) which uses a neural network to track the
    animal's body parts.

Either case can be followed by the
[Linearization notebook](./24_Linearization.ipynb) if the user wants to
linearize the position data for later use.

## 3. LFP Pipeline

This series of notebooks covers the process of LFP analysis. The
[LFP](./30_LFP.ipynb) covers the extraction of the LFP in specific bands from
the raw data. The [Theta](./31_Theta.ipynb) notebook shows specifically how to
extract the theta band power and phase from the LFP data. Finally the
[Ripple Detection](./32_Ripple_Detection.ipynb) notebook shows how to detect
ripples in the LFP data.

## 4. Decoding Pipeline

This series of notebooks covers the process of decoding the position of the
animal from spiking data. It relies on the position data from the Position
pipeline and the output of spike sorting from the Spike Sorting pipeline.
Decoding can be from sorted or from unsorted data using spike waveform features
(so-called clusterless decoding).

The first notebook
([Extracting Clusterless Waveform Features](./40_Extracting_Clusterless_Waveform_Features.ipynb))
in this series shows how to retrieve the spike waveform features used for
clusterless decoding.

The second notebook ([Clusterless Decoding](./41_Decoding_Clusterless.ipynb))
shows a detailed example of how to decode the position of the animal from the
spike waveform features. The third notebook
([Decoding](./42_Decoding_SortedSpikes.ipynb)) shows how to decode the position
of the animal from the sorted spikes.

## Developer note

The `py_scripts` directory contains the same notebook data in `.py` form to
facilitate GitHub PR reviews. To update them, run the following from the root
Spyglass directory

```bash
pip install jupytext
jupytext --to py:light notebooks/*ipynb
mv notebooks/*py notebooks/py_scripts
black .
```

Unfortunately, jupytext-generated py script are not black-compliant by default.

You can ensure black compliance with the `pre-commit` hook by running

```bash
pip install pre-commit
```

This will run black whenever you commit changes to the repository.
