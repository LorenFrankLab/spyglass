# Tutorial Notebooks

There are several paths one can take to these notebooks. The notebooks have
two-digits in their names, the first of which indicates it's 'batch', as
described in the categories below.

<!-- TODO: Add links when names are finalized. -->

## 0. Intro

Everyone should complete the Setup and Insert Data notebooks. Data Sync is an
optional additional tool for collaborators that want to share analysis files.

## 1. Electrophysiology

For folks running ephys analysis, one could use the either one or both of the
following...

1. Spike Sorting, and optionally the Curation notebooks
2. LFP, and optionally Theta notebooks

## 2. Position

For folks tracking animal positon, use of either the Trodes or DLC (DeepLabCut)
notebooks depends on preferred tracking method. Either case should be followed
by Info and Linearization notebooks.

## 3. Combo

The remaining notebooks make use of both ephys and position data for further
processing.

- Ripple Detection: Uses LFP and Position information
- Extract Marks: Comparing actual and mental positon using unclustered spikes
  and spike waveform features.
- Decoding: Uses either spike sorted of clusterless ephys anaysis to look at
  mental position.

<!-- CBroz: Did I get this right? -->