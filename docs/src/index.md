# Spyglass

![Figure 1](./images/fig1.png)

**Spyglass** is an open-source software framework designed to offer reliable
and reproducible analysis of neuroscience data and sharing of the results
with collaborators and the broader community.

Features of Spyglass include:

+ **Standardized data storage** - Spyglass uses the open-source
  [Neurodata Without Borders: Neurophysiology (NWB:N)](https://www.nwb.org/)
  format to ingest and store processed data. NWB:N is a standard set by the BRAIN
  Initiative for neurophysiological data ([Rübel et al., 2022](https://doi.org/10.7554/elife.78362)).
+ **Reproducible analysis** - Spyglass uses [DataJoint](https://datajoint.com/)
  to ensure that all analysis is reproducible. DataJoint is a data management
  system that automatically tracks dependencies between data and analysis code. This
  ensures that all analysis is reproducible and that the results are
  automatically updated when the data or analysis code changes.
+ **Common analysis tools** - Spyglass provides easy usage of the open-source packages
  [SpikeInterface](https://github.com/SpikeInterface/spikeinterface),
  [Ghostipy](https://github.com/kemerelab/ghostipy), and [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
  for common analysis tasks. These packages are well-documented and have active
  developer communities.
+ **Interactive data visualization** - Spyglass uses [figurl](https://github.com/flatironinstitute/figurl)
  to create interactive data visualizations that can be shared with collaborators
  and the broader community. These visualizations are hosted on the web
  and can be viewed in any modern web browser. The interactivity allows users to
  explore the data and analysis results in detail.
+ **Sharing results** - Spyglass enables sharing of data and analysis results via
  [Kachery](https://github.com/flatironinstitute/kachery-cloud), a
  decentralized content addressable data sharing platform. Kachery Cloud allows
  users to access the database and pull data and analysis results directly
  to their local machine.
+ **Pipeline versioning** - Processing and analysis of data in neuroscience is
  often dynamic, requiring new features. Spyglass uses *Merge tables* to ensure that
  analysis pipelines can be versioned. This allows users to easily use and compare
  results from different versions of the analysis pipeline while retaining
  the ability to access previously generated results.
+ **Cautious Delete** - Spyglass uses a `cautious delete` feature to ensure
  that data is not accidentally deleted by other users. When a user deletes data,
  Spyglass will first check to see if the data belongs to another team of users.
  This enables teams of users to work collaboratively on the same database without
  worrying about accidentally deleting each other's data.

## Getting Started

This site hosts both [installation instructions](./installation.md) and
[tutorials](./notebooks/index.md) to help you get started with Spyglass. We
recommend running the notebooks yourself. They can be downloaded from GitHub
[here](https://github.com/LorenFrankLab/spyglass).

## Diving Deeper

The [API Reference](./api/index.md) provides a detailed description of all the
tables and class functions in Spyglass via python docstrings. Potential
contributors should also read the [Developer Guide](./contribute.md). Those
interested in in hosting a Spyglass instance for their own data should read the
[database management guide](./misc/database_management.md).

We have a series of additional docs under the [misc](./misc/index.md) folder
that may be helpful. Our [changelog](./CHANGELOG.md) highlights the changes that
have been made to Spyglass over time and the [copyright](./LICENSE.md) page
contains license information.

## Citing Spyglass

Kyu Hyun Lee, Eric Denovellis, Ryan Ly, Jeremy Magland, Jeff Soules, Alison
Comrie, Jennifer Guidera, Rhino Nevers, Daniel Gramling, Philip Adenekan, Ji
Hyun Bak, Emily Monroe, Andrew Tritt, Oliver Rübel, Thinh Nguyen, Dimitri
Yatsenko, Joshua Chu, Caleb Kemere, Samuel Garcia, Alessio Buccino, Emily Aery
Jones, Lisa Giocomo, and Loren Frank. 'Spyglass: A Data Analysis Framework for
Reproducible and Shareable Neuroscience Research.' (2022) Society for
Neuroscience, San Diego, CA.

<!-- TODO: Convert ccf file and insert here  -->
