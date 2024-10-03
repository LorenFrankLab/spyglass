# spyglass

[![Tests](https://github.com/LorenFrankLab/spyglass/actions/workflows/test-conda.yml/badge.svg)](https://github.com/LorenFrankLab/spyglass/actions/workflows/test-conda.yml)
[![PyPI version](https://badge.fury.io/py/spyglass-neuro.svg)](https://badge.fury.io/py/spyglass-neuro)

![Spyglass Figure](docs/src/images/fig1.png)

[Demo](https://spyglass.hhmi.2i2c.cloud/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FLorenFrankLab%2Fspyglass-demo&urlpath=lab%2Ftree%2Fspyglass-demo%2Fnotebooks%2F02_Insert_Data.ipynb&branch=main)
|
[Installation](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/)
| [Docs](https://lorenfranklab.github.io/spyglass/) |
[Tutorials](https://github.com/LorenFrankLab/spyglass/tree/master/notebooks) |
[Citation](#citation)

`spyglass` is a data analysis framework that facilitates the storage, analysis,
visualization, and sharing of neuroscience data to support reproducible
research. It is designed to be interoperable with the NWB format and integrates
open-source tools into a coherent framework.

Try out a demo
[here](https://spyglass.hhmi.2i2c.cloud/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FLorenFrankLab%2Fspyglass-demo&urlpath=lab%2Ftree%2Fspyglass-demo%2Fnotebooks%2F02_Insert_Data.ipynb&branch=main)!

Features of Spyglass include:

- **Standardized data storage** - Spyglass uses the open-source
    [Neurodata Without Borders: Neurophysiology (NWB:N)](https://www.nwb.org/)
    format to ingest and store processed data. NWB:N is a standard set by the
    BRAIN Initiative for neurophysiological data
    ([Rübel et al., 2022](https://doi.org/10.7554/elife.78362)).
- **Reproducible analysis** - Spyglass uses [DataJoint](https://datajoint.com/)
    to ensure that all analysis is reproducible. DataJoint is a data management
    system that automatically tracks dependencies between data and analysis
    code. This ensures that all analysis is reproducible and that the results
    are automatically updated when the data or analysis code changes.
- **Common analysis tools** - Spyglass provides easy usage of the open-source
    packages [SpikeInterface](https://github.com/SpikeInterface/spikeinterface),
    [Ghostipy](https://github.com/kemerelab/ghostipy), and
    [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) for common analysis
    tasks. These packages are well-documented and have active developer
    communities.
- **Interactive data visualization** - Spyglass uses
    [figurl](https://github.com/flatironinstitute/figurl) to create interactive
    data visualizations that can be shared with collaborators and the broader
    community. These visualizations are hosted on the web and can be viewed in
    any modern web browser. The interactivity allows users to explore the data
    and analysis results in detail.
- **Sharing results** - Spyglass enables sharing of data and analysis results
    via [Kachery](https://github.com/flatironinstitute/kachery-cloud), a
    decentralized content addressable data sharing platform. Kachery Cloud
    allows users to access the database and pull data and analysis results
    directly to their local machine.
- **Pipeline versioning** - Processing and analysis of data in neuroscience is
    often dynamic, requiring new features. Spyglass uses *Merge tables* to
    ensure that analysis pipelines can be versioned. This allows users to easily
    use and compare results from different versions of the analysis pipeline
    while retaining the ability to access previously generated results.
- **Cautious Delete** - Spyglass uses a `cautious delete` feature to ensure that
    data is not accidentally deleted by other users. When a user deletes data,
    Spyglass will first check to see if the data belongs to another team of
    users. This enables teams of users to work collaboratively on the same
    database without worrying about accidentally deleting each other's data.

Documentation can be found at -
[https://lorenfranklab.github.io/spyglass/](https://lorenfranklab.github.io/spyglass/)

## Installation

For installation instructions see -
[https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/)

Typical installation time is: 5-10 minutes

## Tutorials

The tutorials for `spyglass` are currently in the form of Jupyter Notebooks and
can be found in the
[notebooks](https://github.com/LorenFrankLab/spyglass/tree/master/notebooks)
directory. We strongly recommend running the notebooks yourself.

## Contributing

See the
[Developer's Note](https://lorenfranklab.github.io/spyglass/latest/contribute/)
for contributing instructions found at -
[https://lorenfranklab.github.io/spyglass/latest/contribute/](https://lorenfranklab.github.io/spyglass/latest/contribute/)

## License/Copyright

License and Copyright notice can be found at
[https://lorenfranklab.github.io/spyglass/latest/LICENSE/](https://lorenfranklab.github.io/spyglass/latest/LICENSE/)

## System requirements

Spyglass has been tested on Linux Ubuntu 20.04 and MacOS 10.15. It has not been
tested on Windows and likely will not work.

No specific hardware requirements are needed to run spyglass. However, the
amount of data that can be stored and analyzed is limited by the available disk
space and memory. GPUs are required for some of the analysis tools, such as
DeepLabCut.

See [pyproject.toml](pyproject.toml), [environment.yml](environment.yml), or
[environment_dlc.yml](environment_dlc.yml) for software dependencies.

See
[spec-file.txt](https://github.com/LorenFrankLab/spyglass-demo/blob/main/spec-file/spec-file.txt)
for the conda environment used in the demo.

## Citation

> Lee, K.H.\*, Denovellis, E.L.\*, Ly, R., Magland, J., Soules, J., Comrie,
> A.E., Gramling, D.P., Guidera, J.A., Nevers, R., Adenekan, P., Brozdowski, C.,
> Bray, S., Monroe, E., Bak, J.H., Coulter, M.E., Sun, X., Broyles, E., Shin,
> D., Chiang, S., Holobetz, C., Tritt, A., Rübel, O., Nguyen, T., Yatsenko, D.,
> Chu, J., Kemere, C., Garcia, S., Buccino, A., Frank, L.M., 2024. Spyglass: a
> data analysis framework for reproducible and shareable neuroscience research.
> bioRxiv.
> [10.1101/2024.01.25.577295](https://doi.org/10.1101/2024.01.25.577295).

*\* Equal contribution*

See paper related code [here](https://github.com/LorenFrankLab/spyglass-paper).
