[![Import test](https://github.com/LorenFrankLab/spyglass/actions/workflows/workflow.yml/badge.svg)](https://github.com/LorenFrankLab/spyglass/actions/workflows/workflow.yml)

# spyglass

`spyglass` is a data analysis framework that facilitates the storage, analysis, visualization, and sharing of neuroscience data to support reproducible research. It is designed to be interoperable with the NWB format and integrates open-source tools into a coherent framework.

Documentation can be found at - [https://lorenfranklab.github.io/spyglass/](https://lorenfranklab.github.io/spyglass/)

### Installing from pip

Install spyglass

```bash
pip install spyglass-neuro
```

Some functions may take advantage of the latest changes to spike interface, which currently has a slow release cycle. To get the latest changes:

```bash
pip install git+https://github.com/SpikeInterface/spikeinterface.git
```

The Frank Lab typically uses mountainsort, although spyglass uses spikeinterface, which allows for any spike sorter. To install mountainsort:

```bash
pip install mountainsort4
```

Spyglass uses the package `ghostipy` for filtering of signals:

```bash
pip install ghostipy
```

WARNING: If you are on an M1 Mac, you need to install pyfftw via conda BEFORE installing ghostipy

```bash
conda install -c conda-forge pyfftw
```

Finally, if you want to decode on the GPU, you must install cupy:
```bash
conda install -c conda-forge cupy
```

## Setup

See the documentation for setup instructions - [https://lorenfranklab.github.io/spyglass/type/html/installation.html](https://lorenfranklab.github.io/spyglass/type/html/installation.html)

## Tutorials

The tutorials for `spyglass` is currently in the form of Jupyter Notebooks and can be found in the [notebooks](https://github.com/LorenFrankLab/spyglass/tree/master/notebooks) directory. We strongly recommend opening them in the context of `jupyterlab`.

## Contributing

See the [Developer's Note](https://lorenfranklab.github.io/spyglass/type/html/developer_notes.html) for contributing instructions found at - [https://lorenfranklab.github.io/spyglass/type/html/how_to_contribute.html](https://lorenfranklab.github.io/spyglass/type/html/how_to_contribute.html)

## License/Copyright

License and Copyright notice can be found at [https://lorenfranklab.github.io/spyglass/type/html/copyright.html](https://lorenfranklab.github.io/spyglass/type/html/copyright.html)

## Citation

Kyu Hyun Lee, Eric Denovellis, Ryan Ly, Jeremy Magland, Jeff Soules, Alison Comrie, Jennifer Guidera, Rhino Nevers, Daniel Gramling, Philip Adenekan, Ji Hyun Bak, Emily Monroe, Andrew Tritt, Oliver Rübel, Thinh Nguyen, Dimitri Yatsenko, Joshua Chu, Caleb Kemere, Samuel Garcia, Alessio Buccino, Emily Aery Jones, Lisa Giocomo, and Loren Frank. Spyglass: A Data Analysis Framework for Reproducible and Shareable Neuroscience Research. Neuroscience Meeting Planner. San Diego, CA: Society for Neuroscience, 2022.
