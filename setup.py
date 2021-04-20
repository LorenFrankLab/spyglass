import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nwb_datajoint",
    version="0.2.2",
    author="Loren Frank, Eric Denovellis, Kyu Hyun Lee",
    author_email="loren@phy.ucsf.edu",
    description="Code for generating Datajoint pipeline for Loren Frank's lab at UCSF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LorenFrankLab/nwb_datajoint",
    packages=setuptools.find_packages(),
    install_requires=[
        'jupyterlab',
        'pydotplus',
        'dask',
        'labbox-ephys>=0.5',
        'labbox-ephys-widgets-jp>=0.1',
        'mountainsort4',
        'spikeinterface>=0.12',
        'pynwb>=1.4',
        'datajoint==0.13.*',
        'ghostipy',
        'pymysql>=1.0.*',
        'h5py==2.10.*',
        'ndx-franklab-novela @ git+git://github.com/LorenFrankLab/ndx-franklab-novela'
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
