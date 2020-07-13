import setuptools

setuptools.setup(
    name="nwb_datajoint",
    version="0.1.0",
    author="Loren Frank, Eric Denovellis",
    author_email="eric.denovellis@ucsf.edu",
    description="NWB helper code for Loren Frank's lab at UCSF",
    url="https://github.com/LorenFrankLab/nwb_datajoint",
    packages=setuptools.find_packages(),
    install_requires=[
        'pynwb',
        'hdmf',
        'pandas',
        'networkx',
        'python-intervals',
        'matplotlib',
        'numpy',
        'scipy',
        'python-dateutil',
        'datajoint',
        'ghostipy',
        'kachery==0.6.4'
    ],
)
