import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nwb_datajoint",
    version="0.2.1",
    author="Loren Frank, Eric Denovellis, Kyu Hyun Lee",
    author_email="loren@phy.ucsf.edu",
    description="Code for generating Datajoint pipeline for Loren Frank's lab at UCSF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LorenFrankLab/nwb_datajoint",
    packages=setuptools.find_packages(),
    # this needs to be updated
    install_requires=[
        'pynwb',
        'hdmf',
        'pandas',
        'networkx',
        'python-intervals',
        'matplotlib',
        'numpy>=1.19.4',
        'scipy',
        'python-dateutil',
        'datajoint',
        'ghostipy',
        'kachery>=0.6.4'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
