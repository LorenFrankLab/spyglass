from setuptools import setup, find_packages
  
setup(
    install_requires=[
        'kachery-client>=1.1.7',
        'mountainsort4',
        'spikeextractors',
        'pynwb>=2.0.0,<3',
        'hdmf>=3.1.1,<4',
        'datajoint>=0.13.*',
        'ghostipy',
        'pymysql>=1.0.*',
        'h5py==2.10.*',
        'sortingview>=0.7.3'
    ]
)
