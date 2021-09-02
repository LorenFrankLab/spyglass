from setuptools import setup, find_packages
  
setup(
    packages=find_packages(),
    include_package_data = True,
    install_requires=[
        'pynwb',
        'sortingview>=0.6'
    ]
)
