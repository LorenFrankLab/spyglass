import pytest
import sys
import os
from .fixtures._datajoint_server import datajoint_server


thisdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thisdir)

def pytest_addoption(parser):
    parser.addoption('--current', action='store_true', dest="current",
                 default=False, help="run only tests marked as current")

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "current: for convenience -- mark one test as current"
    )

    markexpr_list = []

    if config.option.current:
        markexpr_list.append('current')

    if len(markexpr_list) > 0:
        markexpr = ' and '.join(markexpr_list)
        setattr(config.option, 'markexpr', markexpr)
