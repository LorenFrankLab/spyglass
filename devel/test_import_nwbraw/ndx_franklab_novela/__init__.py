import os

from pynwb import load_namespaces

# Set path of the namespace.yaml file to the expected install location
ndx_franklab_novela_specpath = os.path.join(
    os.path.dirname(__file__),
    'spec',
    'ndx-franklab-novela.namespace.yaml'
)

# If the extension has not been installed yet but we are running directly from
# the git repo
if not os.path.exists(ndx_franklab_novela_specpath):
    ndx_franklab_novela_specpath = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..',
        'spec',
        'ndx-franklab-novela.namespace.yaml'
    ))

# Load the namespace
load_namespaces(ndx_franklab_novela_specpath)
