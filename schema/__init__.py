# specify path to our extension's .yaml files
#import os.path
# fl_ns_path = os.path.join(os.path.dirname(__file__), 'franklab.namespace.yaml')
# fl_ext_path = os.path.join(os.path.dirname(__file__), 'franklab.extensions.yaml')

from .create_franklab_spec import ns_path as fl_ns_path
from .create_franklab_spec import ext_path as fl_ext_path