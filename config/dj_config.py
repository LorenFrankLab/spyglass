import datajoint as dj

# define the hostname and port to connect
dj.config['database.host'] = 'lmf-db.cin.ucsf.edu'
dj.config['database.port'] = 3306

# to prevent connection error for pymysql>=0.10; may omit in future
dj.config['database.use_tls'] = False

# CHANGE HERE: define user name
dj.config['database.user'] = 'your_user_name'

# change password; this will prompt you to type in a new password
dj.set_password()

# next, set up external stores
# (read about them here: https://docs.datajoint.io/python/admin/5-blob-config.html)
import os
import pathlib

assert os.getenv('NWB_DATAJOINT_BASE_DIR') is not None, 'environment variable NWB_DATAJOINT_BASE_DIR must be set'

data_dir = pathlib.Path(os.environ['NWB_DATAJOINT_BASE_DIR'])
raw_dir = data_dir / 'raw'
analysis_dir = data_dir / 'analysis'

dj.config['stores'] = {
    'raw': {
        'protocol': 'file',
        'location': str(raw_dir),
        'stage': str(raw_dir)
    },
    'analysis': {
        'protocol': 'file',
        'location': str(analysis_dir),
        'stage': str(analysis_dir)   
    }
}

# finally, save these settings
dj.config.save_global()