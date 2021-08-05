import datajoint as dj
from pathlib import Path

data_dir = Path('.')

# NOTE these env vars are set in the GitHub Action directly
# os.environ['NWB_DATAJOINT_BASE_DIR'] = str(data_dir)
# os.environ['KACHERY_STORAGE_DIR'] = str(data_dir / 'kachery-storage')
# os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'

raw_dir = data_dir / 'raw'
analysis_dir = data_dir / 'analysis'

dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'tutorial'
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
dj.config.save_global()
