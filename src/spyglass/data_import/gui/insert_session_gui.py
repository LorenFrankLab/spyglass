from PySide6.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout
import sys
import datajoint as dj
import os
from pathlib import Path


class MainWindow(QMainWindow):


    def __init__(self):
        super().__init__()
        from spyglass.common.common_lab import Institution

        institutions = Institution.fetch("institution_name").tolist()
        institutions.insert(0, "")

        self.institution_combobox = QComboBox()
        self.institution_combobox.addItems(institutions)

        layout = QVBoxLayout()
        layout.addWidget(self.institution_combobox)

        container = QWidget()
        container.setLayout(layout)

        self.institution_combobox.activated.connect(self.current_text)

        self.setCentralWidget(container)

    def current_text(self, _):
        ctext = self.institution_combobox.currentText()
        print("Current text", ctext)


def setup_datajoint(base_dir_str):
    # set dirs
    base_dir = Path(base_dir_str)
    if (base_dir).exists() is False:
        os.mkdir(base_dir)
    raw_dir = base_dir / 'raw'
    if (raw_dir).exists() is False:
        os.mkdir(raw_dir)
    analysis_dir = base_dir / 'analysis'
    if (analysis_dir).exists() is False:
        os.mkdir(analysis_dir)
    recording_dir = base_dir / 'recording'
    if (recording_dir).exists() is False:
        os.mkdir(recording_dir)
    sorting_dir = base_dir / 'sorting'
    if (sorting_dir).exists() is False:
        os.mkdir(sorting_dir)
    waveforms_dir = base_dir / 'waveforms'
    if (waveforms_dir).exists() is False:
        os.mkdir(waveforms_dir)
    tmp_dir = base_dir / 'tmp'
    if (tmp_dir).exists() is False:
        os.mkdir(tmp_dir)
    kachery_cloud_dir = base_dir / '.kachery_cloud'
    if (kachery_cloud_dir).exists() is False:
        os.mkdir(kachery_cloud_dir)

    # set dj config
    dj.config['database.host'] = 'localhost'
    dj.config['database.user'] = 'root'
    dj.config['database.password'] = 'tutorial'
    dj.config['database.port'] = 3306
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

    # set env vars
    os.environ['SPYGLASS_BASE_DIR'] = str(base_dir)
    os.environ['SPYGLASS_RECORDING_DIR'] = str(recording_dir)
    os.environ['SPYGLASS_SORTING_DIR'] = str(sorting_dir)
    os.environ['SPYGLASS_WAVEFORMS_DIR'] = str(waveforms_dir)
    os.environ['SPYGLASS_TEMP_DIR'] = str(tmp_dir)
    os.environ['KACHERY_CLOUD_DIR'] = str(kachery_cloud_dir)

    os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'

    dj.config["enable_python_native_blobs"] = True

    # temporarily prepopulate the database
    from spyglass.common.common_lab import Institution
    Institution.insert1({"institution_name": "UCSF"}, skip_duplicates=True)
    Institution.insert1({"institution_name": "UC Berkeley"}, skip_duplicates=True)
    Institution.insert1({"institution_name": "UCLA"}, skip_duplicates=True)


setup_datajoint("/Users/rly/Documents/NWB/spyglass-workspace")
app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()