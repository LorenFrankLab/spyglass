import os
from pathlib import Path

import spyglass as nd


def main():

    data_dir = Path(
        "/stelmo/nwb"
    )  # CHANGE ME TO THE BASE DIRECTORY FOR DATA STORAGE ON YOUR SYSTEM

    os.environ["SPYGLASS_BASE_DIR"] = str(data_dir)
    os.environ["KACHERY_STORAGE_DIR"] = str(data_dir / "kachery-storage")
    os.environ["SPIKE_SORTING_STORAGE_DIR"] = str(data_dir / "spikesorting")

    # session_id = 'jaq_01'
    # nwb_file_name = (nd.common.Session() & {'session_id': session_id}).fetch1('nwb_file_name')
    nwb_file_name = "beans20190718-trim_.nwb"

    nd.common.SpikeSorting().populate()


if __name__ == "__main__":
    main()
