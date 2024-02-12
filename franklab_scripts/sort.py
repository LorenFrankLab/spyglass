import os
from pathlib import Path

import spyglass as sg


def main():
    # CHANGE ME TO THE BASE DIRECTORY FOR DATA STORAGE ON YOUR SYSTEM
    base_dir = Path("/stelmo/nwb")

    os.environ["SPYGLASS_BASE_DIR"] = str(base_dir)
    os.environ["KACHERY_STORAGE_DIR"] = str(base_dir / "kachery-storage")
    os.environ["SPYGLASS_SORTING_DIR"] = str(base_dir / "spikesorting")

    # session_id = "jaq_01"
    # nwb_file_name = (sg.common.Session() & {"session_id": session_id}).fetch1(
    #     "nwb_file_name"
    # )

    sg.common.SpikeSorting().populate()


if __name__ == "__main__":
    main()
