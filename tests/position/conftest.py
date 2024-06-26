"""
The following lines are not used in the course of regular pose processing and
can be removed so long as other functionality is not impacted.

position_merge.py: 106-107, 110-123, 139-262
dlc_decorators.py: 11, 16-18, 22
dlc_reader.py    :
    24, 38, 44-45, 51, 57-58, 61, 70, 74, 80-81, 135-137, 146, 149-162, 214,
    218
dlc_utils.py     :
    58, 61, 69, 72, 97-100, 104, 149-161, 232-235, 239-241, 246, 259, 280,
    293-305, 310-316, 328-341, 356-373, 395, 404, 480, 487-488, 530, 548-561,
    594-601, 611-612, 641-657, 682-736, 762-772, 787, 809-1286
"""

from itertools import product as iter_product

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def dlc_video_params(sgp):
    sgp.v1.DLCPosVideoParams.insert_default()
    params_key = {"dlc_pos_video_params_name": "five_percent"}
    sgp.v1.DLCPosVideoParams.insert1(
        {
            **params_key,
            "params": {
                "percent_frames": 0.05,
                "incl_likelihood": True,
                "processor": "opencv",
            },
        },
        skip_duplicates=True,
    )
    yield params_key


@pytest.fixture(scope="session")
def dlc_video_selection(sgp, dlc_key, dlc_video_params, populate_dlc):
    s_key = {**dlc_key, **dlc_video_params}
    sgp.v1.DLCPosVideoSelection.insert1(s_key, skip_duplicates=True)
    yield dlc_key


@pytest.fixture(scope="session")
def populate_dlc_video(sgp, dlc_video_selection):
    sgp.v1.DLCPosVideo.populate(dlc_video_selection)
    yield sgp.v1.DLCPosVideo()


@pytest.fixture(scope="session")
def populate_evaluation(sgp, populate_model):
    sgp.v1.DLCEvaluation.populate()
    yield


def generate_led_df(leds, inc_vals=False):
    """Returns df with all combinations of 1 and np.nan for each led.

    If inc_vals is True, the values will be incremented by 1 for each non-nan"""
    all_vals = list(zip(*iter_product([1, np.nan], repeat=len(leds))))
    n_rows = len(all_vals[0])
    indices = np.random.uniform(1.6223e09, 1.6224e09, n_rows)

    data = dict()
    for led, values in zip(leds, all_vals):
        data.update(
            {
                (led, "video_frame_id"): {
                    i: f for i, f in zip(indices, range(n_rows + 1))
                },
                (led, "x"): {i: v for i, v in zip(indices, values)},
                (led, "y"): {i: v for i, v in zip(indices, values)},
            }
        )
    df = pd.DataFrame(data)

    if not inc_vals:
        return df

    count = [0]

    def increment_count():
        count[0] += 1
        return count[0]

    def process_value(x):
        return increment_count() if x == 1 else x

    return df.applymap(process_value)
