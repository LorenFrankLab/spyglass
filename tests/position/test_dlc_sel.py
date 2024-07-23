def test_dlc_video_default(sgp):
    expected_default = {
        "dlc_pos_video_params_name": "default",
        "params": {
            "incl_likelihood": True,
            "percent_frames": 1,
            "video_params": {"arrow_radius": 20, "circle_radius": 6},
        },
    }

    # run twice to trigger fetch existing
    assert sgp.v1.DLCPosVideoParams.get_default() == expected_default
    assert sgp.v1.DLCPosVideoParams.get_default() == expected_default


def test_dlc_video_populate(populate_dlc_video):
    assert len(populate_dlc_video) > 0, "DLCPosVideo table is empty"
