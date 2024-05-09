def test_dlcvideo_default(sgp):
    assert sgp.v1.DLCPosVideoParams.get_default() == {
        "dlc_pos_video_params_name": "default",
        "params": {
            "incl_likelihood": True,
            "percent_frames": 1,
            "video_params": {"arrow_radius": 20, "circle_radius": 6},
        },
    }
