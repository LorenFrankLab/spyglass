#!/usr/bin/env python3

import spyglass.common as sgc

nwb_file_name = "RN2_20191110_.nwb"

sgc.SessionGroup.add_group("group1", "Group1", skip_duplicates=True)
sgc.SessionGroup.add_session_to_group(
    nwb_file_name, "group1", skip_duplicates=True
)
print(sgc.SessionGroup.get_group_sessions("group1"))
