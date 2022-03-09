#!/usr/bin/env python3

import nwb_datajoint.common as ndc

nwb_file_name = 'RN2_20191110_.nwb'

ndc.SessionGroup.add_group('group1', 'Group1', skip_duplicates=True)
ndc.SessionGroup.add_session_to_group(nwb_file_name, 'group1', skip_duplicates=True)
print(
    ndc.SessionGroup.get_group_sessions('group1')
)