#!/usr/bin/env python3

import spyglass.common as sgc

nwb_file_name = "RN2_20191110_.nwb"

sgc.SortGroup().set_group_by_shank(
    nwb_file_name=nwb_file_name, references=None, omit_ref_electrode_group=False
)

print(sgc.SortGroup & {"nwb_file_name": nwb_file_name})
print(sgc.SortGroup.SortGroupElectrode & {"nwb_file_name": nwb_file_name})
