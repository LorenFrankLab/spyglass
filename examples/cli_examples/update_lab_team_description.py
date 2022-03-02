#!/usr/bin/env python3

import nwb_datajoint.common as ndc

ndc.LabTeam.update1({
    'team_name': 'LorenLab',
    'team_description': 'New team description'
})