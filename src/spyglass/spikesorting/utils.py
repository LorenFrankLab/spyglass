import numpy as np
import spikeinterface as si

from spyglass.common.common_ephys import Electrode
from spyglass.utils import logger


def get_group_by_shank(
    nwb_file_name: str,
    references: dict = None,
    omit_ref_electrode_group=False,
    omit_unitrode=True,
):
    """Divides electrodes into groups based on their shank position.

    * Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a
      single group
    * Electrodes from probes with multiple shanks (e.g. polymer probes) are
      placed in one group per shank
    * Bad channels are omitted

    Parameters
    ----------
    nwb_file_name : str
        the name of the NWB file whose electrodes should be put into
        sorting groups
    references : dict, optional
        If passed, used to set references. Otherwise, references set using
        original reference electrodes from config. Keys: electrode groups.
        Values: reference electrode.
    omit_ref_electrode_group : bool
        Optional. If True, no sort group is defined for electrode group of
        reference.
    omit_unitrode : bool
        Optional. If True, no sort groups are defined for unitrodes.
    """
    # get the electrodes from this NWB file
    electrodes = (
        Electrode()
        & {"nwb_file_name": nwb_file_name}
        & {"bad_channel": "False"}
    ).fetch()

    e_groups = list(np.unique(electrodes["electrode_group_name"]))
    e_groups.sort(key=int)  # sort electrode groups numerically

    sort_group = 0
    sg_keys, sge_keys = list(), list()
    for e_group in e_groups:
        sg_key, sge_key = dict(), dict()
        sg_key["nwb_file_name"] = sge_key["nwb_file_name"] = nwb_file_name

        # for each electrode group, get a list of the unique shank numbers
        shank_list = np.unique(
            electrodes["probe_shank"][
                electrodes["electrode_group_name"] == e_group
            ]
        )
        sge_key["electrode_group_name"] = e_group

        # get the indices of all electrodes in this group / shank and set their
        # sorting group
        for shank in shank_list:
            sg_key["sort_group_id"] = sge_key["sort_group_id"] = sort_group

            match_names_bool = np.logical_and(
                electrodes["electrode_group_name"] == e_group,
                electrodes["probe_shank"] == shank,
            )

            if references:  # Use 'references' if passed
                sort_ref_id = references.get(e_group, None)
                if not sort_ref_id:
                    raise Exception(
                        f"electrode group {e_group} not a key in "
                        + "references, so cannot set reference"
                    )
            else:  # otherwise use reference from config
                shank_elect_ref = electrodes["original_reference_electrode"][
                    match_names_bool
                ]
                if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                    sort_ref_id = shank_elect_ref[0]
                else:
                    ValueError(
                        f"Error in electrode group {e_group}: reference "
                        + "electrodes are not all the same"
                    )
            sg_key["sort_reference_electrode_id"] = sort_ref_id

            # Insert sort group and sort group electrodes
            match_elec = electrodes[electrodes["electrode_id"] == sort_ref_id]
            ref_elec_group = match_elec["electrode_group_name"]  # group ref

            n_ref_groups = len(ref_elec_group)
            if n_ref_groups == 1:  # unpack single reference
                ref_elec_group = ref_elec_group[0]
            elif int(sort_ref_id) > 0:  # multiple references
                raise Exception(
                    "Should have found exactly one electrode group for "
                    + f"reference electrode, but found {n_ref_groups}."
                )

            if omit_ref_electrode_group and (
                str(e_group) == str(ref_elec_group)
            ):
                logger.warn(
                    f"Omitting electrode group {e_group} from sort groups "
                    + "because contains reference."
                )
                continue
            shank_elect = electrodes["electrode_id"][match_names_bool]

            # omit unitrodes if indicated
            if omit_unitrode and len(shank_elect) == 1:
                logger.warn(
                    f"Omitting electrode group {e_group}, shank {shank} "
                    + "from sort groups because unitrode."
                )
                continue

            sg_keys.append(sg_key)
            sge_keys.extend(
                [
                    {
                        **sge_key,
                        "electrode_id": elect,
                    }
                    for elect in shank_elect
                ]
            )
            sort_group += 1
    return sg_keys, sge_keys
