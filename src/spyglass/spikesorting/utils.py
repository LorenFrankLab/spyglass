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
    sg_keys = list()
    sge_keys = list()
    for e_group in e_groups:
        sg_key = dict()
        sge_key = dict()
        sg_key["nwb_file_name"] = sge_key["nwb_file_name"] = nwb_file_name
        # for each electrode group, get a list of the unique shank numbers
        shank_list = np.unique(
            electrodes["probe_shank"][
                electrodes["electrode_group_name"] == e_group
            ]
        )
        sge_key["electrode_group_name"] = e_group
        # get the indices of all electrodes in this group / shank and set their sorting group
        for shank in shank_list:
            sg_key["sort_group_id"] = sge_key["sort_group_id"] = sort_group
            # specify reference electrode. Use 'references' if passed, otherwise use reference from config
            if not references:
                shank_elect_ref = electrodes["original_reference_electrode"][
                    np.logical_and(
                        electrodes["electrode_group_name"] == e_group,
                        electrodes["probe_shank"] == shank,
                    )
                ]
                if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                    sg_key["sort_reference_electrode_id"] = shank_elect_ref[0]
                else:
                    ValueError(
                        f"Error in electrode group {e_group}: reference "
                        + "electrodes are not all the same"
                    )
            else:
                if e_group not in references.keys():
                    raise Exception(
                        f"electrode group {e_group} not a key in "
                        + "references, so cannot set reference"
                    )
                else:
                    sg_key["sort_reference_electrode_id"] = references[e_group]
            # Insert sort group and sort group electrodes
            reference_electrode_group = electrodes[
                electrodes["electrode_id"]
                == sg_key["sort_reference_electrode_id"]
            ][
                "electrode_group_name"
            ]  # reference for this electrode group
            if len(reference_electrode_group) == 1:  # unpack single reference
                reference_electrode_group = reference_electrode_group[0]
            elif (int(sg_key["sort_reference_electrode_id"]) > 0) and (
                len(reference_electrode_group) != 1
            ):
                raise Exception(
                    "Should have found exactly one electrode group for "
                    + "reference electrode, but found "
                    + f"{len(reference_electrode_group)}."
                )
            if omit_ref_electrode_group and (
                str(e_group) == str(reference_electrode_group)
            ):
                logger.warn(
                    f"Omitting electrode group {e_group} from sort groups "
                    + "because contains reference."
                )
                continue
            shank_elect = electrodes["electrode_id"][
                np.logical_and(
                    electrodes["electrode_group_name"] == e_group,
                    electrodes["probe_shank"] == shank,
                )
            ]
            if (
                omit_unitrode and len(shank_elect) == 1
            ):  # omit unitrodes if indicated
                logger.warn(
                    f"Omitting electrode group {e_group}, shank {shank} "
                    + "from sort groups because unitrode."
                )
                continue
            sg_keys.append(sg_key)
            for elect in shank_elect:
                sge_key["electrode_id"] = elect
                sge_keys.append(sge_key.copy())
            sort_group += 1
