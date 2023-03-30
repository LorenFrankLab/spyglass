import numpy as np
import pathlib
import probeinterface as pi

# from spyglass.common import Probe


def create_probe(probe_json_path: str):
    """Create a spyglass Probe object (with its Shanks and Electrodes) from a ProbeInterface file."""
    probe_group = pi.read_probeinterface(probe_json_path)

    for p in probe_group.probes:
        assert (
            len(np.unique(p.shank_ids)) == 1
        )  # multi-shank is not yet supported in this code
        assert p.ndim == 2  # 1-D and 3-D probes are not yet supported in this code
        assert p.si_units == "um"

        probe_dict = dict()
        probe_dict["probe_type"] = p.annotations["probe_name"]
        probe_dict["probe_description"] = str(p.annotations)
        probe_dict["num_shanks"] = 1
        probe_dict["contact_side_numbering"] = "False"

        # probe type already exists in Probe table
        if probe_dict["probe_type"] in Probe.fetch("probe_type"):
            continue  # skip the entire probe

        print(f"Prepopulating table Probe with data {probe_dict}")
        Probe.insert1(probe_dict)

        shank_dict = dict()
        shank_dict["probe_type"] = probe_dict["probe_type"]
        shank_dict["probe_shank"] = 0
        Probe.Shank.insert1(shank_dict)

        for elect_id, (p, s, sp) in enumerate(
            zip(p.contact_positions, p.contact_shapes, p.contact_shape_params)
        ):
            elect_dict = dict()
            elect_dict["probe_type"] = probe_dict["probe_type"]
            elect_dict["probe_shank"] = shank_dict["probe_shank"]
            if s == "square":
                elect_dict["contact_size"] = sp["width"]
            elif s == "circle":
                elect_dict["contact_size"] = sp["radius"]
            elif s == "rect":
                if "width" in sp and "height" not in sp:
                    elect_dict["contact_size"] = sp["width"]
                else:
                    raise ValueError(
                        "Rectangular contact shape must have width and not height."
                    )
            else:
                raise ValueError(
                    f"Contact shapes that are not rect, square, or circle are not yet supported: {s}"
                )
            elect_dict["probe_electrode"] = elect_id
            elect_dict["rel_x"] = p[0]
            elect_dict["rel_y"] = p[1]
            Probe.Electrode.insert1(elect_dict)


def create_probes():
    create_probe(pathlib.Path(__file__).resolve().parent / "NP1_standard_config.json")
