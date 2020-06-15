import logging
import pathlib
import sys

import numpy as np
from franklab_nwb_extensions import fl_extension  # noqa
from pynwb import NWBHDF5IO
from tqdm import tqdm

import datajoint as dj
from pipeline import experiment

IGNORED_LFP_FIELDS = ['electrodes', 'data']


def run_ingest(nwb_dir):
    logging.info(dj.config)
    file_paths = pathlib.Path(nwb_dir).joinpath().glob("*.nwb")
    for file_path in file_paths:
        absolute_path = str(file_path.absolute())
        with NWBHDF5IO(absolute_path, 'r', load_namespaces=True) as io:
            nwbfile = io.read()

            # Experimenter Info
            experiment.Lab.insert1(
                (nwbfile.lab, nwbfile.institution), skip_duplicates=True)
            experiment.Experimenter.insert1(
                (nwbfile.experimenter), skip_duplicates=True)

            # Subject Info
            subj = nwbfile.subject.fields
            # rename description to subject description to maintain uniqueness
            subj['subject_description'] = subj.pop('description')
            subj.pop('weight')  # ignore weight
            experiment.Subject.insert1(subj, skip_duplicates=True)

            # Session Info
            experiment.Session.insert1(
                (nwbfile.subject.subject_id,
                 nwbfile.session_id,
                 nwbfile.experimenter,
                 nwbfile.experiment_description), skip_duplicates=True)

            # Probe Info
            probe_insertions = [
                dict(subject_id=nwbfile.subject.subject_id,
                     session_id=nwbfile.session_id,
                     insertion_number=int(series['group_name']))
                for _, series
                in nwbfile.ec_electrodes.to_dataframe().iterrows()]

            experiment.ProbeInsertion.insert(
                probe_insertions, skip_duplicates=True)

            # LFP Info
            lfp_dict = nwbfile.acquisition['LFP']['electrical_series'].fields

            lfp_entry = {
                f'lfp_{key_name}': value for key_name, value in lfp_dict.items()
                if key_name not in IGNORED_LFP_FIELDS}
            lfp_entry['lfp_timestamps'] = np.array(lfp_entry['lfp_timestamps'])

            for probe_insertion in tqdm(
                    experiment.ProbeInsertion.fetch('KEY'), desc='LFP Info'):
                experiment.LFP.insert1(dict(**lfp_entry, **probe_insertion),
                                       allow_direct_insert=True,
                                       skip_duplicates=True)

            # LFP data
            lfp_data = np.array(lfp_dict['data'])
            lfp_electrode_table = (
                nwbfile.acquisition['LFP']['electrical_series']
                .fields['electrodes'])
            for lfp_ind, row in enumerate(lfp_electrode_table):
                insertion_number = int(row["group_name"].values[0])
                electrode_id = int(row.index.values[0])

                lfp = (experiment.LFP &
                       f'session_id="{nwbfile.session_id}"' &
                       f'insertion_number={insertion_number}'
                       ).fetch1('KEY')
                lfp_channel = dict(**lfp,
                                   electrode_id=electrode_id,
                                   lfp=lfp_data[:, lfp_ind])
                experiment.LFP.Channel.insert1(
                    lfp_channel, allow_direct_insert=True,
                    skip_duplicates=True)


if __name__ == "__main__":
    sys.exit(run_ingest(sys.argv[1]))
