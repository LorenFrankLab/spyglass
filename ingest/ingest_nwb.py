import sys

import numpy as np
from franklab_nwb_extensions import fl_extension  # noqa
from pynwb import NWBHDF5IO
from tqdm import tqdm

from pipeline import experiment

IGNORED_LFP_FIELDS = ['electrodes', 'data']


def main():

    with NWBHDF5IO(
            '../data/original_data/bon_04.nwb', 'r',
            load_namespaces=True) as io:
        nwbfile = io.read()

        # Experimenter Info
        experiment.Lab.insert1(
            (nwbfile.lab, nwbfile.institution), skip_duplicates=True)
        experiment.Experimenter.insert1(
            (nwbfile.experimenter), skip_duplicates=True)

        # Subject Info
        subj = nwbfile.subject.fields
        subj['subject_description'] = subj.pop('description')
        subj.pop('weight')
        experiment.Subject.insert1(subj)

        # Session Info
        experiment.Session.insert1(
            (nwbfile.subject.subject_id,
             nwbfile.session_id,
             nwbfile.experimenter,
             nwbfile.experiment_description)
        )

        # Probe Info
        probe_insertions = [
            dict(subject_id=nwbfile.subject.subject_id,
                 session_id=nwbfile.session_id,
                 insertion_number=int(series['group_name']))
            for _, series in nwbfile.ec_electrodes.to_dataframe().iterrows()]

        experiment.ProbeInsertion.insert(
            probe_insertions, skip_duplicates=True)

        # LFP Info
        lfp_dict = nwbfile.acquisition['LFP']['electrical_series'].fields

        lfp_entry = {f'lfp_{key_name}': value
                     for key_name, value in lfp_dict.items()
                     if key_name not in IGNORED_LFP_FIELDS}

        lfp_entry['lfp_timestamps'] = np.array(lfp_entry['lfp_timestamps'])

        for probe_insertion in tqdm(experiment.ProbeInsertion.fetch('KEY')):
            experiment.LFP.insert1(dict(**lfp_entry, **probe_insertion),
                                   allow_direct_insert=True,
                                   skip_duplicates=True)

        # LFP data
        lfp_data = np.array(lfp_dict['data'])
        lfp_electrode_table = (
            nwbfile.acquisition['LFP']['electrical_series']
            .fields['electrodes'])
        for idx, row in tqdm(enumerate(lfp_electrode_table)):
            lfp = (experiment.LFP &
                   f'insertion_number={int(row["group_name"].values[0])}'
                   ).fetch1('KEY')

            lfp_channel = dict(**lfp,
                               electrode_id=int(row.index.tolist()[0]),
                               lfp=lfp_data[idx])
            experiment.LFP.Channel.insert1(
                lfp_channel, allow_direct_insert=True)


if __name__ == "__main__":
    sys.exit(main())
