#!/usr/bin/env python

import os

def garbage_collection(*,
    sha1_hashes_to_keep,
    nwb_datajoint_base_dir,
    kachery_storage_dir,
    relative_path
):
    # we should be very careful with these checks
    assert isinstance(nwb_datajoint_base_dir, str)
    assert nwb_datajoint_base_dir != ''
    assert kachery_storage_dir == nwb_datajoint_base_dir + '/kachery-storage'
    assert os.path.exists(kachery_storage_dir)

    kachery_storage_recycling_dir = nwb_datajoint_base_dir + '/kachery-storage-recycling'
    if not os.path.exists(kachery_storage_recycling_dir):
        os.mkdir(kachery_storage_recycling_dir)

    path = os.path.join(kachery_storage_dir, relative_path)
    assert os.path.exists(path)
    path_recycling = os.path.join(kachery_storage_recycling_dir, relative_path)
    if not os.path.exists(path_recycling):
        os.mkdir(path_recycling)
    
    for fname in os.listdir(path):
        filepath = path + '/' + fname
        filepath_recycling = path_recycling + '/' + fname
        if os.path.isfile(filepath):
            if len(fname) == 40:
                hash0 = fname
                if filepath.endswith(f'{hash0[0]}{hash0[1]}/{hash0[2]}{hash0[3]}/{hash0[4]}{hash0[5]}/{hash0}'):
                    if hash0 not in sha1_hashes_to_keep:
                        print(f'Recycling file: {hash0}')
                        os.rename(filepath, filepath_recycling)
        elif os.path.isdir(filepath):
            relp = os.path.join(relative_path, fname)
            garbage_collection(
                sha1_hashes_to_keep=sha1_hashes_to_keep,
                nwb_datajoint_base_dir=nwb_datajoint_base_dir,
                kachery_storage_dir=kachery_storage_dir,
                relative_path=relp
            )


def main():
    sha1_hashes_to_keep = set()
    # Here is where we need to populate the sha1_hashes_to_keep set.
    # sha1_hashes_to_keep.add('6253ee04e09a5145eae0ced9c26ce73b91876de4')
    garbage_collection(
        sha1_hashes_to_keep=sha1_hashes_to_keep,
        nwb_datajoint_base_dir=os.environ['NWB_DATAJOINT_BASE_DIR'],
        kachery_storage_dir=os.environ['KACHERY_STORAGE_DIR'],
        relative_path='sha1'
    )

if __name__ == '__main__':
    main()
