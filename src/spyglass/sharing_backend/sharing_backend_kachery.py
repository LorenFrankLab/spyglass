from typing import Union
import os
from kachery_cloud.TaskBackend import TaskBackend
import kachery_cloud as kcl


# This is a simple example script demonstrating a backend service
# that will listen for requests on remote computers to upload files
# that are stored locally.
#
# You can store some content locally via
# uri = kcl.store_text_local('random-text-00001')
# Now this file is stored locally but not in the cloud
# On a remote computer you can send a request to upload
# the file. See the file_share_remote_test.py example
# and paste in the desired url and the project ID
# for this backend.

# uri is obtained from kcl.store_*_local(fname) on this computer
def kachery_store_shared_file(*, uri: str):
    """uploads the shared file when requested. Note that the uri must be sent by the task on the client

    Parameters
    ----------
    uri : str
        the uri for the file

    Raises
    ------
    Exception
        Raises Unable to load file if file can't be loaded
    """
    # impose restrictions on uri here
    if uri != '' and uri is not None:
        fname = kcl.load_file(uri, local_only=True) # requires kachery-cloud >= 0.1.19
        if fname is not None:
            print(f'storing {fname} in cloud')
            kcl.store_file(fname)
    else:
        raise Exception(f'Unable to load file: {uri}')

def start_backend(*, project_id: Union[str, None]=None):
    X = TaskBackend(project_id=project_id)
    X.register_task_handler(
        task_type='action',
        task_name='kachery_store_shared_file.1',
        task_function=kachery_store_shared_file
    )
    print('Starting kachery cloud backend')

    # Backend will listen for requests to upload a file to kachery cloud
    X.run()

if __name__ == '__main__':
    if 'KACHERY_CLOUD_PROJECT' in os.environ:
        start_backend(project_id=os.environ['KACHERY_CLOUD_PROJECT'])
    else:
        start_backend()