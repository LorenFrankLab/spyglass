import os
import shutil
import tempfile
import time


class TemporaryDirectory():
    def __init__(self, remove: bool=True, prefix: str='tmp'):
        self._remove = remove
        self._prefix = prefix

    def __enter__(self) -> str:
        if 'KACHERY_STORAGE_DIR' in os.environ:
            storage_dir = os.getenv('KACHERY_STORAGE_DIR')
        else:
            storage_dir = None
        if storage_dir is not None:
            dirpath = os.path.join(storage_dir, 'tmp')
            if not os.path.exists(dirpath):
                try:
                    os.mkdir(dirpath)
                except:
                    # maybe somebody else created this directory
                    if not os.path.exists:
                        raise Exception(f'Unexpected problem creating temporary directory: {dirpath}')
        else:
            dirpath = None
        self._path = str(tempfile.mkdtemp(prefix=self._prefix, dir=dirpath))
        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._remove:
            _rmdir_with_retries(self._path, num_retries=5)

    def path(self):
        return self._path


def _rmdir_with_retries(dirname: str, num_retries: int, delay_between_tries: float=1):
    for retry_num in range(1, num_retries + 1):
        if not os.path.exists(dirname):
            return
        try:
            shutil.rmtree(dirname)
            break
        except: # pragma: no cover
            if retry_num < num_retries:
                print('Retrying to remove directory: {}'.format(dirname))
                time.sleep(delay_between_tries)
            else:
                raise Exception('Unable to remove directory after {} tries: {}'.format(num_retries, dirname))
