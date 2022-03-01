from kachery_client._daemon_connection import _kachery_temp_dir

# This will be $KACHERY_TEMP_DIR if this env variable set
# Otherwise, if not set, this will be tempfile.gettempdir() + '/kachery-tmp'
kachery_temp_dir = _kachery_temp_dir()

# Note that in the future, this _kachery_temp_dir() will be exposed in a better way
# from kachery_client. Probably something like:
#
# import kachery_client as kc
# kc.TemporaryDirectory.get_temp_dir()