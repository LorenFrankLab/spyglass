from hdmf.utils import docval, call_docval_func, get_docval
from pynwb import register_class
from pynwb.device import Device


@register_class('HeaderDevice', 'ndx-franklab-novela')
class HeaderDevice(Device):
    """Configuration comes from SpikeGadgets recording system. Represented as HeaderDevice in NWB"""

    __nwbfields__ = ('headstage_serial', 'headstage_smart_ref_on', 'realtime_mode', 'headstage_auto_settle_on',
                     'timestamp_at_creation', 'conntroller_firmware_version', 'controller_serial',
                     'save_displayed_chan_only', 'headstage_firmwavare_version', 'qt_version', 'compile_date',
                     'compile_time', 'file_prefix', 'headstage_gyro_sensor_on', 'headstage_mag_sensor_on',
                     'trodes_version', 'headstage_accel_sensor_on', 'commit_head', 'system_time_at_creation',
                     'file_path')

    @docval(*get_docval(Device.__init__) + (
            {'name': 'headstage_serial', 'type': 'str', 'doc': 'headstage_serial from header global configuration'},
            {'name': 'headstage_smart_ref_on', 'type': 'str', 'doc': 'headstage_smart_ref_on from header global configuration'},
            {'name': 'realtime_mode', 'type': 'str', 'doc': 'realtime_mode from header global configuration'},
            {'name': 'headstage_auto_settle_on', 'type': 'str', 'doc': 'headstage_auto_settle_on from header global configuration'},
            {'name': 'timestamp_at_creation', 'type': 'str', 'doc': 'timestamp_at_creation from header global configuration'},
            {'name': 'controller_firmware_version', 'type': 'str', 'doc': 'conntroller_firmware_version from header global configuration'},
            {'name': 'controller_serial', 'type': 'str', 'doc': 'conntroller_serial from header global configuration'},
            {'name': 'save_displayed_chan_only', 'type': 'str', 'doc': 'save_displayed_chan_only from header global configuration'},
            {'name': 'headstage_firmware_version', 'type': 'str', 'doc': 'headstage_firmware_version from header global configuration'},
            {'name': 'qt_version', 'type': 'str', 'doc': 'qt_version from header global configuration'},
            {'name': 'compile_date', 'type': 'str', 'doc': 'compile_date from header global configuration'},
            {'name': 'compile_time', 'type': 'str', 'doc': 'compile_time from header global configuration'},
            {'name': 'file_prefix', 'type': 'str', 'doc': 'file_prefix from header global configuration'},
            {'name': 'headstage_gyro_sensor_on', 'type': 'str', 'doc': 'headstage_gyro_sensor_on from header global configuration'},
            {'name': 'headstage_mag_sensor_on', 'type': 'str', 'doc': 'headstage_mag_sensor_on from header global configuration'},
            {'name': 'trodes_version', 'type': 'str', 'doc': 'trodes_version from header global configuration'},
            {'name': 'headstage_accel_sensor_on', 'type': 'str', 'doc': 'headstage_accel_sensor_on from header global configuration'},
            {'name': 'commit_head', 'type': 'str', 'doc': 'commit_head from header global configuration'},
            {'name': 'system_time_at_creation', 'type': 'str', 'doc': 'system_time_at_creation from header global configuration'},
            {'name': 'file_path', 'type': 'str', 'doc': 'file_path from header global configuration'}
            ))
    def __init__(self, **kwargs):
        super().__init__(**{kwargs_item: kwargs[kwargs_item]
                            for kwargs_item in kwargs.copy()
                            if kwargs_item not in
                                    ['headstage_serial', 'headstage_smart_ref_on', 'realtime_mode', 'headstage_auto_settle_on',
                                     'timestamp_at_creation', 'controller_firmware_version', 'controller_serial',
                                     'save_displayed_chan_only', 'headstage_firmware_version', 'qt_version', 'compile_date',
                                     'compile_time', 'file_prefix', 'headstage_gyro_sensor_on', 'headstage_mag_sensor_on',
                                     'trodes_version', 'headstage_accel_sensor_on', 'commit_head', 'system_time_at_creation',
                                     'file_path']
                            })
        call_docval_func(super(HeaderDevice, self).__init__, kwargs)
        self.headstage_serial = kwargs['headstage_serial']
        self.headstage_smart_ref_on = kwargs['headstage_smart_ref_on']
        self.realtime_mode = kwargs['realtime_mode']
        self.headstage_auto_settle_on = kwargs['headstage_auto_settle_on']
        self.timestamp_at_creation = kwargs['timestamp_at_creation']
        self.controller_firmware_version = kwargs['controller_firmware_version']
        self.controller_serial = kwargs['controller_serial']
        self.save_displayed_chan_only = kwargs['save_displayed_chan_only']
        self.headstage_firmware_version = kwargs['headstage_firmware_version']
        self.qt_version = kwargs['qt_version']
        self.compile_date = kwargs['compile_date']
        self.compile_time = kwargs['compile_time']
        self.file_prefix = kwargs['file_prefix']
        self.headstage_gyro_sensor_on = kwargs['headstage_gyro_sensor_on']
        self.headstage_mag_sensor_on = kwargs['headstage_mag_sensor_on']
        self.trodes_version = kwargs['trodes_version']
        self.headstage_accel_sensor_on = kwargs['headstage_accel_sensor_on']
        self.commit_head = kwargs['commit_head']
        self.system_time_at_creation = kwargs['system_time_at_creation']
        self.file_path = kwargs['file_path']
