from hdmf.utils import docval, call_docval_func, get_docval
from pynwb import register_class
from pynwb.device import Device


@register_class('CameraDevice', 'ndx-franklab-novela')
class CameraDevice(Device):
    """Represented as CameraDevice in NWB"""

    __nwbfields__ = ('meters_per_pixel', 'camera_name', 'model', 'lens')

    @docval(*get_docval(Device.__init__) + (
            {'name': 'meters_per_pixel', 'type': float, 'doc': 'meters per pixel'},
            {'name': 'camera_name', 'type': str, 'doc': 'name of camera'},
            {'name': 'model', 'type': str, 'doc': 'model of this camera device'},
            {'name': 'lens', 'type': str, 'doc': 'lens info'},
            ))
    def __init__(self, **kwargs):
        super().__init__(**{kwargs_item: kwargs[kwargs_item]
                            for kwargs_item in kwargs.copy()
                            if kwargs_item not in ['meters_per_pixel', 'camera_name', 'model', 'lens']
                            })
        call_docval_func(super(CameraDevice, self).__init__, kwargs)
        self.meters_per_pixel = kwargs['meters_per_pixel']
        self.camera_name = kwargs['camera_name']
        self.model = kwargs['model']
        self.lens = kwargs['lens']


