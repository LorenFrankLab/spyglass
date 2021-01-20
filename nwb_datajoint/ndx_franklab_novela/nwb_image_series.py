from hdmf import docval
from hdmf.utils import get_docval, call_docval_func
from pynwb import register_class
from pynwb.core import MultiContainerInterface
from pynwb.device import Device
from pynwb.image import ImageSeries


@register_class('NwbImageSeries', 'ndx-franklab-novela')
class NwbImageSeries(ImageSeries, MultiContainerInterface):
    ''' Extension of ImageSeries object in NWB '''

    @docval(*get_docval(ImageSeries .__init__) + (
            {'name': 'devices', 'type': (list, tuple), 'doc': 'devices used to record video', 'default': list()},
    ))
    def __init__(self, **kwargs):
        super().__init__(**{kwargs_item: kwargs[kwargs_item]
                            for kwargs_item in kwargs.copy()
                            if kwargs_item != 'devices'
                            })
        call_docval_func(super(NwbImageSeries, self).__init__, kwargs)
        self.devices = kwargs['devices']

    __clsconf__ = [
        {
            'attr': 'devices',
            'type': Device,
            'add': 'add_device',
            'get': 'get_device'
        }
    ]
