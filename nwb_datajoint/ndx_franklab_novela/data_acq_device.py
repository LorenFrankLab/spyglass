from hdmf import docval
from hdmf.utils import get_docval, call_docval_func
from pynwb import register_class
from pynwb.device import Device


@register_class('DataAcqDevice', 'ndx-franklab-novela')
class DataAcqDevice(Device):
    ''' Representation of Probe object in NWB '''

    @docval(*get_docval(Device.__init__) + (
            {'name': 'system', 'type': 'str', 'doc': 'system of device'},
            {'name': 'amplifier', 'type': 'str', 'doc': 'amplifier', 'default': ''},
            {'name': 'adc_circuit', 'type': 'str', 'doc': 'adc_circuit', 'default': ''},
    ))
    def __init__(self, **kwargs):
        super().__init__(**{kwargs_item: kwargs[kwargs_item]
                            for kwargs_item in kwargs.copy()
                            if kwargs_item != 'system'
                            if kwargs_item != 'amplifier'
                            if kwargs_item != 'adc_circuit'
                            })
        call_docval_func(super(DataAcqDevice, self).__init__, kwargs)
        self.system = kwargs['system']
        self.amplifier = kwargs['amplifier']
        self.adc_circuit = kwargs['adc_circuit']

    __nwbfields__ = ('system', 'amplifier', 'adc_circuit')
