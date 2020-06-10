from hdmf import docval
from hdmf.utils import get_docval, call_docval_func
from pynwb import register_class
from pynwb.ecephys import ElectrodeGroup


@register_class('NwbElectrodeGroup', 'ndx-franklab-novela')
class NwbElectrodeGroup(ElectrodeGroup):
    ''' Representation of custom ElectrodeGroup object in NWB '''

    __nwbfields__ = ('targeted_location', 'targeted_x', 'targeted_y', 'targeted_z', 'units')

    @docval(*get_docval(ElectrodeGroup.__init__) + (
            {'name': 'targeted_location', 'type': 'str', 'doc': 'predicted location'},
            {'name': 'targeted_x', 'type': 'float', 'doc': 'predicted x coordinates'},
            {'name': 'targeted_y', 'type': 'float', 'doc': 'predicted y coordinates'},
            {'name': 'targeted_z', 'type': 'float', 'doc': 'predicted z coordinates'},
            {'name': 'units', 'type': 'str', 'doc': 'units of fields, possible value: um or mm'}))
    def __init__(self, **kwargs):
        super().__init__(**{kwargs_item: kwargs[kwargs_item]
                            for kwargs_item in kwargs.copy()
                            if kwargs_item != 'targeted_location'
                            if kwargs_item != 'targeted_x'
                            if kwargs_item != 'targeted_y'
                            if kwargs_item != 'targeted_z'
                            if kwargs_item != 'units'
                            })
        call_docval_func(super(NwbElectrodeGroup, self).__init__, kwargs)
        self.targeted_location = kwargs['targeted_location']
        self.targeted_x = kwargs['targeted_x']
        self.targeted_y = kwargs['targeted_y']
        self.targeted_z = kwargs['targeted_z']
        self.units = kwargs['units']
