from hdmf.utils import docval, call_docval_func, get_docval
from pynwb import register_class
from pynwb.core import NWBDataInterface


@register_class('AssociatedFiles', 'ndx-franklab-novela')
class AssociatedFiles(NWBDataInterface):
    """ Representation of associated files in NWB """

    __nwbfields__ = ('description', 'content', 'task_epochs')

    @docval(*get_docval(NWBDataInterface.__init__) + (
            {'name': 'description', 'type': 'str', 'doc': 'description of associated file'},
            {'name': 'content', 'type': 'str', 'doc': 'content of associated file'},
            {'name': 'task_epochs', 'type': 'str', 'doc': 'epochs this task belongs to'},
            ))
    def __init__(self, **kwargs):
        super().__init__(**{kwargs_item: kwargs[kwargs_item]
                            for kwargs_item in kwargs.copy()
                            if kwargs_item not in ['description', 'content', 'task_epochs']
                            })
        call_docval_func(super(AssociatedFiles, self).__init__, kwargs)
        self.description = kwargs['description']
        self.content = kwargs['content']
        self.task_epochs = kwargs['task_epochs']

