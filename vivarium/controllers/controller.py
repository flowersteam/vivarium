from functools import reduce
from operator import attrgetter

from vivarium.controllers.dataclass_wrapper import Remote

def set_nested_attr(obj, attr_path, value):
    *path, final = attr_path.split('.')
    target = reduce(getattr, path, obj)
    setattr(target, final, value)
    
class AttributeMapping:
    def __init__(self, remote_attr, ctrl_attr=None, remote_to_ctrl_fn=None, ctrl_to_remote_fn=None):
        self.remote_attr = remote_attr
        self.ctrl_attr = ctrl_attr if ctrl_attr is not None else remote_attr
        self.remote_to_ctrl_fn = remote_to_ctrl_fn if remote_to_ctrl_fn is not None else lambda x: x
        self.ctrl_to_remote_fn = ctrl_to_remote_fn if ctrl_to_remote_fn is not None else lambda x: x
        

class Controller:
    def __init__(self, name, remote, mapping={}):
        self._name = name
        self._remote = remote
        self._mapping = mapping

    @classmethod
    def from_config(cls, name, remote, mapping={}):
        return cls(
            name=name,
            remote=remote,
            mapping=mapping,
        )

    @property
    def name(self):
        return self._name
    
    def to_deal_with(self, attr):
        return not(attr.startswith('_') or attr in self._mapping)

    def __getattr__(self, attr):
        if attr.startswith('_'):
            return object.__getattr__(attr)
        elif attr in self._mapping:
            return self.remote_to_ctrl(attr)

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        elif attr in self._mapping:
            attr, value = self.ctrl_to_remote(attr, value)
            set_nested_attr(self._remote, attr, value)
        else:
            pass

    def remote_to_ctrl(self, attr):
            mapping = self._mapping[attr]
            remote_value = attrgetter(mapping.remote_attr)(self._remote)
            remote_value = remote_value.obj() if isinstance(remote_value, Remote) else remote_value
            return mapping.remote_to_ctrl_fn(remote_value)

    def ctrl_to_remote(self, attr, value=None):
        if attr in self._mapping:
            mapping = self._mapping[attr]
            attr = mapping.remote_attr
            if value is not None:
                value = mapping.ctrl_to_remote_fn(value)
        return (attr, value) if value is not None else attr

    def step(self, time, catch_errors):
        pass
