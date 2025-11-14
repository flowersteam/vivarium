from omegaconf import DictConfig
from dataclasses import field, make_dataclass

import jax.numpy as jnp
import numpy as np

from vivarium.environment.utils import is_jax_md_dataclass


def create_dataclass_from_dict(class_name: str, data: dict):
    """
    Dynamically creates a dataclass from a dictionary, recursively handling nested dictionaries.

    Args:
        class_name (str): The name of the dataclass.
        data (dict): The dictionary to define fields for the dataclass.

    Returns:
        A dataclass instance populated with the dictionary values.
    """
    def process_value(key, value):
        if isinstance(value, (dict, DictConfig)):
            # Recursively create a nested dataclass for dictionaries
            nested_class_name = f"{class_name}_{key.capitalize()}"
            return (key, create_dataclass_from_dict(nested_class_name, value).__class__, field(default_factory=lambda: create_dataclass_from_dict(nested_class_name, value)))
        else:
            return (key, type(value), field(default_factory=type(value)))

    # Create fields for the dataclass
    fields = [process_value(key, value) for key, value in data.items()]
    
    # Dynamically create the dataclass
    DynamicDataclass = make_dataclass(class_name, fields)

    # Add a dummy method to the dataclass
    def getitem(self, idx):
        return DynamicDataclass(**{field: getattr(self, field)[idx] for field in self.__dataclass_fields__.keys()})   
    setattr(DynamicDataclass, "__getitem__", getitem)
    
    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{class_name}' object has no attribute '{key}'")
        return self
    setattr(DynamicDataclass, "set", set)

    # Create an instance of the dataclass
    instance = DynamicDataclass(**{key: (value if not isinstance(value, (dict, DictConfig)) else create_dataclass_from_dict(f"{class_name}_{key.capitalize()}", value)) for key, value in data.items()})
    
    return instance

def update_dataclass_from_change_list(dataclass_instance, change_list):
    for changes in change_list:
        dataclass_instance = update_dataclass(dataclass_instance, changes)
    return dataclass_instance


#TODO: Is it possible to jit part of this?
def update_dataclass(dataclass_instance, changes):
    if isinstance(changes, list):
        for change in changes:
            if change['__idx'] is None:
                if isinstance(dataclass_instance, jnp.ndarray):
                    dataclass_instance = jnp.array(change['__value'])
                elif isinstance(dataclass_instance, np.ndarray):
                    dataclass_instance = np.array(change['__value'])
                else:
                    dataclass_instance = change['__value']
            else:
                if isinstance(dataclass_instance, np.ndarray):
                    dataclass_instance[change['__idx']] = change['__value']
                else:
                    if isinstance(dataclass_instance, jnp.ndarray):
                        dataclass_instance = dataclass_instance.at[change['__idx']].set(change['__value'])
                    else:
                        if change['__idx'] == (None,):
                            dataclass_instance = change['__value']
                        else:
                            dataclass_instance[change['__idx']] = change['__value']
    else:
        for attr, child in changes.items():
            if is_jax_md_dataclass(dataclass_instance): # is_dataclass(dataclass_instance) and hasattr(dataclass_instance, 'set') and callable(getattr(dataclass_instance, 'set')):
                dataclass_instance = dataclass_instance.set(**{attr: update_dataclass(getattr(dataclass_instance, attr), child)})
            else:
                setattr(dataclass_instance, attr, update_dataclass(getattr(dataclass_instance, attr), child))

    return dataclass_instance


class Remote:
    def __init__(self, obj=None, set_obj=False, root=None, path=()):
        self._obj = obj  # Local copy of the data
        self._set_obj = set_obj  # Whether to set self._obj on assignment
        self._path = path  # Path to this object from the root (for operations)

        # Root proxy handles batching
        if root is None:
            self._root = self # Needed?
            self._pending_ops = {}
        else:
            self._root = root

    def __getattr__(self, name):
        try:
            attr = getattr(self._obj, name) if self._obj is not None else None
        except AttributeError as e:
            raise AttributeError(f"{type(self._obj).__name__} has no attribute '{name}'") from e
        return self._wrap(attr, self._path + (name,))

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if self._obj is not None and self._set_obj:
                if is_jax_md_dataclass(self._obj):
                    self._obj = self._obj.set(**{name: value})
                else:
                    setattr(self._obj, name, value)
            full_path = self._path + (name,)
            self._root._record_operation(full_path, value)

    def __getitem__(self, key):
        try:
            item = self._obj[key] if self._obj is not None else None
        except (IndexError, KeyError, TypeError) as e:
            raise type(e)(f"Invalid access at path {self._path}: {e}") from e
        return self._wrap(item, self._path + (key,))

    def __setitem__(self, idx, value):
        self._root._record_operation(self._path, value, idx=idx)

    def obj(self):
        return self._obj

    def _wrap(self, obj, path):
        # Wrap nested objects to propagate proxy behavior
        if isinstance(obj, (np.ndarray, jnp.ndarray)) and len(obj.shape) == 0:
            # for consistency with numpy scalars
            return obj.item()
        if isinstance(obj, (list, dict, np.ndarray, Remote)) or (hasattr(obj, '__dict__')) or obj is None:
            return Remote(obj, set_obj=self._set_obj, root=self._root, path=path)
        return obj  # Primitives don't need wrapping

    def _record_operation(self, path, value, idx=None):
        self._rec(path, value, self._pending_ops, idx)

    def _rec(self, path, value, pending_ops, idx):
        if len(path) == 0:
            return [{'__idx': idx, '__value': value}]
        else:
            if path[0] in pending_ops:
                if len(path) > 1:
                    pending_ops[path[0]].update(self._rec(path[1:], value, pending_ops[path[0]], idx))
                else:
                    pending_ops[path[0]].extend(self._rec(path[1:], value, pending_ops[path[0]], idx))
            else:
                pending_ops[path[0]] = self._rec(path[1:], value, pending_ops[path[0]] if path[0] in pending_ops else {}, idx)
            return pending_ops

    def fetch_changes(self):
        pending_ops = self._pending_ops
        self._pending_ops = {}
        return [pending_ops] if len(pending_ops) > 0 else []
    
    def apply(self, obj=None, changes=None):
        changes = changes or self.fetch_changes()
        if obj is None:
            assert self._obj is not None, 'An object must be provided either in the constructor or as argument of this method.'
            self._obj = update_dataclass_from_change_list(self._obj, changes)
            return self._obj
        obj = update_dataclass_from_change_list(obj, changes)
        return obj
