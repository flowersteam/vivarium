from dataclasses import field, make_dataclass

import jax.numpy as jnp
import numpy as np

from omegaconf import DictConfig
from jax_md.dataclasses import is_dataclass

from vivarium.environment.state import field_accessors


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
            if is_dataclass(dataclass_instance):
                dataclass_instance = dataclass_instance.set(**{attr: update_dataclass(getattr(dataclass_instance, attr), child)})
            else:
                setattr(dataclass_instance, attr, update_dataclass(getattr(dataclass_instance, attr), child))

    return dataclass_instance


class ChangeRecorder:
    def __init__(self, name=None, idx=None):   
        self._name = name
        self.__idx = idx
        self._changes = []
        self._children = {}

    @property
    def _idx(self):
        return self.__idx

    @_idx.setter
    def _idx(self, value):
        if isinstance(value, jnp.ndarray):
            if len(value.shape) > 0:
                value = value.tolist()
            else:
                value = value.item()
        self.__idx = value

    def store_change(self, value):
        self._changes.append({'__idx': self._idx, '__value': value})

    def fetch_changes(self):
        
        if self._changes:
            changes = self._changes
            self._changes = []
        else:
            changes = {}
            for attr, child in self._children.items():
                changes[attr] = child.fetch_changes()
            self._children = {}
        return changes

    def __getitem__(self, idx):
        self._idx = idx
        return self

    def __setitem__(self, idx, value):
        self._idx = idx
        self.store_change(value)

    def __getattr__(self, attr):
        if attr.startswith('_'):
            return super().__getattribute__(attr)
        if attr not in self._children and attr != '__iter__':
            if attr == '__iter__':
                print('__iter__!!')
            self._children[attr] = ChangeRecorder(name=attr)
        return self._children[attr]
    
    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            super().__setattr__(attr, value)
        else:
            getattr(self, attr).store_change(value)


def create_property(field_name, rigid_body_field):
    @property
    def prop(self):
        if self._is_rigid_body:
            return getattr(getattr(self._state.entity_state, field_name), rigid_body_field)[self._entity_idx]
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    return self._state.entity_state.orientation[self._entity_idx]
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                return getattr(self._state.entity_state, field_name)[self._entity_idx]
            else:
                return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")

    @prop.setter
    def prop(self, value, idx=None):
        if idx is None:
            idx = self._entity_idx
        else:
            idx = (self._entity_idx, idx)
        if self._is_rigid_body:
            getattr(getattr(self._change_recorder.entity_state, field_name), rigid_body_field)[idx] = value
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    self._change_recorder.entity_state.orientation[idx] = value
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                getattr(self._change_recorder.entity_state, field_name)[idx] = value
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
    return prop

@field_accessors
class DataclassWrapper:
    def __init__(self, dataclass_instance=None):
        object.__setattr__(self, '_root_change_recorder', ChangeRecorder())
        object.__setattr__(self, '_last_change_recorder', self._root_change_recorder)
        object.__setattr__(self, '_dataclass_instance', dataclass_instance)
        object.__setattr__(self, '_nested_fields', [])

    def _reinit(self):
        object.__setattr__(self, '_last_change_recorder', self._root_change_recorder)
        object.__setattr__(self, '_nested_fields', [])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        #TODO: The possibility to use this class with an internal state might not be needed, 
        # EntityState does this (but its usecase is the notebook controller, not raw state as here)
        if self._dataclass_instance is not None:
            if attr in self._dataclass_instance.__dict__:
                self._reinit()
            leaf = self._get_if_leaf(attr)
            if leaf is not None:
                print(f"Returning leaf {attr}")
                return leaf
            self._nested_fields.append(attr)
        
        self._last_change_recorder = getattr(self._last_change_recorder, attr)
        return self

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        getattr(self._last_change_recorder, attr).store_change(value)
    
    def __getitem__(self, idx):
        self._last_change_recorder = self._last_change_recorder[idx]
        return self
    
    def __setitem__(self, idx, value):
        self._last_change_recorder[idx] = value

    def _get_if_leaf(self, attr):
        dataclass_instance = self._dataclass_instance
        for field in self._nested_fields:
            dataclass_instance = getattr(dataclass_instance, field)
        if isinstance(getattr(dataclass_instance, attr), (np.ndarray, jnp.ndarray)):
            self._reinit()
            return getattr(dataclass_instance, attr)
        else:
            return None

    def set(self, value):
        self._last_change_recorder.store_change(value)
        return self

    def update_dataclass(self, dataclass_instance, changes):
        dataclass_instance = update_dataclass(dataclass_instance, changes)
        return dataclass_instance
    
    def fetch_changes(self):
        changes = self._root_change_recorder.fetch_changes()
        self._root_change_recorder = ChangeRecorder()
        self._last_change_recorder = self._root_change_recorder
        return changes

    def apply(self, dataclass_instance=None):
        dataclass_instance = dataclass_instance or self._dataclass_instance
        assert dataclass_instance is not None, 'A dataclass instance must be provided either in the constructor or as argument of this method.'
        changes = self.fetch_changes()
        dataclass_instance = update_dataclass(dataclass_instance, changes)
        if self._dataclass_instance is not None:
            self._dataclass_instance = dataclass_instance
        return dataclass_instance

  
class SimulatorParametersWrapper:
    def __init__(self, simulator_parameters):
        object.__setattr__(self, '_simulator_parameters', simulator_parameters)
        object.__setattr__(self, '_change_recorder', ChangeRecorder())

    def __getattr__(self, attr):
        return getattr(self._simulator_parameters, attr)

    def _setitem(self, attr, value, idx=None):
        setattr(self._change_recorder, attr, value)

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        self._setitem(attr, value)

    def fetch_changes(self):
        changes = self._change_recorder.fetch_changes()
        return changes

    def apply_to_state(self, simulator):  # TODO: change method name
        changes = self.fetch_changes()
        self._simulator = update_dataclass(simulator, changes)
        self._change_recorder = ChangeRecorder()
        return self._simulator


class EntityWrapper:
    """
    Wraps a State into an interface to manipulate a single entitty (usually from a SimulatorController)
    Modifying attributes of an EntityWrapper will not change the state immediately, but will record changes
    that can be applied to the state later using the `apply_to_state` method. This is useful for batch updates
    during client-server interactions.
    """
    position_center = create_property('position', 'center')
    momentum_center = create_property('momentum', 'center')
    force_center = create_property('force', 'center')
    mass_center = create_property('mass', 'center')

    position_orientation = create_property('position', 'orientation')
    momentum_orientation = create_property('momentum', 'orientation')
    force_orientation = create_property('force', 'orientation')
    mass_orientation = create_property('mass', 'orientation')

    def __init__(self, state, ent_idx, entity_type):
        object.__setattr__(self, '_state', state)
        object.__setattr__(self, '_entity_idx', ent_idx)
        object.__setattr__(self, '_entity_type_idx', state.entity_state.entity_type_idx[ent_idx])
        object.__setattr__(self, '_is_rigid_body', self._state.entity_state.is_rigid_body())
        object.__setattr__(self, '_change_recorder', ChangeRecorder())
        object.__setattr__(self, '_entity_type', entity_type)
        object.__setattr__(self, '_entity_fields', ['entity_subtype', 'diameter', 'friction',
                               'exists', 'entity_idx', 'entity_type',
                               'position', 'momentum', 'force', 'mass',
                               'position_center', 'position_orientation',
                               'momentum_center', 'momentum_orientation',
                               'force_center', 'force_orientation',
                               'mass_center', 'mass_orientation'])

    def __getattr__(self, attr):
        if attr in self._entity_fields:
            return getattr(self._state.entity_state, attr)[self._entity_idx]
        return getattr(getattr(self._state, self._entity_type), attr)[self._state.entity_state.entity_type_idx[self._entity_idx]]

    def _setitem(self, attr, value, idx=None):
        entity_state_idx = self._entity_idx if idx is None else (self._entity_idx, idx)
        x_state_idx = self._state.entity_state.entity_type_idx[self._entity_idx] if idx is None else (self._state.entity_state.entity_type_idx[self._entity_idx], idx)
        if attr in self._entity_fields:
            if attr.endswith('_center') or attr.endswith('_orientation'):
                field_name, rigid_body_field = attr.split('_', 1)
                p = create_property(field_name, rigid_body_field)
                p.fset(self, value, idx)
            else:
                getattr(self._change_recorder.entity_state, attr)[entity_state_idx] = value
        else:
            getattr(getattr(self._change_recorder, self._entity_type), attr)[x_state_idx] = value

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        self._setitem(attr, value)

    def apply_to_state(self, state):
        changes = self._change_recorder.fetch_changes()
        self._state = update_dataclass(state, changes)
        self._change_recorder = ChangeRecorder()
        return self._state
    
    def set_state(self, state):
        self._state = state


class EntityList:
    def __init__(self, state, entity_type, entity_type_idx, entity_wrapper_list=None):
        self._state = state
        self._entity_type = entity_type
        self._entity_list = entity_wrapper_list or [EntityWrapper(state, idx, entity_type) for idx, type in enumerate(state.entity_state.entity_type) if type == entity_type_idx]

    def __getitem__(self, idx):
        return self._entity_list[idx]
    
    def __setitem__(self, idx, value):
        raise NotImplementedError('Setting values directly is not supported.')
    
    def __iter__(self):
        return iter(self._entity_list)
    
    def __len__(self):
        return len(self._entity_list)
    
    def __repr__(self):
        return repr(self._entity_list)

    def apply_to_state(self, state):
        for entity in self._entity_list:
            state = entity.apply_to_state(state)
        return state
    
    def set_state(self, state):
        for entity in self._entity_list:
            entity.set_state(state)

    def fetch_changes(self):
        changes = []
        for e in self._entity_list:
            c = e._change_recorder.fetch_changes()
            if c:
                changes.append(c)
        return changes
