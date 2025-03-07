import numpy as np
import jax.numpy as jnp


def update_state_from_change_list(state, change_list):
    for changes in change_list:
        state = update_state(state, changes)
    return state


#TODO: Is it possible to jit part of this?
def update_state(state, changes):
    if isinstance(changes, list):
        for change in changes:
            if change['__idx'] is None:
                state = change['__value']
            else:
                if isinstance(state, np.ndarray):
                    state[change['__idx']] = change['__value']
                else:
                    state = state.at[change['__idx']].set(change['__value'])
    else:
        for attr, child in changes.items():
            state = state.set(**{attr: update_state(getattr(state, attr), child)})
    return state


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
            return getattr(getattr(self._state.entity_state, field_name), rigid_body_field)[self._ent_idx]
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    return self._state.entity_state.orientation[self._ent_idx]
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                return getattr(self._state.entity_state, field_name)[self._ent_idx]
            else:
                return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")

    @prop.setter
    def prop(self, value, idx=None):
        if idx is None:
            idx = self._ent_idx
        else:
            idx = (self._ent_idx, idx)
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


class DataclassWrapper:
    def __init__(self, state=None):
        object.__setattr__(self, '_root_change_recorder', ChangeRecorder())
        object.__setattr__(self, '_last_change_recorder', self._root_change_recorder)
        object.__setattr__(self, '_state', state)
        object.__setattr__(self, '_nested_fields', [])

    def _reinit(self):
        object.__setattr__(self, '_last_change_recorder', self._root_change_recorder)
        object.__setattr__(self, '_nested_fields', [])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        #TODO: The possibility to use this class with an internal state might not be needed, 
        # EntityState does this (but its usecase is the notebook controller, not raw state as here)
        if self._state is not None:
            if attr in self._state.__dict__:
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
        state = self._state
        for field in self._nested_fields:
            state = getattr(state, field)
        if isinstance(getattr(state, attr), (np.ndarray, jnp.ndarray)):
            self._reinit()
            return getattr(state, attr)
        else:
            return None

    def set(self, value):
        self._last_change_recorder.store_change(value)
        return self

    def update_state(self, state, changes):
        state = update_state(state, changes)
        return state
    
    def fetch_changes(self):
        changes = self._root_change_recorder.fetch_changes()
        self._root_change_recorder = ChangeRecorder()
        self._last_change_recorder = self._root_change_recorder
        return changes

    def apply(self, state=None):
        state = state or self._state
        assert state is not None, 'State must be provided either in the constructor or as argument of this method.'
        changes = self.fetch_changes()
        state = update_state(state, changes)
        if self._state is not None:
            self._state = state
        return state


class SimulatorStateWrapper:
    def __init__(self, state):
        object.__setattr__(self, '_state', state)
        object.__setattr__(self, '_change_recorder', ChangeRecorder())

    def __getattr__(self, attr):
        return getattr(self._state.simulator_state, attr)

    def _setitem(self, attr, value, idx=None):
        setattr(self._change_recorder.simulator_state, attr, value)

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        self._setitem(attr, value)

    def fetch_changes(self):
        changes = self._change_recorder.fetch_changes()
        return changes

    def apply_to_state(self, state):
        changes = self.fetch_changes()
        self._state = update_state(state, changes)
        self._change_recorder = ChangeRecorder()
        return self._state
    
    def set_state(self, state):
        self._state = state


class EntityWrapper:
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
        object.__setattr__(self, '_ent_idx', ent_idx)
        object.__setattr__(self, '_is_rigid_body', self._state.entity_state.is_rigid_body())
        object.__setattr__(self, '_change_recorder', ChangeRecorder())
        object.__setattr__(self, '_entity_type', entity_type)
        object.__setattr__(self, '_entity_type_attr', entity_type.name.lower() + '_state')
        object.__setattr__(self, '_entity_fields', ['ent_subtype', 'diameter', 'friction',
                               'exists', 'entity_idx', 'entity_type',
                               'position', 'momentum', 'force', 'mass',
                               'position_center', 'position_orientation',
                               'momentum_center', 'momentum_orientation',
                               'force_center', 'force_orientation',
                               'mass_center', 'mass_orientation'])

    def __getattr__(self, attr):
        if attr in self._entity_fields:
            return getattr(self._state.entity_state, attr)[self._ent_idx]
        return getattr(getattr(self._state, self._entity_type_attr), attr)[self._state.entity_state.entity_idx[self._ent_idx]]

    def _setitem(self, attr, value, idx=None):
        entity_state_idx = self._ent_idx if idx is None else (self._ent_idx, idx)
        x_state_idx = self._state.entity_state.entity_idx[self._ent_idx] if idx is None else (self._state.entity_state.entity_idx[self._ent_idx], idx)
        if attr in self._entity_fields:
            if attr.endswith('_center') or attr.endswith('_orientation'):
                field_name, rigid_body_field = attr.split('_', 1)
                p = create_property(field_name, rigid_body_field)
                p.fset(self, value, idx)
            else:
                getattr(self._change_recorder.entity_state, attr)[entity_state_idx] = value
        else:
            getattr(getattr(self._change_recorder, self._entity_type_attr), attr)[x_state_idx] = value

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        self._setitem(attr, value)

    def apply_to_state(self, state):
        changes = self._change_recorder.fetch_changes()
        self._state = update_state(state, changes)
        self._change_recorder = ChangeRecorder()
        return self._state
    
    def set_state(self, state):
        self._state = state


class EntityList:
    def __init__(self, state, entity_type, entity_wrapper_list=None):
        self._state = state
        self._entity_type = entity_type.name.lower() + 's'
        self._entity_list = entity_wrapper_list or [EntityWrapper(state, idx, entity_type) for idx, type in enumerate(state.entity_state.entity_type) if type == entity_type.value]

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
