import jax.numpy as jnp

def update_state_from_change_list(state, change_list):
    for changes in change_list:
        state = update_state(state, changes)
    return state

def update_state(state, changes):
    if isinstance(changes, list):
        for change in changes:
            if change['__idx'] is None:
                state = change['__value']
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

    def store_change(self, attr, value):
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
        return ChangeRecorder(name=self._name, idx=idx)

    def __setitem__(self, idx, value):
        self._idx = idx
        self.store_change(self._name, value)

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
            self.store_change(attr, value)

def create_property(field_name, rigid_body_field):
    @property
    def prop(self):
        if self._is_rigid_body:
            return getattr(getattr(self._state.entities, field_name), rigid_body_field)[self._ent_idx]
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    return self._state.entities.orientation[self._ent_idx]
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                return getattr(self._state.entities, field_name)[self._ent_idx]
            else:
                return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")

    @prop.setter
    def prop(self, value):
        if self._is_rigid_body:
            getattr(getattr(self._change_recorder.entities, field_name), rigid_body_field)[self._ent_idx] = value
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    self._change_recorder.entities.orientation[self._ent_idx] = value
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                getattr(self._change_recorder.entities, field_name)[self._ent_idx] = value
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
    return prop


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
        self._state = state
        self._ent_idx = ent_idx
        self._is_rigid_body = self._state.entities.is_rigid_body()
        self._change_recorder = ChangeRecorder()
        self._entity_type = entity_type.name.lower() + 's'
        self._entity_fields = ['ent_subtype', 'diameter', 'friction',
                               'exists', 'entity_idx', 'entity_type',
                               'position', 'momentum', 'force', 'mass',
                               'position_center', 'position_orientation',
                               'momentum_center', 'momentum_orientation',
                               'force_center', 'force_orientation',
                               'mass_center', 'mass_orientation']

    def __getattr__(self, attr):
        if attr in self._entity_fields:
            return getattr(self._state.entities, attr)[self._ent_idx]
        return getattr(getattr(self._state, self._entity_type), attr)[self._state.entities.entity_idx[self._ent_idx]]

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            super().__setattr__(attr, value)
            return
        if attr in self._entity_fields:
            if attr.endswith('_center') or attr.endswith('_orientation'):
                field_name, rigid_body_field = attr.split('_', 1)
                p = create_property(field_name, rigid_body_field)
                p.fset(self, value)
            else:
                getattr(self._change_recorder.entities, attr)[self._ent_idx] = value
        else:
            getattr(getattr(self._change_recorder, self._entity_type), attr)[self._state.entities.entity_idx[self._ent_idx]] = value

    def update_state(self, state):
        changes = self._change_recorder.fetch_changes()
        self._state = update_state(state, changes)
        self._change_recorder = ChangeRecorder()
        return self._state


class EntityList:
    def __init__(self, state, entity_type):
        self._state = state
        self._entity_type = entity_type.name.lower() + 's'
        self._entity_list = [EntityWrapper(state, idx, entity_type) for idx, type in enumerate(state.entities.entity_type) if type == entity_type.value]
    
    def __getitem__(self, idx):
        return self._entity_list[idx]
    
    def __setitem__(self, idx, value):
        raise NotImplementedError('Setting values directly is not supported.')
    
    def __iter__(self):
        return iter(self._entity_list)
    
    def __len__(self):
        return len(self._entity_list)

    def update_state(self, state):
        for entity in self._entity_list:
            state = entity.update_state(state)
        return state