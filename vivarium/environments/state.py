from collections.abc import Iterable

import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass
from jax_md.rigid_body import RigidBody
from jax_md import simulate

from vivarium.utils.scene_configs import import_class


def to_rigid_body_state(entity_state):
    for field in ['position', 'momentum', 'mass', 'force', 'previous_force']:
        val = getattr(entity_state, field)
        if val is None:
            continue
        if field == 'position':
            orientation = getattr(entity_state, 'orientation')
        else:
            if field == 'mass':
                orientation = jnp.ones(val.shape[0])
            else:
                orientation = jnp.zeros(val.shape[0])
        entity_state = entity_state.set(**{field: RigidBody(center=val, orientation=orientation)})
    return entity_state


@md_dataclass
class BaseEntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_type_idx: jnp.array
    exists: jnp.array
    previous_force: jnp.array

    @classmethod
    def create(cls, params, order, **kwargs):
        entity_type = []
        entity_type_idx = []
        exists = []
        fields = [field.name for field in cls.__dataclass_fields__.values()
                  if field.name not in ['entity_type', 'entity_type_idx', 'exists', 'momentum', 'force', 'previous_force']]
        for i, t in enumerate(order):
            entity_params = params['state_data'][t]['kwargs']
            n = entity_params['n_exists']
            entity_type.extend([i] * n)
            entity_type_idx.extend(range(n))
            exists.extend([1] * entity_params['n_exists'] + [0] * (n - entity_params['n_exists']))

        kwargs['mass'] = [[m] for m in kwargs['mass']]
        return cls(entity_type=jnp.array(entity_type), 
                   entity_type_idx=jnp.array(entity_type_idx),
                   exists=jnp.array(exists),
                   momentum=None,
                   force=jnp.zeros_like(jnp.array(kwargs['position'])),
                   previous_force=jnp.zeros_like(jnp.array(kwargs['position'])),
                   **{attr: jnp.array(values) for attr, values in kwargs.items() if attr in fields},
                   )

    def is_rigid_body(self):
        return hasattr(self.position, 'center')

    def count(self):
        return self.entity_type_idx.shape[0]

    def __getattr__(self, name):
        prefix, suffix = name.split('_', 1)
        if prefix == "unified":
            if suffix == 'orientation':
                if self.is_rigid_body():
                    return self.position.orientation
                return self.orientation
            if self.is_rigid_body():
                return getattr(self, suffix).center
            return getattr(self, suffix)
        if suffix in ['center', 'orientation']:
            if self.is_rigid_body():
                return getattr(getattr(self, prefix), suffix)
            if suffix == 'center':
                return getattr(self, prefix)
            else:  # Necessarily 'orientation'
                return self.orientation if prefix == 'position' else jnp.zeros_like(self.orientation)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
          

@md_dataclass
class EntityState(BaseEntityState):
    entity_subtype: jnp.array
    diameter: jnp.array
    friction: jnp.array
    orientation: jnp.array

    @classmethod
    def create(cls, params):
        order = params['state_data']['entity_state']['entity_types']
        ent_subtype = []
        fields = [field.name for field in cls.__dataclass_fields__.values()]
        kwargs = {}
        for entity_type in order:
            attributes = params['state_data'][entity_type]['kwargs']
            for f in fields:
                if f in attributes:
                    if f not in kwargs:
                        kwargs[f] = []
                    if not isinstance(attributes[f], Iterable) or len(attributes[f]) == 1:
                        attributes[f] = [attributes[f]] * attributes['n_exists']
                    kwargs[f].extend(attributes[f])
            for st, n in attributes['subtype_to_n']:
                ent_subtype.extend([st] * n)
        base_instance = BaseEntityState.create(params, order, **kwargs)
        return cls(entity_subtype=jnp.array(ent_subtype), **base_instance.__dict__,
                   **{attr: jnp.array(val) for attr, val in kwargs.items() if attr not in base_instance.__dict__})


@md_dataclass
class BaseParticleState:
    entity_idx: jnp.array

    @classmethod
    def create(cls, entity_idx_offset, n_entities):
        return cls(entity_idx=jnp.array(range(entity_idx_offset, entity_idx_offset + n_entities)))
    
    def count(self):
        return self.entity_idx.shape[0]
    

@md_dataclass
class ParticleState(BaseParticleState):

    @classmethod
    def create(cls, entity_idx_offset, n_entities):
        return cls(entity_idx=jnp.array(range(entity_idx_offset, entity_idx_offset + n_entities)))

    @classmethod
    def _create(cls, entity_idx_offset, params, entity_type_field, **kwargs):
        cls_params = params['state_data'][entity_type_field]['kwargs']
        fields = [field.name for field in cls.__dataclass_fields__.values()]
        cls_kwargs = {attr: jnp.array(val) for attr, val in cls_params.items() if attr in fields}
        base_instance = BaseParticleState.create(entity_idx_offset, cls_params['n_exists'])
        return cls(**cls_kwargs, **base_instance.__dict__, **kwargs)
    

class BaseState:
    dt: jnp.float32
    collision_alpha: jnp.float32
    collision_eps: jnp.float32

    def e_cond(self, etype):
        if isinstance(etype, str):
            etype = self.entity_type_to_int(etype)
        return self.entity_state.entity_type == etype

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.entity_state, name)
            if isinstance(value, RigidBody):
                return RigidBody(
                    center=value.center[self.e_cond(e_type)],
                    orientation=value.orientation[self.e_cond(e_type)],
                )
            else:
                return value[self.e_cond(e_type)]

        return wrapper


def create_state_cls(base_state_cls, entity_types, **kwargs):  
    # First make a "copy" of the base class. This is just for pytest, otherwise modify base_state_cls in a test function will have side effect on others. 
    class State(base_state_cls):
        __annotations__ = base_state_cls.__annotations__.copy()
        
    for field, cls in kwargs.items():
        State.__annotations__[field] = cls

    def entity_type_to_int(self, name):
        if name not in entity_types:
            raise ValueError(f"Entity type '{name}' not found in entity_types.")
        return entity_types.index(name)
    
    def entity_type_to_str(self, idx):
        if idx < 0 or idx >= len(entity_types):
            raise ValueError(f"Entity type index '{idx}' out of range.")
        return entity_types[idx]

    
    State.entity_type_to_int = entity_type_to_int
    State.entity_type_to_str = entity_type_to_str

    return md_dataclass(State)


def create_state_cls_from_params(params):
    etype_order = params['entity_state']['entity_types']

    State = create_state_cls(
        base_state_cls=import_class(params['state']['cls']),
        entity_types=etype_order,
        entity_state=import_class(params['entity_state']['cls']),
        **{field: import_class(params[field]['cls']) for field in etype_order}
    )

    return State

def create_state(params):

    CustomState = create_state_cls_from_params(params['state_data'])

    etype_order = params['state_data']['entity_state']['entity_types']

    entity_idx_offset = 0
    etype_instance = {}
    for field in etype_order:
        cls = import_class(params['state_data'][field]['cls'])
        etype_instance[field] = cls.create(entity_idx_offset, params, field)
        entity_idx_offset += params['state_data'][field]['kwargs']['n_exists']

    entity_state_cls = import_class(params['state_data']['entity_state']['cls'])
    state = CustomState(**{attr: jnp.array(val) for attr, val in params['state_data']['state']['kwargs'].items()},
                    entity_state = entity_state_cls.create(params),  #, etype_order), 
                    **etype_instance)
    
    return state


def class_to_string(cls):
    return cls.__name__.lower().replace('state', '_state')
