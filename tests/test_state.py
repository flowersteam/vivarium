import enum
import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import simulate
import pytest

# class EntityType(enum.Enum):
#     BRAITENBERG = 0
#     OBJECT = 1


@md_dataclass
class BaseEntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_type_idx: jnp.array
    # exists: jnp.array
    # previous_force: jnp.array

    @classmethod
    def create(cls, type_to_kwargs, order):
        entity_type = []
        entity_type_idx = []
        fields = [field.name for field in cls.__dataclass_fields__.values() if field.name not in ['entity_type', 'entity_type_idx','momentum', 'force']]
        kwargs_all_entities = {f: [] for f in fields}
        for i, t in enumerate(order):
            #TODO Generate random positions if not provided (but needs box_size and a random key)
            kwargs = type_to_kwargs[t]
            n = len(kwargs['position'])
            entity_type.extend([i] * n)
            entity_type_idx.extend(range(n))
            for f in fields:
                values = kwargs[f]
                kwargs_all_entities[f].extend(values)
        return cls(entity_type=jnp.array(entity_type), 
                   entity_type_idx=jnp.array(entity_type_idx),
                   **{attr: jnp.array(values) for attr, values in kwargs_all_entities.items()}, # if attr in fields},
                   momentum=None,
                   force=jnp.zeros_like(jnp.array(kwargs_all_entities['position']))
                   )

    def is_rigid_body(self):
        return hasattr(self.position, 'center')

    # def __getattr__(self, name):
    #     prefix, suffix = name.split('_', 1)
    #     if prefix == "unified":
    #         if suffix == 'orientation':
    #             if isinstance(self.position, RigidBody):
    #                 return self.position.orientation
    #             return self.orientation
    #         if isinstance(getattr(self, suffix), RigidBody):
    #             return getattr(self, suffix).center
    #         return getattr(self, suffix)
    #     if suffix in ['center', 'orientation']:
    #         if isinstance(self.position, RigidBody):
    #             return getattr(getattr(self, prefix), suffix)
    #         if suffix == 'center':
    #             return getattr(self, prefix)
    #         else:  # Necessarily 'orientation'
    #             return self.orientation if prefix == 'position' else jnp.zeros_like(self.orientation)
    #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


@md_dataclass
class EntityState(BaseEntityState):
    entity_subtype: jnp.array
    # diameter: jnp.array
    # friction: jnp.array

    @classmethod
    def create(cls, params, order=None):
        order = params[params['entity_state_cls']]['order']
        ent_subtype = []
        for entity_type in order:
            attributes = params[entity_type]
            for st, n in attributes['subtype_to_n']:
                ent_subtype.extend([st] * n)
        base_instance = BaseEntityState.create(params, order)
        return cls(entity_subtype=jnp.array(ent_subtype), **base_instance.__dict__)

@md_dataclass
class BaseParticleState:
    entity_idx: jnp.array

    @classmethod
    def create(cls, entity_idx_offset, n_entities):
        return cls(entity_idx=jnp.array(range(entity_idx_offset, entity_idx_offset + n_entities)))
    
@md_dataclass
class ParticleState(BaseParticleState):
    # pass

    @classmethod
    def _create(cls, entity_idx_offset, state_params, **kwargs):
        cls_params = state_params[cls]
        fields = [field.name for field in cls.__dataclass_fields__.values()]
        cls_kwargs = {attr: jnp.array(val) for attr, val in cls_params.items() if attr in fields}
        base_instance = BaseParticleState.create(entity_idx_offset, len(cls_params['position']))
        return cls(**cls_kwargs, **base_instance.__dict__, **kwargs)


@md_dataclass
class BraitenbergState(ParticleState):
    prox: jnp.array
    prox_sensed_ent_type: jnp.array
    prox_sensed_ent_idx: jnp.array
    motor: jnp.array
    proximity_map_dist: jnp.array
    proximity_map_theta: jnp.array
    behavior: jnp.array
    params: jnp.array
    sensed: jnp.array
    wheel_diameter: jnp.array
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array
    
    @classmethod
    def create(cls, entity_idx_offset, state_params):
        n_entities = len(state_params[cls]['position'])
        n_total_entities = sum([len(state_params[_cls]['position']) for _cls in state_params[state_params['entity_state_cls']]['order']])
        n_behaviors = state_params[cls]['n_behaviors']
        n_subtypes = 0
        for c, p in state_params.items():
            if isinstance(p, dict) and 'subtype_to_n' in p:
                n_subtypes = max(n_subtypes, max([stn[0] + 1 for stn in p['subtype_to_n']]))
        proximity_map_dist = jnp.zeros((n_entities, n_total_entities))
        proximity_map_theta = jnp.zeros((n_entities, n_total_entities))
        return cls._create(entity_idx_offset, state_params, 
                           prox=jnp.zeros((n_entities, 2)),
                           motor=jnp.zeros((n_entities, 2)),
                           prox_sensed_ent_type=jnp.zeros((n_entities, 2), dtype=int),
                           prox_sensed_ent_idx=jnp.zeros((n_entities, 2), dtype=int),
                           proximity_map_dist=proximity_map_dist,
                           proximity_map_theta=proximity_map_theta,
                           behavior=jnp.full(n_behaviors, 5, dtype=int),
                           params= jnp.zeros((n_behaviors, 2, 3)),
                           sensed=jnp.ones((n_behaviors, n_subtypes), dtype=int) #TODO: Check if this is correct
                           )
        # return base_instance
    #     return cls(proximity_map_dist=proximity_map_dist, **base_instance.__dict__)

@md_dataclass
class ObjectState(ParticleState):
    @classmethod
    def create(cls, entity_idx_offset, params):
        return cls._create(entity_idx_offset, params)


class BaseState:
    time: jnp.int32
    box_size: jnp.int32


class State(BaseState):
    max_agents: jnp.int32
    max_objects: jnp.int32
    neighbor_radius: jnp.float32
    dt: jnp.float32  # Give a more explicit name
    collision_alpha: jnp.float32
    collision_eps: jnp.float32
    # ent_sub_types: dict


def create_state_cls(base_state_cls, **kwargs):    
    for field, cls in kwargs.items():
        base_state_cls.__annotations__[field] = cls
    return md_dataclass(base_state_cls)


def create_state(params):

    etype_order = params[params['entity_state_cls']]['order']

    CustomState = create_state_cls(
        params['state_cls'],
        entity_state=params['entity_state_cls'],
        **{class_to_string(cls): cls for cls in etype_order}
    )

    entity_idx_offset = 0
    etype_instance = {}
    for cls in etype_order:
        assert class_to_string(cls) in CustomState.__annotations__
        etype_instance[class_to_string(cls)] = cls.create(entity_idx_offset, params)
        entity_idx_offset += len(params[cls]['position'])

    state = CustomState(**{attr: jnp.array(val) for attr, val in params[params['state_cls']].items()},
                    entity_state = params['entity_state_cls'].create(params, etype_order), 
                    **etype_instance)
    
    return state


def class_to_string(cls):
    return cls.__name__.lower().replace('state', '_state')


params = {
    'state_cls': BaseState,
    'entity_state_cls': EntityState,
    BaseState: {
        'time': 0,
        'box_size': 100
    },
    EntityState: {
        'order': [BraitenbergState, ObjectState]
    },
    BraitenbergState: {
        'mass': [1, 2],
        'position': [[0, 0], [1, 1]],
        # 'orientation': [0, 1],
        'n_behaviors': 4,
        'subtype_to_n': [(0, 1), (1, 1)],
        'wheel_diameter': 2.0,
        'proxs_dist_max': 20.0,
        'proxs_cos_min': 0.0
    },
    ObjectState: {
        'mass': [1],
        'position': [[1, 2]],
        # 'orientation': [0, 1],
        'subtype_to_n': [(2, 1)]
    }
}

def test_class_to_string():
    assert class_to_string(BraitenbergState) == 'braitenberg_state'
    assert class_to_string(ObjectState) == 'object_state'


@pytest.mark.parametrize("entity_type_cls, entity_idx_offset", [
    (BraitenbergState, 0),
    (ObjectState, 2),
])
def test_create_entity_type_state(entity_type_cls, entity_idx_offset):
    cls = entity_type_cls
    state = cls.create(entity_idx_offset, params)

    assert jnp.equal(state.entity_idx, jnp.array(range(entity_idx_offset, entity_idx_offset + len(params[cls]['position'])))).all()

    if entity_type_cls == BraitenbergState:
        assert jnp.equal(state.prox, jnp.array([[0, 0], [0, 0]])).all()
        assert state.sensed.shape == (4, 3)


def test_create_entity_state():

    entity_state = EntityState.create(params)
    
    assert jnp.equal(entity_state.entity_type, jnp.array([0, 0, 1])).all()
    assert jnp.equal(entity_state.entity_type_idx, jnp.array([0, 1, 0])).all()
    assert jnp.equal(entity_state.entity_subtype, jnp.array([0, 1, 2])).all()
    assert jnp.equal(entity_state.mass, jnp.array([1, 2, 1])).all()
    assert jnp.equal(entity_state.position, jnp.array([[0, 0], [1, 1], [1, 2]])).all()
    # assert jnp.equal(entity_state.orientation, jnp.array([0, 1, 0])).all()
    assert jnp.equal(entity_state.force, jnp.array([[0, 0], [0, 0], [0, 0]])).all()




def test_create_state():

    state = create_state(params)

    assert jnp.equal(state.entity_state.entity_type, jnp.array([0, 0, 1])).all()
    assert jnp.equal(state.entity_state.entity_type_idx, jnp.array([0, 1, 0])).all()
    assert jnp.equal(state.entity_state.entity_subtype, jnp.array([0, 1, 2])).all()
    assert jnp.equal(state.entity_state.mass, jnp.array([1, 2, 1])).all()
    assert jnp.equal(state.entity_state.position, jnp.array([[0, 0], [1, 1], [1, 2]])).all()
    assert jnp.equal(state.braitenberg_state.prox, jnp.array([[0, 0], [0, 0]])).all()

