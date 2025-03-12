import jax.numpy as jnp
from jax_md.rigid_body import RigidBody
from jax_md import simulate
from jax_md.dataclasses import dataclass as md_dataclass
from dataclasses import make_dataclass, dataclass
import enum

class EntityType(enum.Enum):
    BRAITENBERG = 0
    OBJECT = 1


@md_dataclass
class BaseEntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_type_idx: jnp.array
    # exists: jnp.array
    # previous_force: jnp.array

    @classmethod
    def create(cls, type_to_kwargs):
        entity_type = []
        entity_type_idx = []
        kwargs_all_entities = {k: [] for k in type_to_kwargs[list(type_to_kwargs.keys())[0]].keys()}
        for t, kwargs in type_to_kwargs.items():
            #TODO Generate random positions if not provided (but needs box_size and a random key)
            n = len(kwargs['mass'])
            entity_type.extend([t.value] * n)
            entity_type_idx.extend(range(n))
            for attr, values in kwargs.items():
                kwargs_all_entities[attr].extend(values)
        return cls(entity_type=jnp.array(entity_type), 
                   entity_type_idx=jnp.array(entity_type_idx),
                   **{attr: jnp.array(values) for attr, values in kwargs_all_entities.items()},
                   momentum=None)
            


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
    def create(cls, type_to_kwargs):
        ent_subtype = []
        for entity_type, attributes in type_to_kwargs.items():
            for st, n in attributes['subtype_to_n']:
                ent_subtype.extend([st] * n)
        for entity_type, attributes in type_to_kwargs.items():
            del attributes['subtype_to_n']
            assert 'subtype_to_n' not in type_to_kwargs[entity_type]
        base_instance = BaseEntityState.create(type_to_kwargs)
        return cls(entity_subtype=jnp.array(ent_subtype), **base_instance.__dict__)

@md_dataclass
class BaseParticleState:
    entity_idx: jnp.array

    @classmethod
    def create(cls, entity_idx_offset, n_entities):
        return cls(entity_idx=jnp.array(range(entity_idx_offset, entity_idx_offset + n_entities)))
    
@md_dataclass
class ParticleState(BaseParticleState):
    pass


@md_dataclass
class BraitenbergState(ParticleState):
    prox: jnp.array
    # prox_sensed_ent_type: jnp.array
    # prox_sensed_ent_idx: jnp.array
    # motor: jnp.array
    # proximity_map_dist: jnp.array
    # proximity_map_theta: jnp.array
    # behavior: jnp.array
    # params: jnp.array
    # sensed: jnp.array
    # wheel_diameter: jnp.array
    # proxs_dist_max: jnp.array
    # proxs_cos_min: jnp.array




@md_dataclass
class ObjectState(ParticleState):
    pass


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


def test_create_entity_state():

    entity_state = EntityState.create(
        {
        EntityType.BRAITENBERG: {
            'mass': [1, 2],
            'position': [[0, 0], [1, 1]],
            # 'orientation': [0, 1],
            'force': [[0, 0], [1, 1]],
            'subtype_to_n': [(0, 1), (1, 1)]
            },
        EntityType.OBJECT: {
            'mass': [1],
            'position': [[1, 2]],
            # 'orientation': [0, 1],
            'force': [[0, 0], [1, 1]],
            'subtype_to_n': [(2, 1)]
            }
        })
    
    assert jnp.equal(entity_state.entity_type, jnp.array([0, 0, 1])).all()
    assert jnp.equal(entity_state.entity_type_idx, jnp.array([0, 1, 0])).all()
    assert jnp.equal(entity_state.entity_subtype, jnp.array([0, 1, 2])).all()
    assert jnp.equal(entity_state.mass, jnp.array([1, 2, 1])).all()
    assert jnp.equal(entity_state.position, jnp.array([[0, 0], [1, 1], [1, 2]])).all()
    # assert jnp.equal(entity_state.orientation, jnp.array([0, 1, 0])).all()
    assert jnp.equal(entity_state.force, jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]])).all()


def test_create_state():

    MyState = create_state_cls(
        State,
        entity_state=EntityState,
        object_state=ObjectState,
        agent_state=BraitenbergState
    )
    # class MyState:
    #     time: jnp.int32
    #     box_size: jnp.int32
    # MyState.__annotations__['sub1'] = MdC1
    # state_cls = md_dataclass(MyState)
    # state = state_cls(time=jnp.array(0), box_size=jnp.array(100), sub1=MdC1(a=1, b=2))
    # assert state.time == 0
