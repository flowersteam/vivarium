import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass
from jax_md.rigid_body import RigidBody
from jax_md import simulate


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
    orientation: jnp.array
    entity_type: jnp.array
    entity_type_idx: jnp.array
    exists: jnp.array
    entity_subtype: jnp.array
    previous_force: jnp.array
    diameter: jnp.array
    friction: jnp.array

    def add_new_entities(self, positions, orientations, mass, diameter, friction, exists, entity_type, entity_subtype, entity_type_idx):
        return self.set(
            position = jnp.vstack([self.position, positions]),
            orientation = jnp.hstack([self.orientation, orientations]),
            mass = jnp.vstack([self.mass, mass.reshape((-1, 1))]),
            diameter = jnp.hstack([self.diameter, diameter]),
            friction = jnp.hstack([self.friction, friction]),
            force = jnp.vstack([self.force, jnp.zeros_like(positions)]),
            previous_force = jnp.vstack([self.previous_force, jnp.zeros_like(positions)]),
            exists = jnp.hstack([self.exists, exists]),
            entity_subtype = jnp.hstack([self.entity_subtype, entity_subtype]),
            entity_type = jnp.hstack([self.entity_type, entity_type]),
            entity_type_idx = jnp.hstack([self.entity_type_idx, entity_type_idx])
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
class BaseParticleState:
    entity_type: jnp.array
    entity_idx: jnp.array

    def count(self):
        return self.entity_idx.shape[0]
    

# TODO: There might be a better way to map field names to entity type integer
# See https://docs.jax.dev/en/latest/pytrees.html#explicit-key-paths
def field_accessors(cls):
    """
    Decorator to add `field_name` and `field` methods to a class.
    """
    def field_name(self, cls):
        """
        Get the attribute name of the class from the dataclass fields.
        Args:
            cls: The class to search for in the dataclass fields.
        Returns:
            str: The attribute name of the class in the dataclass fields.
        """
        attr_name = None
        for field_name, field_value in self.__dataclass_fields__.items():
            if field_value.type == cls:
                attr_name = field_name
        return attr_name

    def field(self, cls):
        """
        Get the entity type state from the class.
        Args:
            cls: The class to search for in the dataclass fields.
        Returns:
            dataclass: The entity type state from the class.
        """
        attr_name = self.field_name(cls)
        if attr_name is None:
            raise ValueError(f"Class {cls} not found in dataclass fields.")
        return getattr(self, attr_name)

    cls.field_name = field_name
    cls.field = field
    return cls


@field_accessors
class BaseState:
    time: jnp.ndarray

    def entity_type_to_int(self, entity_type):
        return getattr(self, entity_type).entity_type

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


def create_state_cls(base_state_cls, update_fns):
    # First make a "copy" of the base class. This is just for pytest, otherwise modify base_state_cls in a test function will have side effect on others. 
    class State(base_state_cls):
        __annotations__ = base_state_cls.__annotations__.copy()
    State.__annotations__['entity_state'] = BaseEntityState
    for fn in update_fns:
        State = fn(State)
    State = md_dataclass(State)
    return State
