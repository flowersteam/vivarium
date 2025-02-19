import jax.numpy as jnp
from jax import vmap

from jax_md.dataclasses import dataclass as md_dataclass, fields


@vmap
def normal(theta):
    """Returns the cos and the sin of an angle

    :param theta: angle in radians
    :return: cos and sin
    """
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def distance(displacement_fn, point1, point2):
    """Returns the distance between two points

    :param displacement_fn: displacement function (typically a jax_md.space function)
    :param point1: point 1
    :param point2: point 2
    :return: distance between the two points
    """
    diff = displacement_fn(point1, point2)
    squared_diff = jnp.sum(jnp.square(diff))
    return jnp.sqrt(squared_diff)


def relative_position(displ, theta):
    """
    Compute the relative distance and angle from a source particle to a target particle
    :param displ: Displacement vector (jnp arrray with shape (2,) from source to target
    :param theta: Orientation of the source particle (in the reference frame of the map)
    :return: dist: distance from source to target.
    relative_theta: relative angle of the target in the reference frame of the source particle (front direction at angle 0)
    """
    dist = jnp.linalg.norm(displ)
    norm_displ = displ / dist
    theta_displ = jnp.arccos(norm_displ[0]) * jnp.sign(jnp.arcsin(norm_displ[1]))
    relative_theta = theta_displ - theta
    return dist, relative_theta


def rigid_body_to_point_particle(module):

    @md_dataclass
    class EntityState(module.EntityState):
        orientation: jnp.array

    def convert(rigid_body_state, point_particle_field):
        if point_particle_field in ['position', 'force', 'previous_force', 'mass']:
            return getattr(rigid_body_state, point_particle_field).center
        if point_particle_field == 'orientation':
            return rigid_body_state.position.orientation
        return getattr(rigid_body_state, point_particle_field)
    
    state_fields = [f.name for f in fields(module.State)]
    entities_state_fields = [f.name for f in fields(EntityState)]

    rigid_body_init_entities = module.init_entities
    rigid_body_init_state = module.init_state


    def init_entities_from_rigid_body(rigid_body_entity_state):
        return EntityState(**{field: convert(rigid_body_entity_state, field) for field in entities_state_fields})

    def init_entities(*args, **kwargs):
        rigid_body_entity_state = rigid_body_init_entities(*args, **kwargs)

        return init_entities_from_rigid_body(rigid_body_entity_state)


    def init_state_from_rigid_body(rigid_body_state):

        kwargs = {field: convert(rigid_body_state, field) for field in state_fields}
        kwargs['entities'] = init_entities_from_rigid_body(rigid_body_state.entities)

        return module.State(**kwargs)

    def init_state(*args, **kwargs):
        rigid_body_state = rigid_body_init_state(*args, **kwargs)

        return init_state_from_rigid_body(rigid_body_state)

    return init_state, init_entities

