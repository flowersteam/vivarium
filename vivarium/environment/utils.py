import jax.numpy as jnp
from jax import lax, vmap, random

from jax_md.dataclasses import dataclass as md_dataclass, fields, is_dataclass
from jax_md import space


@vmap
def normal(theta):
    """Returns the cos and the sin of an angle

    :param theta: angle in radians
    :return: cos and sin
    """
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def is_jax_md_dataclass(instance):
    """Check if an instance is a jax md dataclass

    :param instance: instance to check
    :return: True if instance is a jax md dataclass, False otherwise
    """
    return is_dataclass(instance) and hasattr(instance, 'set') and callable(getattr(instance, 'set'))


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


proximity_map = vmap(vmap(relative_position, (0, None)), (0, 0))


# Deprecated?
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
        kwargs['entity_state'] = init_entities_from_rigid_body(rigid_body_state.entity_state)

        return module.State(**kwargs)

    def init_state(*args, **kwargs):
        rigid_body_state = rigid_body_init_state(*args, **kwargs)

        return init_state_from_rigid_body(rigid_body_state)

    return init_state, init_entities


def generate_random_positions(n, position_range, key=random.PRNGKey(0)):
    """
    Generate random positions within a given as a jax array
    :param n: number of positions to generate
    :param position_range: range of positions (x_min, x_max, y_min, y_max)
    :param key: jax random key
    :return: array of random positions
    """
    x_min, x_max, y_min, y_max = position_range
    key_x, key_y = random.split(key)
    x_positions = random.uniform(key_x, shape=(n,), minval=x_min, maxval=x_max)
    y_positions = random.uniform(key_y, shape=(n,), minval=y_min, maxval=y_max)
    return jnp.stack([x_positions, y_positions], axis=-1)


def generate_random_orientations(n, orientation_range, key=random.PRNGKey(0)):
    """
    Generate random orientations within the range [0, 2*pi) as a jax array
    :param n: number of orientations to generate
    :param key: jax random key
    :return: array of random orientations
    """
    min, max = orientation_range
    return random.uniform(key, shape=(n,), minval=min, maxval=max)


def are_two_positions_close(position1, position2, atol):
    """
    Check if two positions are close to each other
    :param position1: first position
    :param position2: second position
    :param atol: absolute tolerance
    :return: boolean array indicating if the positions are close
    """
    return jnp.linalg.norm(position1 - position2, axis=-1) < atol


def is_position_close(position, idx, other_positions, atol):
    closes = vmap(are_two_positions_close, (None, 0, None))(position, other_positions, atol)
    closes = closes.at[idx].set(False)
    return jnp.any(closes, axis=-1)


def neighbors_entity_mask(neighbors_idx, source_mask, target_mask, neighbor_mask):
    target_idx = jnp.where(
        target_mask,
        jnp.arange(target_mask.shape[0], dtype=int),
        -1
    )
    neighbor_mask = jnp.where(
        source_mask[:, jnp.newaxis],
        neighbor_mask,
        False
    )
    return neighbor_mask & jnp.isin(neighbors_idx, target_idx)


def masked_apply(fn, arr, mask, default=0.):
    """
    Note: This was done to potentially speed up jnp.where(mask, fn(arr), arr), hoping it would avoid computing fn(arr) where mask is False.
    However, quick timing tests show that this is not faster than the jnp.where approach.

    Apply a function to elements of an array where the mask is True,
    and return the original value where the mask is False.
    Args:
        fn: Function to apply to the elements of arr.
        arr: Input array.
        mask: Boolean mask array of the same shape as arr.
        default: Default value to return where mask is False.
    Returns:
        jnp.array: Array with fn applied where mask is True, and original values where mask is False.
    """
    def apply_if_masked(x, m):
        # Apply fn if m is True, else return x unchanged
        return lax.cond(m, fn, lambda _: default, x)

    # Vectorize over the array
    return vmap(apply_if_masked)(arr, mask)


def get_relative_displacement(all_positions, source_orientations, source_mask, neighbors, displacement_fn):
    """Get all infos relative to distance and orientation between all agents and their neighbors

    :param state: state
    :param braitenberg_state: braitenberg agents' state
    :param agents_neighs_idx: idx all agents neighbors
    :param displacement_fn: jax md function enabling to know the distance between points
    :return: distance array, angles array, distance map for all agents, angles map for all agents
    """

    dR = space.map_neighbor(displacement_fn)(
        all_positions[source_mask], all_positions[neighbors]
    )

    dist, theta = proximity_map(dR, source_orientations)

    return dist, theta


def type_mask(entity_state, exists=1, entity_type=-1, subtype=-1):
    mask_entity_type = lax.cond(
        entity_type == -1,
        lambda: entity_state.exists == exists,
        lambda: jnp.logical_and(
            entity_state.exists == exists,
            entity_state.entity_type == entity_type
            )
    )
    mask_subtype = lax.cond(
        subtype == -1,
        lambda: entity_state.exists == exists,
        lambda: jnp.logical_and(
            entity_state.exists == exists,
            entity_state.entity_subtype == subtype
        )
    )
    mask = jnp.logical_and(mask_entity_type, mask_subtype)
    return mask
