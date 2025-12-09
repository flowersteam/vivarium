import logging

import jax
import jax.numpy as jnp

from jax_md import rigid_body, util


lg = logging.getLogger(__name__)

f32 = util.f32

SPACE_NDIMS = 2


def to_rigid_body(position):
    return rigid_body.RigidBody(center=position, orientation=jnp.zeros(position.shape[0]))


def handle_rigid_body(force_fn):
    def wrapped_force_fn(state, neighbor, exists_mask):
        force = force_fn(state, neighbor, exists_mask)
        return to_rigid_body(force) if state.entity_state.is_rigid_body() and not isinstance(force,rigid_body.RigidBody) else force
    return wrapped_force_fn


def sum_forces(force_list):
    if isinstance(force_list[0], rigid_body.RigidBody):
        return rigid_body.RigidBody(center=jnp.array([f.center for f in force_list]).sum(0), 
                                    orientation=jnp.array([f.orientation for f in force_list]).sum(0))
    return jnp.array(force_list).sum(0)


def sum_force_fns(displacement, force_fns):
    fns = [fn(displacement) for fn in force_fns]
    def force_fn(state, neighbor, exists_mask):
        force = sum_forces([fn(state, neighbor, exists_mask) for fn in fns])
        return force
    return force_fn


def count_masked_values(x, mask, num_values):
    """
    For each integer value in x, count how many times it corresponds 
    to True values in mask.
    
    Args:
        x: array of integers, shape (N,), with values in [0, num_values)
        mask: array of booleans, shape (N,)
        num_values: the range of possible values [0, num_values).
                   Must be a concrete integer (not a traced value).
    
    Returns:
        counts: array of shape (num_values,) where counts[i] is the number
                of times value i appears in x where mask is True
    """
    # Use segment_sum to count occurrences
    # Convert mask to integers (True -> 1, False -> 0)
    counts = jax.ops.segment_sum(
        mask.astype(jnp.int32),
        x,
        num_segments=num_values
    )
    
    return counts

count_masked_values = jax.jit(count_masked_values, static_argnums=(2,))