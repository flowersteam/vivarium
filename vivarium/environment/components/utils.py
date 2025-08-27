import logging

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
