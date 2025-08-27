import jax.numpy as jnp

from vivarium.environment.components.component import Component
from vivarium.environment.components.utils import SPACE_NDIMS, handle_rigid_body


@handle_rigid_body
def friction_force(state, neighbor, exists_mask):
    """Compute the friction force on the system

    :param state: current state of the system
    :param exists_mask: mask to specify which particles exist
    :return: friction force on the system
    """
    cur_vel = state.entity_state.unified_momentum / state.entity_state.unified_mass
    # stack the mask to give it the same shape as cur_vel (that has 2 rows for forward and angular velocities)
    mask = jnp.stack([exists_mask] * 2, axis=1)
    cur_vel = jnp.where(mask, cur_vel, 0.0)
    return -jnp.tile(state.entity_state.friction, (SPACE_NDIMS, 1)).T * cur_vel


class FrictionComponent(Component):
    def __init__(self, name, precedence, mask_fn):
        super().__init__(name, precedence)
        self.mask_fn = mask_fn

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'mask_fn': self.mask_fn.to_config(state)
        })
        return config

    def get_step_function(self, state, neighbor_manager, key):
        def state_fn(state, neighbor, key):
            mask = self.mask_fn(state)
            force = friction_force(state, neighbor, mask)
            if state.entity_state.is_rigid_body():
                force = force.set(
                    center=state.entity_state.force.center + force.center,
                    orientation=state.entity_state.force.orientation + force.orientation
                )
            else:
                force = state.entity_state.force + force
            entity_state=state.entity_state.set(force=force)
            return state.set(entity_state=entity_state)
        return state_fn
