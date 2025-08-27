import jax.numpy as jnp

from vivarium.environment.components.component import Component
from vivarium.environment.components.utils import to_rigid_body


class ResetForceComponent(Component):

    def get_step_function(self, state, neighbor_manager, key):
        def fn(state, neighbor, key):
            if state.entity_state.is_rigid_body():
                zeros = to_rigid_body(jnp.zeros_like(state.entity_state.force.center))
            else:
                zeros = jnp.zeros_like(state.entity_state.force)
            return state.set(entity_state=state.entity_state.set(force=zeros))
        return fn
