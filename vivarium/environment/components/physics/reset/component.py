import jax.numpy as jnp

from vivarium.environment.components.component import Component


class ResetForceComponent(Component):

    def get_step_function(self, state, neighbor_manager, key):
        def fn(state, neighbor, key):
            zeros = jnp.zeros_like(state.entity_state.force)
            return state.set(entity_state=state.entity_state.set(force=zeros))
        return fn
