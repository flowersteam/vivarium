import jax.numpy as jnp

from vivarium.environment.components.component import Component
from vivarium.environment.utils import get_relative_displacement


class ProximityMapComponent(Component):

    def init_state_fn(self, state, neighbor_manager, key):
        fn = self.get_step_function(state, neighbor_manager, key)
        state = fn(state, neighbor_manager.neighbors, key)
        return state

    def update_state_cls(self, state_cls):
        state_cls.__annotations__['distance_map'] = jnp.ndarray
        state_cls.__annotations__['orientation_map'] = jnp.ndarray
        state_cls.distance_map = None
        state_cls.orientation_map = None
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        source_mask = jnp.full(state.entity_state.exists.shape, True, dtype=bool)
        def step_fn(state, neighbors, key):

            all_dist, all_relative_theta = (
                get_relative_displacement(
                    state.entity_state.position,
                    state.entity_state.orientation,
                    source_mask,
                    neighbors.idx,
                    displacement_fn=neighbor_manager.displacement
                )
            )

            return state.set(
                distance_map = all_dist,
                orientation_map = all_relative_theta
            )

        return step_fn
